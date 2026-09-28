"""Local agent loop: a tool-calling model answers questions from Regista tools only.

The model chooses tools and writes the answer; it never computes numbers. After
generation a deterministic check requires every number in the answer to appear
in the tool outputs (or the question). On failure the model gets one retry with
the violation noted; if it still fails, the answer is returned marked
"unverified". The check is never skipped. Citations and caveats are collected
deterministically from the tool results, so the model cannot invent evidence.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from regista.agent.grounding import check_numbers
from regista.agent.tools import TOOL_NAMES, Toolbox, ToolError

SYSTEM_PROMPT = """You are Regista, a football analyst. You answer questions about one match \
using only the Regista tools.

Rules:
- Get every fact from a tool call. Never compute, estimate, or guess a number yourself; \
only repeat numbers exactly as the tools return them (you may round them).
- Times are match clocks such as "62:00". "45:00" is the start of the second half; \
first-half stoppage time is written "45+2:00".
- If a question is about something Regista does not model (for example xG, shot quality, \
player names, the score, injuries), call check_capability and then say plainly that it is \
not available. Do not guess.
- When you report a formation, also report its margin and say when it is a close call; \
exact formation labels are noisy.
- Passing-option values are model estimates, not facts.
- To compare the two teams, call the tool once for each team and compare the results.
- For "when" questions about tactical changes, use find_moments; it returns the clock of \
each detected moment.
- To find which player passed most, use get_team_passing.
- Do not describe values with qualitative words such as "high", "low", "intense", or \
"clear-cut" unless a tool gives that label; state the values and say which is higher.
- For a "why" question, or any question that states a fact, first check that fact with \
the tools (the team, the time, the player, the formation). If the tools contradict it, say \
that the premise is not supported and give what the tools show instead (the right team, \
time, player, or formation). The match has only two halves.
- If a tool returns an error, fix the arguments and call it again, or explain in plain words \
what could not be computed. Never reply with the raw error.
- Keep answers short: two to four sentences, with the match clock of what you describe.
"""

MALFORMED_PROMPT = (
    "Your last reply could not be parsed. Call one tool at a time with valid JSON arguments, "
    "or answer in plain text."
)

ERROR_RETRY_PROMPT = (
    "Your reply repeated a tool error instead of answering. Call the tool again with corrected "
    "arguments, or explain in plain words what could not be computed and why."
)
MAX_ERROR_RETRY_ROUNDS = 3

RETRY_PROMPT = (
    "Your answer contains numbers that do not appear in the tool results: {numbers}. "
    "Rewrite the answer using only numbers that appear in the tool results (you may round "
    "them), or remove those numbers. Do not call more tools."
)


class ProviderError(RuntimeError):
    """The model server rejected or could not parse the model's output."""


@dataclass
class ProviderReply:
    content: str
    tool_calls: list[tuple[str, dict]]
    duration_s: float
    truncated: bool = False  # generation stopped at the length/context limit


class Provider(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ProviderReply: ...


class OllamaProvider:
    """Chat through a local Ollama server. No paid API."""

    def __init__(
        self,
        model: str,
        host: str | None = None,
        think: bool | str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        keep_alive: str = "10m",
        num_ctx: int = 16384,  # the system prompt, tool schemas, and results exceed 4k tokens
    ):
        import ollama

        self.name = model
        self.think = think
        self.keep_alive = keep_alive
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self.client = ollama.Client(host=host)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ProviderReply:
        import ollama

        start = time.perf_counter()
        try:
            r = self.client.chat(
                model=self.name,
                messages=messages,
                tools=tools,
                think=self.think,
                options=self.options,
                keep_alive=self.keep_alive,
            )
        except ollama.ResponseError as e:
            raise ProviderError(str(e)) from e
        calls = [
            (c.function.name, dict(c.function.arguments or {}))
            for c in (r.message.tool_calls or [])
        ]
        return ProviderReply(
            r.message.content or "",
            calls,
            time.perf_counter() - start,
            truncated=r.done_reason == "length",
        )

    def unload(self) -> None:
        """Free the model's memory (used between agent and judge runs)."""
        self.client.generate(model=self.name, prompt="", keep_alive=0)


def _arg_model(fn) -> type[BaseModel]:
    hints = get_type_hints(fn)
    fields = {}
    for name, p in inspect.signature(fn).parameters.items():
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (hints[name], default)
    return create_model(f"{fn.__name__}_args", **fields)


def tool_schemas(toolbox: Toolbox) -> tuple[list[dict], dict[str, tuple[Any, type[BaseModel]]]]:
    """Function-calling schemas for every tool, and name -> (callable, argument model)."""
    schemas, registry = [], {}
    for name in TOOL_NAMES:
        fn = getattr(toolbox, name)
        model = _arg_model(fn)
        params = model.model_json_schema()
        params.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (fn.__doc__ or "").strip(),
                    "parameters": params,
                },
            }
        )
        registry[name] = (fn, model)
    return schemas, registry


class AgentAnswer(BaseModel):
    question: str
    match: str
    answer_text: str
    citations: list[dict]
    tools_used: list[dict]
    caveats: list[str]
    status: str  # "verified" or "unverified"
    ungrounded_numbers: list[str]
    retried: bool
    model: str
    latency_s: float


@dataclass
class _Trace:
    outputs: list[Any] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    provider_errors: list[str] = field(default_factory=list)


def _for_model(result: Any, seen_notes: list[str]) -> Any:
    """What the model sees of a tool result: evidence and already-seen notes are dropped.

    Evidence is collected into the citations separately, and repeated notes waste
    context; the numbers the model may cite are all kept.
    """
    if not isinstance(result, dict):
        return result
    out = {k: v for k, v in result.items() if k != "evidence"}
    if "notes" in out:
        out["notes"] = [n for n in out["notes"] if n not in seen_notes]
    if "moments" in out:
        out["moments"] = [{k: v for k, v in m.items() if k != "evidence"} for m in out["moments"]]
    return out


def echoes_error(content: str, errors: list[str]) -> bool:
    """True if a reply is empty, a JSON error object, or a tool error message repeated verbatim."""
    c = content.strip()
    if not c:
        return True
    if c.startswith("{"):
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            return True
    return any(c.strip("\"' .") == e.strip("\"' .") for e in errors if e)


class Agent:
    def __init__(self, toolbox: Toolbox, provider: Provider, max_rounds: int = 6):
        self.toolbox = toolbox
        self.provider = provider
        self.max_rounds = max_rounds
        self.schemas, self.registry = tool_schemas(toolbox)

    def _run_tool(self, name: str, args: dict, trace: _Trace) -> str:
        seen_notes = list(trace.caveats)
        if name not in self.registry:
            result: Any = {"error": f"unknown tool {name!r}; tools: {list(self.registry)}"}
        else:
            fn, model = self.registry[name]
            try:
                parsed = model(**args)
                out = fn(**parsed.model_dump())
                result = out.model_dump() if isinstance(out, BaseModel) else out
            except (ToolError, ValidationError, TypeError) as e:
                result = {"error": str(e)}
        trace.outputs.append(result)
        trace.tools_used.append(
            {
                "tool": name,
                "arguments": args,
                "error": "error" in result if isinstance(result, dict) else False,
            }
        )
        if isinstance(result, dict):
            for ev in result.get("evidence", []) or []:
                if ev not in trace.citations:
                    trace.citations.append(ev)
            for m in result.get("moments", []) or []:
                for ev in m.get("evidence", []):
                    if ev not in trace.citations:
                        trace.citations.append(ev)
            for note in result.get("notes", []) or []:
                if note not in trace.caveats:
                    trace.caveats.append(note)
            if result.get("available") is False and result.get("reason") not in trace.caveats:
                trace.caveats.append(result["reason"])
        return json.dumps(_for_model(result, seen_notes), default=str)

    def _chat(self, messages: list[dict], tools: list[dict] | None, trace: _Trace) -> ProviderReply:
        """One model turn; a malformed reply gets one retry with the problem stated."""
        try:
            return self.provider.chat(messages, tools=tools)
        except ProviderError as e:
            trace.provider_errors.append(str(e))
            messages.append({"role": "user", "content": MALFORMED_PROMPT})
            return self.provider.chat(messages, tools=tools)

    def _tool_rounds(self, messages: list[dict], trace: _Trace, rounds: int) -> tuple[str, bool]:
        """Let the model call tools for up to ``rounds`` turns; return (final text, truncated)."""
        for _ in range(rounds):
            reply = self._chat(messages, self.schemas, trace)
            if not reply.tool_calls:
                return reply.content, reply.truncated
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"function": {"name": n, "arguments": a}} for n, a in reply.tool_calls
                    ],
                }
            )
            for name, args in reply.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": self._run_tool(name, args, trace),
                    }
                )
        reply = self._chat(messages, None, trace)
        return reply.content, reply.truncated

    def answer(self, question: str, match: str) -> AgentAnswer:
        start = time.perf_counter()
        trace = _Trace()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Match: {match}\nQuestion: {question}"},
        ]
        couldnt_compute = False
        try:
            content, truncated = self._tool_rounds(messages, trace, self.max_rounds)
        except ProviderError as e:
            content, truncated, couldnt_compute = "", False, True
            content = f"I couldn't compute this: the model's reply could not be parsed ({e})."
            trace.caveats.append("The model's reply could not be parsed; no answer was computed.")
        errors = [o["error"] for o in trace.outputs if isinstance(o, dict) and "error" in o]
        if not couldnt_compute and echoes_error(content, errors):
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": ERROR_RETRY_PROMPT},
            ]
            try:
                content, truncated = self._tool_rounds(messages, trace, MAX_ERROR_RETRY_ROUNDS)
            except ProviderError:
                content = ""
            errors = [o["error"] for o in trace.outputs if isinstance(o, dict) and "error" in o]
            if echoes_error(content, errors):
                couldnt_compute = True
                reason = errors[-1] if errors else "the model gave no answer"
                content = f"I couldn't compute this: {reason}."
                trace.caveats.append(
                    "A tool returned an error and the answer could not be computed."
                )

        sources = [*trace.outputs, question, match]
        check = check_numbers(content, sources)
        retried = False
        if not check.grounded and not couldnt_compute:
            retried = True
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": RETRY_PROMPT.format(numbers=", ".join(check.ungrounded)),
                },
            ]
            try:
                reply = self._chat(messages, None, trace)
                content, truncated = reply.content, reply.truncated
            except ProviderError:
                pass  # keep the first answer; it stays unverified
            check = check_numbers(content, sources)
        if truncated:
            trace.caveats.append("The answer was cut off at the model's length limit.")
        return AgentAnswer(
            question=question,
            match=match,
            answer_text=content.strip(),
            citations=trace.citations,
            tools_used=trace.tools_used,
            caveats=trace.caveats,
            status="verified"
            if check.grounded and not truncated and not couldnt_compute
            else "unverified",
            ungrounded_numbers=check.ungrounded,
            retried=retried,
            model=self.provider.name,
            latency_s=round(time.perf_counter() - start, 2),
        )
