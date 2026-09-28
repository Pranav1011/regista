import "./style.css";
import {
  type Alert, type AnswerItem, type Card, type Instant, type MatchData, before, clock, indexAt, instantAt,
  loadIndex, loadMatch, parseClock,
} from "./data";
import { Pitch } from "./pitch";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {},
  ...children: (Node | string)[]) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

const TYPE_TEXT: Record<Alert["type"], string> = {
  back_line_change: "Back line", press_change: "Press", line_height_shift: "Line height",
};
const OPTION_WINDOW_S = 1.2; // show passing options this close to a pass moment

const pitch = new Pitch($<HTMLCanvasElement>("pitch"));
const state = { data: null as MatchData | null, pos: 0, playing: false, speed: 4, last: 0,
                showAll: false, lines: true, options: true, activeQuestion: -1 };

function now(): Instant {
  return instantAt(state.data!.manifest, Math.floor(state.pos));
}

function describe(a: Alert): string {
  if (a.type === "back_line_change") return `back ${a.before} → ${a.after}`;
  if (a.type === "press_change") return `press intensity ${a.before.toFixed(2)} → ${a.after.toFixed(2)}`;
  return `line height ${a.before.toFixed(1)} m → ${a.after.toFixed(1)} m`;
}

function seek(at: Instant) {
  state.pos = indexAt(state.data!.manifest, at);
  render(true);
}

// ---- cards: the latest usable trailing-window state at or before now, per team
function currentCards(): Record<"home" | "away", Card | null> {
  const t = now();
  const out: Record<"home" | "away", Card | null> = { home: null, away: null };
  for (const c of state.data!.cards) {
    if (c.period !== t.period || !c.usable || c.t > t.t) continue;
    if (!out[c.team] || out[c.team]!.t < c.t) out[c.team] = c;
  }
  return out;
}

function renderCards(cards: Record<"home" | "away", Card | null>) {
  const box = $("cards");
  const nodes: Node[] = [el("span", { className: "col-head" }), el("span", { className: "col-head", textContent: "Without the ball" }),
    el("span", { className: "col-head", textContent: "With the ball" })];
  for (const team of ["home", "away"] as const) {
    nodes.push(el("span", { className: "row-head" },
      el("span", { className: `key key-${team}` }), team === "home" ? "Home" : "Away"));
    for (const phase of ["out", "in"] as const) {
      const s = cards[team]?.[phase];
      if (!s) {
        nodes.push(el("span", { className: "cell empty", textContent: "Not enough of this half yet" }));
        continue;
      }
      const cell = el("span", { className: "cell" },
        el("span", { className: "shape", textContent: s.label }),
        el("span", { className: "detail", textContent: `runner-up ${s.runner_up}` }),
        el("span", { className: "detail", textContent: `margin ${s.margin.toFixed(3)}` }));
      if (s.close_call) cell.append(el("span", { className: "close", textContent: "close call" }));
      nodes.push(cell);
    }
  }
  box.replaceChildren(...nodes);
}

function renderAlerts() {
  const t = now();
  const list = $("alerts");
  const alerts = [...state.data!.alerts]
    .filter((a) => state.showAll || before({ period: a.period, t: a.emit_t }, t))
    .sort((a, b) => (state.showAll ? b.severity - a.severity : (b.period - a.period) || (b.emit_t - a.emit_t)));
  if (!alerts.length) {
    list.replaceChildren(el("li", { className: "empty", textContent: "None yet. Moments appear when their evidence is complete." }));
    return;
  }
  list.replaceChildren(...alerts.map((a) => {
    const future = !before({ period: a.period, t: a.emit_t }, t);
    const btn = el("button", { type: "button", onclick: () => seek({ period: a.period, t: a.emit_t }) },
      el("span", { className: "when", textContent: a.emit_clock }),
      el("span", {},
        el("span", { className: "what" }, el("span", { className: "team-mark", textContent: a.team === "home" ? "Home " : "Away " }),
          `${TYPE_TEXT[a.type]}: ${describe(a)}`),
        el("span", { className: "evidence", textContent: `estimated start ${a.start_clock} · severity ${a.severity.toFixed(2)}` })));
    btn.querySelector(".team-mark")!.setAttribute("data-team", a.team);
    return el("li", { className: future ? "future" : "" }, btn);
  }));
}

function renderTimeline() {
  const { manifest, alerts, total } = state.data!;
  const box = $("timeline");
  const t = now();
  const ticks: Node[] = manifest.periods.slice(1).map((p) => {
    const split = el("span", { className: "period-split" });
    split.style.left = `${(100 * p.first_index) / total}%`;
    return split;
  });
  for (const a of alerts) {
    const future = !before({ period: a.period, t: a.emit_t }, t);
    if (future && !state.showAll) continue;
    const i = indexAt(manifest, { period: a.period, t: a.emit_t });
    const b = el("button", { type: "button", className: "tick", title: `${a.emit_clock} ${TYPE_TEXT[a.type]} (${a.team})`,
      ariaLabel: `${a.emit_clock}: ${a.team} ${TYPE_TEXT[a.type]}`, onclick: () => seek({ period: a.period, t: a.emit_t }) });
    b.style.left = `${(100 * i) / total}%`;
    b.dataset.team = a.team;
    if (future) b.dataset.future = "";
    ticks.push(b);
  }
  box.replaceChildren(...ticks);
}

function renderAnswers() {
  const { answers } = state.data!;
  const box = $("questions");
  const prov = $("ask-provenance");
  $("answer").replaceChildren();
  if (!answers) {
    prov.textContent = "No pre-generated answers for this match. Run `regista serve` for live questions.";
    box.replaceChildren();
    return;
  }
  prov.textContent = `Pre-generated by the Regista agent (${answers.model}, ${answers.generated}). Citations seek the replay.`;
  box.replaceChildren(...answers.items.map((item, k) => el("button", {
    type: "button", textContent: item.question, onclick: () => showAnswer(k),
  })));
  state.activeQuestion = -1;
}

function showAnswer(k: number) {
  renderAnswer(state.data!.answers!.items[k]);
  state.activeQuestion = k;
  $("questions").querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === k)));
}

function renderAnswer(item: AnswerItem) {
  const cites = el("div", { className: "cites" }, ...item.citations.map((c) =>
    el("button", { type: "button", textContent: `${c.clock_start}–${c.clock_end}`,
      ariaLabel: `Seek to ${c.clock_start}`, onclick: () => seek(parseClock(c.clock_start, c.period)) })));
  const status = el("p", { className: "status", textContent: item.status === "verified"
    ? "Every number checked against the tool results." : "Unverified: a number could not be matched to the tool results." });
  status.dataset.status = item.status;
  $("answer").replaceChildren(el("p", { textContent: item.answer_text }), cites, status,
    el("ul", { className: "caveats" }, ...item.caveats.map((c) => el("li", { textContent: c }))));
}

function overlay() {
  const t = now();
  const cards = currentCards();
  const moment = state.options
    ? state.data!.passes.find((p) => p.period === t.period && Math.abs(p.t - t.t) <= OPTION_WINDOW_S) ?? null
    : null;
  return {
    cards,
    ov: {
      backLines: {
        home: state.lines ? cards.home?.out ? Number(cards.home.out.label.split("-")[0]) : null : null,
        away: state.lines ? cards.away?.out ? Number(cards.away.out.label.split("-")[0]) : null : null,
      },
      moment,
    },
  };
}

let lastPanelsAt = -1;
function render(forcePanels = false) {
  const data = state.data;
  if (!data) return;
  const i = Math.floor(state.pos);
  const t = now();
  const { cards, ov } = overlay();
  pitch.draw(data, i, state.pos - i, ov);
  $("clock").textContent = clock(t.period, t.t);
  $("period-label").textContent = t.period === 1 ? "First half" : "Second half";
  const scrub = $<HTMLInputElement>("scrub");
  scrub.value = String(i);
  scrub.setAttribute("aria-valuetext", clock(t.period, t.t));
  const second = Math.floor(state.pos / data.manifest.fps);
  if (forcePanels || second !== lastPanelsAt) {
    lastPanelsAt = second;
    renderCards(cards);
    renderAlerts();
    renderTimeline();
  }
}

function tick(ts: number) {
  if (state.playing && state.data) {
    const dt = state.last ? (ts - state.last) / 1000 : 0;
    state.pos = Math.min(state.pos + dt * state.data.manifest.fps * state.speed, state.data.total - 1);
    if (state.pos >= state.data.total - 1) setPlaying(false);
    render();
  }
  state.last = ts;
  requestAnimationFrame(tick);
}

function setPlaying(on: boolean) {
  state.playing = on;
  const b = $("play");
  b.textContent = on ? "Pause" : "Play";
  b.setAttribute("aria-label", on ? "Pause" : "Play");
}

async function open(id: string, split: string) {
  $("loading").hidden = false;
  setPlaying(false);
  state.data = await loadMatch(id);
  const { manifest, total } = state.data;
  const tag = $("match-split");
  tag.textContent = split;
  tag.dataset.split = split;
  $("credits-data").textContent = `${manifest.credits[manifest.source] ?? ""}.`;
  const scrub = $<HTMLInputElement>("scrub");
  scrub.max = String(total - 1);
  state.pos = 0;
  $("loading").hidden = true;
  renderAnswers();
  render(true);
}

/** Local mode (`regista serve`): free-form questions to the local agent. */
async function enableLocalAsk() {
  const r = await fetch("api/health").catch(() => null);
  if (!r || !r.ok || !(r.headers.get("content-type") ?? "").includes("json")) return;
  const health = (await r.json()) as { model: string };
  const form = $<HTMLFormElement>("ask-form");
  form.hidden = false;
  form.onsubmit = async (e) => {
    e.preventDefault();
    const input = $<HTMLInputElement>("ask-input");
    const question = input.value.trim();
    if (!question || !state.data) return;
    const m = state.data.manifest;
    $("answer").replaceChildren(el("p", { className: "status", textContent: `Asking ${health.model} locally…` }));
    const res = await fetch("api/ask", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, match: `${m.source}/${m.match_id}` }) });
    if (!res.ok) {
      $("answer").replaceChildren(el("p", { className: "status", textContent: `The agent could not answer (${res.status}).` }));
      return;
    }
    renderAnswer((await res.json()) as AnswerItem);
  };
}

async function init() {
  const index = await loadIndex();
  const select = $<HTMLSelectElement>("match-select");
  select.replaceChildren(...index.map((m) => el("option", { value: m.id, textContent: m.title })));
  const pick = (id: string) => open(id, index.find((m) => m.id === id)!.split);
  select.onchange = () => pick(select.value);
  $("play").onclick = () => setPlaying(!state.playing);
  $<HTMLSelectElement>("speed").onchange = (e) => { state.speed = Number((e.target as HTMLSelectElement).value); };
  $<HTMLInputElement>("scrub").oninput = (e) => { state.pos = Number((e.target as HTMLInputElement).value); render(true); };
  $<HTMLInputElement>("show-all").onchange = (e) => { state.showAll = (e.target as HTMLInputElement).checked; render(true); };
  $<HTMLInputElement>("toggle-lines").onchange = (e) => { state.lines = (e.target as HTMLInputElement).checked; render(true); };
  $<HTMLInputElement>("toggle-options").onchange = (e) => { state.options = (e.target as HTMLInputElement).checked; render(true); };
  document.addEventListener("keydown", (e) => {
    if (!state.data || (e.target as HTMLElement).closest("select, input[type=range]")) return;
    const step = (e.shiftKey ? 30 : 5) * state.data.manifest.fps;
    if (e.key === " ") { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.key === "ArrowRight") { state.pos = Math.min(state.pos + step, state.data.total - 1); render(true); }
    else if (e.key === "ArrowLeft") { state.pos = Math.max(state.pos - step, 0); render(true); }
  });
  // deep links: ?match=metrica-3&t=62:00 (a clock of 45:00 or later means the second half)
  const params = new URLSearchParams(location.search);
  const wanted = index.find((m) => m.id === params.get("match"));
  const first = wanted ?? index.find((m) => m.split === "held out") ?? index[0];
  select.value = first.id;
  await pick(first.id);
  const at = params.get("t");
  if (at) {
    const stoppage = /^45\+/.test(at);
    const minutes = Number(at.split(/[:+]/)[0]);
    seek(parseClock(at, stoppage || minutes < 45 ? 1 : 2));
  }
  requestAnimationFrame(tick);
  await enableLocalAsk();
}

init().catch((err) => {
  $("loading").textContent = `Could not load the match data: ${err.message}`;
});
