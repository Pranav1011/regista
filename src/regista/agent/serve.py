"""Local viewer server: the built static viewer plus free-form questions to the agent.

``regista serve`` serves ``viewer/dist`` and ``POST /api/ask`` (question + match)
answered by the frozen local model through Ollama. The hosted demo has no server
and shows pre-generated answers instead. Binds to localhost only.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from regista import io
from regista.agent.loop import Agent, OllamaProvider
from regista.agent.tools import Toolbox

MAX_QUESTION_CHARS = 500


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    match: str = Field(pattern=r"^[a-z]+/[A-Za-z0-9_-]+$")


def create_app(viewer_dist: Path, frozen_path: Path, store_root: Path | None = None) -> FastAPI:
    frozen = json.loads(Path(frozen_path).read_text())
    toolbox = Toolbox(store_root or io.data_dir() / "store")
    agent = Agent(toolbox, OllamaProvider(frozen["model"], think=frozen.get("think")))
    lock = threading.Lock()  # one question at a time: a single local model
    app = FastAPI(title="Regista local viewer")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "model": frozen["model"], "matches": toolbox.matches()}

    @app.post("/api/ask")
    def ask(body: Ask) -> dict:
        if body.match not in toolbox.matches():
            raise HTTPException(404, f"unknown match {body.match!r}")
        with lock:
            answer = agent.answer(body.question, body.match)
        return answer.model_dump()

    if not (viewer_dist / "index.html").exists():
        raise FileNotFoundError(f"{viewer_dist} has no build; run `npm run build` in viewer/")
    app.mount("/", StaticFiles(directory=viewer_dist, html=True), name="viewer")
    return app
