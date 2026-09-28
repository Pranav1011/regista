"""Local viewer server tests (no model is called). Synthetic data only."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from synthetic_match import MatchSpec, make_match

from regista.agent.serve import create_app
from regista.store import build_store

REPO = Path(__file__).resolve().parent.parent
CONFIGS = [
    json.loads((REPO / "eval" / n).read_text())
    for n in ("params_phase1.json", "calibration_phase1.json", "params_phase2.json")
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("s")
    build_store(
        make_match(MatchSpec(period_s=300.0, fps=5.0)), *CONFIGS, "synthetic", "v1", root / "store"
    )
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>viewer</title>")
    frozen = root / "frozen.json"
    frozen.write_text(json.dumps({"model": "test-model", "think": False}))
    return TestClient(create_app(dist, frozen, root / "store"))


def test_health_static_and_input_validation(client):
    h = client.get("/api/health").json()
    assert h["model"] == "test-model" and h["matches"] == ["synthetic/v1"]
    assert "viewer" in client.get("/").text
    assert (
        client.post("/api/ask", json={"question": "hi", "match": "synthetic/v1"}).status_code == 422
    )
    assert (
        client.post("/api/ask", json={"question": "What shape?", "match": "../etc"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/ask", json={"question": "What shape?", "match": "synthetic/nope"}
        ).status_code
        == 404
    )
