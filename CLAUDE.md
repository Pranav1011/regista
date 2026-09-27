# Regista: project context for Claude Code

Regista turns football match data into tactical insight a coach can question.
Pipeline: match video -> player/ball tracking -> 2D pitch coordinates ->
tactical analytics (formations, passing, pressing, shape) -> an analyst agent
that answers questions and flags tactical changes, grounded in the data.

It is a flagship portfolio project. The bar: every claim measured, every
result reproducible with one command, and a demo that makes sense in 60 seconds.

## Architecture: four layers, one contract

```
vision/     perception + pitch mapping  (Phase 3)  --writes-->  canonical frames
ingest/     open-data adapters          (Phase 1)  --writes-->  canonical frames
analytics/  tactical metrics            (Phase 1)  --reads--->  canonical frames
agent/      MCP server + alerts + evals (Phase 2)  --reads--->  analytics outputs
```

The contract is `src/regista/schema.py`. Rules that follow from it:
- `analytics/` never imports from `vision/` or `ingest/`. It takes a validated
  frame table and returns DataFrames or dataclasses.
- Every producer calls `schema.validate_frames()` before writing.
- Coordinates: metres, pitch-centre origin, home attacks +x in every period.

## Hard rules

1. **Zero cost.** Only free resources: open datasets, local compute, free
   Kaggle/Colab GPU sessions, GitHub Actions on this public repo, Hugging Face
   Spaces free tier. No paid APIs. The agent must work against a local model
   (Ollama) or through an MCP client; keep the LLM provider behind an interface.

2. **License policy.** This repo is AGPL-3.0 (required because we use
   Ultralytics YOLO). Before copying or adapting ANY code, read its LICENSE.
   - AGPL-compatible (MIT, BSD, Apache-2.0, GPL-3.0, AGPL-3.0): allowed. Keep
     the original notice, add `# Adapted from <repo>/<path> (<license>)` at the
     top of the file, and add a row to `THIRD_PARTY_NOTICES.md`.
   - No license file: all rights reserved. Read for ideas only; write an
     original implementation.

3. **Data policy.** Never commit data, video, frames, or model weights.
   - StatsBomb open data: non-commercial use; attribution plus logo required.
   - Metrica sample data: acknowledge the source in anything public.
   - SkillCorner open data: MIT; credit SkillCorner.
   - SoccerNet videos: under NDA. Never redistribute, never use frames in the
     README, demos, or GIFs. Local benchmarking only.
   - Public demo media (README GIFs, demo video, live demo) uses only footage we
     have rights to (our own recordings) or animated tracking-data replays.
     No broadcast clips.

4. **Honesty policy.**
   - Every number in README/docs is produced by a script in `eval/` and is
     reproducible with one command. No hand-typed metrics.
   - No silent fallbacks. If a stage fails, raise. Never substitute synthetic
     or placeholder data in a real run.
   - Synthetic data is allowed only in tests and must be labelled as such.
   - When ground truth does not exist for a claim, say so; do not invent it.

5. **Scope discipline.** Work only on the current phase in `docs/ROADMAP.md`.

## Conventions

- Python 3.11+, `uv`, `ruff` (lint + format), `pytest`, type hints.
- Analytics are pure functions over DataFrames; I/O lives at the edges.
- Every analytics function gets a synthetic known-answer test
  (e.g. a noisy 4-4-2 must be detected as 4-4-2).
- Hold-out discipline: never tune parameters on the data you report on.
- Small commits, conventional-commit messages.

## Commands

```
uv sync --group dev        # install
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

## Layout

```
src/regista/schema.py      canonical frame contract
src/regista/ingest/        Metrica / SkillCorner / StatsBomb adapters (via kloppy)
src/regista/analytics/     kinematics, possession, passes, formations, shape
src/regista/vision/        Phase 3: detection, tracking, teams, homography
src/regista/agent/         Phase 2: MCP server, alerting, evals
eval/                      reproducible evaluation scripts -> eval/reports/
data/                      local only, gitignored
docs/ROADMAP.md            phases and definitions of done
THIRD_PARTY_NOTICES.md     every borrowed piece of code/data and its terms
```
