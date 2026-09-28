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

## Design Context

### Users
Football coaches and analysts reviewing a match, and recruiters or engineers
opening a portfolio demo from a README link. They want to replay the match on a
2D pitch, see each team's shape and how clear-cut it is, jump to the tactical
moments Regista flagged, and read agent answers whose evidence they can check by
seeking the replay. They arrive curious but sceptical: the interface must earn
trust quickly, on a laptop, often in daylight.

### Brand Personality
An analyst's instrument: **precise, calm, trustworthy**. The interface should
evoke confidence in the evidence rather than excitement. It states uncertainty
plainly ("close call", "model estimate") and never oversells a number.

### Aesthetic Direction
- Light theme: tinted off-white paper, a muted grass-green pitch, ink-dark text
  tinted toward the brand hue (no pure black or white).
- Dense but legible, like a broadcast analysis desk or a scientific instrument:
  numbers and evidence front and centre, restrained decoration, clear hierarchy.
- Team identity: red home, blue away, always paired with a non-colour cue
  (filled vs ringed markers and shirt numbers).
- Anti-references: generic SaaS dashboards (card grids, big KPI numbers,
  gradient accents); neon esports or betting (glow on black, aggressive motion);
  FIFA or video-game HUDs (glossy chrome); toy or playful styling (bubbly
  shapes, emoji, bouncy motion).
- Accessibility: WCAG 2.2 AA contrast, full keyboard control of replay and
  timeline, visible focus, reduced-motion support, nothing conveyed by colour
  alone.

### Design Principles
1. **Evidence first.** Every claim on screen links to the moment it came from;
   citations seek the replay.
2. **Honest uncertainty.** Margins, "close call", and "model estimate" are shown
   where they apply, in the same visual weight as the claim.
3. **Causal by default.** During playback, show only what was knowable at that
   time: cards and alerts appear when their data exists.
4. **Quiet surface, strong structure.** Typography and spacing carry the
   hierarchy; colour is reserved for teams and for state.
5. **Never by colour alone.** Every colour-coded meaning has a shape, text, or
   position cue too.
