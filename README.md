# Regista

Football match intelligence: from match video to tactical insight a coach can
ask questions about.

> **Status:** Phase 1, tactical analytics on open tracking data. Early
> development; every metric below comes from `eval/` and is reproducible.

## What it does (planned)

- Tracks players and the ball from video and maps them onto a 2D pitch.
- Measures formations, roles, passing networks, pressing, and team shape
  over time.
- An analyst agent answers questions ("how did their press change after
  60'?") and flags tactical shifts live. Every answer cites the frames it is
  based on.

## Architecture

```
match video ─▶ perception ─▶ pitch mapping ─┐
                                            ├─▶ canonical frames ─▶ tactical analytics ─▶ analyst agent
open tracking data ─▶ ingest adapters ──────┘
```

Every source, whether open data today or CV output later, is converted to one
canonical schema (`src/regista/schema.py`). The analytics layer never knows
where its data came from. Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quick start

```bash
git clone https://github.com/Pranav1011/regista && cd regista
uv sync --group dev
uv run pytest
```

## Demo

Demo video and live link land with Phase 2.

## Results

Populated by `eval/phase1.py` once Phase 1 lands.

## Credits and data terms

Built on open-source work and open data. Full list with licenses:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Tracking data courtesy of
Metrica Sports and SkillCorner. No match video is stored in or distributed
from this repository.

## License

AGPL-3.0 (see `LICENSE`), matching Ultralytics YOLO's license.
