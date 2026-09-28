# Third-party code, models, and data

Every borrowed piece of code, model, or dataset is listed here with its terms.
"Verified" means someone read the actual LICENSE or terms on the date shown.
Re-check terms before reuse elsewhere.

## Code and models

| Source | License / terms | Status | Use in Regista |
|---|---|---|---|
| [PySport/kloppy](https://github.com/PySport/kloppy) | BSD-3-Clause | verified 2026-09-27 | Dependency: loading and standardising tracking/event data |
| [Friends-of-Tracking-Data-FoTD/LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | MIT | verified 2026-09-27 | Pitch control adapted in `src/regista/analytics/passing_options.py` (from `Metrica_PitchControl.py`, published default parameters); otherwise reference only |
| [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 (Nano–Large). XL/2XL use a separate Platform Model License, so do not use them | verified 2026-09-27 | Phase 3 detector candidate (benchmarked against YOLO) |
| [roboflow/sports](https://github.com/roboflow/sports) | MIT | verified 2026-09-27 | Phase 3 reference (team clustering, pitch keypoints, radar) |
| [roboflow/supervision](https://github.com/roboflow/supervision) | MIT | verified 2026-09-27 | Phase 3 dependency (tracking, annotation) |
| [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) + [TrackLab](https://github.com/TrackingLaboratory/tracklab) | GPL-3.0 (sn-gamestate), MIT (TrackLab) | verified 2026-09-27 | Phase 3 benchmark harness only; keep outside `src/` |
| [duckdb/duckdb](https://github.com/duckdb/duckdb) | MIT | verified 2026-09-27 | Phase 2 match store queries |
| [pydantic/pydantic](https://github.com/pydantic/pydantic) | MIT | verified 2026-09-27 | Phase 2 typed tool inputs and outputs |
| [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | MIT | verified 2026-09-27 | Phase 2 MCP server |
| [ollama/ollama-python](https://github.com/ollama/ollama-python) | MIT | verified 2026-09-27 | Phase 2 local model client |
| [fastapi/fastapi](https://github.com/fastapi/fastapi) | MIT | verified 2026-09-27 | Phase 2 local viewer server (`regista serve`) |
| [encode/uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause | verified 2026-09-27 | Phase 2 local viewer server |
| [vitejs/vite](https://github.com/vitejs/vite) | MIT | verified 2026-09-27 | Phase 2 viewer build tool |
| [microsoft/TypeScript](https://github.com/microsoft/TypeScript) | Apache-2.0 | verified 2026-09-27 | Phase 2 viewer language |
| [Instrument Sans / Instrument Serif](https://fontsource.org/fonts/instrument-sans) via @fontsource | OFL-1.1 | verified 2026-09-27 | Phase 2 viewer typefaces, self-hosted in the build |
| [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) | MIT | verified 2026-09-27 | Pitch plots |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | verified 2026-09-27 | Phase 3 detector candidate. This is why the repo itself is AGPL-3.0 |

## Models (run locally via Ollama, never redistributed)

| Model | License (as shipped with the model) | Status | Use in Regista |
|---|---|---|---|
| llama3.1:8b | Llama 3.1 Community License | verified 2026-09-28 | Phase 2 agent candidate (baseline) |
| qwen3.5:9b | Apache-2.0 | verified 2026-09-28 | Phase 2 agent candidate |
| gemma4:12b | Apache-2.0 | verified 2026-09-28 | Phase 2 agent candidate |
| gpt-oss:20b | Apache-2.0 | verified 2026-09-28 | Phase 2 agent candidate |

## Data

| Source | Terms | Status | Use in Regista |
|---|---|---|---|
| [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | No formal license; README asks for responsible use and acknowledgement in public work | verified 2026-09-27 | Dev + pass-detection validation. Not redistributed |
| [SkillCorner/opendata](https://github.com/SkillCorner/opendata) | MIT | verified 2026-09-27 | Dev + role validation + noisy-data stress test |
| [StatsBomb open data](https://github.com/hudl/open-data) | Public non-commercial use; credit StatsBomb and show their logo | verified 2026-09-27 | Later: formation checks via 360 freeze frames |
| [SoccerNet](https://www.soccer-net.org/data) videos | NDA; no redistribution (copyrighted broadcasts) | verified 2026-09-27 | Local benchmarking only. Never in repo, README, or demos |

Data credits: tracking data from Metrica Sports and SkillCorner; event data
from StatsBomb (where used).
