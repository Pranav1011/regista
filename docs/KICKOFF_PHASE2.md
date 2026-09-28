# Claude Code kickoff: Phase 2

Paste everything below the line into Claude Code from the repo root.
Start in plan mode.

---

Read `CLAUDE.md`, `docs/ROADMAP.md`, `docs/DESIGN.md`, and
`eval/reports/phase1.md` first; CLAUDE.md is binding. Phase 2 goal: an analyst
agent grounded in the Phase 1 analytics, plus a viewer anyone can open in a
browser. No computer vision in this phase.

Propose a plan and wait for my approval. Your plan should also update the
Phase 2 section of `docs/ROADMAP.md` to match this document. Then implement
step by step, committing after each step. Stop at the three checkpoints.

## Design principles
- **Deterministic core, LLM at the edge.** Analytics and moment detectors
  decide what happened. The LLM chooses tools and writes the answer, and it
  never computes a number itself.
- **Evidence-backed answers.** Every answer carries citations (match, period,
  frame range, match clock). The viewer turns each citation into a seek link.
- **Live means causal.** During replay, detectors see only data up to the
  current frame.
- **One tool layer, many clients.** The same Python tool functions back the
  MCP server, the local agent loop, and the eval harness.
- **Zero cost.** The model runs locally via Ollama and the hosted demo is a
  static site. No paid API.
- **Held-out discipline continues.** Design prompts, thresholds, and question
  templates on Metrica games 1–2. Evaluate once on game 3 (corrected labels)
  and on the SkillCorner matches.

## Step 1: Match store (`src/regista/store/`)
- `regista build-store --source metrica --game 3` precomputes everything for
  one match into `data/store/<source>/<match_id>/`:
  - canonical frames and possession
  - v2 passes
  - formation windows with runner-up and margins
  - shape windows and pass networks
  - pitch control at pass moments, with the Platt display probability
- Query the store with DuckDB over parquet. Write a manifest recording
  package version, params hash, source, and creation time.
- Tests: on a synthetic match, reading from the store must give the same
  result as computing directly.

## Step 2: Pressing metrics (`analytics/pressing.py`)
Tools and alerts need this, and it has no ground truth.
- Per frame in opponent possession, compute the nearest defender's distance to
  the ball carrier and the number of defenders within 5 m.
- Per window, compute press intensity: the share of opponent-possession frames
  with a defender within a fixed distance of the carrier, split by pitch third.
- Fix the thresholds a priori and don't tune them. Record them and the
  reasoning in DESIGN.md.
- Sanity check, reported as such and not as accuracy: per-window correlation
  with Metrica CHALLENGE + RECOVERY counts.

## Step 3: Moment detectors (`src/regista/moments/`)
Each detector emits moments with: type, team, start frame, emit frame,
evidence (values before and after), and severity.
- **Back-line change.** Out-of-possession windows only, per the ADR. Fires when
  the back-line count differs from the last stable state for 2 or more
  consecutive windows, with the margin above a threshold.
- **Press change.** Fires when press intensity shifts by more than a threshold
  between trailing windows.
- **Line height shift.** Fires when defensive line height changes by more than
  a threshold between trailing windows.

Rules:
- Streaming implementation: detectors consume frames in order. Offline results
  come from running the streaming version over the whole match.
- Causality test: output up to frame t must be identical whether or not the
  data after t exists.
- Choose thresholds on games 1–2 to hit a readable alert rate (propose a
  target in your plan), then freeze them.
- Report alert counts for game 3 and every SkillCorner match, plus detection
  delay: emit frame minus start frame.
- Known-answer tests on synthetic matches:
  - an injected back four → back five switch, a press drop, and a line drop
    must each fire at the right time
  - a clean match must fire nothing

**Checkpoint A:** show me game 3's full alert list (clock, type, team,
evidence) and alert counts per match.

## Step 4: Tool layer + MCP server (`src/regista/agent/`)
- `tools.py`: typed (pydantic), read-only functions over the store:
  - `get_match_overview`
  - `get_formation(team, phase, t_from, t_to)`
  - `get_shape`, `get_press_stats`
  - `get_pass_network(team, t_from, t_to, top_k)`
  - `get_player_passes`
  - `get_passing_options(pass_id)`
  - `find_moments(type, team, t_from, t_to)`
- Every result includes `evidence` (a list of match_id, period, frame range,
  and clock) plus reliability signals: margin, relative_margin, and the note
  that exact template labels are noisy at 5 minutes.
- Times are given as match clock plus period, and are validated. Out-of-range
  times raise a clear error.
- Unsupported questions get an explicit "not available" result. This covers
  xG, shot quality, and player names (Metrica is anonymised).
- `mcp_server.py`: a stdio MCP server exposing the tools. `docs/MCP.md`
  explains how to connect it to Claude Desktop. Add a test that lists the
  tools and calls one.

## Step 5: Agent loop (`agent/loop.py`)
- Provider interface with an Ollama implementation. No paid API.
- System prompt: answer only from tool results, cite evidence, say what cannot
  be known, and never state a number that did not come from a tool.
- Output schema: `{answer_text, citations[], tools_used[], caveats[]}`.
- Number-grounding check (deterministic, after generation): every number in
  `answer_text` must appear in the tool outputs, allowing for rounding.
  - On failure, retry once with the violation noted.
  - If it still fails, return the answer marked "unverified".
  - Never silently skip the check.
- Model choice: run the Step 6 dev set on 2–3 small tool-calling models that
  run on this laptop via Ollama. Pick one on accuracy and latency, and record
  the choice as an ADR.

## Step 6: Agent eval (`eval/agent/`)
- The question bank is generated from templates, with answers computed from
  the store (never hand-written). Categories:
  - **lookup:** formation in a window, line height in a half, top pass pairs
  - **comparison:** e.g. which team pressed more after 60'
  - **temporal:** e.g. when did away switch to a back five
  - **multi-step:** needs two tools combined
  - **unanswerable:** xG, player names, things Regista doesn't model, times
    outside the match. The correct behaviour is to decline.
  - **reliability:** questions about low-margin windows. The answer must flag
    the ambiguity.
- Split: develop prompts and templates on games 1–2. Run the frozen prompt and
  model once on game 3 plus SkillCorner. Report both splits.
- Metrics:
  - answer accuracy per category (exact match or within tolerance)
  - tool-selection accuracy
  - citation validity: the cited frames contain the evidence
  - number-grounding rate
  - abstention accuracy on unanswerable questions
  - latency p50 and p95
- Written match summaries: LLM-as-judge with a rubric (faithful to the tool
  data, covers the flagged moments, states caveats).
  - Run pairwise judgments in both orders and report order agreement
    (position bias).
  - I will hand-label 30 items to measure judge–human agreement. Build a small
    CLI labelling script for that.
- `uv run python eval/agent/run.py --split dev|test` regenerates
  `eval/reports/phase2_agent.md`.

**Checkpoint B:** show me dev-set results per category and the ten worst
failures before the test run.

## Step 7: Viewer v1 (`viewer/`)
A static web app (Vite + TypeScript, canvas rendering). It reads JSON produced
by `regista export-viewer --game 3`.
- **Replay:** frames downsampled to 10 fps, quantised and compressed,
  interpolated for smooth playback. Play, pause, scrub, and speed controls.
- **Pitch:**
  - red home, blue away, and the ball; each player also carries a text or
    shape cue so team identity isn't colour alone
  - optional back-line shape lines
  - passing-option arrows at pass moments with the Platt probability,
    labelled "model estimate"
- **Formation cards:** per team and phase, with runner-up and margin. Show
  "close call" when the margin is below the median.
- **Alert timeline:** markers from the causal detectors; click to seek. During
  playback, each alert appears only when its emit frame is reached.
- **Chat panel:**
  - Hosted build: pre-generated agent answers to about 10 suggested questions,
    labelled "pre-generated by the Regista agent (model, date)". Every
    citation seeks the replay.
  - Local mode (`regista serve`, FastAPI + Ollama): free-form questions.
- **Credits** visible in the UI: Metrica Sports, SkillCorner, LaurieOnTracking.
- Report bundle size, data size, and load time.

## Step 8: Deploy + docs
- Add a GitHub Actions workflow that builds the viewer and publishes it to
  GitHub Pages. Do not enable Pages or create any public surface yourself.
  Tell me when it's ready and I'll enable it.
- README:
  - The Demo section gets the live link and a 20–30 s GIF of the viewer
    (tracking replay only). If capturing the GIF isn't possible here, tell me
    and I'll record it.
  - Extend the generated results block with Phase 2 headline numbers.
- DESIGN.md ADRs for: store design, causal detectors, number-grounding check,
  model choice, and the static hosted demo.

**Checkpoint C:** show me a local preview of the build before touching the
README.

## Acceptance criteria
- CI is green: causality tests, known-answer tests for tools and detectors,
  and the MCP server test.
- Every Phase 2 number is reproducible with one command per eval. The test
  split runs once, with prompt and model frozen.
- No paid API is used anywhere, and the hosted demo is static.
- `THIRD_PARTY_NOTICES.md` covers every new dependency, with its license
  checked.
