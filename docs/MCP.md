# Using Regista from an MCP client (Claude Desktop)

Regista ships a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server. It exposes read-only tools over precomputed match stores: formations,
shape, pressing, pass networks, passing options, and detected tactical moments.
Every result carries evidence (match, period, frame range, match clock).

## 1. Build the match stores

```bash
uv sync --group dev --extra agent
uv run regista ingest metrica --game 3           # open data, downloads on first run
uv run regista build-store --source metrica --game 3
```

Stores land in `data/store/<source>/<match_id>/` (local only, never committed).
Repeat for other matches, e.g. `--source skillcorner --match 1886347`.

## 2. Check the server starts

```bash
uv run --extra agent regista mcp
```

It speaks MCP over stdin/stdout, so it waits silently; stop it with Ctrl+C.

## 3. Add it to Claude Desktop

Open Claude Desktop's settings, go to Developer, and edit the config file
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows). Add:

```json
{
  "mcpServers": {
    "regista": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/regista",
        "run", "--extra", "agent", "regista", "mcp"
      ]
    }
  }
}
```

Replace the path with where you cloned the repo, then restart Claude Desktop.
The Regista tools appear in the tools menu. If `uv` is not on the app's `PATH`,
use its absolute path (`which uv`) as `command`.

## Tools

| Tool | What it returns |
|---|---|
| `list_matches` | Matches that have a store |
| `get_match_overview` | Periods and clocks, pass counts, detected moments, typical formations |
| `get_formation` | 5-minute formation windows for a team and phase, with runner-up, margin, and a "close call" flag |
| `get_shape` | Line height, length, width, compactness (median of 5-minute medians) |
| `get_press_stats` | Press intensity (defender within 5 yd of the carrier), overall and by pitch third |
| `get_pass_network` | Pass counts between teammates (inferred from tracking) and mean positions |
| `get_player_passes` | A player's attempts, completions, receptions, and pass ids |
| `get_passing_options` | At one pass: a model success estimate for every option (labelled "model estimate") |
| `find_moments` | Back-line changes, press changes, and line-height shifts from the causal detectors |
| `check_capability` | Whether a topic is supported (xG, shot quality, and player names are not) |

Matches are named `<source>/<match_id>`, e.g. `metrica/3`. Times are match
clocks: `"62:00"`, or `"45+2:00"` for first-half stoppage time. A time outside
the recorded match returns an error message saying which range is valid.

Example prompts:

- "In metrica/3, what formation did the away team use out of possession
  between 60:00 and 75:00? How confident is that label?"
- "When did metrica/3's home team press most, and in which third?"
- "List the tactical moments in metrica/3 and cite the frames."

Data credits: Metrica Sports sample data; SkillCorner open data (MIT).
