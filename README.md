# MarketCanvas-Env

A deterministic 800×600 design canvas with text, shapes, and image placeholders.
Gymnasium and MCP share the same state, actions, and reward function.

## 1. Set up with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the
following from the repository root. All commands below assume a macOS/Linux shell.

```sh
uv sync --locked --group dev
```

The project selects Python 3.12 through `.python-version`. This command creates
`.venv` and installs the project and development dependencies from `uv.lock`.
Use `uv run` for project commands; activating the environment is optional.
The demo and example scripts need no model account or API key.

## 2. Run the demo

```sh
uv run --locked python demo.py
```

The demo builds a fixed Summer Sale banner and prints the actions, final state,
and reward breakdown. The final reward is `1.0`.

To also save an image:

```sh
uv run --locked python demo.py --png artifacts/demo/banner.png
```

Open the PNG to inspect the canvas. Add `--json` for machine-readable output.
The demo uses `data/tasks/summer_sale.yaml`; use the scripts below for other tasks.

## 3. Run the examples

Five tasks live in [data/tasks/](data/tasks/). Each has a successful solution and
two reward-hacking examples in [data/trajectories/](data/trajectories/).

```sh
# Print rewards for all 15 trajectories
uv run --locked python scripts/run_examples.py

# Run every trajectory for one task
uv run --locked python scripts/run_examples.py --task two_column

# Run one trajectory and save its PNG and JSON report
uv run --locked python scripts/run_examples.py \
  --task webinar --trajectory missing_cta --output-dir artifacts/reward
```

JSON reports include actions, final state, and reward details. Files are written
under `<output-dir>/<task>/<trajectory>.png` and `.json`; matching files are overwritten.

To replay a saved solution through the actual MCP server:

```sh
uv run --locked python scripts/mcp_client.py --task two_column
```

This script starts the server, replays `well_done`, prints the result, and exits.
Use `--trajectory` to select another saved run. It is a protocol check; an interactive
LLM client chooses its own actions using the connections below.

## 4. Run the MCP server

```sh
uv run --locked python -m marketcanvas_env.mcp_server
```

The server waits for MCP messages over **stdio**. It has no HTTP address, port,
or interactive text prompt. Stop this standalone process with Ctrl+C.

**Codex and Claude Code start their own server process.** You can skip the standalone
command when connecting either client. Each server process owns a separate canvas.

| Tool | Purpose |
| --- | --- |
| `get_canvas_state` | Read elements, task requirements, and episode progress |
| `execute_action` | Add, move, update, delete, or finish |
| `get_current_reward` | Inspect the current reward breakdown |
| `reset_canvas` | Start a new episode with default or supplied task constraints |

MCP currently returns semantic JSON. PNG export is available through the demo and
example script; `render_canvas` is not implemented yet.

## 5. Connect Codex

With Codex CLI installed and signed in, register the server **from the repository root**:

```sh
codex mcp add marketcanvas -- "$PWD/.venv/bin/python" -m marketcanvas_env.mcp_server
codex mcp list
codex
```

The shell expands `$PWD` to your checkout's absolute path. In the new Codex session,
use `/mcp` to inspect the connection. For the Codex IDE extension, restart the extension
and open a new conversation; it shares MCP configuration on the same host.
See the [official Codex MCP guide](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

## 6. Connect Claude Code

With Claude Code installed and signed in, run **from the repository root**:

```sh
claude mcp add --transport stdio --scope local marketcanvas \
  -- "$PWD/.venv/bin/python" -m marketcanvas_env.mcp_server
claude mcp list
claude
```

This registers the server privately for this project. In Claude Code, use `/mcp`
to inspect the connection. See the [official Claude Code MCP guide](https://code.claude.com/docs/en/mcp).
The server itself needs no model API key; model access is handled by your chosen client.

## 7. Ask the agent to design a banner

Send this to either connected client:

> Read `data/tasks/webinar.yaml` and pass its prompt and constraints to MarketCanvas
> `reset_canvas`. Keep those requirements fixed. Use the MCP tools to design the
> banner, inspect the reward, and improve the layout. Choose your own actions rather
> than replaying a saved trajectory. Finish before the action budget runs out and
> report the final score. Use MCP for all canvas edits; do not change project files.

For a custom prompt, the agent can translate it into the supported TaskSpec constraints
before resetting the canvas. The environment scores those constraints deterministically;
no LLM runs inside the reward function. Built-in prompt parsing is not implemented,
so prompt-only reset or `demo.py --prompt ...` raises `NotImplementedError`
(reported as `not_implemented` through MCP). See [task authoring](docs/REWARD.md).

## Checks and further reading

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

[WRITEUP.md](WRITEUP.md) explains the design and scaling choices.
[Reward equations](docs/REWARD_EQUATIONS.md) map the score to code;
[reward-hacking analysis](docs/REWARD_HACKING.md) describes known loopholes.
A perfect score means the implemented checks pass, not that the design is visually ideal.
See [MCP details](docs/MCP.md) for protocol behavior and [docs/INDEX.md](docs/INDEX.md)
for the full documentation map.
