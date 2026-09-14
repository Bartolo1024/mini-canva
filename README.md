# MarketCanvas-Env

A deterministic 800×600 design canvas with text, rectangles/buttons, and colored image
placeholders. Gymnasium and MCP share the same actions, episode state, and structured reward.

From the checkout, with Python 3.12 and uv installed:

```sh
uv sync --locked --group dev
uv run --locked python demo.py --png artifacts/demo/summer_sale.png
```

The demo adds a headline, image placeholder, and yellow button, then finishes. It prints
the action trace, semantic state, and reward breakdown: rewards are `0, 0, 0, 1`.
Open `artifacts/demo/summer_sale.png` to inspect the result. Add `--json` for one JSON object.
`--png` creates parent directories and overwrites the named file.

The default prompt is:

> Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast.

The default demo loads that YAML's authored constraints and draws a fixed Summer Sale
layout. An explicit `--prompt` requests parsing, which is not implemented yet—even for
the default phrase. For new tasks, convert the loaded TaskSpec to a dictionary with
`task = load_task("webinar.yaml").model_dump(mode="json")`, then pass it to
`env.reset(options=task)` or MCP `reset_canvas`, with `prompt` metadata and `constraints`.
Explicit constraints bypass parsing; `options={"target": task}` remains an alternative.
Every prompt needs structured constraints before scoring.
Task constraints contribute equally: no `weight` fields or configurable task/quality mixture.
Exact duplicate structured requirements are rejected. See the
[TaskSpec guide](docs/REWARD.md) for supported requirements and prompt interpretation.
Shared heuristic settings live under `reward` in
[environment_config.yaml](src/marketcanvas_env/config/environment_config.yaml); restart the
Python/MCP process after edits. The packaged defaults preserve existing scores.
The demo is a fixed example policy. No model training or model calls are involved.

## Task data

Five task definitions live in `data/tasks`: the default Summer Sale task, webinar,
newsletter, event, and two-column layout. Each has three action trajectories under
`data/trajectories/<task_name>`: `well_done.yaml` and two representative attacks.
Trajectories contain ordered actions ending in `finish`; states and rewards are computed
by replay. Start with [the data guide](docs/DATA.md),
[default_task.yaml](data/tasks/default_task.yaml), and
[its successful trajectory](data/trajectories/default_task/well_done.yaml).

## Inspect the reward

Start with [the equations and code map](docs/REWARD_EQUATIONS.md), then
[compute_reward_breakdown](src/marketcanvas_env/reward/evaluator.py).

```sh
uv run --locked python scripts/run_examples.py
uv run --locked python scripts/run_examples.py --task two_column
uv run --locked python scripts/run_examples.py --task webinar --trajectory missing_cta --output-dir artifacts/reward
```

The same script can replay all 15 trajectories, one task, or one trajectory. It prints
their reward components. Add `--output-dir artifacts/reward` to export PNG/JSON pairs
under `<output-dir>/<task_name>/`; JSON includes the action trace, final state, and
reward explanations. Tasks and actions come from `data/`, not from the script.

The [reward-hacking analysis](docs/REWARD_HACKING.md) maps attacks to tests and records
unmet safeguards. Broader snapshot regressions and historical experiments live under
`tests/reward_support/`; see [development checks](docs/DEVELOPMENT.md) and
[reward experiments](docs/REWARD_EXPERIMENTS.md).

Reward combines explicit task satisfaction and generic bounds, overlap, contrast, and
validity. A score of 1 means the implemented checks pass; it does not establish professional
design quality. Roles are caller-supplied, images are solid placeholders, and visibility and
background contrast use rectangle approximations. Contrast is a soft rule in the default task.

## Verify and connect

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv run --locked python scripts/mcp_client.py --task two_column
```

The MCP script starts a real stdio subprocess and replays an authored trajectory from `data/`.
Add `--trajectory missing_cta` with `--task webinar` to select a representative attack;
the default trajectory is `well_done`.
For an LLM choosing its own actions, follow the [Codex connection and live-demo guide](docs/MCP.md).
Each server process owns one canvas; stdout carries protocol messages. MCP currently exposes
JSON state/reward and editing/reset tools. The [render_canvas tool](docs/MCP_RENDER_CANVAS.md)
is specified but not implemented.

Start reading with [WRITEUP.md](WRITEUP.md), then [core.py](src/marketcanvas_env/core.py),
[reward/evaluator.py](src/marketcanvas_env/reward/evaluator.py), and the thin
[Gymnasium](src/marketcanvas_env/env.py) and [MCP](src/marketcanvas_env/mcp_server.py) adapters.
[Reward authoring](docs/REWARD.md), [Gymnasium spaces](docs/GYMNASIUM.md),
[rendering](docs/RENDERING.md), and [tests](docs/DEVELOPMENT.md) provide details.
The [documentation index](docs/INDEX.md) maps the current guides; [project status](docs/PROJECT_PLAN.md)
separates implemented features from pending work.
The two utilities in `scripts/` require the checkout; the installed runtime package
does not depend on them. The required `demo.py` remains a separate fixed demonstration.
