# Documentation map

MarketCanvas implements a deterministic canvas, structured task scoring, Gymnasium,
RGB/PNG rendering, and four MCP tools. Start with the root [README](../README.md)
for commands and [WRITEUP](../WRITEUP.md) for the submission overview.

## Read by purpose

| Guide | What it explains |
| --- | --- |
| [Project status](PROJECT_PLAN.md) | Implemented scope, design choices, and remaining work |
| [Core](CORE.md) | Elements, actions, state, validation, IDs, and episode lifecycle |
| [Gymnasium](GYMNASIUM.md) | Scored API, reset, observation/action encoding, and termination |
| [Tasks and trajectories](DATA.md) | Five tasks, fifteen replayable examples, and adding a task |
| [Reward overview](../WRITEUP.md#reward-construction-and-loopholes) | Compact equations, variable definitions, measured examples, and remaining loopholes |
| [TaskSpec authoring](REWARD.md) | Supported constraints, selection rules, and global configuration |
| [Reward equations](REWARD_EQUATIONS.md) | Computation order, equations, and corresponding functions |
| [Reward hacking](REWARD_HACKING.md) | Measured attacks, defenses, failed safeguards, and regression tests |
| [Reward experiments](REWARD_EXPERIMENTS.md) | Historical formula comparisons and why the simple formula was retained |
| [Rendering](RENDERING.md) | RGB/PNG APIs, drawing rules, and visual limitations |
| [MCP](MCP.md) | Connect Codex, use the tools, and distinguish live design from replay |
| [Development and tests](DEVELOPMENT.md) | Locked setup, validation commands, packaging, and test coverage |
| [Glossary](GLOSSARY.md) | Terms used by this implementation |

[render_canvas](MCP_RENDER_CANVAS.md) is a proposed implementation contract, not an
available MCP tool. Supply authored YAML constraints directly at reset; the
[prompt-parser extension point](REWARD.md#prompt-interpretation) raises
`NotImplementedError`. General prose interpretation and model calls are absent.

## Follow one computation

```text
Gymnasium action → codec.decode_action ─┐
MCP execute_action → canonical JSON ────┴→ MarketCanvasEnv.execute_action
                                           ↓
                                    CanvasCore.apply_action
                                           ↓
                                    state + episode status
                                           ↓
                         terminal only: compute_reward_breakdown
```

State and reward diagnostics can be read without stepping. Rendering reads a
snapshot on demand and does not score or edit it.

For a code review, read [env.py](../src/marketcanvas_env/env.py) and
[core.py](../src/marketcanvas_env/core.py), then follow the
[equation-to-function map](REWARD_EQUATIONS.md) through the reward directory.
[models.py](../src/marketcanvas_env/models.py) owns canonical validation;
[codec.py](../src/marketcanvas_env/codec.py) owns Gymnasium conversion;
[mcp_server.py](../src/marketcanvas_env/mcp_server.py) owns transport.

Predict a small trace before reading its test: add ID 1, delete it, add ID 2,
then finish. Deletion never reuses an ID. An unknown-ID edit spends an attempt;
a malformed request does not. This distinction is shared by both interfaces.
