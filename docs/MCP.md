# MCP: connect Codex to the canvas

The server exposes the existing scored environment through stdio. It runs no
model calls; the connected client chooses actions. There are four tools today.
[render_canvas](MCP_RENDER_CANVAS.md) is specified but not implemented, so current
MCP observations contain semantic JSON and reward diagnostics, not images.

## Connect Codex

From the checkout, prepare the environment and register the server once.
Replace the absolute Python path with your checkout location:

```sh
uv sync --locked --group dev
codex mcp add marketcanvas -- /absolute/path/to/verita_take_home/.venv/bin/python -m marketcanvas_env.mcp_server
```

Restart the Codex IDE extension and start a new conversation, or run `codex`
in the terminal and use `/mcp` to inspect the active connection. The CLI and IDE
share MCP configuration on the same host. These steps follow the
[official Codex MCP guide](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
MarketCanvas itself requires no model API key; model access belongs to the client.

Codex launches the subprocess automatically. Do not start a second server expecting
the client to attach to it: there is no HTTP port or URL. For another stdio client,
configure the same executable and `-m marketcanvas_env.mcp_server` arguments.

The standalone command below waits for MCP protocol messages, not human input:

```sh
uv run --locked python -m marketcanvas_env.mcp_server
```

## Interactive design

Send this to the connected client:

> Use only MarketCanvas MCP tools for this design session. Reset to the default
> task and read its TaskSpec. Keep those requirements unchanged. Choose your own
> elements, positions, sizes, and colors; inspect reward diagnostics and revise
> the design. Finish before exhausting the action budget, then report the final
> score and weaknesses. Do not edit project files, run Python, or replay reference
> trajectories. If the tools are unavailable, report the connection problem.

This is an LLM-driven workflow. [demo.py](../demo.py) is a fixed Gymnasium policy;
[examples/mcp_client.py](../examples/mcp_client.py) is a real MCP client that replays
an authored trajectory. Use that replay as an account-free protocol check:

```sh
uv run --locked python -m examples.mcp_client --benchmark two_column
uv run --locked python -m examples.mcp_client --benchmark webinar --trajectory missing_cta
```

These example commands require the source checkout. The installed server and its
bundled data/configuration work independently of the working directory.

## Tools and task input

| Tool | Arguments | Successful result |
| --- | --- | --- |
| `get_canvas_state` | `{}` | Canonical state, TaskSpec, and progress |
| `execute_action` | `{"action": ...}` | State, reward, termination flags, and info |
| `get_current_reward` | `{}` | Pure diagnostic reward breakdown |
| `reset_canvas` | Optional `seed`; `prompt`/`constraints` or alternative `target` | State and reset info |

Discovery supplies canonical action and TaskSpec JSON schemas. Pass action objects,
not encoded Gymnasium tuples or JSON strings. For example, finish is
`{"action":{"op":"finish"}}`; see [CORE.md](CORE.md) for editing payloads.

The server performs a default reset on startup. `reset_canvas({})` restores the
default task. Pass the task YAML's fields directly as `{"prompt": ..., "constraints": [...]}`.
Supplying `constraints`, even `[]`, bypasses parsing; prompt metadata is optional.
The existing `target` argument accepts an already translated TaskSpec and cannot
be combined with `prompt` or `constraints`. Prompt-only reset returns the
`not_implemented` tool error without changing the episode: the future parser has
no implementation, including for saved prompts. There is no task-name loader tool.
An external LLM can propose constraints for human review, then use the approved
structure. The reward reads constraints, never prompt text; keep the spec fixed
during design. See [prompt interpretation](REWARD.md#prompt-interpretation) and
[saved tasks](DATA.md).

Explicit null prompt/constraints/target is invalid. Omitted/null seed is accepted.
Reset can abandon an episode but does not award its score.

## Errors, lifecycle, and isolation

Schema/target/lifecycle errors return `isError=true` with JSON text
`{"code": ..., "message": ...}`, preserving state and budget. Canonical integer
fields reject numeric strings, floats, and booleans. Protocol-level failures and
unexpected programming errors retain the SDK error surface.

A semantic failure, such as an unknown element ID, is a normal tool result with
`info.error` and consumes one attempt. Nonterminal action rewards are zero;
finish or budget exhaustion issues one terminal reward. Post-terminal actions
fail with `episode_done` until reset. State/reward reads remain available.

One process owns one canvas. A lock serializes all calls, including reads/reset.
Replay assumes a specified action order, not racing requests. Separate processes
provide separate episodes. There is no session multiplexing, retry deduplication,
or idempotency key: retrying a mutation spends another attempt. Stdout carries
protocol messages; diagnostics go to stderr.

[test_mcp.py](../tests/test_mcp.py) checks actual SDK stdio calls and direct-API
parity. A protocol replay does not establish that a particular LLM client has
completed a live design session.
