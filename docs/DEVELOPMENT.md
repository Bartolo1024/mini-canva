# Development and verification

From the repository root with Python 3.12 and uv available:

```sh
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

[pyproject.toml](../pyproject.toml) declares the stack and [uv.lock](../uv.lock)
pins resolved dependencies. Development sync installs the source package editably.
Update the lock deliberately when changing dependencies. If the default uv cache
is unavailable, set `UV_CACHE_DIR=/tmp/marketcanvas-uv-cache` for these commands.

## Tests and review

| Tests | Coverage |
| --- | --- |
| `test_models`, `test_config`, `test_core`, `test_geometry` | Strict validation, actions, configuration, IDs, lifecycle, isolation, and generated deterministic replay |
| `test_reward_constraints`, `test_reward_primitives`, `test_reward_scenarios` | Task authoring, exact numerical primitives, ranking, and task conditioning |
| `test_reward_adversarial`, `test_reward_adversarial_primitives` | Tiny/offscreen/hidden content, duplicate spam, overlap dilution, malformed snapshots, and known exploits |
| `test_codec`, `test_env` | Space membership, encoding round trips, Gymnasium checker, and scored/core parity |
| `test_rendering`, `test_render_env` | Pixel geometry, fonts/clipping, PNG repeatability, and observational purity |
| `test_mcp` | Real SDK stdio discovery/calls, direct parity, errors, locking, and process isolation |
| `test_demo`, `test_task_data` | CLI behavior and legal deterministic replay of the compact public dataset |
| `test_prompt_parser` | Unimplemented parsing boundary, explicit constraints bypass, and reset validation |

The reset/parser refactor passes **765 tests with nine documented expected failures**.
Actual MCP stdio tests passed outside the sandbox after initialization timed out
inside it. Ruff passes, and the fifteen-trajectory reward table is unchanged.

Test files live in [tests](../tests). Strict expected failures mark unmet reward
safeguards; unexpected passes fail the suite. Use `pytest --runxfail` to expose
those assertions as ordinary failures. A passing normal suite does not mean the
[known reward exploits](REWARD_HACKING.md) are solved.

Earlier checkout verification passed the full suite and fifteen-trajectory reward
review. Re-run the commands above for the current revision; this does not replace
the pending final clean-install audit.

```sh
uv run --locked python demo.py --png artifacts/demo/summer_sale.png
uv run --locked python -m examples.reward.review
uv run --locked python -m examples.reward.render --all
uv run --locked python -m examples.reward.adversarial --markdown
uv run --locked python -m examples.mcp_client --benchmark two_column
```

Review outputs are generated from current code; do not copy old weighted reward
tables as current results. The MCP example is a trajectory replay, not an LLM
policy. [MCP.md](MCP.md) describes the separate interactive Codex workflow.

## Packaging and submission boundary

`uv build` produces a wheel and source archive. Runtime source lives under
`src/marketcanvas_env`; task/trajectory YAMLs are bundled from `data/` and global
configuration from the package. Their loaders do not rely on the working directory.
Example scripts and the broad raw regression fixture require the source checkout.

Earlier checks verified installed-wheel task replay and MCP/direct parity. Final
fresh-install verification of the complete submission remains pending; follow
[PROJECT_PLAN.md](PROJECT_PLAN.md), rather than interpreting an editable-install
test pass as completion.

## Repository automation

`AGENTS.md`, `.codex/agents/`, and `.agents/skills/` are operational instructions;
they remain in their discoverable locations. `tooling/codex-setup/` contains setup
source copies. Project guides live only in `docs/`, apart from root README/WRITEUP.

`.codex/hooks.json` configures a SessionStart context reminder. Its activation/trust
must be checked in the client; file presence alone does not prove activation.
Review its absolute checkout path after moving the repository. It does not gate
implementation authorization.

`.pre-commit-config.yaml` defines check-only Ruff hooks. To validate/install them
in a writable Git checkout:

```sh
uv run --locked pre-commit validate-config
uv run --locked pre-commit install
```

Tests run explicitly, not automatically on every edit. Hook installation and Codex
hook trust are separate operations; neither is asserted by this documentation.
