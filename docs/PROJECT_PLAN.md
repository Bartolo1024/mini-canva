# Project status and design decisions

The required environment, reward, Gymnasium and MCP interfaces, demo, and submission
write-up are implemented (M1–M8). RGB/PNG rendering is the implemented optional
visual output. Final clean-install submission verification (M9) remains pending.
This documentation consolidation does not complete M9.

## Assignment and delivered scope

| Requirement | Current implementation |
| --- | --- |
| Deterministic canvas with text, shapes, and image placeholders | [Core and canonical schemas](CORE.md) |
| Semantic state, properties, relationships, and actions | [Core](CORE.md), [Gymnasium encoding](GYMNASIUM.md) |
| Bounded, inspectable, task-driven terminal reward | [TaskSpec](REWARD.md), [equations](REWARD_EQUATIONS.md), [adversarial analysis](REWARD_HACKING.md) |
| MCP access for an LLM client | [Four stdio tools and Codex connection](MCP.md) |
| Programmatic demo for a mock task | [demo.py](../demo.py) loads Summer Sale YAML; explicit prompt parsing remains unimplemented |
| State/action rationale, reward loopholes, 10,000-rollout discussion | [WRITEUP.md](../WRITEUP.md) |
| Optional image output | [On-demand RGB and PNG](RENDERING.md) |

Five task YAMLs and fifteen action trajectories are separated from source and global
configuration; see [DATA.md](DATA.md). A new task using supported constraint kinds
requires no reward-code change. Reset accepts task YAML fields directly; explicit
constraints bypass parsing. Prompt-only reset raises `NotImplementedError` until
the deterministic parser is implemented. Reward reads only structured requirements.

## Decisions to preserve

- One deterministic core serves both Gymnasium and MCP. Rendering and transport
  stay outside transition and reward logic.
- Use semantic add/move/full-update/delete/finish actions on an 800×600 canvas.
  Packaged limits are 32 elements and 64 attempts, configured in YAML.
- Expose target, stable IDs, relationships, and episode progress. Schema errors
  preserve budget; semantic failures consume it. Finish and budget exhaustion
  terminate and issue one reward. Reads remain pure.
- Task constraints contribute equally; there are no prompt-specific weights.
  Reward is `clamp(G * (T + Q) - 1, -1, 1)`. Alignment/color preferences belong to
  TaskSpec; global heuristic settings live in the configuration YAML.
- Use the official MCP Python SDK with stdio and one canvas per server process.
  An LLM client chooses actions; the server itself makes no model calls.
- Pillow supplies fixed text metrics and on-demand pixels. Images are colored
  placeholders. Rectangle-based scoring has documented visual loopholes.
- Gymnasium Text/OneOf spaces need a suitable policy adapter for PPO. Model
  training and large-scale infrastructure are discussion topics, not implemented features.

Python 3.12, Gymnasium, Pydantic, MCP, NumPy, Pillow, and PyYAML are the runtime
stack. pytest, Hypothesis, Ruff, and pre-commit support development.
[pyproject.toml](../pyproject.toml) declares dependencies; [uv.lock](../uv.lock)
records resolved versions. The build backend is pinned to uv_build 0.10.4.

## Remaining work

- [ ] Implement and verify the proposed [render_canvas tool](MCP_RENDER_CANVAS.md).
  Current MCP observations contain JSON only.
- [ ] Demonstrate a live Codex session choosing actions through MCP. The provided
  Python MCP client currently replays authored trajectories.
- [ ] Complete M9 from a fresh locked installation: verify installed imports and
  bundled data outside the checkout; run demo/review/render commands from their
  documented checkout context; exercise actual MCP discovery, edits, and errors.
- [ ] Run final lint/format/full-suite checks on the submission revision, inspect
  representative outputs, recheck write-up pagination, and review the intended
  repository contents. Record unresolved findings and execution restrictions.

General prompt parsing and task selection by name through MCP are future options,
not existing capabilities. A client can propose a TaskSpec for human review and
pass the approved structure to reset; keep it fixed while designing. Known reward
exploits remain explicit limitations, not completed anti-hacking guarantees.
Publishing or sending the repository is separate from local verification.
