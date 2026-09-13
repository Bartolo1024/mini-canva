---
name: marketcanvas-design
description: Design or implement MarketCanvas MDP, canvas state/actions, rendering boundaries, and MCP contracts. Use for this project's environment or interface work.
---

Read `AGENTS.md` and `docs/PROJECT_PLAN.md` from the project root. Respect the current preparation-only scope until the user requests implementation.

Before changing contracts, settle observable state, action validation, and episode lifecycle together. State must expose the target specification, stable element IDs, spatial relationships, and progress needed to predict transitions and termination. Keep JSON serialization deterministic and avoid leaking mutable internal objects.

Use semantic actions first. A caller must be able to create all required properties, correct a layout, and finish an episode. Define atomic invalid actions and whether they consume the action budget. Separate task termination from the external step limit.

Keep one transition path for both Gymnasium and MCP. Match actual observations/actions to declared Gymnasium spaces; introduce a documented encoding if structured text actions do not fit training-oriented spaces. Do not claim compatibility with arbitrary PPO libraries without verifying it.

Keep target interpretation deterministic and explicitly limited to supported templates or structured requirements. Unsupported prompts must not silently become the example task.

Reward queries are pure diagnostics. Episode reward is issued once at the episode boundary. Rendering is optional work done on demand, with fixed font assets and stacking rules shared with semantic visibility calculations.

When implementation is authorized, validate contracts with deterministic replay, isolation, JSON round-trips, and an MCP client smoke test. Report remaining assumptions instead of expanding the assignment.
