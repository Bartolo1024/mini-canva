---
name: marketcanvas-review
description: Review MarketCanvas reward hacking, deterministic episode behavior, protocol parity, and take-home deliverables. Use for focused design or implementation review.
---

Read `AGENTS.md`, and `docs/PROJECT_PLAN.md`. During preparation review proposed behavior without implementing tests or fixes.

Build an adversarial comparison against a valid layout: empty canvas, blank headline, duplicate CTAs, tiny/off-canvas text, hidden required elements, complete occlusion, wrong CTA color, illegible overlap, and repeated finish/reward calls. A good score must depend on visible useful content, not only labels or element counts.

Check that legitimate text inside a button is allowed and that contrast uses the actual background under visible text. Explain approximations where multiple backgrounds or partial occlusion complicate measurement. Avoid claiming complete WCAG compliance from a contrast heuristic.

Review repeatability across fresh instances and resets, deterministic ID allocation, stable z-order ties, bounded finite rewards, atomic validation, caller mutation of returned state, episode budget handling, and absence of cross-instance state leaks.

Compare direct API and MCP action sequences. Inspect protocol output cleanliness and whether state/reward reads accidentally mutate or terminate an episode.
