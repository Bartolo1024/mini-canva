# Reward equations and code map

Start at [`compute_reward_breakdown`](../src/marketcanvas_env/reward/evaluator.py).
Its calls follow equations (1)–(5) below. Open the named function to see the same equation
in its docstring. This guide describes current behavior, including its approximations.

Notation: `clip(x)=min(1,max(0,x))`; `[condition]` is 1 when true, otherwise 0.
`u_e` is element usability, `a_e` its constraint visibility multiplier, and `rho_e` raw
contrast. Capital **V** denotes generic validity, not visibility. All heuristic settings
come from [`config.reward`](../src/marketcanvas_env/config/environment_config.yaml).

## Preparation: geometry and usable elements

[`Scene`](../src/marketcanvas_env/reward/scene.py) prepares a read-only snapshot:
normalize elements → invalidate duplicate IDs → identify painted regions → measure visibility.

```text
v_e = area((element_box ∩ canvas) minus union(higher_painted_regions)) / area(element_box)
t_e = area((text_ink ∩ inset ∩ canvas) minus union(higher_painted_regions)) / area(text_ink)
u_e = min(v_e, t_e) for text-bearing elements; otherwise v_e
a_e = clip(u_e / configured_min_visible_ratio)
```

Invalid elements have zero usability. Missing ink or a font below the configured readable
minimum makes `t_e=0`. See `_painted_regions`, `_visible_elements`, `_text_usable`, then
[`geometry.visible_ratio`](../src/marketcanvas_env/reward/geometry.py). Rectangular unions
avoid subtracting overlapping occluders twice. Rendering shares the same text inset/ink helpers.

## (1) Constraint scores

[`evaluate_task_constraints`](../src/marketcanvas_env/reward/evaluator.py) calls
[`evaluate_constraint`](../src/marketcanvas_env/reward/constraints.py) once per TaskSpec row.
The dispatcher selects a named evaluator, then clamps its result to [0,1]. Nonfinite scores
become zero in the outer evaluator. Rows retain their order, hard flags, and explanations.

Scene selectors match all supplied `id`/`role`/`type` fields and sort by
`(-visible_ratio, lexical_id)`.
Most rules use the first match; this is not best-score selection. Missing dependencies still
score zero. `no_overlap` is the exception: an empty selected group scores one.

| TaskSpec kind → function in `constraints.py` | Equation / rule |
| --- | --- |
| `exists` → `evaluate_exists` | `[first exists, valid, and u_first ≥ task threshold]` |
| `absent` → `evaluate_absent` | `[number of matches = 0]` |
| `count` → `evaluate_count` | `[number of matches operator target]` |
| `text_contains` → `evaluate_text_contains` | `[case-folded target is a substring] × a_first` |
| `text_equals` → `evaluate_text_equals` | `[case/whitespace-normalized strings equal] × a_first` |
| `color_family` / `color_exact` → `evaluate_color_family` / `evaluate_color_exact` | `[selected color matches target] × a_first` |
| `region` → `evaluate_region` | `region_score(normalized box center) × a_first` |
| `relative_position` → `evaluate_relative_position` | `clip(1 + edge_gap / smaller_axis_span) × min(a_subject,a_object)` |
| `size_relation` → `evaluate_size_relation` | `[subject metric operator object metric] × min(a_subject,a_object)` |
| `alignment` → `evaluate_alignment` | `clip(1 − position_spread / canvas_span / configured_tolerance) × min(a_selected)` |
| `contrast_min` → `evaluate_contrast_min` | `min(clip(rho_e / task_threshold)) × min(a_selected)` |
| `no_overlap` → `evaluate_no_overlap` | `(1 − max(intersection_area / smaller_area)) × min(a_selected)` |

Region scores use configured side/center falloff spans; compound regions take the minimum.
Pair rules require distinct representatives. A single `no_overlap` selector checks every match;
multiple selectors check their representatives. Count/absence include unusable matches.

## (2) Task satisfaction

[`task_satisfaction`](../src/marketcanvas_env/reward/evaluator.py):

```text
T = sum(s_i) / number_of_constraints; T = 0 if there are no constraints
```

Every requirement contributes equally. TaskSpec rejects exact duplicate structured requirements,
ignoring IDs and hard flags. Overlapping predicates can still implicitly emphasize one topic.
There are no importance weights or prompt-specific branches. Prompt text is metadata; a new
task using supported constraint kinds requires only a new spec, not reward-code changes.

## (3) Generic quality

[`evaluate_quality`](../src/marketcanvas_env/reward/quality.py) calls these functions in order:

| Function | Equation |
| --- | --- |
| `bounds_score` | `B = min(on_canvas_ratio_e)`, additionally limited by `u_e` for text |
| `overlap_score` | `O = 1 − max(collision_ratio(i,j))` |
| `contrast_score` | `C = min(text_readability_score(e)) = min(min(rho_e / configured_target,1) × u_e)` |
| `validity_score` | `V = valid_element_count / element_count` |
| `harmonic_mean` | `Q = 4 / (1/B + 1/O + 1/C + 1/V)`; zero if any component is zero |

Empty canvas: B=0, O=1, C=1, V=0. No text: C=1. Invalid canvas/collection: V=0.
`collision_ratio` exempts images and unlabeled backgrounds; explicit task `no_overlap` does
not use that exemption. Worst-case B/O/C cannot be diluted by adding unrelated good objects.
The diagnostic key `contrast` retains usability multiplication; raw contrast is separate:

```text
Scene.contrast → Scene.effective_background → colors.contrast_ratio
              → colors.relative_luminance → colors.linearize_srgb_channel
rho = (max(L_text,L_background)+0.05) / (min(L_text,L_background)+0.05)
```

Each color function documents its sRGB equation. A shape label uses its own fill. Transparent
text uses the highest lower solid rectangle containing its entire box, or the canvas color.
This approximation misses partial ink backgrounds; it is not a complete readability assessment.

## (4) Hard gate and (5) final reward

[`hard_constraint_gate`](../src/marketcanvas_env/reward/evaluator.py), then `aggregate_reward`:

```text
G = 1 if every hard s_i = 1, otherwise configured hard_failure_gate
R = clamp(G × (T + Q) − 1, −1, 1)
S = G × (T + Q) / 2             # diagnostic score_01
```

No hard rows means G=1. The failure multiplier is at most 0.5, so hard failures cannot earn a
positive reward. The default is 0.4. Episode termination and issuance of reward remain in
the environment adapter; nonterminal steps return zero while diagnostic reads stay pure.

The report contains `reward`, `score_01`, `task_score`, `quality_score`, `hard_gate`,
ordered `constraints` (`id`, `kind`, `score`, `hard`, `explanation`), and `quality`
(`bounds`, `overlap`, `contrast`, `validity`). Invalid TaskSpecs raise validation errors;
malformed canvas snapshots are scored defensively. No timestamps or mutable state references
are returned.

## Follow a concrete run

```sh
uv run --locked python -m examples.reward.review
uv run --locked pytest -q
```

The review replays the 15 public trajectories from [`data/`](../data/).
[`REWARD.md`](REWARD.md) covers authoring/configuration;
[`REWARD_HACKING.md`](REWARD_HACKING.md) records measured attacks and unresolved safeguards;
[`REWARD_EXPERIMENTS.md`](REWARD_EXPERIMENTS.md) explains the aggregation choice.

The equation-order code refactor preserved complete breakdowns on 681 comparisons
(including 600 seeded mutations and all public trajectories), plus 2,988 constraint
score/explanation pairs. This is behavior-preservation evidence, not a robustness guarantee.
