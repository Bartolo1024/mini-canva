# TaskSpec and reward authoring

A task is data: `TaskSpec(prompt, constraints)`. The evaluator reads only structured
constraints and canvas properties. Changing the prompt alone has no scoring effect.
New tasks using supported kinds require no evaluator changes; see [DATA.md](DATA.md).
For computation order and diagnostics, read [REWARD_EQUATIONS.md](REWARD_EQUATIONS.md).

```python
from marketcanvas_env.reward import compute_reward, compute_reward_breakdown
from marketcanvas_env.task_data import load_task, replay_trajectory, trajectory_paths

path = next(p for p in trajectory_paths("summer_sale") if p.stem == "well_done")
episode = replay_trajectory(path)
task = load_task("summer_sale.yaml")
reward = compute_reward(episode["state"], task)
report = compute_reward_breakdown(episode["state"], task)
```

Both evaluators accept a TaskSpec model or dictionary. Invalid task specifications
raise validation errors. Malformed standalone canvas snapshots are scored defensively;
the action API has stricter validation. The supplied TaskSpec determines scoring,
rather than any legacy target embedded in a snapshot.

## Constraint schema

Every constraint has a unique nonempty `id`, a `kind`, optional `hard` (default
false), and fields allowed for that kind. Each contributes equally to T.
Unknown/inapplicable fields, including `weight`, are rejected. Exact normalized
duplicate requirements are rejected regardless of their IDs/hard flags; this does
not detect every logically equivalent or overlapping predicate.

| Kind | Required fields and options |
| --- | --- |
| `exists` | `selector`; optional `min_visible_ratio`. Binary usable-presence test |
| `absent` | `selector`. Binary zero-match test |
| `count` | `selector`, `operator` eq/lte/gte, nonnegative integer `value` |
| `text_contains` | `selector`, string `value`; case-insensitive substring |
| `text_equals` | `selector`, string `value`; case/whitespace-normalized equality |
| `color_family` | `selector`, family `value`: neutral/red/orange/yellow/green/blue/purple |
| `color_exact` | `selector`, six-digit hex `value` |
| `region` | `selector`, `value`: left/center/right/top/bottom or a corner combination |
| `relative_position` | `subject`, `object`, `relation`: above/below/left_of/right_of |
| `alignment` | `value`: horizontal_centers/vertical_centers/left_edges/right_edges; multiple `selectors` or one `selector` with `reference="canvas"` |
| `size_relation` | `subject`, `object`; optional `metric`: width/height/area (default area), and `operator` gt/lt/gte/lte (default gt) |
| `contrast_min` | One `selector` or multiple `selectors`; optional ratio `value` in [1,21], otherwise the global contrast threshold |
| `no_overlap` | One `selector` checks all matches; multiple `selectors` check their representatives |

Selectors combine all supplied `id`, `role`, and `type` fields. An empty selector
matches everything; numeric core IDs compare through string form. Most rules choose
highest occlusion-aware visible-box ratio, then lowest lexical ID. Selection is not
a search for best content/color satisfaction. Count and absence include hidden or
unusable matches. Existence cannot exceed one; uniqueness requires a count constraint.

Soft property/layout scores are attenuated by usable visibility, saturating at the
configured threshold. Exists uses its own binary visibility threshold. Missing
dependencies score zero, except `no_overlap`: a missing selector group gives full
credit because no matching pairs can be checked. There is no N/A aggregation or
global one-to-one requirement assignment. Separate rules can select the same object.

Position rules use box centers/edges, not glyph geometry. Relative placement reaches
full credit at edge contact: it does not impose an unrequested positive gap. Size
comparisons are literal: width 110 is greater than 100. Absolute normalized size
floors, preferred gap intervals, and clutter penalties are not supported.

## Global configuration

All retained scoring heuristics live under `reward` in
[environment_config.yaml](../src/marketcanvas_env/config/environment_config.yaml).
They apply across tasks and load once at startup; restart Python/MCP after edits.
`load_config(path)` validates a file without changing the running configuration.

| YAML setting | Meaning |
| --- | --- |
| `hard_failure_gate` | Hard-failure multiplier, default 0.4; constrained to at most 0.5 |
| `min_visible_ratio` | Default usable-presence threshold / soft visibility saturation, 0.8 |
| `contrast_full_credit_ratio` | Generic quality contrast threshold and default for `contrast_min`, 4.5 |
| `side_region_falloff_span`, `center_region_falloff_span` | Continuous regional scoring spans |
| `alignment_falloff_span` | Alignment decay span as a fraction of canvas size, default 0.5 |
| `min_readable_font_size`, `text_inset` | Text usability minimum and padding shared with rendering, defaults 12 and 4 pixels |
| `max_geometry_magnitude` | Defensive raw-snapshot numeric ceiling, default 1e9 |
| `color_families` | HSV neutral cutoffs and ordered hue boundaries |

The YAML is the single source of heuristic defaults; missing/invalid settings are
rejected. Task-specific existence and contrast thresholds affect their constraints,
not the generic quality baseline. There are no `alpha`/`beta` mixture settings.
sRGB/luminance coefficients and geometric identities remain mathematical definitions
in code. Schema limits, such as legal font sizes and content lengths, are separate
input contracts.

## Visibility and color semantics

The read-only Scene validates geometry, identity, relevant colors, and content.
It clips to the canvas and subtracts the union of higher painted rectangles.
Text uses approximate ink boxes, fixed font metrics, and the shared inset; tiny or
blank text is unusable. Nontext objects have no normalized minimum-size safeguard.

Text elements have transparent backgrounds. Contrast uses the highest lower solid
shape/image containing the entire text box, otherwise the canvas background.
A shape label uses its own fill. This misses partial backgrounds beneath ink:
a pixel-invisible headline can receive full score. Unlabeled backgrounds and image
underlays are exempt from generic collision scoring; explicit task `no_overlap`
does not use that exemption.

These conventions, self-reported roles, disjoint spam, and missing applicability
handling are documented in [REWARD_HACKING.md](REWARD_HACKING.md). A score of one
means the implemented checks pass, not that the design is professionally usable.

## Prompt interpretation

Every design prompt must be translated into structured constraints before scoring,
as shown in [the task YAMLs](../data/tasks). The reward never interprets prose.

Pass the YAML task dictionary directly to Gymnasium reset:

```python
from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.task_data import load_task

env = MarketCanvasEnv()
task = load_task("webinar.yaml").model_dump(mode="json")
env.reset(options=task)  # {"prompt": ..., "constraints": [...]}
```

An explicit `constraints` key supplies already translated requirements and bypasses
parsing, even when its value is `[]`. The optional `prompt` is metadata; constraints
alone use an empty prompt. The existing `options={"target": task}` form remains an
alternative and cannot be combined with `prompt` or `constraints`. Reset with no
options loads the Summer Sale YAML.

[`parse_prompt`](../src/marketcanvas_env/prompt_parser.py) is an unimplemented
extension point. Supplying only a prompt calls it and raises `NotImplementedError`,
including for prompts present in the YAML files. There is no prompt lookup or prose
interpretation today. MCP reports this as a `not_implemented` tool error.

Each task's `constraints` field carries a TODO for future deterministic parsing.
That stage will return a list of constraint dictionaries derived from prose, which
reset validates as a TaskSpec before scoring. It must reject unsupported or ambiguous
requirements rather than silently ignore them. Today, new tasks author supported
constraints without changing reward code. Keep the resulting spec fixed during
design; neither reset nor reward calls an LLM.
