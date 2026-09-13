# Deterministic canvas core

[`CanvasCore`](../src/marketcanvas_env/core.py) owns one episode: elements, target,
IDs, action budget, and status. It accepts canonical JSON actions and produces
unscored transitions. [`MarketCanvasEnv`](GYMNASIUM.md) adds reward and Gymnasium
conversion; [MCP](MCP.md) calls that same scored interface. Rendering runs separately
on demand. No transition uses randomness or model inference.

```python
from marketcanvas_env import CanvasCore
from marketcanvas_env.task_data import load_task

canvas = CanvasCore()
state, info = canvas.reset(seed=7, options={"target": load_task("default_task.yaml")})
transition = canvas.apply_action(
    {
        "op": "add_element",
        "element": {
            "type": "text",
            "role": "headline",
            "content": "Summer Sale",
            "x": 100,
            "y": 100,
            "width": 600,
            "height": 80,
            "font_size": 32,
            "text_align": "center",
        },
    }
)
assert transition.action_applied and not transition.terminated
serialized = canvas.serialize_state()
```

## State and public API

| Operation | Result |
| --- | --- |
| `reset(seed=None, options=None)` | Detached state and `{"target_source": "default" / "prompt" / "structured"}` |
| `apply_action(payload)` | `CanvasTransition(state, action_applied, error, end_reason)` |
| `get_canvas_state()` | Fresh JSON-compatible snapshot, elements sorted by ID |
| `serialize_state()` | ASCII-escaped JSON, sorted keys and compact separators |
| `get_draw_order()` | Detached elements ordered back to front by `(z_index, id)` |

A transition's `terminated` property is true when `end_reason` is present;
`truncated` is always false. The core never calculates or issues reward.

Canonical snapshots contain `schema_version` (1), `canvas` (800×600, white),
`limits` (`max_elements`, `max_steps`, `max_content_length`), `target`, `elements`,
`relationships`, `steps_taken`, `steps_remaining`, `next_element_id`, and `status`.
`steps_remaining = max_steps - steps_taken`, including after early finish.
Status is `active`, `finished`, or `budget_exhausted`.

Successful adds allocate increasing integer IDs starting at 1. Failed adds and
other actions do not advance the ID counter; deletion never recycles IDs. Reset
clears elements and progress and restarts IDs. Snapshots, draw-order records, and
transition metadata are independent of caller mutations. Serialization contains
no timestamps, random IDs, or hidden allocation history.

## Element schema

[`models.py`](../src/marketcanvas_env/models.py) defines strict Pydantic schemas.
Coordinates are integer pixels from the top-left, increasing rightward/downward.
Rectangles are half-open: `[x, x + width) × [y, y + height)`. Valid off-canvas
geometry is retained for scoring and clipped only when rendered.

| Field | Accepted values | Add default |
| --- | --- | --- |
| `id` | Assigned integer, 1 through `max_steps`; forbidden on add | Next ID |
| `type` | `text`, `shape`, `image` | Required |
| `subtype` | Shape: `rectangle` or `button`; text/image: null | Rectangle for shape, otherwise null |
| `role` | 1–`max_role_length` printable ASCII characters | `none` |
| `x`, `y` | -800–1599; -600–1199 | 0, 0 |
| `width`, `height` | 1–1600; 1–1200 | 200, 60 |
| `z_index` | 0–63 | 0 |
| `color`, `text_color` | `#RRGGBB`, normalized to uppercase | White fill, black text |
| `content` | 0–`max_content_length` printable ASCII characters | Empty string |
| `font_size` | 8–72 pixels | 24 |
| `text_align` | `left`, `center`, `right` | `left` |

Integers reject booleans, floats, numeric strings, and null. Unknown fields are
rejected; colors have no alpha. Element text cannot contain Unicode, tabs, or
newlines. Task metadata has a separate schema; see [TaskSpec](REWARD.md).

`headline` is reserved for text and `cta` for a shape with subtype `button`.
Other role labels have no inferred semantics. Compatibility is a semantic
transition check, so an incompatible role consumes an attempt. Role labels alone
do not establish reward satisfaction.

A button contains its own label. Text and nonblank shape content are drawn,
including labels on rectangles. Image content is metadata; images are colored
placeholders, with no fetching.
Text elements render with transparent backgrounds; solid shapes/images supply
fills. See [rendering](RENDERING.md) for font, clipping, and background rules.

## Actions and validation

| `op` | Other payload fields | Effect |
| --- | --- | --- |
| `add_element` | `element`: type/subtype and optional properties | Apply defaults, allocate ID, add |
| `move_element` | `id`, `new_x`, `new_y` | Replace position |
| `update_element` | `id`, `properties` | Replace all mutable properties atomically |
| `delete_element` | `id` | Remove element |
| `finish` | None | End episode |

An update must supply every mutable field: `role`, `x`, `y`, `width`, `height`,
`z_index`, `color`, `text_color`, `content`, `font_size`, and `text_align`.
ID, type, and subtype cannot change. Read the current record, retain its mutable
properties, then change the desired values; there is no partial patch operation.

Every attempt follows this order:

1. Validate payload structure, types, and ranges. `RequestError("invalid_request", ...)`
   leaves the entire episode unchanged, even if it is already finished.
2. Require an initialized, active episode. Otherwise raise
   `LifecycleError("not_initialized")` or `LifecycleError("episode_done")`.
3. Check applicable semantic preconditions: referenced ID, add capacity, then role
   compatibility. Failures return `unknown_element`, `capacity_exceeded`, or
   `incompatible_role` in the transition without changing elements, target, or IDs.
4. Apply the successful mutation and increment the attempt counter once. Semantic
   failures and successful no-op edits also consume one attempt.
5. A finish sets `finished`; otherwise the last allowed attempt sets
   `budget_exhausted`. Finish takes precedence on the last attempt.

For example, moving absent ID 64 is a normal failed attempt under the default
limits. Supplying `new_x="0"` instead of integer 0 raises a schema error and costs
no attempt. A failed move on attempt 64 still ends the task. Reads remain available
after termination; further actions fail until reset. The scored adapter's exact
reward issuance is described in [GYMNASIUM.md](GYMNASIUM.md).

## Reset and configuration

Reset validates all inputs before replacing state. It accepts a seed (null or a
Python integer 0–2**32−1, excluding booleans) and options containing either `target`
or `prompt`, never both. Explicit null values for those options are invalid.
The seed is validated but does not affect core transitions. Reset during an active
episode abandons it without awarding reward.

Use an explicit TaskSpec for direct core calls, as in the example. For compatibility,
the standalone core's no-options reset still returns the original three-field
`Target` (`headline`, `cta_text`, `cta_color`), and recognizes only the original
Summer Sale phrase without a final period. **The scored environment always uses
TaskSpec**, including default reset. It accepts task YAML fields directly as reset
options and passes the validated TaskSpec into the core; explicit constraints
bypass parsing. Its prompt-only reset calls the
[unimplemented parser](REWARD.md#prompt-interpretation) and raises
`NotImplementedError`. Neither path interprets arbitrary prose.

[`environment_config.yaml`](../src/marketcanvas_env/config/environment_config.yaml)
sets process-wide limits: 32 active elements, 64 attempts, 256 content characters,
64 role characters, and 65,536 serialized TaskSpec characters by default. The frozen
configuration loads once at import; restart after editing it and recheck affected
fixtures. Geometry/style bounds in the table remain schema constants. Task data is
separate from runtime configuration; see [DATA.md](DATA.md).

## Spatial relationships

Relationships are derived from raw, unclipped rectangles for every ordered pair
of distinct elements. They describe geometry, not visibility or reward. Emit only
true predicates, sorted by `(source_id, target_id, relation index)`:

| Index / relation | Predicate for elements a, b |
| --- | --- |
| 0 `overlaps` | Positive intersection area; touching edges do not overlap |
| 1 `contains` | All b edges lie inside/on a edges; equal boxes contain each other |
| 2 `left_of` | `a.x + a.width <= b.x` |
| 3 `above` | `a.y + a.height <= b.y` |
| 4 `center_aligned_x` | Horizontal centers differ by at most 2 pixels |
| 5 `center_aligned_y` | Vertical centers differ by at most 2 pixels |

Each record has `source_id`, `target_id`, and `relation`. Containment and overlap
may both hold. Doubled integer centers avoid rounding. These descriptive alignment
bits do not introduce a global centering reward. There are no nested containers
or independently editable relationship records.
