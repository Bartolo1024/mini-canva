# Gymnasium interface

[`MarketCanvasEnv`](../src/marketcanvas_env/env.py) connects the deterministic
[canvas core](CORE.md) to reward, Gymnasium spaces, and on-demand rendering.
[`codec.py`](../src/marketcanvas_env/codec.py) converts between readable canonical
JSON and built-in Gymnasium spaces without adding transition or scoring logic.

```python
from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.codec import encode_action, decode_observation
from marketcanvas_env.task_data import load_task

env = MarketCanvasEnv()
task = load_task("summer_sale.yaml").model_dump(mode="json")
observation, info = env.reset(seed=7, options=task)
assert decode_observation(observation) == env.get_canvas_state()
action = {
    "op": "add_element",
    "element": {
        "type": "text",
        "role": "headline",
        "content": "Summer Sale",
        "x": 100,
        "y": 100,
        "width": 600,
        "height": 80,
    },
}
observation, reward, terminated, truncated, info = env.step(encode_action(action))
assert reward == 0 and not terminated
observation, reward, terminated, truncated, info = env.step(encode_action({"op": "finish"}))
assert terminated and not truncated
assert reward == env.get_current_reward()["reward"]
```

This intentionally incomplete banner demonstrates episode mechanics, not full task
satisfaction. [DATA.md](DATA.md) describes the complete example trajectories.

## Shared execution and reward

`step(encoded_action)` decodes the action and calls `execute_action(canonical_action)`.
That method calls `CanvasCore.apply_action`, then evaluates reward only if the
transition ends the episode. [MCP](MCP.md) uses `execute_action` too.

The semantic result contains `state`, `reward`, `terminated`, `truncated`, and
`info`; Gymnasium returns `(observation, reward, terminated, truncated, info)`.
The `info` fields are:

| Field | Meaning |
| --- | --- |
| `action_applied` | Whether the action succeeded; finish counts as success |
| `error` | Null or a semantic error code from [CORE.md](CORE.md#actions-and-validation) |
| `end_reason` | Null, `finish`, or `budget_exhausted` |
| `reward_breakdown` | Null until termination, then the full diagnostic report |

Nonterminal reward is zero. Finish or the final allowed attempt yields one final
reward and `terminated=True`, including when that attempt fails semantically.
`truncated` is always false: the action budget belongs to the finite task.
Finish on the last attempt takes precedence over budget exhaustion. There is no
automatic finish when a high score is reached. The environment applies no discount.

`get_canvas_state()` returns a detached snapshot. `get_current_reward()` evaluates
that snapshot without changing state, attempts, target, or RNG. A diagnostic score
is not another trajectory return. The current formula is
`clamp(G * (T + Q) - 1, -1, 1)`; see [equations](REWARD_EQUATIONS.md) and
[TaskSpec/reward reference](REWARD.md) for the authoritative definitions.

Schema-invalid actions raise `RequestError` with code `invalid_request` without
consuming an attempt. Semantic failures are normal step results and consume an
attempt. Actions after termination raise `LifecycleError("episode_done")`; reads
remain available until reset. Calls before reset raise `not_initialized`, after
structural validation. Reset and close do not issue reward.

## Reset and reproducibility

`reset()` always restores [the Summer Sale TaskSpec](../data/tasks/summer_sale.yaml),
including after a previous custom task. Pass a YAML task dictionary directly as
`options={"prompt": text, "constraints": [...]}`. An explicit `constraints` key
bypasses parsing, including `[]`; prompt metadata is optional and defaults to empty.
`options={"target": task}` remains an alias accepting a TaskSpec model or dictionary,
mutually exclusive with `prompt` and `constraints`. Prompt-only reset calls the
[future parser](REWARD.md#prompt-interpretation) and raises `NotImplementedError`;
saved prompts have no special treatment. The legacy three-field Target is not a
scored task.

Invalid options, seed, TaskSpec, or oversized serialized task are rejected before
changing the episode or Gymnasium RNG. Seeds are null or Python integers from 0
through 2**32−1, excluding booleans. Seed initializes Gymnasium's `np_random` through
`super().reset(seed=seed)`; it is not an input to task resolution, canvas transitions,
or reward. Seed action sampling separately with `env.action_space.seed(seed)`.

Limits and global reward policy load once from
[`environment_config.yaml`](../src/marketcanvas_env/config/environment_config.yaml).
Restart after editing it; spaces and codecs assume the same fixed process
configuration. No per-prompt reward weights exist. Task requirements and replayable
actions live separately under `data/`; see [DATA.md](DATA.md).

## Action encoding

`action_space` is `OneOf`; each value is `(branch_index, payload)`:

| Index | Operation | Encoded payload |
| --- | --- | --- |
| 0 | Add | `{"kind": integer, "properties": full_properties}` |
| 1 | Move | `{"id": integer, "new_x": integer, "new_y": integer}` |
| 2 | Full update | `{"id": integer, "properties": full_properties}` |
| 3 | Delete | `{"id": integer}` |
| 4 | Finish | `{}` |

Kinds are text=0, rectangle=1, button=2, image=3. Properties appear in the order
listed by [the element schema](CORE.md#element-schema), starting at `role`.
Roles and content remain printable ASCII strings. Alignment is left=0, center=1,
right=2; colors are `Box(0, 255, shape=(3,), dtype=uint8)` RGB arrays, without scaling.
Integer properties use `Discrete` with the inclusive core bounds and appropriate
start values. Role/content `Text` spaces use the ordered ASCII characters 32–126,
with their configured length limits. Add encoding expands all defaults.

For example, `encode_action({"op": "move_element", "id": 2, "new_x": 320, "new_y": 300})`
is `(1, {"id": 2, "new_x": 320, "new_y": 300})`, and finish is `(4, {})`.
Actions always refer to stable element IDs, never observation slots.

Decoding checks strict scalars before space membership: Python/NumPy integers are
accepted, but booleans, fractions, and out-of-range RGB channels are rejected
before coercion. Space-valid samples can still fail semantically through absent
IDs, incompatible roles, or capacity. Those failures remain ordinary attempts.
Fresh child spaces and fixed key/branch ordering make seeded sampling reproducible.

## Observation encoding

`observation_space` is an ordered `Dict` with these fields. The defaults below use
32 elements and 64 attempts; configured limits determine the actual bounds.

| Field | Space / representation |
| --- | --- |
| `target` | `Text`: complete TaskSpec as sorted compact ASCII-escaped JSON, length 1–65,536 |
| `steps_taken` | Integer 0–64 |
| `next_element_id` | Integer 1–65 |
| `status` | `Discrete(3)`: active=0, finished=1, budget_exhausted=2 |
| `active` | `MultiBinary(32)`, int8; occupied slots then zeros |
| `elements` | Tuple of 32 slots: `id` (0–64), `kind` (0–3), and properties |
| `relationships` | `MultiBinary((32, 32, 6))`, int8 |

Active slots are sorted by ID; deletion compacts slots without reusing IDs.
Relationship axes are source slot, target slot, and the [six predicates](CORE.md#spatial-relationships)
in their documented order. Padding rows/columns and the diagonal are zero.
Padding uses ID 0, text kind, role `none`, position/z-index 0, dimensions 1×1,
white fill, black foreground, empty content, font size 8, and left alignment.

Unicode task metadata round-trips through JSON escapes. Canvas constants, limits,
schema version, and remaining attempts reconstruct from process configuration and
steps taken. Decoding an emitted observation reconstructs the complete canonical
state. Arbitrary samples from the Cartesian observation space can describe
impossible states; decoding is not an episode restoration API.

Built-in `Text` and `OneOf` spaces support Gymnasium checks and explicit conversion.
A PPO trainer still needs a suitable policy/input adapter; space validity alone
does not establish compatibility. For `render_mode="rgb_array"`, `render()`, and
`save_png(path)`, see [RENDERING.md](RENDERING.md).
