# MarketCanvas-Env

## State, actions, and interfaces

MarketCanvas is a deterministic 800×600 canvas with text, rectangle/button shapes,
and images represented by colored boxes with their own geometry, stacking order,
and content metadata.

I chose semantic actions: `add_element`, `move_element`, `update_element`,
`delete_element`, and `finish`. Color changes use `update_element`. These actions
let an agent edit layout without spending steps on mouse movement or focus. They
sacrifice desktop realism by exposing element IDs and direct edits. A mouse interface
would need cursor, selection, and focus state, backed by the same core.

The JSON state includes element properties, spatial relationships, the complete
TaskSpec, and episode progress. The agent can see both its goal and remaining budget.
JSON is easy to inspect but exposes information otherwise inferred from pixels.
Optional rendering provides RGB arrays and PNG export. Gymnasium uses 32 masked
element slots and structured actions; PPO needs an adapter for mixed numeric/text fields.

Gymnasium and MCP share transitions and scoring. MCP exposes `get_canvas_state`,
`execute_action`, `get_current_reward`, and `reset_canvas` over stdio, with one canvas
per server process. Reads do not change state or award reward. Intermediate rewards
are zero; finishing or reaching the 64-attempt budget issues one terminal reward.
`demo.py` demonstrates the Summer Sale task and prints state and reward. The
environment also accepts the other example tasks or a custom TaskSpec through reset.

## Reward construction and loopholes

The reward reads structured constraints, never prompt text. Five [example tasks](data/tasks/)
cover different content and layouts, including two columns. Each has a successful
trajectory and two attacks in [data/trajectories/](data/trajectories/). Adding a task
using supported constraint types requires no reward-code edits or per-task weights.
Constraints can come from YAML or an external agent that translates a custom prompt
into the supported TaskSpec format (I wanted the env to be deterministic). They are supplied at reset and stay fixed for the episode. The environment has no built-in prompt parser: prose alone raises
`NotImplementedError`. Agent-assisted task preparation happens outside the simulator.
Reward computation remains deterministic and makes no LLM calls.

For constraint scores $s_i\in[0,1]$:

$$
T=\frac{1}{n}\sum_{i=1}^{n}s_i,
\qquad Q=\frac{4}{B^{-1}+O^{-1}+C^{-1}+V^{-1}}
$$

The hard gate uses these values:

| Condition | $G$ |
| --- | --- |
| All hard constraints score 1 | 1 |
| Otherwise | 0.4 |

$$
R=\min\left(1,\max\left(-1,G(T+Q)-1\right)\right)
$$

| Variable | Meaning |
| --- | --- |
| $n$ | Number of constraints |
| $s_i$ | Each constraint's satisfaction score |
| $T$ | Equal mean of constraint scores; zero with no constraints |
| $B$ | Worst on-canvas area ratio, also limited by text usability |
| $O$ | One minus the worst forbidden overlap: intersection area divided by the smaller box area |
| $C$ | Worst text score: WCAG contrast ratio divided by 4.5, capped at one, times text usability |
| $V$ | Fraction of valid element records; zero for an invalid canvas |
| $Q$ | Harmonic mean of $B,O,C,V$; zero if any component is zero |
| $G$ | Hard-constraint multiplier; the default failure value caps reward at −0.2 if $T=Q=1$|
| $R$ | Terminal reward in $[-1,1]$ |

An empty canvas has $B=V=0$; no text gives $C=1$; no forbidden pairs gives $O=1$.
Text usability includes font size, clipping, and coverage by higher elements. Contrast
uses WCAG linearized sRGB luminance and the worst exposed background beneath the
clipped text ink; button labels use their own fill.
Heuristic defaults are in the [global YAML config](src/marketcanvas_env/config/environment_config.yaml).

Alignment is task-specific. Its score is
$\max(0,1-d/(\tau L))\,a$: $d$ is the spread of requested centers or edges,
$L$ is the canvas width or height, $\tau=0.5$ is the configured falloff, and $a$
is the smallest selected visibility multiplier. Each multiplier is usable visibility
divided by the configured threshold (default 0.8), capped at one. There is no global
centering bonus. The reward breakdown exposes individual constraints and all components.

An RL policy optimizes the implemented checks. Replaying all 15 saved trajectories
with `python scripts/run_examples.py` gives these terminal rewards:

| Task | Trajectory | T | Q | G | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Summer Sale | `well_done` | 1.000 | 1.000 | 1.0 | +1.000 |
| Summer Sale | `offscreen_headline` | 0.400 | 0.000 | 0.4 | -0.840 |
| Summer Sale | `small_element_spam` | 0.000 | 0.000 | 0.4 | -1.000 |
| Event | `well_done` | 1.000 | 1.000 | 1.0 | +1.000 |
| Event | `center_stacking` | 0.125 | 0.000 | 0.4 | -0.950 |
| Event | `duplicate_cta` | 1.000 | 1.000 | 1.0 | **+1.000** |
| Newsletter | `well_done` | 1.000 | 1.000 | 1.0 | +1.000 |
| Newsletter | `forbidden_cta` | 0.800 | 1.000 | 0.4 | -0.280 |
| Newsletter | `tiny_logo` | 0.800 | 1.000 | 0.4 | -0.280 |
| Two-column | `well_done` | 1.000 | 1.000 | 1.0 | +1.000 |
| Two-column | `centered_layout` | 0.900 | 1.000 | 1.0 | **+0.900** |
| Two-column | `contrast_patch` | 1.000 | 0.533 | 1.0 | **+0.533** |
| Webinar | `well_done` | 1.000 | 1.000 | 1.0 | +1.000 |
| Webinar | `hidden_headline` | 0.556 | 0.000 | 0.4 | -0.778 |
| Webinar | `missing_cta` | 0.667 | 1.000 | 0.4 | -0.333 |

Seven attacks score negatively, but duplicate CTA still earns full reward; centered layout
and contrast patch are penalized but remain positive. Tiny logos now fail the configurable
minimum-area check (default 0.1% of the canvas), and missing hard requirements activate the gate.

**Duplicate CTA — a gap with a trade-off.** The original CTA satisfies the task;
the extra button is readable and does not overlap anything. The task has no exact
count, so neither $T$ nor $Q$ decreases. Explicit hard counts already work. A general
surplus-element penalty remains unimplemented because identifying unnecessary elements
also requires allowing legitimate decorations and repeated controls. Existence is capped,
but that alone does not make duplication worse.

**Contrast patch — detected, but still positive.** Background lookup now checks the
clipped ink area and uses the worst exposed background, ignoring buried layers. It
correctly measures 1:1 and reduces this example from +1.000 to +0.533. The score stays
positive because task requirements pass and soft contrast gives 1:1 partial credit.
A hard contrast requirement would activate the gate. Ink boxes still approximate glyphs;
even a small patch between letters can reduce contrast.

**Centered two-column layout — weak punishment.** Four region constraints score 0.75;
the other six score 1, giving $T=(4\times0.75+6)/10=0.9$. Quality stays perfect and
the region constraints are soft, so $R=0.9$. Continuous scores provide gradual layout
feedback, but give this wrong layout too much credit. Marking required regions hard
in the TaskSpec would prevent a positive reward without changing reward code.

Roles also remain self-reported, and overlapping constraints can implicitly emphasize
one requirement. These examples expose gaps; they do not prove resistance to every
policy strategy. See [reward equations](docs/REWARD_EQUATIONS.md) and
[adversarial results](docs/REWARD_HACKING.md) for further analysis.

## Scaling to 10,000 VLM PPO rollouts

At 10,000 parallel rollouts, I would expect the MarketCanvas simulator itself to stay relatively cheap. The main bottlenecks would be VLM inference, GPU memory, image preprocessing, KV-cache memory, and data transfer. I would keep many logical or vectorized canvas environments in a limited number of CPU workers, while GPU workers process observations in batches.
For the VLM setup, I would use the existing RGB rendering. One 800×600 RGB `uint8` image is 1.44 MB, so 10,000 images are about 14.4 GB before model tensors or trajectory storage. Because of this, images should only be rendered when needed. I would avoid PNG encoding during training, batch image preprocessing, and reduce unnecessary memory copies and long conversation history.
For PPO, rollout workers would collect actions, old log probabilities, value estimates, rewards, and policy version information. A separate learner pool would update the policy and value model. Because PPO is on-policy, the difference between the policy used for rollout generation and the current policy should not become too large.
I would avoid MCP in the PPO training path and use a direct in-process or vectorized API instead, because RPC and serialization would add unnecessary overhead at this scale. Rollouts should also run asynchronously instead of waiting for all environments at every step, so slower episodes do not leave GPU resources unused.
Before increasing the number of parallel environments, I would measure VLM throughput, GPU memory usage, queue delays, and policy staleness.

