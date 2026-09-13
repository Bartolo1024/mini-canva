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

$$
G=\begin{cases}1,&\text{all hard constraints score }1\\0.4,&\text{otherwise}\end{cases}
\qquad R=\operatorname{clamp}\bigl(G(T+Q)-1,-1,1\bigr)
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
uses WCAG linearized sRGB luminance and the selected underlying background.
Heuristic defaults are in the [global YAML config](src/marketcanvas_env/config/environment_config.yaml).

Alignment is task-specific. Its score is
$\max(0,1-d/(\tau L))\,a$: $d$ is the spread of requested centers or edges,
$L$ is the canvas width or height, $\tau=0.5$ is the configured falloff, and $a$
is the smallest selected visibility multiplier. Each multiplier is usable visibility
divided by the configured threshold (default 0.8), capped at one. There is no global
centering bonus. The reward breakdown exposes individual constraints and all components.

An RL policy optimizes these checks, so a high score is not proof of a good design.

| Attack | Current protection and remaining gap |
| --- | --- |
| Tiny or offscreen content | Presence requires usable visibility and a configurable minimum canvas-area fraction (default 0.1%, overridable in TaskSpec). This rejects 1×1 images/logos; area alone does not ensure useful proportions. |
| Duplicate CTAs or create-everything spam | Existence is capped and explicit hard counts work. Without counts or clutter penalties, disjoint spam can tie a clean design. |
| Perfectly aligned, overlapping elements | Worst-pair overlap makes $Q=0$ and cannot be diluted by decoys. Images and unlabeled backgrounds are exempt. |
| Missing required elements | The hard gate prevents positive reward. Missing dependent constraints still add zeros. |
| High contrast or hidden text | Usability accounts for occlusion, but background lookup requires whole-text-box containment. A dark patch under only dark ink can remain undetected at reward 1. |

Roles also remain self-reported. Overlapping constraints can emphasize one requirement
despite equal weights. Changing aggregation cannot separate identical component scores.
See [reward equations](docs/REWARD_EQUATIONS.md) and [adversarial results](docs/REWARD_HACKING.md) for a deeper dive.

## Scaling to 10,000 VLM PPO rollouts

At 10,000 parallel rollouts, I would expect the simulator itself to remain relatively cheap. The dominant bottlenecks would come from model inference, memory, and data movement. Since the environment state is represented as JSON, observations should be relatively lightweight to serialize and batch. I would therefore separate environment execution from model serving: thousands of logical MarketCanvas environments could be maintained in vectorized CPU workers, while JSON observations are submitted asynchronously to a smaller pool of GPU inference workers using continuous batching.
For PPO, rollout workers would collect actions, old log-probabilities, value estimates, and rewards, while a separate learner pool performs policy/value updates and periodically redistributes updated weights. Because PPO is on-policy, policy staleness between rollout workers and learners would need to be bounded. I would also avoid MCP in the PPO training hot path and instead use a direct in-process or vectorized environment API, since serialization and RPC overhead can become significant at this scale.
The current JSON state representation should be preferable to image-based observations for this setup. Processing images (if someone would have images as state) would be substantially harder to scale because it would add image rendering, encoding, preprocessing, GPU memory, and data transfer costs. If visual observations were introduced later, rendering and preprocessing would need to be lazy and batched rather than performed for every environment step.
Asynchronous rollouts are preferable to strict lock-step execution because episode lengths vary, otherwise slow trajectories create stragglers and leave expensive GPU capacity idle. Completed environments should immediately reset and re-enter the inference queue.
A further simplification would be to consider GRPO instead of PPO for this environment. MarketCanvas has a cheap deterministic terminal reward and can easily generate multiple independent trajectories for the same TaskSpec. GRPO can therefore replace the learned value-function baseline with relative rewards across a group of rollouts, removing the critic/value-model path and simplifying the training architecture. The grouped rollouts are naturally parallelizable across inference workers. The trade-off is increased generation cost, since several trajectories must be sampled per task, and group-level synchronization can introduce its own stragglers. For short, verifiable MarketCanvas tasks, however, GRPO may be operationally simpler than PPO while making good use of the environment's deterministic reward signal.
