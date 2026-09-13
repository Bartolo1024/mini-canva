# Project glossary

| Term | Meaning in MarketCanvas |
| --- | --- |
| State | Canvas, elements, target, stable IDs, relationships, and episode progress |
| Observation | State encoded for Gymnasium, or semantic JSON returned through MCP |
| Action | Add, move, fully update, delete, or finish |
| Episode / trajectory | One attempt from reset to finish or budget exhaustion; a stored trajectory records its ordered actions |
| Policy | The program or model choosing actions; it is outside the simulator |
| Deterministic | Identical initial conditions and ordered actions reproduce state and reward |
| Schema failure | Invalid request fields/types/ranges; no attempt is consumed |
| Semantic failure | A valid request that cannot apply, such as an unknown ID; one attempt is consumed |
| Termination | Finish or exhaustion of the task's action budget |
| Truncation | External interruption; the base environment never reports it |
| TaskSpec | Explicit structured requirements consumed by the reward function |
| Selector | Element selection by the conjunction of supplied ID, role, and type |
| Role | Caller-supplied semantic label; it does not prove useful content exists |
| CTA | Call to action, represented by a labeled button shape |
| T / task satisfaction | Unweighted arithmetic mean of constraint scores |
| Q / generic quality | Harmonic mean of bounds, overlap, contrast/readability, and validity |
| G / hard gate | Multiplier applied when any explicitly hard constraint fails |
| Diagnostic reward | The current score read without stepping or awarding an episode return |
| Reward hacking | Optimizing a numerical rule while producing an unwanted design |
| Bounding / ink box | Rectangle approximating an element or its drawn text; ink boxes are not glyph masks |
| Half-open geometry | A box includes its left/top edges and excludes right/bottom; touching boxes have zero intersection area |
| z-index | Drawing depth, with stable IDs breaking ties |
| Contrast | Relative-luminance ratio; one readability factor, not complete accessibility compliance |
| Codec | Conversion between canonical actions/state and Gymnasium spaces |
| MCP | Protocol for discovering tools and exchanging their arguments/results |
| stdio server | A subprocess communicating over standard input/output, launched by its client |
| Rendering | Converting the current semantic canvas into RGB pixels or a PNG |
| LLM / VLM | A language model / a model that also accepts images; neither runs inside MarketCanvas |
| PPO | An RL training method; the write-up discusses possible scaling, but no trainer is implemented |

See [CORE.md](CORE.md) for lifecycle examples and [REWARD_EQUATIONS.md](REWARD_EQUATIONS.md)
for numerical definitions.
