# MarketCanvas Reward — One-Page Summary

The reward measures **task satisfaction and generic canvas usability**. It is
deterministic, independent of LLM inference, and driven by structured TaskSpec
constraints. Prompts must be translated into constraints before scoring. There
are no per-task weights or global centering/color preferences.

## Construction

Each of the \(n\) constraints produces a satisfaction score \(s_i\in[0,1]\):

$$
T=\frac{1}{n}\sum_{i=1}^{n}s_i
\qquad
Q=\frac{4}{B^{-1}+O^{-1}+C^{-1}+V^{-1}}
$$

$$
G=
\begin{cases}
1,&\text{all hard constraints score }1\\
0.4,&\text{otherwise}
\end{cases}
\qquad
\boxed{R=\operatorname{clamp}\!\left(G(T+Q)-1,-1,1\right)}
$$

| Variable | Meaning |
| --- | --- |
| \(T\) | Unweighted mean of the \(n\) constraint scores \(s_i\) |
| \(B\) | Worst on-canvas element-area ratio, further limited by usable visibility for text |
| \(O\) | One minus the worst forbidden overlap: intersection area divided by the smaller element's area |
| \(C\) | Worst text score: contrast ratio capped relative to **4.5:1**, multiplied by usable visibility |
| \(V\) | Fraction of structurally valid element records |
| \(Q\) | Harmonic mean of \(B,O,C,V\); zero if any component is zero |
| \(G\) | Hard-constraint gate; a failure caps reward at **−0.2** with the default multiplier |
| \(R\) | Final reward; clamp restricts it to \([-1,1]\) |

With no constraints, \(T=0\). An empty canvas has \(B=V=0\); no forbidden pairs
gives \(O=1\); no text gives \(C=1\). Invalid canvas structure forces \(V=0\).

Text usability includes font size, inset clipping, canvas clipping, and occlusion.
Required existence uses a default **0.8 usable-visibility threshold**. Heuristic
thresholds and the failure multiplier are globally configurable in YAML. The
breakdown exposes constraint explanations, \(T\), \(B/O/C/V\), \(Q\), and \(G\).
Reward is issued once at termination; diagnostic reads do not award reward.

## Reward-hacking defenses

| Attack | Current defense | Remaining limitation |
| --- | --- | --- |
| Offscreen or covered required content | Clipping and occlusion reduce usable presence | Visibility uses rectangular approximations |
| Microscopic elements | Text/font/inset checks reject tiny text | Tiny images and logos can receive full credit |
| Duplicate CTAs or create-everything spam | Existence is capped; explicit hard counts enforce cardinality | No general clutter penalty; disjoint spam can tie clean designs |
| Perfect-center stacking or overlap dilution | Worst-pair overlap prevents dilution by unrelated objects | Images and unlabeled backgrounds are exempt |
| Omit difficult requirements | Hard gate prevents positive reward | Missing dependencies generally receive additional zeros |
| Perfect contrast but unreadable text | Contrast is multiplied by text usability | Partial backgrounds beneath ink can escape detection |
| Role spoofing or repeated keywords | Selectors check supplied type/role; text scores are capped | Roles remain self-reported; selection is not best-match optimization |
| Repeated reward collection | Reads are pure; post-terminal actions fail | Collectors must distinguish diagnostics from episode returns |

## Remaining risks

**Maximum-reward exploits remain:** a microscopic logo, duplicate CTAs without
a count requirement, and a background patch that makes headline ink invisible.
Contrast lookup requires whole-text-box containment, so it can miss a patch
beneath only the ink.

Equal constraint contributions also depend on task representation: several
overlapping predicates implicitly emphasize one requirement. Exact duplicates
are rejected, but logical equivalence is not detected. There is no global
one-to-one requirement matching.

These are missing measurements, not merely aggregation problems: changing
addition to multiplication cannot distinguish designs with identical \(T\) and
\(Q\). Normalized size checks, conservative background sampling, and bounded
extra-element penalties are possible improvements—not implemented safeguards.

Details: [equations and code map](REWARD_EQUATIONS.md) ·
[measured exploits and regression tests](REWARD_HACKING.md).
