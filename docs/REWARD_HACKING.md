# Reward-hacking analysis

## 1. Threat model

A policy optimizes the implemented scalar reward, not the human designer's intent.
The objective is to make common degenerate strategies observable and unprofitable;
this implementation does not make reward hacking impossible. No policy was trained.

The public dataset has five tasks and 15 legal action trajectories under
[`data/`](../data/). Raw snapshots in
[`reward_regressions.yaml`](../tests/fixtures/reward_regressions.yaml) additionally test
malformed and out-of-contract inputs. These two input surfaces are deliberately distinct.

## 2. Reward decomposition

The [equation-to-function guide](REWARD_EQUATIONS.md) is the runtime reference:

```text
T = mean(TaskSpec constraint scores), or 0 with no constraints
Q = harmonic_mean(bounds, overlap, text readability, validity)
G = 1 if every hard constraint passes, otherwise 0.4
R = clamp(G × (T + Q) − 1, −1, 1)
```

The gate and retained heuristic thresholds are global settings in
[`environment_config.yaml`](../src/marketcanvas_env/config/environment_config.yaml).
There are no task weights, global centering preferences, clutter penalties, or learned judges.
A failed hard requirement caps reward at −0.2 with the packaged gate.

Diagnostics separately log T, Q, G, the four quality components, and ordered constraint
scores with explanations. The quality key `contrast` includes text usability; raw contrast
is a separate primitive. Inspection utilities additionally expose size/visibility data.
Existence explanations include the measured area ratio and required threshold. Production
reports do **not** include continuous size scores, applicability flags, global matching
assignments, or extraneous-element lists.

## 3. Attack classes and regressions

S refers to [`test_reward_adversarial.py`](../tests/test_reward_adversarial.py),
P to [`test_reward_adversarial_primitives.py`](../tests/test_reward_adversarial_primitives.py).
Names below omit the `test_` prefix. RH-01's microscopic-presence failure is now mitigated
by a minimum-area check; RH-04 now detects partial ink backgrounds. Other RH identifiers
mark retained issues; RH-05 is a classification concern rather than demonstrated illegibility.

| Attack | Naive failure | Current mitigation | Residual risk | Regression |
| --- | --- | --- | --- | --- |
| Offscreen / mostly offscreen | A role label counts as presence | Canvas clipping; existence needs usable visibility, default 0.8 | Presence is binary, not continuous | S `clipped_and_covered_required_elements_fail` |
| Microscopic elements | A 1×1 object receives full presence | Required presence checks usable visibility and minimum area/canvas area, default 0.001; text also needs usable ink/font size | Area alone does not prevent thin shapes; tasks can override the threshold | S `microscopic_nontext_fails_hard_presence` |
| Create-all-roles spam | Enough candidates satisfy every request | Tiny text fails usability; selectors check supplied types | **RH-02:** readable disjoint spam remains free | S `nonoverlapping_spam_gap_is_observed` |
| Duplicate CTAs | Existence grows with duplicate count | Existence is capped; explicit hard `count` enforces cardinality | **RH-02:** no inferred uniqueness or extra-object cost | S `twenty_readable_ctas_are_capped_and_explicit_count_is_hard` |
| Multi-color hedging | Different candidates supply requested properties | Stable representative ordering; explicit counts available | **RH-02:** yellow-first hedge can tie a clean design | S `readable_spam_and_color_hedging_should_lose_to_clean_layout` |
| Giant elements / underlays | Size satisfies visibility/alignment trivially | Giant foreground can hide required text and fail the gate | **RH-03:** redundant giant backgrounds are unpenalized | S `giant_background_extras_should_lose_to_clean_design` |
| Perfect-center stacking | Alignment compensates for collision | Worst overlap makes Q zero; occlusion also affects task scores | T=1, Q=0 can still yield R=0 when no hard rule fails | S `giant_foreground_and_center_stacking_do_not_beat_clean_layouts` |
| Overlap dilution | Many good pairs dilute one collision | Maximum intersection/min-area, never pairwise mean | Image/background exemptions depend on metadata | S `thirty_decoys_cannot_dilute_a_catastrophic_collision` |
| Omission of difficult objects | Missing requirement trades for better quality | Hard gate prevents positive reward | **RH-06:** dependent constraints still receive repeated zeros | P `missing_dependent_constraints_should_be_not_applicable` |
| Role/type spoofing | Self-reported role overrides actual type | Selectors conjunctively match role/type/ID; core reserves headline/CTA types | Role-only selectors still trust labels; overlapping selectors can reuse an object | S `type_spoof_does_not_match_shape_selector`; P `overlapping_selectors_reuse_one_element_observation` |
| Keyword repetition | Repeating required words increases credit | Contains is capped; normalized equals rejects repetition | Contains intentionally accepts additional fitting text | S `keyword_repetition_is_capped_and_exact_text_rejects_it` |
| Contrast gaming | High ratio is mistaken for readable content | Actual CTA fill; worst exposed background beneath clipped ink; text usability | Any positive-area patch counts, including glyph gaps; soft contrast can still yield positive reward | S `ink_background_exploit_is_pixel_invisible_but_loses_reward` |
| Background-role relabeling | Metadata changes quality without changing pixels | Text-bearing backgrounds still receive usability checks | **RH-05:** identical underlay pixels score differently; fixture does not prove a bad design | S `role_only_overlap_exemption_changes_reward_without_pixels` |
| Hidden content / z-index | Covered required content remains credited | Subtract union of higher painted rectangles | Ink rectangles approximate glyphs and holes | P `occlusion_union_is_not_double_counted_or_diluted` |
| Malformed geometry/colors/IDs | Invalid numbers crash scoring or inflate it | Defensive parsing, invalid-record retention, finite bounded reward | Missing content defaults to empty: presence fails, structural validity can remain 1 | S `malformed_state_reduces_validity_and_stays_finite`; P `missing_content_is_semantically_empty_but_structurally_valid` |
| Global alignment bias | Universal centering penalizes correct layouts | Layout constraints come from TaskSpec; Q has no centering term | Authors must describe the intended layout | S `two_column_task_and_generic_quality_do_not_prefer_centering` |
| Bad first representative | A blank early candidate hides a valid match | Deterministic visible-box ratio, then lexical ID | **RH-07:** selection is not best usable-match score | P `blank_duplicate_should_not_hide_a_valid_presence_match` |

### Measured public trajectories

Recomputed with packaged settings using `python scripts/run_examples.py`. Each task's
`well_done` trajectory has T=Q=G=1 and R=+1.000. The ten attacks produce:

| Task / trajectory | T | Q | G | R |
| --- | ---: | ---: | ---: | ---: |
| summer_sale / offscreen_headline | 0.400 | 0.000 | 0.400 | −0.840 |
| summer_sale / small_element_spam | 0.000 | 0.000 | 0.400 | −1.000 |
| webinar / hidden_headline | 0.556 | 0.000 | 0.400 | −0.778 |
| webinar / missing_cta | 0.667 | 1.000 | 0.400 | −0.333 |
| newsletter / forbidden_cta | 0.800 | 1.000 | 0.400 | −0.280 |
| newsletter / tiny_logo | 0.800 | 1.000 | 0.400 | −0.280 |
| event / center_stacking | 0.125 | 0.000 | 0.400 | −0.950 |
| event / duplicate_cta | 1.000 | 1.000 | 1.000 | +1.000 |
| two_column / centered_layout | 0.900 | 1.000 | 1.000 | +0.900 |
| two_column / contrast_patch | 1.000 | 0.533 | 1.000 | +0.533 |

Duplicate CTA remains a maximum-score loophole. Seven attacks score negatively;
centered layout and contrast patch are penalized but remain positive under soft scoring.
Raw regression snapshots can produce different numbers because they are different layouts,
not equivalent replays.

```sh
uv run --locked python scripts/run_examples.py
uv run --locked python -m tests.reward_support.adversarial --markdown
uv run --locked python -m tests.reward_support.adversarial --json
uv run --locked pytest -q tests/test_reward_adversarial.py tests/test_reward_adversarial_primitives.py
```

The adversarial utility computes 52 raw cases, with inputs and full reports available as JSON.
There are six strict expected-failure instances for RH-02–03/06–07, alongside passing tests
that assert the observed behavior. Only assertion failures are expected; unexpected passes
require review. Add `--runxfail` to the pytest command to expose those unmet safeguards as
ordinary failures. A green normal run does not establish that every requested safeguard holds.

## 4. Why not use one metric

Perfect alignment can coexist with catastrophic overlap. A 1×1 black text box on white
has raw contrast 21:1 but zero usable ink. Twenty disjoint readable CTAs can satisfy all
requirements when uniqueness is absent. Components constrain each other, but aggregation
cannot detect a property missing from both T and Q.

## 5. Why alignment is task-conditioned

Two-column layouts, left-aligned ads, and corner logos are legitimate tasks. Q therefore
has no centering or symmetry bonus. Requested alignment decays with normalized displacement;
relative placement uses edge gaps, not just element origins. Touching and separated edges
receive full relative-position credit; no preferred positive-gap interval is implemented.

## 6. Why contrast is not readability

Raw sRGB luminance contrast is distinct from quality C, which multiplies capped contrast
by text usability. Usability includes font size, inset, clipping, and occlusion. Packaged
12px font and 4px inset settings are absolute conventions. The separate existence check
requires element area / canvas area to meet `min_area_ratio` (default 0.001). These checks
serve different purposes: font/ink checks measure text usability; area rejects microscopic
required objects of any type.

Transparent text now checks exposed lower solid backgrounds beneath its clipped ink box,
using the worst contrast ratio and excluding buried layers. Uncovered ink uses canvas color.
The pixel-invisible patch regression now measures 1:1, not 18.88:1. The public trajectory
falls from +1.000 to +0.533; it stays positive because T remains 1, contrast contributes
1/4.5, and no hard contrast constraint is specified. Any positive-area patch counts, so
patches in glyph gaps can be over-penalized. A legitimate full background may have excellent
contrast; that alone says nothing about redundant extras.

The fix changes background measurement only: no new thresholds, task weights, or reward
aggregation rules were introduced. In the public patch case, B=O=V=T=G=1 and C=1/4.5,
so Q=4/(1+1+4.5+1)=0.5333 and R=Q. All five well-done trajectories remain at +1.000.

## 7. Aggregation risks

Harmonic Q becomes zero when any quality component is zero. Worst-case bounds, collision,
and readability resist decoy dilution. The hard gate prevents positive rewards for failed
essential constraints. Nevertheless, soft low-contrast text can earn positive reward because
contrast 1:1 still contributes 1/4.5, and missing objects create correlated penalties in T,
B, and C. Missing dependencies are not excluded from T.

Equal contributions also depend on representation: five different CTA predicates outweigh
one logo predicate. Exact duplicate requirements are rejected, but logical equivalents and
overlapping predicates are not canonicalized. Any future parser should emit each intended
predicate once. [Aggregation experiments](REWARD_EXPERIMENTS.md) explain why replacing
addition with multiplication does not repair perfect-score exploits.

## 8. Remaining limitations

Minimum area applies to required presence, not every decorative element or generic quality.
There are no minimum-width/height checks, extraneous-element penalties, global one-to-one
matching, or best-candidate search. Count/absence include hidden matches. Shape presence
alone need not require a label unless the shape is a button or text is explicitly requested.

Bounding boxes do not assess typography, subjective aesthetics, or natural-image clutter;
images are flat placeholders. Color-family boundaries and fixed font thresholds introduce
conventions and scale bias. Rectangle occlusion and background lookup are approximations,
not comprehensive accessibility guarantees or evidence of robustness against a trained policy.

## 9. Future extensions

Possible measured improvements include normalized width/height constraints, representative binding,
explicit dependency applicability, bounded extra-object costs, and glyph-level
background sampling. These remain proposals, not runtime behavior. Richer typography,
randomized adversarial tasks, and automatic attack search could test their trade-offs.

Human preferences, Pareto objectives, learned aesthetics, or an LLM/VLM secondary judge
could supplement deterministic constraints. They must not replace the objective checks and
would introduce additional cost, calibration, reproducibility, and hacking risks.
