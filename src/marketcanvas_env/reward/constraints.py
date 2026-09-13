"""Task constraint equations, dispatched by kind without prompt-specific rules.

Read ``evaluate_constraint`` first, then the named evaluator for the TaskSpec kind.
Each evaluator returns its score and selection explanation; the common wrapper
bounds the score and formats the diagnostic. Selectors preserve Scene's ordering:
most rules use the first representative, never the candidate with the best score.
"""

from typing import TYPE_CHECKING

from marketcanvas_env.config import config
from marketcanvas_env.reward.colors import color_family, parse_color
from marketcanvas_env.reward.geometry import intersection_area
from marketcanvas_env.reward.tasks import Constraint

if TYPE_CHECKING:
    from marketcanvas_env.reward.scene import ElementView, Scene


_NO_MATCH = (0.0, "no usable matching element")


def evaluate_constraint(constraint: Constraint, scene: "Scene") -> tuple[float, str]:
    """s_i = clip(evaluate_kind(constraint, scene), 0, 1), with a selection explanation.

    Constraint validation guarantees a supported kind. Dispatch follows the
    TaskSpec kind only; selectors and values are data, never prompt branches.
    """
    score, detail = _CONSTRAINT_EVALUATORS[constraint.kind](constraint, scene)
    score = clamp_score(score)
    return score, f"{constraint.kind}: {detail}; score {score:.3f}"


def clamp_score(value: float) -> float:
    """Return clip(value, 0, 1), the common constraint score range."""
    return max(0.0, min(1.0, value))


def visibility_score(element: "ElementView") -> float:
    """a(e) = clip(u_e / tau, 0, 1); u_e is usable ratio, tau the configured threshold.

    Invalid elements receive 0. This attenuation is distinct from quality validity.
    """
    return (
        clamp_score(element.usable_ratio / config.reward.min_visible_ratio)
        if element.valid
        else 0.0
    )


def region_score(value: str, x: float, y: float) -> float:
    """Score a normalized center (x, y) against the requested region.

    Left/top: clip((1 - coordinate) / side_span); right/bottom:
    clip(coordinate / side_span). Center: min over both coordinates of
    clip((0.5 - abs(coordinate - 0.5)) / center_span). Compound regions
    take the minimum of their component scores. All clips are to [0, 1].
    """

    def low(v: float) -> float:
        return clamp_score((1 - v) * (1 / config.reward.side_region_falloff_span))

    def high(v: float) -> float:
        return clamp_score(v * (1 / config.reward.side_region_falloff_span))

    scores = {
        "left": low(x),
        "right": high(x),
        "top": low(y),
        "bottom": high(y),
        "center": min(
            clamp_score((0.5 - abs(x - 0.5)) / config.reward.center_region_falloff_span),
            clamp_score((0.5 - abs(y - 0.5)) / config.reward.center_region_falloff_span),
        ),
    }
    return min(scores[part] for part in value.split("-"))


def _select_usable_element(constraint: Constraint, scene: "Scene") -> "ElementView | None":
    """Resolve the first representative, requiring positive usable visibility."""
    matches = scene.select(constraint.selector)
    first = matches[0] if matches else None
    return first if first is not None and visibility_score(first) > 0 else None


def _select_pair(
    constraint: Constraint, scene: "Scene"
) -> "tuple[ElementView, ElementView] | None":
    """Resolve distinct first representatives with valid rectangles; do not search further."""
    subjects, objects = scene.select(constraint.subject), scene.select(constraint.object)
    if not subjects or not objects:
        return None
    subject, obj = subjects[0], objects[0]
    if subject.rect is None or obj.rect is None or subject is obj:
        return None
    return subject, obj


def _select_group(constraint: Constraint, scene: "Scene") -> "list[ElementView]":
    """Resolve one representative per selector, or none if any selector is empty.

    A singular selector supplies one representative for alignment and contrast,
    but supplies every match for no_overlap. Invalid rectangles are retained so
    individual equations can preserve their documented validity behavior.
    """
    if constraint.selectors:
        groups = [scene.select(selector) for selector in constraint.selectors]
    else:
        groups = [scene.select(constraint.selector)]
    if not all(groups):
        return []
    if constraint.kind == "no_overlap" and not constraint.selectors:
        return groups[0]
    return [group[0] for group in groups]


def _with_element_visibility(
    score: float, element: "ElementView", value: object
) -> tuple[float, str]:
    """s = raw_score * a(first); report the same representative used by the equation."""
    return score * visibility_score(element), f"selected {element.id}; requested {value!r}"


def _with_pair_visibility(
    score: float, subject: "ElementView", obj: "ElementView", label: str
) -> tuple[float, str]:
    """s = raw_score * min(a(subject), a(object))."""
    return (
        score * min(visibility_score(subject), visibility_score(obj)),
        f"selected {subject.id} and {obj.id}; {label}",
    )


def _with_group_visibility(
    score: float, selected: "list[ElementView]", label: object
) -> tuple[float, str]:
    """s = raw_score * min(a(e) for e in selected)."""
    return (
        score * min(visibility_score(element) for element in selected),
        f"selected {', '.join(element.id for element in selected)}; {label}",
    )


def evaluate_exists(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[first valid and u >= min_visible_ratio and area/canvas_area >= min_area_ratio]."""
    matches = scene.select(c.selector)
    first = matches[0] if matches else None
    area_ratio = (
        first.rect.area / scene.canvas.area
        if first is not None and first.rect is not None and scene.canvas is not None
        else 0.0
    )
    score = float(
        first is not None
        and first.valid
        and first.usable_ratio >= c.min_visible_ratio
        and area_ratio >= c.min_area_ratio
    )
    detail = (
        f"requires usable visibility >= {c.min_visible_ratio:g}"
        f" and area ratio >= {c.min_area_ratio:g}"
    )
    if first is not None:
        detail += f"; selected {first.id}: visibility {first.usable_ratio:.3f}, area {area_ratio:g}"
    return score, detail


def evaluate_absent(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[number of selector matches == 0], including unusable matching objects."""
    matches = scene.select(c.selector)
    return float(not matches), f"{len(matches)} matching elements; requires none"


def evaluate_count(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[count operator target], with operator in {eq, lte, gte}; count all matches."""
    count = len(scene.select(c.selector))
    score = float(
        {"eq": count == c.value, "lte": count <= c.value, "gte": count >= c.value}[c.operator]
    )
    return score, f"{count} matching elements {c.operator} {c.value}"


def evaluate_relative_position(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = clip(1 + edge_gap / min(relevant_spans)) * min(a(subject), a(object)).

    Above/below use vertical edge gaps and heights; left/right use horizontal
    gaps and widths. Touching or separated boxes receive full geometric credit.
    Missing, identical, or invalid-rectangle representatives give 0.
    """
    pair = _select_pair(c, scene)
    if pair is None:
        return _NO_MATCH
    subject, obj = pair
    a, b = subject.rect, obj.rect
    gap, span = {
        "above": (b.y - a.bottom, min(a.height, b.height)),
        "below": (a.y - b.bottom, min(a.height, b.height)),
        "left_of": (b.x - a.right, min(a.width, b.width)),
        "right_of": (a.x - b.right, min(a.width, b.width)),
    }[c.relation]
    score = clamp_score(1 + gap / span)
    return _with_pair_visibility(score, subject, obj, c.relation)


def evaluate_size_relation(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[subject.metric operator object.metric] * min(a(subject), a(object)).

    Width, height, and area comparisons are literal, with no preferred ratio.
    Missing, identical, or invalid-rectangle representatives give 0.
    """
    pair = _select_pair(c, scene)
    if pair is None:
        return _NO_MATCH
    subject, obj = pair
    av, bv = getattr(subject.rect, c.metric), getattr(obj.rect, c.metric)
    operator = c.operator or "gt"
    score = float({"gt": av > bv, "gte": av >= bv, "lt": av < bv, "lte": av <= bv}[operator])
    return _with_pair_visibility(score, subject, obj, c.metric)


def evaluate_alignment(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = clip(1 - (max(positions) - min(positions)) / canvas_span / tolerance) * min(a(e)).

    Positions are the requested centers or edges of selected representatives.
    An explicit canvas reference appends its corresponding center or edge.
    A missing group or any invalid rectangle gives 0; there is no global alignment.
    """
    selected = _select_group(c, scene)
    if not selected:
        return _NO_MATCH
    score = 0.0
    if all(element.rect is not None for element in selected):
        attr = {
            "horizontal_centers": "cx",
            "vertical_centers": "cy",
            "left_edges": "x",
            "right_edges": "right",
        }[c.value]
        scale = scene.height if attr == "cy" else scene.width
        positions = [getattr(element.rect, attr) for element in selected]
        if c.reference == "canvas":
            positions.append(
                {"cx": scene.width / 2, "cy": scene.height / 2, "x": 0, "right": scene.width}[attr]
            )
        # Normalize first: multiplying scale by span could underflow on raw snapshots.
        score = clamp_score(
            1.0 - (max(positions) - min(positions)) / scale / config.reward.alignment_falloff_span
        )
    return _with_group_visibility(score, selected, c.value or c.kind)


def evaluate_contrast_min(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = min(clip(contrast(e) / task_threshold)) * min(a(e)); missing groups give 0."""
    selected = _select_group(c, scene)
    if not selected:
        return _NO_MATCH
    score = min(clamp_score(scene.contrast(element) / float(c.value)) for element in selected)
    return _with_group_visibility(score, selected, c.value or c.kind)


def evaluate_no_overlap(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = (1 - max(intersection(a, b) / min(area(a), area(b)))) * min(a(e)).

    Distinct pairs with valid rectangles contribute to the maximum, defaulting
    to zero collision when no such pair exists. Unlike other group constraints,
    an empty selector group returns 1: there are no matching pairs to overlap.
    """
    selected = _select_group(c, scene)
    if not selected:
        return 1.0, "no matching pairs overlap"
    collision = max(
        (
            intersection_area(a.rect, b.rect) / min(a.rect.area, b.rect.area)
            for i, a in enumerate(selected)
            for b in selected[i + 1 :]
            if a is not b and a.rect is not None and b.rect is not None
        ),
        default=0.0,
    )
    return _with_group_visibility(1.0 - collision, selected, c.value or c.kind)


def evaluate_text_contains(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[casefold(target) is a substring of casefold(content)] * a(first)."""
    first = _select_usable_element(c, scene)
    if first is None:
        return _NO_MATCH
    score = float(str(c.value).casefold() in first.content.casefold())
    return _with_element_visibility(score, first, c.value)


def evaluate_text_equals(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[normalize(target) == normalize(content)] * a(first).

    Normalization case-folds and collapses whitespace. It does not remove repeats.
    """
    first = _select_usable_element(c, scene)
    if first is None:
        return _NO_MATCH
    score = float(
        " ".join(str(c.value).casefold().split()) == " ".join(first.content.casefold().split())
    )
    return _with_element_visibility(score, first, c.value)


def evaluate_color_family(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[family(selected_color) == target] * a(first); invalid colors give 0.

    Text uses text_color; shapes, buttons, and images use their fill color.
    """
    first = _select_usable_element(c, scene)
    if first is None:
        return _NO_MATCH
    color = first.text_color if first.type == "text" else first.color
    score = float(color_family(color) == c.value) if color is not None else 0.0
    return _with_element_visibility(score, first, c.value)


def evaluate_color_exact(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = 1[selected_color == parsed_target_color] * a(first); invalid colors give 0.

    Text uses text_color; shapes, buttons, and images use their fill color.
    """
    first = _select_usable_element(c, scene)
    if first is None:
        return _NO_MATCH
    color = first.text_color if first.type == "text" else first.color
    score = float(color == parse_color(c.value)) if color is not None else 0.0
    return _with_element_visibility(score, first, c.value)


def evaluate_region(c: Constraint, scene: "Scene") -> tuple[float, str]:
    """s = region_score(target, center_x / canvas_width, center_y / canvas_height) * a(first)."""
    first = _select_usable_element(c, scene)
    if first is None:
        return _NO_MATCH
    score = 0.0
    if first.rect is not None:
        score = region_score(
            str(c.value), first.rect.cx / scene.width, first.rect.cy / scene.height
        )
    return _with_element_visibility(score, first, c.value)


_CONSTRAINT_EVALUATORS = {
    "exists": evaluate_exists,
    "absent": evaluate_absent,
    "count": evaluate_count,
    "relative_position": evaluate_relative_position,
    "size_relation": evaluate_size_relation,
    "alignment": evaluate_alignment,
    "contrast_min": evaluate_contrast_min,
    "no_overlap": evaluate_no_overlap,
    "text_contains": evaluate_text_contains,
    "text_equals": evaluate_text_equals,
    "color_family": evaluate_color_family,
    "color_exact": evaluate_color_exact,
    "region": evaluate_region,
}
