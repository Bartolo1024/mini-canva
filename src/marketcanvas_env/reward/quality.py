"""Equation (3): Q = harmonic_mean(B, O, C, V); no global layout preference.

Read evaluate_quality, then bounds_score -> overlap_score -> contrast_score ->
validity_score. The contrast diagnostic includes text usability, not just WCAG ratio.
"""

from itertools import combinations

from marketcanvas_env.config import config
from marketcanvas_env.reward.geometry import intersection_area, visible_ratio
from marketcanvas_env.reward.scene import ElementView, Scene


def evaluate_quality(scene: Scene) -> tuple[float, dict[str, float]]:
    """Equation (3): compute B, O, C, V in order, then Q = 4 / (1/B + 1/O + 1/C + 1/V).

    Q = 0 if any component is zero. Keep the four components separately inspectable.
    """
    bounds = bounds_score(scene)
    overlap = overlap_score(scene)
    contrast = contrast_score(scene)
    validity = validity_score(scene)
    components = {"bounds": bounds, "overlap": overlap, "contrast": contrast, "validity": validity}
    quality = harmonic_mean(list(components.values()))
    return quality, components


def bounds_score(scene: Scene) -> float:
    """B = min_e b_e; b_e = area(rect_e intersect canvas) / area(rect_e).

    For text-bearing elements b_e is additionally limited by usable visibility u_e.
    Invalid elements contribute 0; an empty canvas has B = 0.
    """
    return min((element_bounds_score(element, scene) for element in scene.elements), default=0.0)


def element_bounds_score(element: ElementView, scene: Scene) -> float:
    """b_e = min(on_canvas_ratio_e, u_e) for text; otherwise b_e = on_canvas_ratio_e."""
    bound = visible_ratio(element.rect, scene.canvas) if element.valid else 0.0
    return min(bound, element.usable_ratio) if element.text_bearing else bound


def overlap_score(scene: Scene) -> float:
    """O = clip(1 - max_{i<j} collision_ratio(i,j), 0, 1); max(empty) = 0.

    Worst-pair aggregation prevents unrelated decoys from diluting a collision.
    """
    worst_collision = max(
        (collision_ratio(a, b) for a, b in combinations(scene.elements, 2)), default=0.0
    )
    return max(0.0, min(1.0, 1.0 - worst_collision))


def collision_ratio(a: ElementView, b: ElementView) -> float:
    """o_ij = area(rect_i intersect rect_j) / min(area_i, area_j) for forbidden pairs.

    Image/unlabeled-background underlays are exempt. Invalid pairs contribute 0
    here; their invalidity is handled by B and V rather than geometric overlap.
    """
    if is_background_underlay(a) or is_background_underlay(b):
        return 0.0
    if not a.valid or not b.valid or a.rect is None or b.rect is None:
        return 0.0
    return intersection_area(a.rect, b.rect) / min(a.rect.area, b.rect.area)


def is_background_underlay(element: ElementView) -> bool:
    """Exempt(e) = [type_e = image] or [role_e = background and not text_bearing_e].

    This is the existing metadata-based pair classification, not an aesthetic test.
    """
    return element.type == "image" or (element.role == "background" and not element.text_bearing)


def contrast_score(scene: Scene) -> float:
    """C = min_{text-bearing e} (min(rho_e / rho_quality, 1) * u_e).

    rho_e is the raw WCAG ratio; u_e is scene text usability. With no text C = 1.
    The report keeps its historical key 'contrast', although C also includes usability.
    """
    return min(
        (text_readability_score(e, scene) for e in scene.elements if e.text_bearing), default=1.0
    )


def text_readability_score(element: ElementView, scene: Scene) -> float:
    """c_e = min(rho_e / configured_contrast_target, 1) * u_e.

    Raw contrast remains separate in Scene.contrast; this product measures usability.
    """
    raw_contrast = scene.contrast(element)
    contrast = min(raw_contrast / config.reward.contrast_full_credit_ratio, 1.0)
    return contrast * element.usable_ratio


def validity_score(scene: Scene) -> float:
    """V = count(valid elements) / count(elements).

    V = 0 for an empty canvas, invalid canvas metadata, or a malformed element collection.
    """
    if not scene.canvas_valid or not scene.structure_valid:
        return 0.0
    return sum(e.valid for e in scene.elements) / len(scene.elements) if scene.elements else 0.0


def harmonic_mean(values: list[float]) -> float:
    """H(x_1,...,x_n) = n / sum(1/x_i); H = 0 if empty or any x_i <= 0."""
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1 / value for value in values)
