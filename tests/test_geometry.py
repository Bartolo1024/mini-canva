"""Rectangle boundaries and ordering without rendering assumptions."""

from marketcanvas_env.geometry import relationship_names, spatial_relationships
from marketcanvas_env.models import Element


def rectangle(eid, x=0, y=0, width=10, height=10):
    return Element(id=eid, type="shape", x=x, y=y, width=width, height=height)


def test_touching_edges_are_separate_but_ordered():
    a, b = rectangle(1), rectangle(2, x=10)
    assert relationship_names(a, b) == ("left_of", "center_aligned_y")
    assert "overlaps" not in relationship_names(b, a)
    assert "above" in relationship_names(a, rectangle(3, y=10))


def test_equal_rectangles_overlap_and_contain_each_other_without_self_relations():
    a, b = rectangle(1), rectangle(2)
    expected = ("overlaps", "contains", "center_aligned_x", "center_aligned_y")
    assert relationship_names(a, b) == expected
    assert relationship_names(b, a) == expected
    relations = spatial_relationships([b, a])
    assert len(relations) == 8
    assert all(r["source_id"] != r["target_id"] for r in relations)
    assert [r["source_id"] for r in relations] == [1] * 4 + [2] * 4


def test_containment_is_directed_and_uses_unclipped_geometry():
    outer = rectangle(1, x=-100, y=-100, width=50, height=50)
    inner = rectangle(2, x=-90, y=-90, width=10, height=10)
    assert "contains" in relationship_names(outer, inner)
    assert "contains" not in relationship_names(inner, outer)
    assert "overlaps" in relationship_names(outer, inner)


def test_center_tolerance_uses_exact_half_pixel_centers():
    a = rectangle(1, width=10)
    assert "center_aligned_x" in relationship_names(a, rectangle(2, x=2, width=10))
    assert "center_aligned_x" not in relationship_names(a, rectangle(2, x=2, width=11))


def test_partial_intersection_does_not_imply_containment():
    relations = relationship_names(rectangle(1), rectangle(2, x=9, y=9))
    assert relations == ("overlaps",)
