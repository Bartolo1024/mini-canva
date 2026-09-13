"""Pixel-level renderer contracts, bounded clipping, and observational purity."""

from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from examples.reward.benchmarks import BENCHMARKS
from marketcanvas_env.rendering import render_rgb, save_png
from marketcanvas_env.reward.scene import Scene, get_font, text_ink_box


def element(**overrides):
    return {
        "id": 1,
        "type": "shape",
        "role": "custom-role",
        "x": 10,
        "y": 20,
        "width": 100,
        "height": 60,
        "color": "#FF0000",
        "text_color": "#000000",
        "content": "",
        "font_size": 24,
        "z_index": 0,
        **overrides,
    }


def state(*elements):
    return {
        "canvas": {"width": 800, "height": 600, "background_color": "#FFFFFF"},
        "elements": list(elements),
    }


def test_rgb_shape_half_open_bounds_and_independent_output():
    snapshot = state(element())
    pixels = render_rgb(snapshot)
    assert pixels.shape == (600, 800, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[20:80, 10:110], np.broadcast_to([255, 0, 0], (60, 100, 3)))
    assert tuple(pixels[20, 110]) == (255, 255, 255)
    assert tuple(pixels[80, 10]) == (255, 255, 255)
    pixels[:] = 0
    assert tuple(render_rgb(snapshot)[0, 0]) == (255, 255, 255)


@pytest.mark.parametrize("identifiers", [(2, 10), ("a", "z"), (1, "a")])
def test_stacking_uses_z_then_stable_id_not_input_order(identifiers):
    lower = element(id=identifiers[0], color="#FF0000")
    higher = element(id=identifiers[1], color="#0000FF")
    for records in ((lower, higher), (higher, lower)):
        assert tuple(render_rgb(state(*records))[25, 15]) == (0, 0, 255)
    lower["z_index"] = 1
    assert tuple(render_rgb(state(higher, lower))[25, 15]) == (255, 0, 0)


def test_huge_offcanvas_geometry_clips_before_allocating(monkeypatch):
    original_new = Image.new
    sizes = []

    def bounded_new(mode, size, *args, **kwargs):
        sizes.append(size)
        assert size[0] <= 800 and size[1] <= 600
        return original_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", bounded_new)
    snapshot = state(
        element(x=-999999900, y=-999999900, width=1000000000, height=1000000000),
        element(id=2, x=1000000000, y=1000000000),
    )
    pixels = render_rgb(snapshot)
    assert sizes
    assert tuple(pixels[99, 99]) == (255, 0, 0)
    assert tuple(pixels[100, 100]) == (255, 255, 255)


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_text_placement_matches_shared_ink_box_and_keeps_fill_transparent(align):
    text = element(type="text", content="Hello", text_align=align, width=200)
    snapshot = state(element(id=2, color="#00FF00", width=200, z_index=-1), text)
    pixels = render_rgb(snapshot)
    ink = text_ink_box(Scene(snapshot).elements[1])
    ys, xs = np.where(np.any(pixels[20:80, 10:210] != [0, 255, 0], axis=2))
    assert len(xs) > 0
    assert xs.min() + 10 >= ink.x
    assert xs.max() + 10 < ink.right
    assert ys.min() + 20 == ink.y
    assert ys.max() + 20 < ink.bottom
    assert tuple(pixels[20, 10]) == (0, 255, 0)
    # Antialiasing produces intermediate foreground/background colors.
    assert np.any((pixels[24:50, 14:206, 1] > 0) & (pixels[24:50, 14:206, 1] < 255))


def test_button_label_clips_to_inset_and_higher_shape_covers_it():
    button = element(subtype="button", content="Very long label", width=40, height=30)
    pixels = render_rgb(state(button))
    assert np.any(pixels[24:46, 14:46, 0] != 255)
    assert np.all(pixels[20:24, 10:50] == [255, 0, 0])
    assert np.all(pixels[20:50, 46:50] == [255, 0, 0])
    covered = render_rgb(state(button, element(id=2, color="#0000FF")))
    assert np.all(covered[20:50, 10:50] == [0, 0, 255])


def test_tiny_text_and_image_content_are_not_drawn():
    snapshot = state(
        element(type="text", content="Hello", width=8, height=8),
        element(id=2, type="image", content="Do not draw this", x=200),
    )
    pixels = render_rgb(snapshot)
    assert np.all(pixels[20:28, 10:18] == 255)
    assert np.all(pixels[20:80, 200:300] == [255, 0, 0])


def test_text_clipped_by_canvas_keeps_same_glyph_position():
    visible = render_rgb(state(element(type="text", content="Hello", x=0, y=0)))
    clipped = render_rgb(state(element(type="text", content="Hello", x=-10, y=-5)))
    np.testing.assert_array_equal(clipped[:55, :90], visible[5:60, 10:100])


def test_fractional_rectangles_round_outward_to_touched_pixels():
    pixels = render_rgb(state(element(x=10.25, y=20.25, width=1, height=1)))
    assert np.all(pixels[20:22, 10:12] == [255, 0, 0])
    assert tuple(pixels[22, 12]) == (255, 255, 255)


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"canvas": {"width": 10**9, "height": 10**9, "background_color": "#FFFFFF"}},
        {**state(), "elements": "invalid"},
        state(element(width=0)),
        state(element(color="not-a-color")),
        state(element(), element()),
        state(element(type="text", content="two\nlines")),
    ],
)
def test_invalid_snapshots_fail_before_image_allocation(snapshot, monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid snapshots must be rejected before image allocation")

    monkeypatch.setattr(Image, "new", unexpected_allocation)
    with pytest.raises(ValueError):
        render_rgb(snapshot)


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmarks_render_repeatably_without_state_mutation(name, tmp_path):
    snapshot = BENCHMARKS[name].state
    before = deepcopy(snapshot)
    pixels = render_rgb(snapshot)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    save_png(snapshot, first)
    save_png(snapshot, second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), pixels)
    assert snapshot == before


def test_png_does_not_create_missing_parent_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_png(state(), tmp_path / "missing" / "canvas.png")


def test_shared_font_is_pillow_default_aileron():
    assert get_font(24).getname() == ("Aileron", "Regular")
