"""Global YAML policies affect scoring and rendering without changing TaskSpec weights."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from marketcanvas_env.config import load_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("hard_failure_gate", -0.1),
        ("hard_failure_gate", 0.51),
        ("min_visible_ratio", 0),
        ("min_visible_ratio", 1.1),
        ("min_area_ratio", 0),
        ("min_area_ratio", -0.1),
        ("min_area_ratio", 1.1),
        ("min_area_ratio", float("nan")),
        ("min_area_ratio", float("inf")),
        ("min_area_ratio", True),
        ("contrast_full_credit_ratio", 0.9),
        ("contrast_full_credit_ratio", 21.1),
        ("side_region_falloff_span", 0),
        ("center_region_falloff_span", 0.51),
        ("alignment_falloff_span", 0),
        ("side_region_falloff_span", 5e-324),
        ("min_readable_font_size", 7),
        ("min_readable_font_size", 73),
        ("text_inset", -1),
        ("max_geometry_magnitude", 0),
        ("max_geometry_magnitude", 1e308),
        ("color_families.neutral_saturation_below", 1.1),
        ("color_families.neutral_value_below", -0.1),
        ("color_families.hue_red_end", 41),
        ("color_families.hue_red_start", 260),
    ],
)
def test_invalid_reward_settings_fail_when_loading_yaml(tmp_path, field, value):
    data = load_config().model_dump()
    target = data["reward"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.4"])
def test_reward_numbers_reject_booleans_nonfinite_and_strings(tmp_path, value):
    data = load_config().model_dump()
    data["reward"]["hard_failure_gate"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(path)


def test_reward_settings_are_required_and_deeply_immutable(tmp_path):
    config = load_config()
    with pytest.raises(ValidationError):
        config.reward.hard_failure_gate = 0.2
    with pytest.raises(ValidationError):
        config.reward.color_families.hue_red_end = 20
    data = config.model_dump()
    del data["reward"]["hard_failure_gate"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="hard_failure_gate"):
        load_config(path)


def test_changed_reward_yaml_reaches_runtime_schema_and_renderer(tmp_path):
    source = Path(__file__).parents[1] / "src" / "marketcanvas_env"
    package = tmp_path / "marketcanvas_env"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
    data = load_config().model_dump()
    data["reward"].update(
        hard_failure_gate=0.25,
        min_visible_ratio=0.5,
        min_area_ratio=0.02,
        contrast_full_credit_ratio=7.0,
        side_region_falloff_span=0.5,
        center_region_falloff_span=0.25,
        alignment_falloff_span=0.25,
        min_readable_font_size=24,
        text_inset=10,
        max_geometry_magnitude=10000.0,
        color_families=dict(
            neutral_saturation_below=0.2,
            neutral_value_below=0.2,
            hue_red_end=20.0,
            hue_orange_end=50.0,
            hue_yellow_end=90.0,
            hue_green_end=180.0,
            hue_blue_end=270.0,
            hue_red_start=330.0,
        ),
    )
    (package / "config" / "environment_config.yaml").write_text(yaml.safe_dump(data))
    code = r"""
import colorsys
from math import isclose

from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.colors import color_family
from marketcanvas_env.reward.constraints import region_score, visibility_score, evaluate_constraint
from marketcanvas_env.reward.geometry import finite_number
from marketcanvas_env.reward.scene import Scene, text_content_box, text_ink_box
from marketcanvas_env.reward.tasks import Constraint, TaskSpec
from marketcanvas_env.rendering import render_rgb

canvas = {"width": 800, "height": 600, "background": "#FFFFFF"}
image = dict(id="logo", type="image", role="logo", color="#000000", x=-40, y=100,
             width=100, height=100, z_index=0)
state = {"canvas": canvas, "elements": [image]}
c = Constraint(id="present", kind="exists", selector={"role": "logo"})
assert c.min_visible_ratio == 0.5
assert c.min_area_ratio == 0.02
assert Constraint.model_json_schema()["properties"]["min_area_ratio"]["default"] == 0.02
assert Constraint.model_json_schema()["properties"]["min_visible_ratio"]["default"] == 0.5
assert evaluate_constraint(c, Scene(state))[0] == 1
smaller = {**image, "x": 0, "width": 80}
assert evaluate_constraint(c, Scene({"canvas": canvas, "elements": [smaller]}))[0] == 0
assert visibility_score(Scene(state).elements[0]) == 1
strict = Constraint(id="strict", kind="exists", selector={"role": "logo"}, min_visible_ratio=0.9)
assert evaluate_constraint(strict, Scene(state))[0] == 0
assert region_score("left", 0.4, 0.5) == 1
assert isclose(region_score("center", 0.2, 0.5), 0.8)
image["x"] = 150  # center 200, distance 200 = configured quarter-canvas tolerance
alignment = Constraint(id="aligned", kind="alignment", selector={"role": "logo"},
                       value="horizontal_centers", reference="canvas")
assert evaluate_constraint(alignment, Scene(state))[0] == 0
missing = TaskSpec(prompt="x", constraints=[dict(id="required", kind="exists",
                   selector={"role": "missing"}, hard=True)])
report = compute_reward_breakdown(state, missing)
assert report["hard_gate"] == 0.25 and report["reward"] == -0.75
assert compute_reward_breakdown(state, missing) == report
assert finite_number(10000) == 10000 and finite_number(10001) is None

text = dict(id="text", type="text", role="body", x=100, y=100, width=150, height=60,
            color=None, text_color="#777777", content="Hi", font_size=20, z_index=0)
text_state = {"canvas": canvas, "elements": [text]}
assert Scene(text_state).elements[0].usable_ratio == 0  # configured 24px minimum
text["font_size"] = 24
scene = Scene(text_state)
e = scene.elements[0]
assert e.usable_ratio == 1
assert text_content_box(e.rect).x == text_ink_box(e).x == 110
assert text_content_box(e.rect).width == 130
report = compute_reward_breakdown(text_state, TaskSpec(prompt="x"))
assert isclose(report["quality"]["contrast"], scene.contrast(e) / 7)
default_contrast = Constraint(id="contrast", kind="contrast_min", selector={"role": "body"})
assert default_contrast.value == 7
assert isclose(evaluate_constraint(default_contrast, scene)[0], scene.contrast(e) / 7)
explicit_contrast = Constraint(id="contrast", kind="contrast_min", selector={"role": "body"}, value=3)
assert explicit_contrast.value == 3
assert evaluate_constraint(explicit_contrast, scene)[0] == 1
pixels = render_rgb(text_state)
assert (pixels[100:110, 100:250] == 255).all()
assert (pixels[110:150, 110:230] != 255).any()
assert (pixels == render_rgb(text_state)).all()
# Right alignment clips a long word to the configured inset in the renderer.
text.update(width=35, text_align="right", content="HHHH")
pixels = render_rgb(text_state)
assert (pixels[100:160, 100:110] == 255).all()
assert (pixels[110:150, 110:125] != 255).any()

for hue, family in [(16,"red"), (45,"orange"), (80,"yellow"), (175,"green"),
                    (265,"blue"), (335,"red")]:
    assert color_family(colorsys.hsv_to_rgb(hue / 360, 1, 1)) == family
assert color_family(colorsys.hsv_to_rgb(60 / 360, 0.18, 1)) == "neutral"
assert color_family(colorsys.hsv_to_rgb(60 / 360, 1, 0.18)) == "neutral"
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
