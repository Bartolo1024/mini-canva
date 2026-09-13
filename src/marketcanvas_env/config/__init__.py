"""One validated, immutable configuration shared by schemas and transitions."""

import math
import sys
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColorFamilyConfig(BaseModel):
    """Global HSV classification policy; RGB/WCAG definitions remain mathematical constants."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    neutral_saturation_below: float = Field(ge=0, le=1)
    neutral_value_below: float = Field(ge=0, le=1)
    hue_red_end: float = Field(gt=0, lt=360)
    hue_orange_end: float = Field(gt=0, lt=360)
    hue_yellow_end: float = Field(gt=0, lt=360)
    hue_green_end: float = Field(gt=0, lt=360)
    hue_blue_end: float = Field(gt=0, lt=360)
    hue_red_start: float = Field(gt=0, lt=360)

    @model_validator(mode="after")
    def ordered_hue_boundaries(self) -> "ColorFamilyConfig":
        boundaries = (
            self.hue_red_end,
            self.hue_orange_end,
            self.hue_yellow_end,
            self.hue_green_end,
            self.hue_blue_end,
            self.hue_red_start,
        )
        if any(a >= b for a, b in pairwise(boundaries)):
            raise ValueError("hue boundaries must be strictly increasing")
        return self


class RewardConfig(BaseModel):
    """Shared scoring conventions, required explicitly in YAML; never prompt weights."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    # At most 1/2 keeps every hard-failure reward nonpositive for T,Q in [0,1].
    hard_failure_gate: float = Field(ge=0, le=0.5)
    min_visible_ratio: float = Field(gt=0, le=1)
    min_area_ratio: float = Field(gt=0, le=1)
    contrast_full_credit_ratio: float = Field(ge=1, le=21)
    side_region_falloff_span: float = Field(gt=0, le=1)
    center_region_falloff_span: float = Field(gt=0, le=0.5)
    alignment_falloff_span: float = Field(gt=0, le=1)
    min_readable_font_size: int = Field(ge=8, le=72)
    text_inset: int = Field(ge=0)
    # Bounded endpoint differences and their area products must remain finite.
    max_geometry_magnitude: float = Field(gt=0, le=math.sqrt(sys.float_info.max) / 4)
    color_families: ColorFamilyConfig

    @model_validator(mode="after")
    def finite_shaping_arithmetic(self) -> "RewardConfig":
        if not math.isfinite(1 / self.side_region_falloff_span):
            raise ValueError("side_region_falloff_span must have a finite reciprocal")
        if self.text_inset > self.max_geometry_magnitude:
            raise ValueError("text_inset cannot exceed max_geometry_magnitude")
        return self


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    max_elements: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_content_length: int = Field(gt=0, le=4096)  # Scene's supported content bound.
    max_role_length: int = Field(ge=4)  # Must fit the default/padding role "none".
    max_task_json_length: int = Field(gt=0)
    reward: RewardConfig


def load_config(path: Path | None = None) -> EnvironmentConfig:
    """Read explicit YAML or the packaged default, independent of the working directory."""
    source = path if path is not None else files(__package__).joinpath("environment_config.yaml")
    return EnvironmentConfig.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))


# Schemas are built at import time. Keep the same config for the whole process.
config = load_config()
