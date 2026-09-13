"""Deterministic sRGB parsing, WCAG contrast, and documented HSV families."""

import colorsys
import re

from marketcanvas_env.config import config

RGB = tuple[float, float, float]


def parse_color(value: object) -> RGB | None:
    """channel = int(hex_pair, 16) / 255; accept six-digit RGB hex or return None."""
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        return None
    return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def linearize_srgb_channel(channel: float) -> float:
    """c_linear = c / 12.92 if c <= 0.04045; otherwise ((c + 0.055) / 1.055)^2.4."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: RGB) -> float:
    """L = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear."""
    linear = tuple(linearize_srgb_channel(channel) for channel in rgb)
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def contrast_ratio(foreground: RGB | None, background: RGB | None) -> float:
    """rho = (max(L_fg, L_bg) + 0.05) / (min(L_fg, L_bg) + 0.05).

    Unknown colors return 0; valid sRGB pairs have ratios in [1, 21]. This function
    calculates contrast only; clipping, size, and readability are handled separately.
    """
    if foreground is None or background is None:
        return 0.0
    a, b = relative_luminance(foreground), relative_luminance(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def color_family(rgb: RGB) -> str:
    """Use the configured global HSV hue bands and neutral thresholds."""
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    policy = config.reward.color_families
    if saturation < policy.neutral_saturation_below or value < policy.neutral_value_below:
        return "neutral"
    hue *= 360
    if hue < policy.hue_red_end or hue >= policy.hue_red_start:
        return "red"
    if hue < policy.hue_orange_end:
        return "orange"
    if hue < policy.hue_yellow_end:
        return "yellow"
    if hue < policy.hue_green_end:
        return "green"
    if hue < policy.hue_blue_end:
        return "blue"
    return "purple"
