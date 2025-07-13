from __future__ import annotations

import vapoursynth as vs
from jetpytools import normalize_seq

from ..enums import ColorRange, ColorRangeT
from ..types import HoldsVideoFormatT, VideoFormatT
from .info import get_depth, get_video_format

__all__ = [
    "get_lowest_value",
    "get_lowest_values",
    "get_neutral_value",
    "get_neutral_values",
    "get_peak_value",
    "get_peak_values",
    "scale_delta",
    "scale_mask",
    "scale_value",
]


def scale_value(
    value: int | float,
    input_depth: int | VideoFormatT | HoldsVideoFormatT,
    output_depth: int | VideoFormatT | HoldsVideoFormatT,
    range_in: ColorRangeT | None = None,
    range_out: ColorRangeT | None = None,
    scale_offsets: bool = True,
    chroma: bool = False,
    family: vs.ColorFamily | None = None,
) -> int | float:
    """
    Converts the value to the specified bit depth, or bit depth of the clip/format specified.

    Args:
        value: Value to scale.
        input_depth: Input bit depth, or clip, frame, format from where to get it.
        output_depth: Output bit depth, or clip, frame, format from where to get it.
        range_in: Color range of the input value
        range_out: Color range of the desired output.
        scale_offsets: Whether or not to apply & map YUV zero-point offsets. Set to True when converting absolute color
            values. Set to False when converting color deltas. Only relevant if integer formats are involved.
        chroma: Whether or not to treat values as chroma values instead of luma.
        family: Which color family to assume for calculations.

    Returns:
        Scaled value.
    """

    out_value = float(value)

    in_fmt = get_video_format(input_depth)
    out_fmt = get_video_format(output_depth)

    if range_in is None:
        if isinstance(input_depth, vs.VideoNode):
            range_in = ColorRange(input_depth)
        elif vs.RGB in (in_fmt.color_family, family):
            range_in = ColorRange.FULL
        else:
            range_in = ColorRange.LIMITED
    else:
        range_in = ColorRange.from_param(range_in, scale_value)

    if range_out is None:
        if isinstance(output_depth, vs.VideoNode):
            range_out = ColorRange(output_depth)
        elif vs.RGB in (out_fmt.color_family, family):
            range_out = ColorRange.FULL
        else:
            range_out = ColorRange.LIMITED
    else:
        range_out = ColorRange.from_param(range_out, scale_value)

    if input_depth == output_depth and range_in == range_out and in_fmt.sample_type == out_fmt.sample_type:
        return out_value

    if vs.RGB in (in_fmt.color_family, out_fmt.color_family, family):
        chroma = False

    input_peak = get_peak_value(in_fmt, chroma, range_in, family)
    input_lowest = get_lowest_value(in_fmt, chroma, range_in, family)
    output_peak = get_peak_value(out_fmt, chroma, range_out, family)
    output_lowest = get_lowest_value(out_fmt, chroma, range_out, family)

    if scale_offsets and in_fmt.sample_type is vs.INTEGER:
        if chroma:
            out_value -= 128 << (in_fmt.bits_per_sample - 8)
        elif range_in.is_limited:
            out_value -= 16 << (in_fmt.bits_per_sample - 8)

    out_value *= (output_peak - output_lowest) / (input_peak - input_lowest)

    if scale_offsets and out_fmt.sample_type is vs.INTEGER:
        if chroma:
            out_value += 128 << (out_fmt.bits_per_sample - 8)
        elif range_out.is_limited:
            out_value += 16 << (out_fmt.bits_per_sample - 8)

    if out_fmt.sample_type is vs.INTEGER:
        out_value = max(min(round(out_value), get_peak_value(out_fmt, range_in=ColorRange.FULL)), 0)

    return out_value


def scale_mask(
    value: int | float,
    input_depth: int | VideoFormatT | HoldsVideoFormatT,
    output_depth: int | VideoFormatT | HoldsVideoFormatT,
) -> int | float:
    """
    Converts the value to the specified bit depth, or bit depth of the clip/format specified.
    Intended for mask clips which are always full range.

    Args:
        value: Value to scale.
        input_depth: Input bit depth, or clip, frame, format from where to get it.
        output_depth: Output bit depth, or clip, frame, format from where to get it.

    Returns:
        Scaled value.
    """

    return scale_value(value, input_depth, output_depth, ColorRange.FULL, ColorRange.FULL)


def scale_delta(
    value: int | float,
    input_depth: int | VideoFormatT | HoldsVideoFormatT,
    output_depth: int | VideoFormatT | HoldsVideoFormatT,
    range_in: ColorRangeT | None = None,
    range_out: ColorRangeT | None = None,
) -> int | float:
    """
    Converts the value to the specified bit depth, or bit depth of the clip/format specified.
    Uses the clip's range (if only one clip is passed) for both depths.
    Intended for filter thresholds.

    Args:
        value: Value to scale.
        input_depth: Input bit depth, or clip, frame, format from where to get it.
        output_depth: Output bit depth, or clip, frame, format from where to get it.
        range_in: Color range of the input value
        range_out: Color range of the desired output.

    Returns:
        Scaled value.
    """

    if isinstance(input_depth, vs.VideoNode) != isinstance(output_depth, vs.VideoNode):
        if isinstance(input_depth, vs.VideoNode):
            clip_range = ColorRange.from_video(input_depth)

        if isinstance(output_depth, vs.VideoNode):
            clip_range = ColorRange.from_video(output_depth)

        range_in = clip_range if range_in is None else range_in
        range_out = clip_range if range_out is None else range_out

    return scale_value(value, input_depth, output_depth, range_in, range_out, False)


def get_lowest_value(
    clip_or_depth: int | VideoFormatT | HoldsVideoFormatT,
    chroma: bool = False,
    range_in: ColorRangeT | None = None,
    family: vs.ColorFamily | None = None,
) -> float:
    """
    Returns the lowest value for the specified bit depth, or bit depth of the clip/format specified.

    Args:
        clip_or_depth: Input bit depth, or clip, frame, format from where to get it.
        chroma: Whether to get luma (default) or chroma plane value.
        range_in: Whether to get limited or full range lowest value.
        family: Which color family to assume for calculations.

    Returns:
        Lowest possible value.
    """

    fmt = get_video_format(clip_or_depth)

    if is_rgb := vs.RGB in (fmt.color_family, family):
        chroma = False

    if fmt.sample_type is vs.FLOAT:
        return -0.5 if chroma else 0.0

    if range_in is None:
        if isinstance(clip_or_depth, vs.VideoNode):
            range_in = ColorRange.from_video(clip_or_depth, func=get_lowest_value)
        elif is_rgb:
            range_in = ColorRange.FULL
        else:
            range_in = ColorRange.LIMITED

    if ColorRange(range_in).is_limited:
        return 16 << get_depth(fmt) - 8

    return 0


def get_lowest_values(
    clip_or_depth: int | VideoFormatT | HoldsVideoFormatT,
    range_in: ColorRangeT | None = None,
    family: vs.ColorFamily | None = None,
) -> list[float]:
    """
    Get the lowest values of all planes of a specified format.
    """

    return normalize_seq(
        [
            get_lowest_value(clip_or_depth, False, range_in, family),
            get_lowest_value(clip_or_depth, True, range_in, family),
        ],
        get_video_format(clip_or_depth).num_planes,
    )


def get_neutral_value(clip_or_depth: int | VideoFormatT | HoldsVideoFormatT) -> float:
    """
    Returns the neutral point value (e.g. as used by std.MakeDiff) for the specified bit depth,
    or bit depth of the clip/format specified.

    Args:
        clip_or_depth: Input bit depth, or clip, frame, format from where to get it.

    Returns:
        Neutral value.
    """

    fmt = get_video_format(clip_or_depth)

    if fmt.sample_type is vs.FLOAT:
        return 0.0

    return 1 << (get_depth(fmt) - 1)


def get_neutral_values(clip_or_depth: int | VideoFormatT | HoldsVideoFormatT) -> list[float]:
    """
    Get the neutral values of all planes of a specified format.
    """

    fmt = get_video_format(clip_or_depth)
    return normalize_seq(get_neutral_value(fmt), fmt.num_planes)


def get_peak_value(
    clip_or_depth: int | VideoFormatT | HoldsVideoFormatT,
    chroma: bool = False,
    range_in: ColorRangeT | None = None,
    family: vs.ColorFamily | None = None,
) -> float:
    """
    Returns the peak value for the specified bit depth, or bit depth of the clip/format specified.

    Args:
        clip_or_depth: Input bit depth, or clip, frame, format from where to get it.
        chroma: Whether to get luma (default) or chroma plane value.
        range_in: Whether to get limited or full range peak value.
        family: Which color family to assume for calculations.

    Returns:
        Highest possible value.
    """

    fmt = get_video_format(clip_or_depth)

    if is_rgb := vs.RGB in (fmt.color_family, family):
        chroma = False

    if fmt.sample_type is vs.FLOAT:
        return 0.5 if chroma else 1.0

    if range_in is None:
        if isinstance(clip_or_depth, vs.VideoNode):
            range_in = ColorRange.from_video(clip_or_depth, func=get_peak_value)
        elif is_rgb:
            range_in = ColorRange.FULL
        else:
            range_in = ColorRange.LIMITED

    if ColorRange(range_in).is_limited:
        return (240 if chroma else 235) << get_depth(fmt) - 8

    return (1 << get_depth(fmt)) - 1


def get_peak_values(
    clip_or_depth: int | VideoFormatT | HoldsVideoFormatT,
    range_in: ColorRangeT | None = None,
    family: vs.ColorFamily | None = None,
) -> list[float]:
    """
    Get the peak values of all planes of a specified format.
    """

    return normalize_seq(
        [get_peak_value(clip_or_depth, False, range_in, family), get_peak_value(clip_or_depth, True, range_in, family)],
        get_video_format(clip_or_depth).num_planes,
    )
