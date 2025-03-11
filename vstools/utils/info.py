from __future__ import annotations

from fractions import Fraction
from typing import Any, SupportsFloat, overload

import vapoursynth as vs
from jetpytools import fallback, mod_x

from ..exceptions import UnsupportedSubsamplingError
from ..functions import depth, disallow_variable_format
from ..types import HoldsVideoFormatT, VideoFormatT

__all__ = [
    'get_var_infos',
    'get_video_format',

    'get_depth', 'get_sample_type', 'get_subsampling', 'get_color_family', 'get_framerate',

    'expect_bits',

    'get_plane_sizes', 'get_resolutions',

    'get_w', 'get_h'
]


def get_var_infos(frame: vs.VideoNode | vs.VideoFrame) -> tuple[vs.VideoFormat, int, int]:
    """Get information from a variable resolution clip or frame."""

    if isinstance(frame, vs.VideoNode) and not (
        frame.width and frame.height and frame.format
    ):
        with frame.get_frame(0) as frame:
            return get_var_infos(frame)

    assert frame.format

    return frame.format, frame.width, frame.height


@disallow_variable_format
def get_video_format(
    value: int | VideoFormatT | HoldsVideoFormatT, /, *, sample_type: int | vs.SampleType | None = None
) -> vs.VideoFormat:
    """Get the video format of a given value."""

    from ..utils.vs_enums import VSPresetVideoFormat

    if sample_type is not None:
        sample_type = vs.SampleType(sample_type)

    if isinstance(value, vs.VideoFormat):
        return value

    if isinstance(value, VSPresetVideoFormat):
        return vs.core.get_video_format(value)

    if isinstance(value, int):
        if value > 32:
            return vs.core.get_video_format(value)

        if sample_type is None:
            sample_type = vs.SampleType(value == 32)

        return vs.core.query_video_format(vs.YUV, sample_type, value)

    assert value.format

    if sample_type is not None:
        return value.format.replace(sample_type=sample_type)

    return value.format


def get_depth(clip: VideoFormatT | HoldsVideoFormatT, /) -> int:
    """Get the bitdepth of a given clip or value."""

    return get_video_format(clip).bits_per_sample


def get_sample_type(clip: VideoFormatT | HoldsVideoFormatT | vs.SampleType, /) -> vs.SampleType:
    """Get the sample type of a given clip."""

    if isinstance(clip, vs.SampleType):
        return clip

    return get_video_format(clip).sample_type


def get_color_family(clip: VideoFormatT | HoldsVideoFormatT | vs.ColorFamily, /) -> vs.ColorFamily:
    """Get the color family of a given clip."""

    if isinstance(clip, vs.ColorFamily):
        return clip

    return get_video_format(clip).color_family


def get_framerate(clip: vs.VideoNode | Fraction | tuple[int, int] | float) -> Fraction:
    """Get the framerate from any object holding it."""

    if isinstance(clip, vs.VideoNode):
        return clip.fps

    if isinstance(clip, Fraction):
        return clip

    if isinstance(clip, tuple):
        return Fraction(*clip)

    return Fraction(clip)


def expect_bits(clip: vs.VideoNode, /, expected_depth: int = 16, **kwargs: Any) -> tuple[vs.VideoNode, int]:
    """
    Expected output bitdepth for a clip.

    This function is meant to be used when a clip may not match the expected input bitdepth.
    Both the dithered clip and the original bitdepth are returned.

    :param clip:            Input clip.
    :param expected_depth:  Expected bitdepth.
                            Default: 16.

    :return:                Tuple containing the clip dithered to the expected depth and the original bitdepth.
    """

    bits = get_depth(clip)

    if bits != expected_depth:
        clip = depth(clip, expected_depth, **kwargs)

    return clip, bits


def get_plane_sizes(frame: vs.VideoNode | vs.VideoFrame, /, index: int) -> tuple[int, int]:
    """Get the size of a given clip's plane using the index."""

    assert frame.format and frame.width

    width, height = frame.width, frame.height

    if index != 0:
        width >>= frame.format.subsampling_w
        height >>= frame.format.subsampling_h

    return width, height


def get_resolutions(clip: vs.VideoNode | vs.VideoFrame) -> tuple[tuple[int, int, int], ...]:
    """Get a tuple containing the resolutions of every plane of the given clip."""

    assert clip.format

    return tuple(
        (plane, *get_plane_sizes(clip, plane)) for plane in range(clip.format.num_planes)
    )


def get_subsampling(clip: VideoFormatT | HoldsVideoFormatT, /) -> str | None:
    """
    Get the subsampling of a clip as a human-readable name.

    :param clip:                Input clip.

    :return:                    String with a human-readable name.

    :raises CustomValueError:   Unknown subsampling.
    """

    fmt = get_video_format(clip)

    if fmt.color_family != vs.YUV:
        return None

    if fmt.subsampling_w == 2 and fmt.subsampling_h == 2:
        return '410'

    if fmt.subsampling_w == 2 and fmt.subsampling_h == 0:
        return '411'

    if fmt.subsampling_w == 1 and fmt.subsampling_h == 1:
        return '420'

    if fmt.subsampling_w == 1 and fmt.subsampling_h == 0:
        return '422'

    if fmt.subsampling_w == 0 and fmt.subsampling_h == 1:
        return '440'

    if fmt.subsampling_w == 0 and fmt.subsampling_h == 0:
        return '444'

    raise UnsupportedSubsamplingError('Unknown subsampling.', get_subsampling)


@overload
def get_w(height: float, ar: SupportsFloat = 16 / 9, /, mod: int = 2) -> int:
    ...


@overload
def get_w(height: float, ref: vs.VideoNode | vs.VideoFrame, /, mod: int | None = None) -> int:
    ...


def get_w(
    height: float, ar_or_ref: vs.VideoNode | vs.VideoFrame | SupportsFloat = 16 / 9, /, mod: int | None = None
) -> int:
    """
    Calculate the width given a height and an aspect ratio.

    Either an aspect ratio (as a float) or a reference clip can be given.
    A mod can also be set, which will ensure the output width is MOD#.

    The output is rounded by default (as fractional output resolutions are not supported anywhere).

    :param height:          Height to use for the calculation.
    :param ar_or_ref:       Aspect ratio or reference clip from which the AR will be calculated.
                            Default: 1.778 (16 / 9).
    :param mod:             Mod for the output width to comply to. If None, do not force it to comply to anything.
                            Default: None.

    :return:                Calculated width.
    """
    if not isinstance(ar_or_ref, SupportsFloat):
        ref = ar_or_ref
        assert ref.format
        aspect_ratio = ref.width / ref.height
        mod = fallback(mod, ref.format.subsampling_w and 2 << ref.format.subsampling_w)
    else:
        aspect_ratio = float(ar_or_ref)

        if mod is None:
            mod = 0 if height % 2 else 2

    width = height * aspect_ratio

    if mod:
        return mod_x(width, mod)

    return round(width)


@overload
def get_h(width: float, ar: SupportsFloat = 16 / 9, /, mod: int = 2) -> int:
    ...


@overload
def get_h(width: float, ref: vs.VideoNode | vs.VideoFrame, /, mod: int | None = None) -> int:
    ...


def get_h(
    width: float, ar_or_ref: vs.VideoNode | vs.VideoFrame | SupportsFloat = 16 / 9, /, mod: int | None = None
) -> int:
    """
    Calculate the height given a width and an aspect ratio.

    Either an aspect ratio (as a float) or a reference clip can be given.
    A mod can also be set, which will ensure the output height is MOD#.

    The output is rounded by default (as fractional output resolutions are not supported anywhere).

    :param width:           Width to use for the calculation.
    :param ar_or_ref:       Aspect ratio or reference clip from which the AR will be calculated.
                            Default: 1.778 (16 / 9).
    :param mod:             Mod for the output width to comply to. If None, do not force it to comply to anything.
                            Default: None.

    :return:                Calculated height.
    """

    if not isinstance(ar_or_ref, SupportsFloat):
        ref = ar_or_ref
        assert ref.format
        aspect_ratio = ref.height / ref.width
        mod = fallback(mod, ref.format.subsampling_h and 2 << ref.format.subsampling_h)
    else:
        aspect_ratio = 1.0 / float(ar_or_ref)

        if mod is None:
            mod = 0 if width % 2 else 2

    height = width * aspect_ratio

    if mod:
        return mod_x(height, mod)

    return round(height)
