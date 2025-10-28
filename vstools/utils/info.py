from __future__ import annotations

from fractions import Fraction
from typing import SupportsFloat, SupportsInt

from jetpytools import CustomNotImplementedError, fallback, mod_x

from ..enums import Dar, Sar
from ..exceptions import UnsupportedColorFamilyError
from ..types import HoldsVideoFormat, VideoFormatLike
from ..vs_proxy import vs

__all__ = [
    "get_color_family",
    "get_depth",
    "get_framerate",
    "get_h",
    "get_plane_sizes",
    "get_resolutions",
    "get_sample_type",
    "get_subsampling",
    "get_var_infos",
    "get_video_format",
    "get_w",
]


def get_var_infos(frame: vs.VideoNode | vs.VideoFrame) -> tuple[vs.VideoFormat, int, int]:
    """
    Get information from a variable resolution clip or frame.
    """

    if isinstance(frame, vs.VideoNode) and not (frame.width and frame.height and frame.format):
        with frame.get_frame(0) as frame:
            return get_var_infos(frame)

    assert frame.format

    return frame.format, frame.width, frame.height


def get_video_format(
    value: SupportsInt | VideoFormatLike | HoldsVideoFormat, /, *, sample_type: int | vs.SampleType | None = None
) -> vs.VideoFormat:
    """
    Retrieve a VapourSynth VideoFormat object from various input types.

    Args:
        value: The format source. This can be:

               - A bidepth format if `value < 32`
               - A unique format ID
               - A VideoFormat-like object
               - An object holding a VideoFormat (i.e., exposing a `format` attribute)

        sample_type: Optional override for the sample type. Accepts either an integer or a SampleType. If None, the
            default or inferred sample type is used.

    Returns:
        A VideoFormat object derived from the input.
    """
    if isinstance(value, vs.VideoFormat):
        return value

    if sample_type is not None:
        sample_type = vs.SampleType(sample_type)

    if isinstance(value, SupportsInt):
        value = int(value)

        if value > 32 or value == 0:
            return vs.core.get_video_format(value)

        if sample_type is None:
            sample_type = vs.SampleType(value == 32)

        return vs.core.query_video_format(vs.YUV, sample_type, value)

    if sample_type is not None:
        return value.format.replace(sample_type=sample_type)

    return value.format


def get_depth(clip: VideoFormatLike | HoldsVideoFormat, /) -> int:
    """
    Get the bitdepth of a given clip or value.
    """

    return get_video_format(clip).bits_per_sample


def get_sample_type(clip: VideoFormatLike | HoldsVideoFormat | vs.SampleType, /) -> vs.SampleType:
    """
    Get the sample type of a given clip.
    """

    if isinstance(clip, vs.SampleType):
        return clip

    return get_video_format(clip).sample_type


def get_color_family(clip: VideoFormatLike | HoldsVideoFormat | vs.ColorFamily, /) -> vs.ColorFamily:
    """
    Get the color family of a given clip.
    """

    if isinstance(clip, vs.ColorFamily):
        return clip

    return get_video_format(clip).color_family


def get_framerate(clip: vs.VideoNode | Fraction | tuple[int, int] | float) -> Fraction:
    """
    Get the framerate from any object holding it.
    """

    if isinstance(clip, vs.VideoNode):
        return clip.fps

    if isinstance(clip, Fraction):
        return clip

    if isinstance(clip, tuple):
        return Fraction(*clip)

    return Fraction(clip)


def get_plane_sizes(frame: vs.VideoNode | vs.VideoFrame, /, index: int) -> tuple[int, int]:
    """
    Get the size of a given clip's plane using the index.
    """

    assert frame.format and frame.width

    width, height = frame.width, frame.height

    if index != 0:
        width >>= frame.format.subsampling_w
        height >>= frame.format.subsampling_h

    return width, height


def get_resolutions(clip: vs.VideoNode | vs.VideoFrame) -> tuple[tuple[int, int, int], ...]:
    """
    Get a tuple containing the resolutions of every plane of the given clip.
    """

    assert clip.format

    return tuple((plane, *get_plane_sizes(clip, plane)) for plane in range(clip.format.num_planes))


def get_subsampling(clip: VideoFormatLike | HoldsVideoFormat, /) -> str:
    """
    Get the subsampling of a clip as a human-readable name.

    Args:
        clip: Input clip.

    Returns:
        String with a human-readable name.

    Raises:
        CustomValueError: Unknown subsampling.
    """

    fmt = get_video_format(clip)

    UnsupportedColorFamilyError.check(fmt, vs.YUV, get_subsampling)

    if fmt.subsampling_w == 2 and fmt.subsampling_h == 2:
        return "410"

    if fmt.subsampling_w == 2 and fmt.subsampling_h == 0:
        return "411"

    if fmt.subsampling_w == 1 and fmt.subsampling_h == 1:
        return "420"

    if fmt.subsampling_w == 1 and fmt.subsampling_h == 0:
        return "422"

    if fmt.subsampling_w == 0 and fmt.subsampling_h == 1:
        return "440"

    if fmt.subsampling_w == 0 and fmt.subsampling_h == 0:
        return "444"

    raise CustomNotImplementedError(
        "This subsampling doesn't have a supported human-readable name", get_subsampling, fmt.name
    )


def get_w(
    height: float,
    ar_or_ref: vs.VideoNode | vs.VideoFrame | SupportsFloat | Dar | Sar = 16 / 9,
    /,
    mod: int | None = None,
) -> int:
    """
    Calculate the width given a height and an aspect ratio.

    Either an aspect ratio (as a float), a reference clip, or a Dar/Sar object can be given.
    A mod can also be set, which will ensure the output width is MOD#.

    The output is rounded by default (as fractional output resolutions are not supported anywhere).

    Args:
        height: Height to use for the calculation.
        ar_or_ref: Aspect ratio, reference clip, or Dar/Sar object from which the AR will be calculated. Default: 1.778
            (16 / 9).
        mod: Mod for the output width to comply to. If None, do not force it to comply to anything. Default: None.

    Returns:
        Calculated width.
    """
    if isinstance(ar_or_ref, (Dar, Sar)):
        aspect_ratio = float(ar_or_ref)
    elif not isinstance(ar_or_ref, SupportsFloat):
        ref = ar_or_ref
        assert ref.format

        aspect_ratio = ref.width / ref.height
        mod = fallback(mod, 2**ref.format.subsampling_w)
    else:
        aspect_ratio = float(ar_or_ref)

    width = height * aspect_ratio

    if mod:
        return mod_x(width, mod)

    return round(width)


def get_h(
    width: float,
    ar_or_ref: vs.VideoNode | vs.VideoFrame | SupportsFloat | Dar | Sar = 16 / 9,
    /,
    mod: int | None = None,
) -> int:
    """
    Calculate the height given a width and an aspect ratio.

    Either an aspect ratio (as a float), a reference clip, or a Dar/Sar object can be given.
    A mod can also be set, which will ensure the output height is MOD#.

    The output is rounded by default (as fractional output resolutions are not supported anywhere).

    Args:
        width: Width to use for the calculation.
        ar_or_ref: Aspect ratio, reference clip, or Dar/Sar object from which the AR will be calculated. Default: 1.778
            (16 / 9).
        mod: Mod for the output width to comply to. If None, do not force it to comply to anything. Default: None.

    Returns:
        Calculated height.
    """
    if isinstance(ar_or_ref, (Dar, Sar)):
        aspect_ratio = 1.0 / float(ar_or_ref)
    elif not isinstance(ar_or_ref, SupportsFloat):
        ref = ar_or_ref
        assert ref.format

        aspect_ratio = ref.height / ref.width
        mod = fallback(mod, 2**ref.format.subsampling_h)
    else:
        aspect_ratio = 1.0 / float(ar_or_ref)

    height = width * aspect_ratio

    if mod:
        return mod_x(height, mod)

    return round(height)
