from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import EnumMeta
from functools import partial, reduce, wraps
from types import NoneType
from typing import Any, Literal, Self, SupportsIndex, overload
from weakref import WeakValueDictionary

from jetpytools import (
    CustomIndexError,
    CustomNotImplementedError,
    CustomStrEnum,
    CustomTypeError,
    FuncExcept,
    cachedproperty,
    normalize_seq,
    to_arr,
)

from vsjetpack import deprecated

from ..enums import ColorRange, ColorRangeLike, Matrix
from ..exceptions import ClipLengthError, UnsupportedColorFamilyError
from ..types import HoldsVideoFormat, Planes, VideoFormatLike, VideoNodeIterable
from ..utils import flatten, get_depth, get_lowest_value, get_peak_value, get_video_format
from ..vs_proxy import core, vs

__all__ = [
    "EXPR_VARS",
    "DitherType",
    "depth",
    "expect_bits",
    "flatten_vnodes",
    "frame2clip",
    "get_b",
    "get_g",
    "get_r",
    "get_u",
    "get_v",
    "get_y",
    "insert_clip",
    "join",
    "limiter",
    "plane",
    "split",
    "stack_clips",
    "stack_planes",
]


EXPR_VARS = [
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
]
"""
Variables to access clips in Expr.
"""


class _EnumDeprecationWarning(DeprecationWarning): ...


class _DitherTypeMeta(EnumMeta):
    @property
    @deprecated(
        '"AUTO" member is deprecated and will be removed in a future version. Please use RANDOM instead',
        category=_EnumDeprecationWarning,
    )
    def AUTO(cls) -> Literal[DitherType.RANDOM]:  # noqa: N802
        return DitherType.RANDOM


class DitherType(CustomStrEnum, metaclass=_DitherTypeMeta):
    """
    Enum for `zimg_dither_type_e` and fmtc `dmode`.
    """

    NONE = "none", 1
    """
    Round to nearest.
    """

    ORDERED = "ordered", 0
    """
    Bayer patterned dither.
    """

    RANDOM = "random", 8
    """
    Pseudo-random noise of magnitude 0.5.
    """

    ERROR_DIFFUSION = "error_diffusion", 6
    """
    Floyd-Steinberg error diffusion.
    """

    FLOYD_STEINBERG = ERROR_DIFFUSION

    SIERRA_2_4A = "sierra_2_4a", 3
    """
    Another type of error diffusion.
    Quick and excellent quality, similar to Floyd-Steinberg.
    """

    STUCKI = "stucki", 4
    """
    Another error diffusion kernel.
    Preserves delicate edges better but distorts gradients.
    """

    ATKINSON = "atkinson", 5
    """
    Another error diffusion kernel.
    Generates distinct patterns but keeps clean the flat areas (noise modulation).
    """

    OSTROMOUKHOV = "ostromoukhov", 7
    """
    Another error diffusion kernel.
    Slow, available only for integer input at the moment. Avoids usual F-S artifacts.
    """

    VOID = RANDOM

    QUASIRANDOM = "quasirandom", 9
    """
    Dither using quasirandom sequences.
    Good intermediary between void, cluster, and error diffusion algorithms.
    """

    _fmtc_dmode: int

    def __new__(cls, value: Any, fmtc_dmode: int) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj._fmtc_dmode = fmtc_dmode

        return obj

    def apply(
        self,
        clip: vs.VideoNode,
        out_fmt: vs.VideoFormat,
        range_in: ColorRange,
        range_out: ColorRange,
        force_fmtc: bool,
    ) -> vs.VideoNode:
        """
        Apply the given DitherType to a clip.
        """
        fmt = get_video_format(clip)

        if not (self.is_fmtc or force_fmtc):
            return clip.resize.Point(
                format=out_fmt,
                dither_type=self.value.lower(),
                range_in=range_in.value_zimg,
                range=range_out.value_zimg,
            )

        # Workaround because fmtc doesn't support FLOAT 16 input
        if fmt.sample_type is vs.FLOAT and fmt.bits_per_sample == 16:
            clip = DitherType.NONE.apply(clip, fmt.replace(bits_per_sample=32), range_in, range_out, False)

        return clip.fmtc.bitdepth(
            dmode=self._fmtc_dmode,
            bits=out_fmt.bits_per_sample,
            fulls=range_in is ColorRange.FULL,
            fulld=range_out is ColorRange.FULL,
        )

    @cachedproperty
    def is_fmtc(self) -> bool:
        """
        Whether the DitherType is exclusive to fmtc.
        """
        return self in (
            DitherType.SIERRA_2_4A,
            DitherType.STUCKI,
            DitherType.ATKINSON,
            DitherType.OSTROMOUKHOV,
            DitherType.QUASIRANDOM,
        )

    @overload
    @staticmethod
    def should_dither(
        in_fmt: VideoFormatLike | HoldsVideoFormat,
        out_fmt: VideoFormatLike | HoldsVideoFormat,
        /,
        in_range: ColorRangeLike | None = None,
        out_range: ColorRangeLike | None = None,
    ) -> bool:
        """
        Automatically determines whether dithering is needed for a given depth/range/sample type conversion.

        Args:
            in_fmt: Input clip, frame or video format.
            out_fmt: Output clip, frame or video format.
            in_range: Input color range.
            out_range: Output color range.

        Returns:
            Whether the clip should be dithered.
        """

    @overload
    @staticmethod
    def should_dither(
        in_bits: int,
        out_bits: int,
        /,
        in_range: ColorRangeLike | None = None,
        out_range: ColorRangeLike | None = None,
        in_sample_type: vs.SampleType | None = None,
        out_sample_type: vs.SampleType | None = None,
    ) -> bool:
        """
        Automatically determines whether dithering is needed for a given depth/range/sample type conversion.

        Args:
            in_bits: Input bitdepth.
            out_bits: Output bitdepth.
            in_range: Input color range.
            out_range: Output color range.
            in_sample_type: Input sample type.
            out_sample_type: Output sample type.

        Returns:
            Whether the clip should be dithered.
        """

    @staticmethod
    @deprecated("should_dither is deprecated and will be removed in a future version.", category=DeprecationWarning)
    def should_dither(
        in_bits_or_fmt: int | VideoFormatLike | HoldsVideoFormat,
        out_bits_or_fmt: int | VideoFormatLike | HoldsVideoFormat,
        /,
        in_range: ColorRangeLike | None = None,
        out_range: ColorRangeLike | None = None,
        in_sample_type: vs.SampleType | None = None,
        out_sample_type: vs.SampleType | None = None,
    ) -> bool:
        """
        Automatically determines whether dithering is needed for a given depth/range/sample type conversion.

        If an input range is specified, an output range *should* be specified, otherwise it assumes a range conversion.

        For an explanation of when dithering is needed:

        - Dithering is NEVER needed if the conversion results in a float sample type.
        - Dithering is ALWAYS needed for a range conversion (i.e. full to limited or vice-versa).
        - Dithering is ALWAYS needed to convert a float sample type to an integer sample type.
        - Dithering is needed when upsampling full range content except when one depth is a multiple of the other,
            when the upsampling is a simple integer multiplication, e.g. for 8 -> 16: (0-255) * 257 -> (0-65535).
        - Dithering is needed when downsampling limited or full range.

        Dithering is theoretically needed when converting from an integer depth greater than 10 to half float,
        despite the higher bit depth, but zimg's internal resampler currently does not dither for float output.

        Args:
            in_bits_or_fmt: Input bitdepth, clip, frame or video format.
            out_bits_or_fmt: Output bitdepth, clip, frame or video format.
            in_range: Input color range.
            out_range: Output color range.
            in_sample_type: Input sample type.
            out_sample_type: Output sample type.

        Returns:
            Whether the clip should be dithered.
        """
        in_fmt = get_video_format(in_bits_or_fmt, sample_type=in_sample_type)
        out_fmt = get_video_format(out_bits_or_fmt, sample_type=out_sample_type)

        in_range = ColorRange.from_param_with_fallback(in_range)
        out_range = ColorRange.from_param_with_fallback(out_range)

        if out_fmt.sample_type is vs.FLOAT:
            return False

        if in_fmt.sample_type is vs.FLOAT:
            return True

        if in_range != out_range:
            return True

        in_bits = in_fmt.bits_per_sample
        out_bits = out_fmt.bits_per_sample

        if in_bits == out_bits:
            return False

        if in_bits > out_bits:
            return True

        return in_range == ColorRange.FULL and bool(out_bits % in_bits)


def depth(
    clip: vs.VideoNode,
    bitdepth: VideoFormatLike | HoldsVideoFormat | int | None = None,
    /,
    sample_type: int | vs.SampleType | None = None,
    *,
    range_in: ColorRangeLike | None = None,
    range_out: ColorRangeLike | None = None,
    dither_type: str | DitherType = DitherType.RANDOM,
    force_fmtc: bool = False,
) -> vs.VideoNode:
    """
    A convenience bitdepth conversion function using only internal plugins if possible.

    This uses exclusively internal plugins except for specific dither_types.

    Example:
        ```py
        rc_8 = vs.core.std.BlankClip(format=vs.YUV420P8)
        rc_10 = depth(src_8, 10)
        print(rc_10.format.name)  # YUV420P10

        rc2_10 = vs.core.std.BlankClip(format=vs.RGB30)
        rc2_8 = depth(src2_10, 8, dither_type=Dither.RANDOM)  # override default dither behavior
        print(rc2_8.format.name)  # RGB24
        ```

    Args:
        clip: Input clip.
        bitdepth: Desired bitdepth of the output clip.
        sample_type: Desired sample type of output clip. Allows overriding default float/integer behavior.
        range_in: Input pixel range (defaults to input `clip`'s range).
        range_out: Output pixel range (defaults to input `clip`'s range).
        dither_type: Dithering algorithm. Allows overriding default dithering behavior.
        force_fmtc: Force usage of fmtc to apply dithering.

    Returns:
        Converted clip with desired bit depth and sample type. `ColorFamily` will be same as input.
    """
    if range_in is not None:
        clip = ColorRange.ensure_presence(clip, range_in)

    in_fmt = get_video_format(clip)
    out_fmt = get_video_format(bitdepth or clip, sample_type=sample_type)

    range_out = ColorRange.from_param_or_video(range_out, clip)
    range_in = ColorRange.from_param_or_video(range_in, clip)

    if (in_fmt.bits_per_sample, in_fmt.sample_type, range_in) == (
        out_fmt.bits_per_sample,
        out_fmt.sample_type,
        range_out,
    ):
        return clip

    new_format = in_fmt.replace(bits_per_sample=out_fmt.bits_per_sample, sample_type=out_fmt.sample_type)

    return DitherType.from_param(dither_type, depth).apply(clip, new_format, range_in, range_out, force_fmtc)


def expect_bits(clip: vs.VideoNode, /, expected_depth: int = 16, **kwargs: Any) -> tuple[vs.VideoNode, int]:
    """
    Expected output bitdepth for a clip.

    This function is meant to be used when a clip may not match the expected input bitdepth.
    Both the dithered clip and the original bitdepth are returned.

    Args:
        clip: Input clip.
        expected_depth: Expected bitdepth. Default: 16.

    Returns:
        Tuple containing the clip dithered to the expected depth and the original bitdepth.
    """
    bits = get_depth(clip)

    if bits != expected_depth:
        clip = depth(clip, expected_depth, **kwargs)

    return clip, bits


_f2c_cache = WeakValueDictionary[int, vs.VideoNode]()


def frame2clip(frame: vs.VideoFrame) -> vs.VideoNode:
    """
    Convert a VideoFrame to a VideoNode.

    Args:
        frame: Input frame.

    Returns:
        1-frame long VideoNode of the input frame.
    """

    key = hash((frame.width, frame.height, frame.format.id))

    if _f2c_cache.get(key, None) is None:
        _f2c_cache[key] = blank_clip = vs.core.std.BlankClip(
            None, frame.width, frame.height, frame.format, 1, 1, 1, [0] * frame.format.num_planes, True
        )
    else:
        blank_clip = _f2c_cache[key]

    frame_cp = frame.copy()

    return vs.core.std.ModifyFrame(blank_clip, blank_clip, lambda n, f: frame_cp)


def get_y(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the luma (Y) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        Y plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not GRAY or YUV.
    """

    UnsupportedColorFamilyError.check(clip, [vs.YUV, vs.GRAY], get_y)

    return plane(clip, 0)


def get_u(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the first chroma (U) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        Y plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not YUV.
    """

    UnsupportedColorFamilyError.check(clip, vs.YUV, get_u)

    return plane(clip, 1)


def get_v(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the second chroma (V) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        V plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not YUV.
    """

    UnsupportedColorFamilyError.check(clip, vs.YUV, get_v)

    return plane(clip, 2)


def get_r(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the red plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        R plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not RGB.
    """

    UnsupportedColorFamilyError.check(clip, vs.RGB, get_r)

    return plane(clip, 0)


def get_g(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the green plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        G plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not RGB.
    """

    UnsupportedColorFamilyError.check(clip, vs.RGB, get_g)

    return plane(clip, 1)


def get_b(clip: vs.VideoNode, /) -> vs.VideoNode:
    """
    Extract the blue plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        B plane of the input clip.

    Raises:
        UnsupportedColorFamilyError: Clip is not RGB.
    """

    UnsupportedColorFamilyError.check(clip, vs.RGB, get_b)

    return plane(clip, 2)


def insert_clip(clip: vs.VideoNode, /, insert: vs.VideoNode, start_frame: int, strict: bool = True) -> vs.VideoNode:
    """
    Replace frames of a longer clip with those of a shorter one.

    The insert clip may NOT exceed the final frame of the input clip.
    This limitation can be circumvented by setting `strict=False`.

    Args:
        clip: Input clip.
        insert: Clip to insert into the input clip.
        start_frame: Frame to start inserting from.
        strict: Throw an error if the inserted clip exceeds the final frame of the input clip. If False, truncate the
            inserted clip instead. Default: True.

    Returns:
        Clip with frames replaced by the insert clip.

    Raises:
        CustomValueError: Insert clip is too long, ``strict=False``, and exceeds the final frame of the input clip.
    """

    if start_frame == 0:
        return vs.core.std.Splice([insert, clip[insert.num_frames :]])

    pre = clip[:start_frame]
    insert_diff = (start_frame + insert.num_frames) - clip.num_frames

    if insert_diff == 0:
        return vs.core.std.Splice([pre, insert])

    if insert_diff < 0:
        return vs.core.std.Splice([pre, insert, clip[insert_diff:]])

    if strict:
        raise ClipLengthError(
            "Inserted clip is too long and exceeds the final frame of the input clip.",
            insert_clip,
            {"clip": clip.num_frames, "diff": insert_diff},
        )

    return vs.core.std.Splice([pre, insert[:-insert_diff]])


@overload
def join(
    luma: vs.VideoNode, chroma: vs.VideoNode, /, *, prop_src: vs.VideoNode | SupportsIndex | None = ...
) -> vs.VideoNode:
    """
    Combine a luma and chroma clip into a YUV clip.

    Args:
        luma: Luma clip (GRAY or YUV).
        chroma: Chroma clip (must be YUV).
        prop_src: Clip or index to copy frame properties from.

    Returns:
        YUV clip with combined planes.
    """


@overload
def join(
    plane0: vs.VideoNode,
    plane1: vs.VideoNode,
    plane2: vs.VideoNode,
    alpha: vs.VideoNode | None = None,
    /,
    *,
    family: vs.ColorFamily = vs.ColorFamily.YUV,
    prop_src: vs.VideoNode | SupportsIndex | None = ...,
) -> vs.VideoNode:
    """
    Combine three single-plane clips into one, with optional alpha.

    Args:
        plane0: First plane.
        plane1: Second plane.
        plane2: Third plane.
        alpha: Optional alpha clip.
        family: Output color family. Defaults to YUV.
        prop_src: Clip or index to copy frame properties from.

    Returns:
        Clip with combined planes.
    """


@overload
def join(
    planes: Sequence[vs.VideoNode],
    family: vs.ColorFamily = vs.ColorFamily.YUV,
    /,
    *,
    prop_src: vs.VideoNode | SupportsIndex | None = ...,
) -> vs.VideoNode:
    """
    Combine a sequence of single-plane clips into one.

    Args:
        planes: Planes to merge.
        family: Output color family. Defaults to YUV.
        prop_src: Optional clip or index to copy frame properties from.

    Returns:
        Clip with combined planes.
    """


@overload
def join(
    clips: Mapping[Planes, vs.VideoNode | None],
    family: vs.ColorFamily = vs.ColorFamily.YUV,
    /,
    *,
    prop_src: vs.VideoNode | SupportsIndex | None = ...,
) -> vs.VideoNode:
    """
    Combine a sequence of single-plane clips into one.

    Args:
        planes: Planes to merge.
        family: Output color family. Defaults to YUV.
        prop_src: Optional clip or index to copy frame properties from.

    Returns:
        Clip with combined planes.
    """


def join(
    clips: vs.VideoNode | Sequence[vs.VideoNode] | Mapping[Planes, vs.VideoNode | None],
    plane1_or_family: vs.VideoNode | vs.ColorFamily | None = None,
    plane2: vs.VideoNode | None = None,
    alpha: vs.VideoNode | None = None,
    /,
    *,
    family: vs.ColorFamily = vs.ColorFamily.YUV,
    prop_src: vs.VideoNode | SupportsIndex | None = None,
) -> vs.VideoNode:
    """
    General interface to combine clips or planes into a single clip.

    Args:
        clips: First plane, sequence of single-plane clips or mapping of planes to clips:
        plane1_or_family: Chroma clip, second plane or color family, depending on usage.
        plane2: Third plane when combining three planes.
        alpha: Optional alpha clip.
        family: Output color family. Defaults to YUV.
        prop_src: Optional clip or index to copy frame properties from.

    Raises:
        CustomIndexError: Invalid plane index.
        CustomTypeError: Invalid input type.

    Returns:
        Clip with combined planes.
    """
    if isinstance(clips, Mapping):
        clips_map = dict[int, vs.VideoNode]()

        for p_key, node in clips.items():
            if node is None:
                continue

            if p_key is None:
                clips_map.update(enumerate(flatten_vnodes(node, split_planes=True)))
            else:
                clips_map.update((i, plane(node, i)) for i in to_arr(p_key))

        return join([clips_map[i] for i in sorted(clips_map)], family, prop_src=prop_src)

    if not isinstance(clips, vs.VideoNode):
        if isinstance(plane1_or_family, vs.ColorFamily):
            family = plane1_or_family

        if (n_clips := len(clips)) == 1:
            return clips[0]
        if n_clips in (2, 3, 4):
            return join(*clips, family=family, prop_src=prop_src)
        else:
            raise CustomIndexError("Too many or not enough clips/planes passed!", join, n_clips)

    if not isinstance(plane1_or_family, vs.VideoNode):
        raise CustomTypeError(func=join)

    if not plane2:
        UnsupportedColorFamilyError.check(
            (family, plane1_or_family),
            vs.YUV,
            join,
            "When combining two clips, color family and chroma clip must be {correct}, not {wrong}.",
        )

        clips = [clips, plane1_or_family]
        planes = [0, 1, 2]

    else:
        clips = [clips, plane1_or_family, plane2]
        planes = [0, 0, 0]

    if not isinstance(prop_src, (vs.VideoNode, NoneType)):
        prop_src = clips[prop_src.__index__()]

    joined = vs.core.std.ShufflePlanes(clips, planes, family, prop_src)

    if alpha:
        joined = joined.std.ClipToProp(alpha, "_Alpha")

    return joined


def plane(clip: vs.VideoNode, index: SupportsIndex, /, strict: bool = True) -> vs.VideoNode:
    """
    Extract a plane from the given clip.

    Args:
        clip: Input clip.
        index: Index of the plane to extract.
        strict: If False, removes `_Matrix` property when the input clip is RGB.

    Returns:
        Grayscale clip of the clip's plane.
    """
    if clip.format.num_planes == 1 and index.__index__() == 0:
        return clip

    if not strict and clip.format.color_family is vs.RGB:
        clip = vs.core.std.RemoveFrameProps(clip, "_Matrix")

    return vs.core.std.ShufflePlanes(clip, index.__index__(), vs.GRAY)


def split(clip: vs.VideoNode, /, strict: bool = True) -> list[vs.VideoNode]:
    """
    Split a clip into a list of individual planes.

    Args:
        clip: Input clip.
        strict: If False, removes `_Matrix` property when the input clip is RGB.

    Returns:
        List of individual planes.
    """
    return [clip] if clip.format.num_planes == 1 else [plane(clip, i, strict) for i in range(clip.format.num_planes)]


def flatten_vnodes(*clips: VideoNodeIterable, split_planes: bool = False) -> Sequence[vs.VideoNode]:
    """
    Flatten an array of VideoNodes.

    Args:
        *clips: An array of clips to flatten into a list.
        split_planes: Optionally split the VideoNodes into their individual planes as well. Default: False.

    Returns:
        Flattened list of VideoNodes.
    """
    nodes = list[vs.VideoNode](flatten(clips))

    if not split_planes:
        return nodes

    return reduce(operator.iadd, map(split, nodes), [])


def stack_clips(clips: Iterable[VideoNodeIterable]) -> vs.VideoNode:
    """
    Recursively stack clips in alternating directions: horizontal → vertical → horizontal → ...

    This function takes a nested sequence of clips (or lists of clips) and stacks them
    alternately along the horizontal and vertical axes at each nesting level.

    Examples:
        - Stack a list of clips horizontally:
          ```py
          from vstools import core, split, vs

          clip = core.std.BlankClip(format=vs.RGB24)
          clips = split(clip)

          stacked = stack_clips(clips)
          ```

        - Stack a list of clips vertically (wrap in another list):
          ```py
          from vstools import core, split, vs

          clip = core.std.BlankClip(format=vs.RGB24)
          clips = split(clip)

          stacked = stack_clips([clips])
          ```

        - Stack the Y plane horizontally with the U and V planes stacked vertically:
          ```py
          from vstools import core, split, vs

          clip = core.std.BlankClip(format=vs.YUV420P8)
          y, u, v = split(clip)

          stacked = stack_clips([y, [u, v]])
          ```

        - Stack multiple YUV clips, with Y planes horizontally and UV planes vertically:
          ```py
          from vstools import core, split, vs

          yuv_clips = [...]  # all must share format and height

          clips = []
          for yuv_clip in yuv_clips:
              y, *uv = split(yuv_clip)
              clips.extend([y, uv])

          stacked = stack_clips(clips)
          ```

        - Using ``append`` instead of ``extend`` (and wrapping the sequence, e.g. ``stack_clips([clips])``)
          changes the stacking layout, since it alters the nesting depth.

    Args:
        clips: A (possibly nested) sequence of clips to be stacked.

    Returns:
        Stacked clips.
    """
    return vs.core.std.StackHorizontal(
        [
            inner_clips
            if isinstance(inner_clips, vs.VideoNode)
            else (
                vs.core.std.StackVertical(
                    [clipa if isinstance(clipa, vs.VideoNode) else stack_clips(clipa) for clipa in inner_clips]
                )
            )
            for inner_clips in clips
        ]
    )


@overload
def stack_planes(
    clip: vs.VideoNode,
    shift_float_chroma: bool = True,
    offset_chroma: Literal["min", "max", False] | float = False,
    mode: Literal["h", "v"] = "h",
    write_plane_name: Literal[False] = False,
) -> vs.VideoNode: ...


@overload
def stack_planes(
    clip: vs.VideoNode,
    shift_float_chroma: bool = True,
    offset_chroma: Literal["min", "max", False] | float = False,
    mode: Literal["h", "v"] = "h",
    write_plane_name: bool = ...,
    alignment: int = 7,
    scale: int = 1,
) -> vs.VideoNode: ...


def stack_planes(
    clip: vs.VideoNode,
    shift_float_chroma: bool = True,
    offset_chroma: Literal["min", "max", False] | float = False,
    mode: Literal["h", "v"] = "h",
    write_plane_name: bool = False,
    alignment: int = 7,
    scale: int = 1,
) -> vs.VideoNode:
    """
    Split a clip into its individual planes and stack them visually for inspection.

    Args:
        clip: Input clip to be split and visually stacked.
        shift_float_chroma: If True, shift U and V by +0.5 when working in float YUV formats.
        offset_chroma: Apply chroma plane offseting:

               - "min": match luma minimum
               - "max": match luma maximum
               - float: add value directly
               - False: no chroma offseting
        mode: Stacking direction:

               - "h": horizontal (default)
               - "v": vertical
        write_plane_name: If True, overlays the short plane name ("Y", "U", "V", "R", "G", ...) on each plane.
        alignment: Text alignment for plane labels (only used if `write_plane_name=True`).
        scale: Font scale for plane labels (only used if `write_plane_name=True`).

    Returns:
        A clip containing the stacked planes.
    """
    if clip.format.color_family is vs.GRAY:
        return clip

    if clip.format.sample_type is vs.FLOAT:
        clip = depth(clip, 32)

    if clip.format.color_family is vs.YUV:
        if clip.format.sample_type is vs.FLOAT and shift_float_chroma:
            clip = core.std.Expr(clip, ["", "x 0.5 +"])

        def offset_uv_planes(value: float, plane_stats: str) -> list[vs.VideoNode]:
            planes = split(clip)

            if clip.format.sample_type is vs.FLOAT:
                value += 0.5

            planes[1:] = [
                p.std.PlaneStats().akarin.Expr(f"x {value} x.PlaneStats{plane_stats} - +") for p in planes[1:]
            ]
            return planes

        match offset_chroma:
            case "min":
                planes = offset_uv_planes(get_lowest_value(clip, True), "Min")
            case "max":
                planes = offset_uv_planes(get_peak_value(clip, True), "Max")
            case False:
                planes = split(clip)
            case _:
                planes = split(core.std.Expr(clip, ["", f"x {offset_chroma} +"]))
    else:
        planes = split(clip)

    if write_plane_name:
        planes = [c.text.Text(k, alignment, scale) for k, c in zip(clip.format.name, planes)]

    org: VideoNodeIterable

    dim = {"h": "h", "v": "w"}[mode]

    match getattr(clip.format, f"subsampling_{dim}"):
        case 2:
            blank = planes[1].std.BlankClip(keep=True)
            org = [planes[0], (blank, *planes[1:], blank)]
        case 1:
            org = [planes[0], planes[1:]]
        case 0:
            org = planes
        case _:
            raise CustomNotImplementedError

    if mode == "v":
        org = [org]

    stacked = stack_clips(org)

    if clip.format.color_family == vs.RGB:
        return core.std.RemoveFrameProps(stacked, Matrix.prop_key)

    return stacked


@overload
def limiter(
    clip: vs.VideoNode,
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    mask: bool = False,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Wraps [vszip.Limiter](https://github.com/dnjulek/vapoursynth-zip/wiki/Limiter)
    but only processes if clip format is not integer, a min/max val is specified or tv_range is True.

    Args:
        clip: Clip to process.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        mask: Float chroma range from -0.5/0.5 to 0.0/1.0.
        planes: Which planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clamped clip.
    """


@overload
def limiter[**P](
    _func: Callable[P, vs.VideoNode],
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    mask: bool = False,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> Callable[P, vs.VideoNode]:
    """
    Wraps [vszip.Limiter](https://github.com/dnjulek/vapoursynth-zip/wiki/Limiter)
    but only processes if clip format is not integer, a min/max val is specified or tv_range is True.

    This is the decorator implementation.

    Args:
        _func: Function that returns a VideoNode to be processed.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        mask: Float chroma range from -0.5/0.5 to 0.0/1.0.
        planes: Which planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Decorated function.
    """


@overload
def limiter[**P](
    *,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    tv_range: bool = False,
    mask: bool = False,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Wraps [vszip.Limiter](https://github.com/dnjulek/vapoursynth-zip/wiki/Limiter)
    but only processes if clip format is not integer, a min/max val is specified or tv_range is True.

    This is the decorator implementation.

    Args:
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        mask: Float chroma range from -0.5/0.5 to 0.0/1.0.
        planes: Which planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Decorated function.
    """


def limiter[**P](
    clip_or_func: vs.VideoNode | Callable[P, vs.VideoNode] | None = None,
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    mask: bool = False,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode | Callable[P, vs.VideoNode] | Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Wraps [vszip.Limiter](https://github.com/dnjulek/vapoursynth-zip/wiki/Limiter)
    but only processes if clip format is not integer, a min/max val is specified or tv_range is True.

    Args:
        clip_or_func: Clip to process or function that returns a VideoNode to be processed.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        mask: Float chroma range from -0.5/0.5 to 0.0/1.0.
        planes: Which planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clamped clip.
    """
    if callable(clip_or_func):
        func_ = clip_or_func

        @wraps(func_)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
            return limiter(
                func_(*args, **kwargs),
                min_val,
                max_val,
                tv_range=tv_range,
                mask=mask,
                planes=planes,
                func=func or func_,
            )

        return _wrapper

    func = func or limiter
    clip = clip_or_func

    if clip is None:
        return partial(limiter, min_val=min_val, max_val=max_val, tv_range=tv_range, planes=planes, func=func)

    if all([clip.format.sample_type == vs.INTEGER, min_val is None, max_val is None, tv_range is False]):
        return clip

    if not (min_val == max_val is None):
        from ..utils import get_lowest_values, get_peak_values

        min_val = normalize_seq(min_val or get_lowest_values(clip, clip), clip.format.num_planes)
        max_val = normalize_seq(max_val or get_peak_values(clip, clip), clip.format.num_planes)

    return clip.vszip.Limiter(min_val, max_val, tv_range, mask, planes)
