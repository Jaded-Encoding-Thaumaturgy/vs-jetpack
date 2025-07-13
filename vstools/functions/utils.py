from __future__ import annotations

import string
from functools import partial, wraps
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union, overload
from weakref import WeakValueDictionary

import vapoursynth as vs
from jetpytools import CustomIndexError, CustomStrEnum, CustomValueError, FuncExceptT, P, normalize_seq

from ..enums import ColorRange, ColorRangeT, Matrix
from ..exceptions import ClipLengthError, InvalidColorFamilyError
from ..types import ConstantFormatVideoNode, HoldsVideoFormatT, PlanesT, VideoFormatT
from .check import check_variable_format
from .clip import shift_clip

__all__ = [
    "EXPR_VARS",
    "DitherType",
    "depth",
    "depth_func",
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
    "sc_detect",
    "split",
    "stack_clips",
]


EXPR_VARS = (alph := list(string.ascii_lowercase))[(idx := alph.index("x")) :] + alph[:idx]
"""
Variables to access clips in Expr.
"""


class DitherType(CustomStrEnum):
    """
    Enum for `zimg_dither_type_e` and fmtc `dmode`.
    """

    AUTO = "auto"
    """
    Choose automatically.
    """

    NONE = "none"
    """
    Round to nearest.
    """

    ORDERED = "ordered"
    """
    Bayer patterned dither.
    """

    RANDOM = "random"
    """
    Pseudo-random noise of magnitude 0.5.
    """

    ERROR_DIFFUSION = "error_diffusion"
    """
    Floyd-Steinberg error diffusion.
    """

    ERROR_DIFFUSION_FMTC = "error_diffusion_fmtc"
    """
    Floyd-Steinberg error diffusion.
    Modified for serpentine scan (avoids worm artefacts).
    """

    SIERRA_2_4A = "sierra_2_4a"
    """
    Another type of error diffusion.
    Quick and excellent quality, similar to Floyd-Steinberg.
    """

    STUCKI = "stucki"
    """
    Another error diffusion kernel.
    Preserves delicate edges better but distorts gradients.
    """

    ATKINSON = "atkinson"
    """
    Another error diffusion kernel.
    Generates distinct patterns but keeps clean the flat areas (noise modulation).
    """

    OSTROMOUKHOV = "ostromoukhov"
    """
    Another error diffusion kernel.
    Slow, available only for integer input at the moment. Avoids usual F-S artefacts.
    """

    VOID = "void"
    """
    A way to generate blue-noise dither and has a much better visual aspect than ordered dithering.
    """

    QUASIRANDOM = "quasirandom"
    """
    Dither using quasirandom sequences.
    Good intermediary between void, cluster, and error diffusion algorithms.
    """

    def apply(
        self, clip: vs.VideoNode, fmt_out: vs.VideoFormat, range_in: ColorRange, range_out: ColorRange
    ) -> ConstantFormatVideoNode:
        """
        Apply the given DitherType to a clip.
        """

        from ..utils import get_video_format

        assert self != DitherType.AUTO, CustomValueError("Cannot apply AUTO.", self.__class__)

        fmt = get_video_format(clip)
        clip = ColorRange.ensure_presence(clip, range_in)

        if not self.is_fmtc:
            return clip.resize.Point(
                format=fmt_out.id,
                dither_type=self.value.lower(),
                range_in=range_in.value_zimg,
                range=range_out.value_zimg,
            )

        if fmt.sample_type is vs.FLOAT:
            if self == DitherType.OSTROMOUKHOV:
                raise CustomValueError("Ostromoukhov can't be used for float input.", self.__class__)

            # Workaround because fmtc doesn't support FLOAT 16 input
            if fmt.bits_per_sample < 32:
                clip = DitherType.NONE.apply(clip, fmt.replace(bits_per_sample=32), range_in, range_out)

        return clip.fmtc.bitdepth(
            dmode=_dither_fmtc_types.get(self),
            bits=fmt_out.bits_per_sample,
            fulls=range_in is ColorRange.FULL,
            fulld=range_out is ColorRange.FULL,
        )

    @property
    def is_fmtc(self) -> bool:
        """
        Whether the DitherType is applied through fmtc.
        """

        return self in _dither_fmtc_types

    @overload
    @staticmethod
    def should_dither(
        in_fmt: VideoFormatT | HoldsVideoFormatT,
        out_fmt: VideoFormatT | HoldsVideoFormatT,
        /,
        in_range: ColorRangeT | None = None,
        out_range: ColorRangeT | None = None,
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
        in_range: ColorRangeT | None = None,
        out_range: ColorRangeT | None = None,
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
            in_bits: Input bitdepth.
            out_bits: Output bitdepth.
            in_sample_type: Input sample type.
            out_sample_type: Output sample type.
            in_range: Input color range.
            out_range: Output color range.

        Returns:
            Whether the clip should be dithered.
        """

    @staticmethod
    def should_dither(
        in_bits_or_fmt: int | VideoFormatT | HoldsVideoFormatT,
        out_bits_or_fmt: int | VideoFormatT | HoldsVideoFormatT,
        /,
        in_range: ColorRangeT | None = None,
        out_range: ColorRangeT | None = None,
        in_sample_type: vs.SampleType | None = None,
        out_sample_type: vs.SampleType | None = None,
    ) -> bool:
        from ..utils import get_video_format

        in_fmt = get_video_format(in_bits_or_fmt, sample_type=in_sample_type)
        out_fmt = get_video_format(out_bits_or_fmt, sample_type=out_sample_type)

        in_range = ColorRange.from_param(in_range, (DitherType.should_dither, "in_range"))
        out_range = ColorRange.from_param(out_range, (DitherType.should_dither, "out_range"))

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


_dither_fmtc_types: dict[DitherType, int] = {
    DitherType.SIERRA_2_4A: 3,
    DitherType.STUCKI: 4,
    DitherType.ATKINSON: 5,
    DitherType.ERROR_DIFFUSION_FMTC: 6,
    DitherType.OSTROMOUKHOV: 7,
    DitherType.VOID: 8,
    DitherType.QUASIRANDOM: 9,
}


def depth(
    clip: vs.VideoNode,
    bitdepth: VideoFormatT | HoldsVideoFormatT | int | None = None,
    /,
    sample_type: int | vs.SampleType | None = None,
    *,
    range_in: ColorRangeT | None = None,
    range_out: ColorRangeT | None = None,
    dither_type: str | DitherType = DitherType.AUTO,
) -> ConstantFormatVideoNode:
    """
    A convenience bitdepth conversion function using only internal plugins if possible.

    This uses exclusively internal plugins except for specific dither_types.
    To check whether your DitherType uses fmtc, use `DitherType.is_fmtc`.

        >>> src_8 = vs.core.std.BlankClip(format=vs.YUV420P8)
        >>> src_10 = depth(src_8, 10)
        >>> src_10.format.name
        'YUV420P10'

        >>> src2_10 = vs.core.std.BlankClip(format=vs.RGB30)
        >>> src2_8 = depth(src2_10, 8, dither_type=Dither.RANDOM)  # override default dither behavior
        >>> src2_8.format.name
        'RGB24'

    Args:
        clip: Input clip.
        bitdepth: Desired bitdepth of the output clip.
        sample_type: Desired sample type of output clip. Allows overriding default float/integer behavior. Accepts
            ``vapoursynth.SampleType`` enums ``vapoursynth.INTEGER`` and ``vapoursynth.FLOAT`` or their values, ``0``
            and ``1`` respectively.
        range_in: Input pixel range (defaults to input `clip`'s range).
        range_out: Output pixel range (defaults to input `clip`'s range).
        dither_type: Dithering algorithm. Allows overriding default dithering behavior.
            See [Dither][vstools.DitherType].

            When integer output is desired but the conversion may produce fractional values,
            defaults to DitherType.VOID if it is available via the fmtc VapourSynth plugin,
            or to Floyd-Steinberg DitherType.ERROR_DIFFUSION for 8-bit output
            or DitherType.ORDERED for higher bit depths.
            In other cases, defaults to DitherType.NONE, or round to nearest.
            See [DitherType.should_dither][vstools.DitherType.should_dither] for more information.

    Returns:
        Converted clip with desired bit depth and sample type. ``ColorFamily`` will be same as input.
    """

    from ..utils import get_video_format

    assert check_variable_format(clip, depth)

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

    dither_type = DitherType(dither_type)

    if dither_type is DitherType.AUTO:
        should_dither = DitherType.should_dither(in_fmt, out_fmt, range_in, range_out)

        if hasattr(clip, "fmtc"):
            dither_type = DitherType.VOID
        else:
            dither_type = DitherType.ERROR_DIFFUSION if out_fmt.bits_per_sample == 8 else DitherType.ORDERED
        dither_type = dither_type if should_dither else DitherType.NONE

    new_format = in_fmt.replace(bits_per_sample=out_fmt.bits_per_sample, sample_type=out_fmt.sample_type)

    return dither_type.apply(clip, new_format, range_in, range_out)


_f2c_cache = WeakValueDictionary[int, ConstantFormatVideoNode]()


def frame2clip(frame: vs.VideoFrame) -> ConstantFormatVideoNode:
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
            None, frame.width, frame.height, frame.format.id, 1, 1, 1, [0] * frame.format.num_planes, True
        )
    else:
        blank_clip = _f2c_cache[key]

    frame_cp = frame.copy()

    return vs.core.std.ModifyFrame(blank_clip, blank_clip, lambda n, f: frame_cp)


def get_y(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the luma (Y) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        Y plane of the input clip.

    Raises:
        CustomValueError: Clip is not GRAY or YUV.
    """

    InvalidColorFamilyError.check(clip, [vs.YUV, vs.GRAY], get_y)

    return plane(clip, 0)


def get_u(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the first chroma (U) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        Y plane of the input clip.

    Raises:
        CustomValueError: Clip is not YUV.
    """

    InvalidColorFamilyError.check(clip, vs.YUV, get_u)

    return plane(clip, 1)


def get_v(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the second chroma (V) plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        V plane of the input clip.

    Raises:
        CustomValueError: Clip is not YUV.
    """

    InvalidColorFamilyError.check(clip, vs.YUV, get_v)

    return plane(clip, 2)


def get_r(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the red plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        R plane of the input clip.

    Raises:
        CustomValueError: Clip is not RGB.
    """

    InvalidColorFamilyError.check(clip, vs.RGB, get_r)

    return plane(clip, 0)


def get_g(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the green plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        G plane of the input clip.

    Raises:
        CustomValueError: Clip is not RGB.
    """

    InvalidColorFamilyError.check(clip, vs.RGB, get_g)

    return plane(clip, 1)


def get_b(clip: vs.VideoNode, /) -> ConstantFormatVideoNode:
    """
    Extract the blue plane of the given clip.

    Args:
        clip: Input clip.

    Returns:
        B plane of the input clip.

    Raises:
        CustomValueError: Clip is not RGB.
    """

    InvalidColorFamilyError.check(clip, vs.RGB, get_b)

    return plane(clip, 2)


def insert_clip(
    clip: vs.VideoNode, /, insert: vs.VideoNode, start_frame: int, strict: bool = True
) -> ConstantFormatVideoNode:
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
def join(luma: vs.VideoNode, chroma: vs.VideoNode, family: vs.ColorFamily | None = None) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single YUV clip.

    Args:
        luma: Luma clip, GRAY or YUV.
        chroma: Chroma clip, must be YUV.

    Returns:
        YUV clip of combined planes.
    """


@overload
def join(
    y: vs.VideoNode, u: vs.VideoNode, v: vs.VideoNode, family: Literal[vs.ColorFamily.YUV]
) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single YUV clip.

    Args:
        y: Y plane.
        u: U plane.
        v: V plane.

    Returns:
        YUV clip of combined planes.
    """


@overload
def join(
    y: vs.VideoNode, u: vs.VideoNode, v: vs.VideoNode, alpha: vs.VideoNode, family: Literal[vs.ColorFamily.YUV]
) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single YUV clip.

    Args:
        y: Y plane.
        u: U plane.
        v: V plane.
        alpha: Alpha clip.

    Returns:
        YUV clip of combined planes with an alpha clip attached.
    """


@overload
def join(
    r: vs.VideoNode, g: vs.VideoNode, b: vs.VideoNode, family: Literal[vs.ColorFamily.RGB]
) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single RGB clip.

    Args:
        r: R plane.
        g: G plane.
        b: B plane.

    Returns:
        RGB clip of combined planes.
    """


@overload
def join(
    r: vs.VideoNode, g: vs.VideoNode, b: vs.VideoNode, alpha: vs.VideoNode, family: Literal[vs.ColorFamily.RGB]
) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single RGB clip.

    Args:
        r: R plane.
        g: G plane.
        b: B plane.
        alpha: Alpha clip.

    Returns:
        RGB clip of combined planes with an alpha clip attached.
    """


@overload
def join(*planes: vs.VideoNode, family: vs.ColorFamily | None = None) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single clip.

    Args:
        *planes: Planes to combine.
        family: Output clip family. Default: first clip or detected from props if GRAY and len(planes) > 1.

    Returns:
        Clip of combined planes.
    """


@overload
def join(planes: Iterable[vs.VideoNode], family: vs.ColorFamily | None = None) -> ConstantFormatVideoNode:
    """
    Join a list of planes together to form a single clip.

    Args:
        planes: Planes to combine.
        family: Output clip family. Default: first clip or detected from props if GRAY and len(planes) > 1.

    Returns:
        Clip of combined planes.
    """


@overload
def join(
    planes: Mapping[PlanesT, vs.VideoNode | None], family: vs.ColorFamily | None = None
) -> ConstantFormatVideoNode:
    """
    Join a map of planes together to form a single clip.

    Args:
        planes: Planes to combine.
        family: Output clip family. Default: first clip or detected from props if GRAY and len(planes) > 1.

    Returns:
        Clip of combined planes.
    """


def join(*_planes: Any, **kwargs: Any) -> vs.VideoNode:
    """
    Join a list of planes together to form a single clip.

    Args:
        _planes: Planes to combine.

    Returns:
        Combined planes.
    """

    from ..functions import flatten_vnodes, to_arr
    from ..utils import get_color_family, get_video_format

    family: vs.ColorFamily | None = kwargs.get("family")

    if isinstance(_planes[-1], vs.ColorFamily):
        family = _planes[-1]
        _planes = _planes[:-1]

    if isinstance(_planes[0], Mapping):
        planes_map = dict[int, vs.VideoNode]()

        for p_key, node in _planes[0].items():
            if node is None:
                continue

            if p_key is None:
                planes_map |= dict(enumerate(flatten_vnodes(node, split_planes=True)))
            else:
                planes_map |= {i: plane(node, min(i, node.format.num_planes - 1)) for i in to_arr(p_key)}

        return join(*(planes_map[i] for i in sorted(planes_map.keys())))

    planes = list[vs.VideoNode](
        _planes[0] if (isinstance(_planes[0], Iterable) and not isinstance(_planes[0], vs.VideoNode)) else _planes
    )

    n_clips = len(planes)

    if not n_clips:
        raise CustomIndexError("Not enough clips/planes passed!", join)

    if n_clips == 1 and (family is None or family is (planes[0].format and planes[0].format.color_family)):
        return planes[0]

    if family is None:
        family = get_color_family(planes[0])

    if n_clips == 2:
        other_family = get_color_family(planes[1])

        if family in {vs.GRAY, vs.YUV}:
            InvalidColorFamilyError.check(
                other_family, vs.YUV, join, '"chroma" clip must be {correct} color family, not {wrong}!'
            )

            if family is vs.GRAY:
                planes[0] = get_y(planes[0])

            return vs.core.std.ShufflePlanes(planes, [0, 1, 2], other_family)

    if n_clips in {3, 4}:
        if family is vs.GRAY:
            for pplane in planes[:3]:
                if (fmt := get_video_format(pplane)).num_planes > 1:
                    family = fmt.color_family
                    break
            else:
                matrix = Matrix.from_video(planes[0])
                family = vs.RGB if matrix is Matrix.RGB else vs.YUV

        clip = vs.core.std.ShufflePlanes(planes[:3], [0, 0, 0], family)

        if n_clips == 4:
            clip = clip.std.ClipToProp(planes[-1], "_Alpha")

        return clip
    elif n_clips > 4:
        raise CustomValueError("Too many clips or planes passed!", join)

    raise CustomValueError("Not enough clips or planes passed!", join)


def plane(clip: vs.VideoNode, index: int, /, strict: bool = True) -> ConstantFormatVideoNode:
    """
    Extract a plane from the given clip.

    Args:
        clip: Input clip.
        index: Index of the plane to extract.

    Returns:
        Grayscale clip of the clip's plane.
    """

    assert check_variable_format(clip, plane)

    if clip.format.num_planes == 1 and index == 0:
        return clip

    if not strict and clip.format.color_family is vs.RGB:
        clip = vs.core.std.RemoveFrameProps(clip, "_Matrix")

    return vs.core.std.ShufflePlanes(clip, index, vs.GRAY)


def split(clip: vs.VideoNode, /) -> list[ConstantFormatVideoNode]:
    """
    Split a clip into a list of individual planes.

    Args:
        clip: Input clip.

    Returns:
        List of individual planes.
    """

    assert check_variable_format(clip, split)

    return [clip] if clip.format.num_planes == 1 else [plane(clip, i, False) for i in range(clip.format.num_planes)]


depth_func = depth


def stack_clips(
    clips: Sequence[
        vs.VideoNode
        | Sequence[
            vs.VideoNode
            | Sequence[vs.VideoNode | Sequence[vs.VideoNode | Sequence[vs.VideoNode | Sequence[vs.VideoNode]]]]
        ]
    ],
) -> vs.VideoNode:
    """
    Stack clips in the following repeating order: hor->ver->hor->ver->...

    Args:
        clips: Sequence of clips to stack recursively.

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
def limiter(
    clip: vs.VideoNode,
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Wraps `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`.Limiter but only processes
    if clip format is not integer, a min/max val is specified or tv_range is True.

    Args:
        clip: Clip to process.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        planes: Planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clamped clip.
    """


@overload
def limiter(
    _func: Callable[P, ConstantFormatVideoNode],
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> Callable[P, ConstantFormatVideoNode]:
    """
    Wraps `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`.Limiter but only processes
    if clip format is not integer, a min/max val is specified or tv_range is True.

    This is the decorator implementation.

    Args:
        _func: Function that returns a VideoNode to be processed.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        planes: Planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Decorated function.
    """


@overload
def limiter(
    *,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    tv_range: bool = False,
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> Callable[[Callable[P, ConstantFormatVideoNode]], Callable[P, ConstantFormatVideoNode]]:
    """
    Wraps `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`.Limiter but only processes
    if clip format is not integer, a min/max val is specified or tv_range is True.

    This is the decorator implementation.

    Args:
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        planes: Planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Decorated function.
    """


def limiter(
    clip_or_func: vs.VideoNode | Callable[P, ConstantFormatVideoNode] | None = None,
    /,
    min_val: float | Sequence[float] | None = None,
    max_val: float | Sequence[float] | None = None,
    *,
    tv_range: bool = False,
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> Union[
    ConstantFormatVideoNode,
    Callable[P, ConstantFormatVideoNode],
    Callable[[Callable[P, ConstantFormatVideoNode]], Callable[P, ConstantFormatVideoNode]],
]:
    """
    Wraps `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`.Limiter but only processes
    if clip format is not integer, a min/max val is specified or tv_range is True.

    Args:
        clip_or_func: Clip to process or function that returns a VideoNode to be processed.
        min_val: Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane
            individually.
        max_val: Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane
            individually.
        tv_range: Changes min/max defaults values to LIMITED.
        planes: Planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clamped clip.
    """
    if callable(clip_or_func):
        _func = clip_or_func

        @wraps(_func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> ConstantFormatVideoNode:
            return limiter(
                _func(*args, **kwargs), min_val, max_val, tv_range=tv_range, planes=planes, func=func or _func
            )

        return _wrapper

    func = func or limiter
    clip = clip_or_func

    if clip is None:
        return partial(limiter, min_val=min_val, max_val=max_val, tv_range=tv_range, planes=planes, func=func)

    assert check_variable_format(clip, func)

    if all([clip.format.sample_type == vs.INTEGER, min_val is None, max_val is None, tv_range is False]):
        return clip

    if not (min_val == max_val is None):
        from ..utils import get_lowest_values, get_peak_values

        min_val = normalize_seq(min_val or get_lowest_values(clip, clip), clip.format.num_planes)
        max_val = normalize_seq(max_val or get_peak_values(clip, clip), clip.format.num_planes)

    return clip.vszip.Limiter(min_val, max_val, tv_range, planes)


def sc_detect(clip: vs.VideoNode, threshold: float = 0.1) -> ConstantFormatVideoNode:
    assert check_variable_format(clip, sc_detect)

    stats = vs.core.std.PlaneStats(shift_clip(clip, -1), clip)

    return vs.core.akarin.PropExpr(
        [clip, stats, stats[1:]],
        lambda: {
            "_SceneChangePrev": f"y.PlaneStatsDiff {threshold} > 1 0 ?",
            "_SceneChangeNext": f"z.PlaneStatsDiff {threshold} > 1 0 ?",
        },
    )
