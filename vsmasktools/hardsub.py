from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Type

from vsexprtools import ExprOp, ExprToken, expr_func, norm_expr
from vskernels import Bilinear, Catrom, Point
from vsrgtools import box_blur, median_blur
from vssource import IMWRI, Indexer
from vstools import (
    ColorRange, ConstantFormatVideoNode, CustomOverflowError, FileNotExistsError, FilePathType, FrameRangeN,
    FrameRangesN, Matrix, VSFunctionNoArgs, check_variable, core, depth, fallback, get_lowest_value, get_neutral_value,
    get_neutral_values, get_peak_value, get_y, iterate, limiter, normalize_ranges, replace_ranges, scale_delta,
    scale_value, vs, vs_object
)

from .abstract import BoundingBox, DeferredMask, GeneralMask
from .edge import SobelStd
from .morpho import Morpho
from .types import GenericMaskT, XxpandMode
from .utils import max_planes, normalize_mask

__all__ = [
    'CustomMaskFromFolder',
    'CustomMaskFromRanges',

    'HardsubMask',
    'HardsubSignFades',
    'HardsubSign',
    'HardsubLine',
    'HardsubLineFade',
    'HardsubASS',

    'bounded_dehardsub',
    'diff_hardsub_mask',

    'get_all_sign_masks'
]


class _base_cmaskcar(vs_object):
    clips: list[vs.VideoNode]

    def __vs_del__(self, core_id: int) -> None:
        self.clips.clear()


@dataclass
class CustomMaskFromClipsAndRanges(GeneralMask, _base_cmaskcar):
    """Abstract CustomMaskFromClipsAndRanges interface"""

    processing: VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = field(
        default=core.lazy.std.Binarize, kw_only=True
    )
    idx: Indexer | Type[Indexer] = field(default=IMWRI, kw_only=True)

    def get_mask(self, ref: vs.VideoNode, /, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Get the constructed mask

        :param ref:         Reference clip.
        :param **kwargs:    Keyword arguments passed to `replace_ranges` function.
        :return:            Constructed mask
        """
        assert check_variable(ref, self.get_mask)

        mask = vs.core.std.BlankClip(
            ref,
            format=ref.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id,
            keep=True, color=0
        )

        matrix = Matrix.from_video(ref)

        for maskclip, mask_ranges in zip(self.clips, self.frame_ranges(ref)):
            maskclip = Point.resample(
                maskclip.std.AssumeFPS(ref), mask, matrix,
                range_in=ColorRange.FULL, range=ColorRange.FULL
            )
            maskclip = self.processing(maskclip)
            maskclip = vs.core.std.Loop(maskclip, mask.num_frames)

            mask = replace_ranges(mask, maskclip, mask_ranges, **kwargs)

        return mask

    @abstractmethod
    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        ...


@dataclass
class CustomMaskFromFolder(CustomMaskFromClipsAndRanges):
    """A helper class for creating a mask clip from a folder of images."""

    folder_path: FilePathType

    def __post_init__(self) -> None:
        if not (folder_path := Path(str(self.folder_path))).is_dir():
            raise FileNotExistsError('"folder_path" must be an existing path directory!', self.get_mask)

        self.files = list(folder_path.glob('*'))

        self.clips = [self.idx.source(file, bits=-1) for file in self.files]

    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        return [
            [(other[-1] if other else end, end)]
            for (*other, end) in (map(int, name.stem.split('_')) for name in self.files)
        ]


@dataclass
class CustomMaskFromRanges(CustomMaskFromClipsAndRanges):
    """
    A helper class for creating a mask clip from a mapping of file paths
    and their corresponding frame ranges
    """

    ranges: Mapping[FilePathType, FrameRangeN | FrameRangesN]

    def __post_init__(self) -> None:
        self.clips = [self.idx.source(str(file), bits=-1) for file in self.ranges.keys()]

    def frame_ranges(self, clip: vs.VideoNode) -> list[list[tuple[int, int]]]:
        return [normalize_ranges(clip, ranges) for ranges in self.ranges.values()]


class HardsubMask(DeferredMask):
    """Abstract HardsubMask interface"""

    bin_thr: float = 0.75

    def get_progressive_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode]
    ) -> tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        """
        Dehardsub using multiple superior hardsubbed sources and one inferior non-subbed source.

        :param hardsub:     Hardsub master source (eg Wakanim RU dub).
        :param ref:         Non-subbed reference source (eg CR, Funi, Amazon).
        :param partials:    Sources to use for partial dehardsubbing (eg Waka DE, FR, SC).

        :return:            Dehardsub stages and masks used for progressive dehardsub.
        """
        assert check_variable(hardsub, self.get_progressive_dehardsub)

        masks = [self.get_mask(hardsub, ref)]
        partials_dehardsubbed = [hardsub]
        dehardsub_masks = list[ConstantFormatVideoNode]()
        partials = partials + [ref]

        thr = scale_value(self.bin_thr, 32, masks[-1])

        for p in partials:
            masks.append(
                ExprOp.SUB.combine(masks[-1], self.get_mask(p, ref))
            )
            dehardsub_masks.append(
                iterate(expr_func([masks[-1]], f"x {thr} < 0 x ?"), core.lazy.std.Maximum, 4).std.Inflate()
            )
            partials_dehardsubbed.append(
                partials_dehardsubbed[-1].std.MaskedMerge(p, dehardsub_masks[-1])
            )

            masks[-1] = masks[-1].std.MaskedMerge(masks[-1].std.Invert(), masks[-2])

        return partials_dehardsubbed, dehardsub_masks

    def apply_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Dehardsub using multiple superior hardsubbed sources and one inferior non-subbed source.

        :param hardsub:     Hardsub master source (eg Wakanim RU dub).
        :param ref:         Non-subbed reference source (eg CR, Funi, Amazon).
        :param partials:    Sources to use for partial dehardsubbing (eg Waka DE, FR, SC).

        :return:            Dehardsubbed clip.
        """
        if partials:
            partials_dehardsubbed, _ = self.get_progressive_dehardsub(hardsub, ref, partials)
            dehardsub = partials_dehardsubbed[-1]
        else:
            dehardsub = hardsub.std.MaskedMerge(ref, self.get_mask(hardsub, ref))

        return replace_ranges(hardsub, dehardsub, self.ranges)


class HardsubSignFades(HardsubMask):
    """
    Helper for hardsub scene filtering, typically used for de-hardsubbing signs during fades or hard-to-catch signs.
    Originally written by Kageru from Kagefunc:
    `https://github.com/Irrational-Encoding-Wizardry/kagefunc`
    """

    highpass: float
    expand: int
    edgemask: GenericMaskT
    expand_mode: XxpandMode

    def __init__(
        self,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        highpass: float = 0.0763,
        expand: int = 8,
        edgemask: GenericMaskT = SobelStd,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        *,
        blur: bool = False,
        refframes: int | list[int | None] | None = None
    ) -> None:
        """
        :param ranges:          The frame ranges that the mask should be applied to.
        :param bound:           An optional bounding box that defines the area of the frame where the mask will be applied.
                                If None, the mask applies to the whole frame.
        :param highpass:        Highpass threshold. Lower this value if the sign isn't fully de-hardsubbed,
                                but be cautious as it may also capture more artifacts.
        :param expand:          Number of expand iterations.
        :param edgemask:        Edge mask used for finding subtitles.
        :param expand_mode:     Specifies the XxpandMode used for mask growth
        :param blur:            Whether to apply a box blur effect to the mask.
        :param refframes:       A list of reference frames used in building the final mask for each specified range.
                                Must have the same length as `ranges`.
        """
        self.highpass = highpass
        self.expand = expand
        self.edgemask = edgemask
        self.expand_mode = expand_mode

        super().__init__(ranges, bound, blur=blur, refframes=refframes)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        clipedge, refedge = (
            box_blur(normalize_mask(self.edgemask, x, **kwargs))
            for x in (clip, ref)
        )

        highpass = scale_delta(self.highpass, 32, clip)

        mask = median_blur(
            norm_expr([clipedge, refedge], f'x y - {highpass} < 0 {ExprToken.RangeMax} ?', func=self.__class__)
        )

        return max_planes(Morpho.inflate(Morpho.expand(mask, self.expand, mode=self.expand_mode), iterations=4))


class HardsubSign(HardsubMask):
    """
    Hardsub scenefiltering helper using `Zastin <https://github.com/kgrabs>`_'s hardsub mask.
    """

    thr: float
    minimum: int
    expand: int
    inflate: int
    expand_mode: XxpandMode

    def __init__(
        self,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        thr: float = 0.06,
        minimum: int = 1,
        expand: int = 8,
        inflate: int = 7,
        expand_mode: XxpandMode = XxpandMode.RECTANGLE,
        *,
        blur: bool = False,
        refframes: int | list[int | None] | None = None
    ) -> None:
        """
        :param ranges:          The frame ranges that the mask should be applied to.
        :param bound:           An optional bounding box that defines the area of the frame where the mask will be applied.
                                If None, the mask applies to the whole frame.
        :param thr:             Binarization threshold, [0, 1] (Default: 0.06).
        :param minimum:         std.Minimum iterations (Default: 1).
        :param expand:          std.Maximum iterations (Default: 8).
        :param inflate:         std.Inflate iterations (Default: 7).
        :param expand_mode:     Specifies the XxpandMode used for mask growth (Default: XxpandMode.RECTANGLE).
        :param blur:            Whether to apply a box blur effect to the mask.
        :param refframes:       A list of reference frames used in building the final mask for each specified range.
                                Must have the same length as `ranges`.
        """
        self.thr = thr
        self.minimum = minimum
        self.expand = expand
        self.inflate = inflate
        self.expand_mode = expand_mode
        super().__init__(ranges, bound, blur=blur, refframes=refframes)

    @limiter
    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable(clip, self._mask)

        hsmf = norm_expr([clip, ref], 'x y - abs', func=self.__class__)
        hsmf = Bilinear.resample(hsmf, clip.format.replace(subsampling_w=0, subsampling_h=0))

        hsmf = ExprOp.MAX(hsmf, split_planes=True)

        hsmf = Morpho.binarize(hsmf, self.thr)
        hsmf = Morpho.minimum(hsmf, iterations=self.minimum, func=self.__class__)
        hsmf = Morpho.expand(hsmf, self.expand, mode=self.expand_mode, func=self.__class__)
        hsmf = Morpho.inflate(hsmf, iterations=self.inflate, func=self.__class__)

        return hsmf


class HardsubLine(HardsubMask):
    """
    Helper for de-hardsubbing white text with black border subtitles.
    Originally written by Kageru from Kagefunc:
    `https://github.com/Irrational-Encoding-Wizardry/kagefunc`
    """

    expand: int | None

    def __init__(
        self,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        expand: int | None = None,
        *,
        blur: bool = False,
        refframes: int | list[int | None] | None = None
    ) -> None:
        """
        :param ranges:          The frame ranges that the mask should be applied to.
        :param bound:           An optional bounding box that defines the area of the frame where the mask will be applied.
                                If None, the mask applies to the whole frame.
        :param expand:          std.Maximum iterations. Default is automatically adjusted based on the width of the clip.
        :param blur:            Whether to apply a box blur effect to the mask.
        :param refframes:       A list of reference frames used in building the final mask for each specified range.
                                Must have the same length as `ranges`.
        """
        self.expand = expand

        super().__init__(ranges, bound, blur=blur, refframes=refframes)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        expand_n = fallback(self.expand, clip.width // 200)

        y_range = get_peak_value(clip) - get_lowest_value(clip)
        uv_range = get_peak_value(clip, chroma=True) - get_lowest_value(clip, chroma=True)

        uv_abs = f' {get_neutral_value(clip)} - abs '
        yexpr = f'x y - abs {y_range * 0.7} > 255 0 ?'
        uv_thr = uv_range * 0.8
        uvexpr = f'x {uv_abs} {uv_thr} < y {uv_abs} {uv_thr} < and 255 0 ?'

        right = core.resize.Point(clip, src_left=4)

        subedge = norm_expr(
            [clip, right], (yexpr, uvexpr), format=clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=8),
            func=self.__class__
        )

        subedge = ExprOp.MIN(Catrom.resample(subedge, vs.YUV444P8), split_planes=True)

        clip_y, ref_y = get_y(clip), depth(get_y(ref), clip)

        clips = [box_blur(clip_y), box_blur(ref_y)]
        diff = norm_expr(
            clips,
            'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?',
            upper=scale_value(0.8, 32, clip),
            lower=scale_value(0.2, 32, clip),
            mindiff=y_range * 0.1,
            format=vs.GRAY8,
            func=self.__class__
        )
        diff = Morpho.maximum(diff, iterations=2, func=self.__class__)

        mask = subedge.hysteresis.Hysteresis(diff)
        mask = iterate(mask, core.std.Maximum, expand_n)
        mask = box_blur(mask.std.Inflate().std.Inflate())

        return depth(mask, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


class HardsubLineFade(HardsubLine):
    """
    A specialized version of HardsubLine with a weight for selecting a frame within the specified frame ranges.
    """

    ref_float: float

    def __init__(
        self,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        expand: int | None = None,
        refframe: float = 0.5,
        *,
        blur: bool = False,
    ) -> None:
        """
        :param ranges:          The frame ranges that the mask should be applied to.
        :param bound:           An optional bounding box that defines the area of the frame where the mask will be applied.
                                If None, the mask applies to the whole frame.
        :param expand:          std.Maximum iterations. Default is automatically adjusted based on the width of the clip.
        :param refframe:        Reference frame weight. Must be between 0 and 1.
        :param blur:            Whether to apply a box blur effect to the mask.
        :param refframes:       A list of reference frames used in building the final mask for each specified range.
                                Must have the same length as `ranges`.
        """
        if refframe < 0 or refframe > 1:
            raise CustomOverflowError('"refframe" must be between 0 and 1!', self.__class__)

        self.ref_float = refframe

        super().__init__(ranges, bound, expand, blur=blur, refframes=None)

    def get_mask(self, clip: vs.VideoNode, /, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        self.refframes = [
            r[0] + round((r[1] - r[0]) * self.ref_float)
            for r in normalize_ranges(ref, self.ranges)
        ]

        return super().get_mask(clip, ref)


class HardsubASS(HardsubMask):
    """A helper for de-hardsubbing using an ASS subtitle file to generate a hardsub mask."""

    filename: str
    fontdir: str | None

    def __init__(
        self,
        filename: FilePathType,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        *,
        fontdir: str | None = None,
        blur: bool = False,
        refframes: int | list[int | None] | None = None
    ) -> None:
        """
        :param filename:        Subtitle file.
        :param ranges:          The frame ranges that the mask should be applied to.
        :param bound:           An optional bounding box that defines the area of the frame where the mask will be applied.
                                If None, the mask applies to the whole frame.
        :param blur:            Whether to apply a box blur effect to the mask.
        :param refframes:       A list of reference frames used in building the final mask for each specified range.
                                Must have the same length as `ranges`.
        """
        self.filename = str(filename)
        self.fontdir = fontdir
        super().__init__(ranges, bound, blur=blur, refframes=refframes)

    @limiter
    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        mask = core.sub.TextFile(ref, self.filename, fontdir=self.fontdir, blend=False).std.PropToClip('_Alpha')

        mask = mask.std.Binarize(1)

        mask = iterate(mask, core.lazy.std.Maximum, 3)
        mask = iterate(mask, core.lazy.std.Inflate, 3)

        return mask


def bounded_dehardsub(
    hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask], partials: list[vs.VideoNode] | None = None
) -> ConstantFormatVideoNode:
    assert check_variable(hrdsb, bounded_dehardsub)
    for sign in signs:
        hrdsb = sign.apply_dehardsub(hrdsb, ref, partials)

    return hrdsb


def diff_hardsub_mask(a: vs.VideoNode, b: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
    assert check_variable(a, diff_hardsub_mask)
    assert check_variable(b, diff_hardsub_mask)

    return a.std.BlankClip(color=get_neutral_values(a), keep=True).std.MaskedMerge(
        a.std.MakeDiff(b), HardsubLine(**kwargs).get_mask(a, b)
    )


@limiter
def get_all_sign_masks(hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask]) -> ConstantFormatVideoNode:
    assert check_variable(hrdsb, get_all_sign_masks)
    assert check_variable(ref, get_all_sign_masks)

    mask = core.std.BlankClip(
        ref, format=ref.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id, keep=True
    )

    for sign in signs:
        mask = replace_ranges(mask, ExprOp.ADD.combine(mask, max_planes(sign.get_mask(hrdsb, ref))), sign.ranges)

    return mask
