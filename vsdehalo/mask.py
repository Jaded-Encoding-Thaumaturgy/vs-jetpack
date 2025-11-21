"""
This module implements dehalo functions with complex masking abilities.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Sequence

from jetpytools import CustomIndexError, FuncExcept, normalize_seq

from vsaa import NNEDI3, SuperSamplerProcess
from vsdenoise import Prefilter
from vsexprtools import ExprOp, norm_expr
from vsmasktools import (
    Coordinates,
    MaskLike,
    Morpho,
    Robinson3,
    XxpandMode,
    grow_mask,
    normalize_mask,
)
from vsrgtools import BlurMatrixBase, box_blur, contrasharpening_dehalo
from vsscale import pre_ss as pre_supersample
from vstools import (
    ConvMode,
    FunctionUtil,
    OneDimConvMode,
    Planes,
    UnsupportedColorFamilyError,
    VSFunctionPlanesArgs,
    VSObjectABC,
    check_progressive,
    get_y,
    join,
    limiter,
    normalize_planes,
    scale_mask,
    split,
    vs,
)

from .alpha import IterArr, dehalo_alpha

__all__ = ["fine_dehalo", "fine_dehalo2"]


class FineDehalo[**P, R]:
    """
    Class decorator that wraps the [fine_dehalo][vsdehalo.fine_dehalo] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    masks: Masks
    """
    The generated masks.
    """

    def __init__(self, fine_dehalo: Callable[P, R]) -> None:
        self._func = fine_dehalo

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    def mask(
        self,
        clip: vs.VideoNode,
        # fine_dehalo mask specific params
        rx: int = 2,
        ry: int | None = None,
        edgemask: MaskLike = Robinson3,
        thmi: float | Sequence[float] = 80,
        thma: float | Sequence[float] = 128,
        thlimi: float | Sequence[float] = 50,
        thlima: float | Sequence[float] = 100,
        exclude: bool | Sequence[bool] = True,
        edgeproc: float | Sequence[float] = 0.0,
        # Misc params
        planes: Planes = 0,
        func: FuncExcept | None = None,
    ) -> vs.VideoNode:
        """
        The fine_dehalo mask.

        The parameters `thmi`, `thma`, `thlimi`, `thlima`, `exclude` and `edgeproc` can be configured per plane.

        Args:
            clip: Source clip.
            rx: Horizontal radius for halo removal.
            ry: Vertical radius for halo removal. Defaults to `rx` if not set.
            edgemask: Edge detection object to use. Defaults to `Robinson3`.
            thmi: Minimum threshold for sharp edge selection; isolates only the strongest (line-like) edges.
            thma: Maximum threshold for sharp edge selection; filters out weaker edges.
            thlimi: Minimum threshold for including edges that were previously ignored.
            thlima: Maximum threshold for the inclusion of additional, less distinct edges.
            exclude: Whether to exclude edges that are too close together.
            edgeproc: If greater than 0, adds the edge mask into the final processing. Defaults to 0.0.
            planes: Planes to process.
            func: An optional function to use for error handling.

        Returns:
            Mask clip.
        """
        return self.Masks(clip, rx, ry, edgemask, thmi, thma, thlimi, thlima, exclude, edgeproc, planes, func).MAIN

    class Masks(VSObjectABC, Mapping[str, vs.VideoNode]):
        """
        Class for creating and storing intermediate masks used in the [fine_dehalo][vsdehalo.fine_dehalo] function.

        Each step of the masking pipeline is stored with a descriptive key, allowing for
        debugging or further processing.
        """

        _names = ("EDGES", "SHARP_EDGES", "LARGE_EDGES", "IGNORE_DETAILS", "SHRINK", "SHRINK_EDGES_EXCL", "MAIN")

        def __init__(
            self,
            clip: vs.VideoNode,
            # fine_dehalo mask specific params
            rx: int = 2,
            ry: int | None = None,
            edgemask: MaskLike = Robinson3,
            thmi: float | Sequence[float] = 80,
            thma: float | Sequence[float] = 128,
            thlimi: float | Sequence[float] = 50,
            thlima: float | Sequence[float] = 100,
            exclude: bool | Sequence[bool] = True,
            edgeproc: float | Sequence[float] = 0.0,
            # Misc params
            planes: Planes = 0,
            func: FuncExcept | None = None,
        ) -> None:
            """
            Initialize the mask generation process.

            The parameters `thmi`, `thma`, `thlimi`, `thlima`, `exclude` and `edgeproc` can be configured per plane.

            Args:
                clip: Source clip.
                rx: Horizontal radius for halo removal.
                ry: Vertical radius for halo removal. Defaults to `rx` if not set.
                edgemask: Edge detection object to use. Defaults to `Robinson3`.
                thmi: Minimum threshold for sharp edge selection; isolates only the strongest (line-like) edges.
                thma: Maximum threshold for sharp edge selection; filters out weaker edges.
                thlimi: Minimum threshold for including edges that were previously ignored.
                thlima: Maximum threshold for the inclusion of additional, less distinct edges.
                exclude: Whether to exclude edges that are too close together.
                edgeproc: If greater than 0, adds the edge mask into the final processing. Defaults to 0.0.
                planes: Planes to process.
                func: An optional function to use for error handling.
            """
            func = func or self.__class__

            UnsupportedColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

            thmif, thmaf, thlimif, thlimaf = [
                [scale_mask(x, 8, clip) for x in th]
                for th in [normalize_seq(th, clip.format.num_planes) for th in (thmi, thma, thlimi, thlima)]
            ]
            exclude = normalize_seq(exclude, clip.format.num_planes)
            edgeproc = normalize_seq(edgeproc, clip.format.num_planes)
            planes = normalize_planes(clip, planes)
            work_clip = get_y(clip) if planes == [0] else clip

            # Main edges #
            # Basic edge detection, thresholding will be applied later.
            edges = normalize_mask(edgemask, work_clip, work_clip, func=func)

            # Keeps only the sharpest edges (line edges)
            strong = norm_expr(
                edges,
                "x {thmif} - {thmaf} {thmif} - / mask_max *",
                planes,
                func=func,
                thmif=thmif,
                thmaf=thmaf,
            )

            # Extends them to include the potential halos
            large = Morpho.expand(strong, rx, ry, planes=planes, func=func)

            # Exclusion zones #
            # When two edges are close from each other (both edges of a single
            # line or multiple parallel color bands), the halo removal
            # oversmoothes them or makes seriously bleed the bands, producing
            # annoying artifacts. Therefore we have to produce a mask to exclude
            # these zones from the halo removal.

            # Includes more edges than previously, but ignores simple details
            light = norm_expr(
                edges,
                "x {thlimif} - {thlimaf} {thlimif} - / mask_max *",
                planes,
                func=func,
                thlimif=thlimif,
                thlimaf=thlimaf,
            )

            # To build the exclusion zone, we make grow the edge mask, then shrink
            # it to its original shape. During the growing stage, close adjacent
            # edge masks will join and merge, forming a solid area, which will
            # remain solid even after the shrinking stage.
            # Mask growing
            shrink = Morpho.expand(light, rx, ry, XxpandMode.ELLIPSE, planes=planes, func=func)

            # At this point, because the mask was made of a shades of grey, we may
            # end up with large areas of dark grey after shrinking. To avoid this,
            # we amplify and saturate the mask here (actually we could even
            # binarize it).
            shrink = norm_expr(shrink, "x 4 *", planes, func=func)
            shrink = Morpho.inpand(shrink, rx, ry, XxpandMode.ELLIPSE, planes=planes, func=func)

            # This mask is almost binary, which will produce distinct
            # discontinuities once applied. Then we have to smooth it.
            shrink = box_blur(shrink, passes=2, planes=planes)

            # Final mask building #

            # Previous mask may be a bit weak on the pure edge side, so we ensure
            # that the main edges are really excluded. We do not want them to be
            # smoothed by the halo removal.
            shr_med = (
                norm_expr([strong, shrink], tuple("x y max" * ex for ex in exclude), planes, func=func)
                if any(exclude)
                else strong
            )

            # Subtracts masks and amplifies the difference to be sure we get 255
            # on the areas to be processed.
            mask = norm_expr([large, shr_med], "x y - 2 *", planes, func=func)

            # If edge processing is required, adds the edgemask
            if any(edgeproc):
                mask = norm_expr([mask, strong], "x y {edgeproc} 0.66 * * +", planes, func=func, edgeproc=edgeproc)

            # Smooth again and amplify to grow the mask a bit, otherwise the halo
            # parts sticking to the edges could be missed.
            # Also clamp to legal ranges
            mask = norm_expr(box_blur(mask, planes=planes), "x 2 * 0 mask_max clip", planes, func=func)

            self._edges = edges
            self._strong = strong
            self._large = large
            self._light = light
            self._shrink = shrink
            self._shr_med = shr_med
            self._main = mask

        def __getitem__(self, index: str) -> vs.VideoNode:
            index = index.upper()

            if index in self._names:
                return getattr(self, index)

            raise KeyError

        def __iter__(self) -> Iterator[str]:
            yield from self._names

        def __len__(self) -> int:
            return len(self._names)

        @property
        def EDGES(self) -> vs.VideoNode:  # noqa: N802
            return self._edges

        @property
        def SHARP_EDGES(self) -> vs.VideoNode:  # noqa: N802
            return self._strong

        @property
        def LARGE_EDGES(self) -> vs.VideoNode:  # noqa: N802
            return self._large

        @property
        def IGNORE_DETAILS(self) -> vs.VideoNode:  # noqa: N802
            return self._light

        @property
        def SHRINK(self) -> vs.VideoNode:  # noqa: N802
            return self._shrink

        @property
        def SHRINK_EDGES_EXCL(self) -> vs.VideoNode:  # noqa: N802
            return self._shr_med

        @property
        def MAIN(self) -> vs.VideoNode:  # noqa: N802
            return self._main


@FineDehalo
def fine_dehalo(
    clip: vs.VideoNode,
    # dehalo_alpha params
    blur: IterArr[float]
    | VSFunctionPlanesArgs
    | tuple[float | list[float] | VSFunctionPlanesArgs, ...] = Prefilter.GAUSS(sigma=1.4),
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    ss: float | tuple[float, ...] = 1.5,
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    # fine_dehalo mask specific params
    rx: int = 2,
    ry: int | None = None,
    edgemask: MaskLike = Robinson3,
    thmi: float | Sequence[float] = 80,
    thma: float | Sequence[float] = 128,
    thlimi: float | Sequence[float] = 50,
    thlima: float | Sequence[float] = 100,
    exclude: bool | Sequence[bool] = True,
    edgeproc: float | Sequence[float] = 0.0,
    # Final post processing
    contra: float = 0.0,
    # Misc params
    pre_ss: int = 1,
    planes: Planes = 0,
    attach_masks: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Halo removal function based on `dehalo_alpha`, enhanced with additional masking and optional contra-sharpening
    to better preserve important line detail while effectively reducing halos.

    The parameter `ss` can be configured per iteration while `blur`, `lowsens`, `highsens`, `darkstr` and `brightstr`
    can be configured per plane and per iteration. You can specify:

    - A single value: applies to all iterations and all planes.
    - A tuple of values: interpreted as iteration-wise.
    - A list inside the tuple: interpreted as per-plane for a specific iteration.

    For example:
        `blur=(1.4, [1.4, 1.65], [1.5, 1.4, 1.45])` implies 3 iterations:

        - 1st: 1.4 for all planes
        - 2nd: 1.4 for luma, 1.65 for both chroma planes
        - 3rd: 1.5 for luma, 1.4 for U, 1.45 for V

    The parameters `thmi`, `thma`, `thlimi`, `thlima`, `exclude` and `edgeproc` can be configured per plane.

    Example usage:
        ```py
        dehalo = fine_dehalo(clip, (2.0, 1.4), brightstr=(0.85, 0.25))
        # Getting the masks of the last fine_dehalo call:
        dehalo_mask = fine_dehalo.masks.MAIN
        ```

    Args:
        clip: Source clip.
        blur: Standard deviation of the Gaussian kernel if float or custom blurring function
            to use in place of the default implementation.
        lowsens: Lower sensitivity threshold — dehalo is fully applied below this value.
        highsens: Upper sensitivity threshold — dehalo is completely skipped above this value.
        ss: Supersampling factor to reduce aliasing artifacts.
        darkstr: Strength factor for suppressing dark halos.
        brightstr: Strength factor for suppressing bright halos.
        rx: Horizontal radius for halo removal.
        ry: Vertical radius for halo removal. Defaults to `rx` if not set.
        edgemask: Edge detection object to use. Defaults to [Robinson3][vsmasktools.Robinson3].
        thmi: Minimum threshold for sharp edge selection; isolates only the strongest (line-like) edges.
        thma: Maximum threshold for sharp edge selection; filters out weaker edges.
        thlimi: Minimum threshold for including edges that were previously ignored.
        thlima: Maximum threshold for the inclusion of additional, less distinct edges.
        exclude: Whether to exclude edges that are too close together.
        edgeproc: If greater than 0, adds the edge mask into the final processing. Defaults to 0.0.
        contra: Contra-sharpening level in [contrasharpening_dehalo][vsrgtools.contrasharpening_dehalo].
        pre_ss: Scaling factor for supersampling before processing.
            If > 1, supersamples the clip with NNEDI3, applies dehalo processing, and then downscales back with Point.
        planes: Planes to process.
        attach_masks: Stores the masks as frame properties in the output clip.
            The prop names are `FineDehaloMask` + the masking step.
        func: An optional function to use for error handling.
        **kwargs: Additionnal advanced parameters.

    Returns:
        Dehaloed clip.
    """
    func_util = FunctionUtil(clip, func or fine_dehalo, planes)

    assert check_progressive(clip, func_util.func)

    if pre_ss > 1:
        if pre_ss & (pre_ss - 1) != 0:
            raise CustomIndexError("`pre_ss` has to be a power of 2.", func_util.func, pre_ss)

        return pre_supersample(
            clip,
            sp=SuperSamplerProcess[NNEDI3](
                function=lambda clip: fine_dehalo(
                    clip,
                    blur,
                    lowsens,
                    highsens,
                    ss,
                    darkstr,
                    brightstr,
                    rx,
                    ry,
                    edgemask,
                    thmi,
                    thma,
                    thlimi,
                    thlima,
                    exclude,
                    edgeproc,
                    contra,
                    planes=planes,
                    attach_masks=attach_masks,
                    func=func_util.func,
                    **kwargs,
                ),
                tff=any(p in func_util.norm_planes for p in [1, 2]),
            ),
            rfactor=pre_ss,
            mod=1,
            planes=planes,
            func=func_util.func,
        )

    fine_dehalo.masks = fine_dehalo.Masks(
        func_util.work_clip, rx, ry, edgemask, thmi, thma, thlimi, thlima, exclude, edgeproc, planes, func_util.func
    )

    dehaloed = dehalo_alpha(
        func_util.work_clip, blur, lowsens, highsens, ss, darkstr, brightstr, planes, attach_masks, func, **kwargs
    )

    if contra:
        dehaloed = contrasharpening_dehalo(dehaloed, func_util.work_clip, contra, planes=planes)

    y_merge = func_util.work_clip.std.MaskedMerge(dehaloed, fine_dehalo.masks.MAIN, planes)

    out = func_util.return_clip(y_merge)

    if attach_masks:
        out = out.std.CopyFrameProps(dehaloed)

        for k, v in fine_dehalo.masks.items():
            out = out.std.ClipToProp(v, "FineDehaloMask" + "".join(w.title() for w in k.split("_")))

    return out


def fine_dehalo2(
    clip: vs.VideoNode,
    mode: OneDimConvMode = ConvMode.HV,
    radius: int = 2,
    mask_radius: int = 2,
    brightstr: float = 1.0,
    darkstr: float = 1.0,
    dark: bool | None = True,
    *,
    attach_masks: bool = False,
) -> vs.VideoNode:
    """
    Halo removal function for 2nd order halos.

    Args:
        clip: Source clip.
        mode: Horizontal/Vertical or both ways.
        radius: Radius for the fixing convolution.
        mask_radius: Radius for mask growing.
        brightstr: Strength factor for bright halos.
        darkstr: Strength factor for dark halos.
        dark: Whether to filter for dark or bright haloing. None for disable merging with source clip.
        attach_masks: Stores the masks as frame properties in the output clip.
            The prop names are `FineDehalo2MaskV` and `FineDehalo2MaskH`.

    Returns:
        Dehaloed clip.
    """
    func = fine_dehalo2

    work_clip, *chroma = split(clip)

    mask_h = mask_v = None

    if mode in {ConvMode.HV, ConvMode.VERTICAL}:
        mask_h = BlurMatrixBase([1, 2, 1, 0, 0, 0, -1, -2, -1], ConvMode.V)(work_clip, divisor=4, saturate=False)

    if mode in {ConvMode.HV, ConvMode.HORIZONTAL}:
        mask_v = BlurMatrixBase([1, 0, -1, 2, 0, -2, 1, 0, -1], ConvMode.H)(work_clip, divisor=4, saturate=False)

    if mask_h and mask_v:
        mask_h2 = norm_expr([mask_h, mask_v], "x 3 * y -", func=func)
        mask_v2 = norm_expr([mask_v, mask_h], "x 3 * y -", func=func)
        mask_h, mask_v = mask_h2, mask_v2
    elif mask_h:
        mask_h = norm_expr(mask_h, "x 3 *", func=func)
    elif mask_v:
        mask_v = norm_expr(mask_v, "x 3 *", func=func)

    if mask_h:
        mask_h = grow_mask(mask_h, mask_radius, coord=Coordinates.VERTICAL, multiply=1.8, func=func)
    if mask_v:
        mask_v = grow_mask(mask_v, mask_radius, coord=Coordinates.HORIZONTAL, multiply=1.8, func=func)

    mask_h = mask_h and limiter(mask_h, mask=True, func=func)
    mask_v = mask_v and limiter(mask_v, mask=True, func=func)

    fix_weights = list(range(-1, -radius - 1, -1))
    fix_rweights = list(reversed(fix_weights))
    fix_zeros, fix_mweight = [0] * radius, 10 * (radius + 2)

    fix_h_conv = [*fix_weights, *fix_zeros, fix_mweight, *fix_zeros, *fix_rweights]
    fix_v_conv = [*fix_rweights, *fix_zeros, fix_mweight, *fix_zeros, *fix_weights]

    fix_h = ExprOp.convolution("x", fix_h_conv, mode=ConvMode.HORIZONTAL)(work_clip, func=func)
    fix_v = ExprOp.convolution("x", fix_v_conv, mode=ConvMode.VERTICAL)(work_clip, func=func)

    dehaloed = work_clip

    for fix, mask in [(fix_h, mask_v), (fix_v, mask_h)]:
        if mask:
            dehaloed = dehaloed.std.MaskedMerge(fix, mask)

    if dark is not None:
        dehaloed = norm_expr([work_clip, dehaloed], f"x y {'max' if dark else 'min'}")

    if darkstr != brightstr != 1.0:
        dehaloed = norm_expr(
            [work_clip, dehaloed],
            "x x y - dup {brightstr} * dup1 {darkstr} * ? -",
            func=func,
            darkstr=darkstr,
            brightstr=brightstr,
        )

    out = dehaloed if not chroma else join([dehaloed, *chroma])

    return out if not attach_masks else out.std.SetFrameProps(FineDehalo2MaskV=mask_v, FineDehalo2MaskH=mask_h)
