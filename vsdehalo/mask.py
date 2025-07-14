from __future__ import annotations

from math import sqrt
from typing import Callable, Generic, Iterator, Mapping

from jetpytools import P, R

from vsaa import NNEDI3
from vsexprtools import ExprOp, combine, norm_expr
from vsmasktools import EdgeDetect, EdgeDetectT, Morpho, PrewittTCanny, Robinson3, XxpandMode, grow_mask
from vsrgtools import (
    BlurMatrix,
    BlurMatrixBase,
    contrasharpening,
    contrasharpening_dehalo,
)
from vstools import (
    ConstantFormatVideoNode,
    ConvMode,
    FuncExceptT,
    FunctionUtil,
    InvalidColorFamilyError,
    OneDimConvModeT,
    PlanesT,
    check_progressive,
    check_variable,
    cround,
    get_peak_value,
    get_y,
    join,
    limiter,
    normalize_planes,
    scale_delta,
    scale_mask,
    split,
    to_arr,
    vs,
    vs_object,
)

from .alpha import IterArr, VSFunctionPlanesArgs, _limit_dehalo, dehalo_alpha

__all__ = ["base_dehalo_mask", "fine_dehalo", "fine_dehalo2"]


def base_dehalo_mask(
    clip: vs.VideoNode,
    expand: float = 0.5,
    iterations: int = 2,
    brz0: float = 0.31,
    brz1: float = 1.0,
    shift: int = 8,
    multi: float = 1.0,
) -> vs.VideoNode:
    """
    Based on `muvsfunc.YAHRmask`, stand-alone version with some tweaks. Adopted from jvsfunc.

    Args:
        clip: Input clip.
        expand: Expansion of edge mask.
        iterations: Protects parallel lines and corners that are usually damaged by strong dehaloing.
        brz0: Adjusts the internal line thickness.
        brz1: Adjusts the internal line thickness.
        shift: 8-bit corrective shift value for fine-tuning expansion.
        multi: Final pixel value multiplier.

    Returns:
        Dehalo mask.
    """

    assert check_progressive(clip, base_dehalo_mask)

    y = get_y(clip)

    y = NNEDI3(noshift=True).supersample(y)

    exp_edges = norm_expr(
        [y, Morpho.maximum(y, iterations=2)],
        "y x - {shift} - range_half *",
        shift=scale_delta(shift, 8, y),
        func=base_dehalo_mask,
    )

    edgemask = PrewittTCanny.edgemask(exp_edges, sigma=sqrt(expand * 2), mode=-1, multi=16)

    halo_mask = Morpho.maximum(exp_edges, iterations=iterations, func=base_dehalo_mask)
    halo_mask = Morpho.minimum(halo_mask, iterations=iterations, func=base_dehalo_mask)
    halo_mask = Morpho.binarize(halo_mask, brz0, 1.0, 0.0)

    if brz1 != 1.0:
        halo_mask = Morpho.inflate(halo_mask, iterations=2, func=base_dehalo_mask)
        halo_mask = Morpho.binarize(halo_mask, brz1)

    mask = norm_expr(
        [edgemask, BlurMatrix.BINOMIAL()(halo_mask)], "x y min {multi} *", multi=multi, func=base_dehalo_mask
    )

    return vs.core.resize.Point(mask, clip.width, clip.height)


class FineDehalo(Generic[P, R]):
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
        edgemask: EdgeDetectT = Robinson3,
        thmi: int = 80,
        thma: int = 128,
        thlimi: int = 50,
        thlima: int = 100,
        exclude: bool = True,
        edgeproc: float = 0.0,
        # Misc params
        planes: PlanesT = 0,
        func: FuncExceptT | None = None,
    ) -> ConstantFormatVideoNode:
        """
        The fine_dehalo mask.

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

    class Masks(Mapping[str, ConstantFormatVideoNode], vs_object):
        """
        Class for creating and storing intermediate masks used in the `fine_dehalo` function.

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
            edgemask: EdgeDetectT = Robinson3,
            thmi: int = 80,
            thma: int = 128,
            thlimi: int = 50,
            thlima: int = 100,
            exclude: bool = True,
            edgeproc: float = 0.0,
            # Misc params
            planes: PlanesT = 0,
            func: FuncExceptT | None = None,
        ) -> None:
            """
            Initialize the mask generation process.

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

            InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

            work_clip = get_y(clip) if planes == [0] else clip
            thmif, thmaf, thlimif, thlimaf = [scale_mask(x, 8, clip) for x in [thmi, thma, thlimi, thlima]]
            planes = normalize_planes(clip, planes)
            peak = get_peak_value(clip)

            # Main edges #
            # Basic edge detection, thresholding will be applied later.
            edges = EdgeDetect.ensure_obj(edgemask, func).edgemask(work_clip)

            # Keeps only the sharpest edges (line edges)
            strong = norm_expr(edges, f"x {thmif} - {thmaf - thmif} / {peak} *", planes, func=func)

            # Extends them to include the potential halos
            large = Morpho.expand(strong, rx, ry, planes=planes, func=func)

            # Exclusion zones #
            # When two edges are close from each other (both edges of a single
            # line or multiple parallel color bands), the halo removal
            # oversmoothes them or makes seriously bleed the bands, producing
            # annoying artifacts. Therefore we have to produce a mask to exclude
            # these zones from the halo removal.

            # Includes more edges than previously, but ignores simple details
            light = norm_expr(edges, f"x {thlimif} - {thlimaf - thlimif} / {peak} *", planes, func=func)

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
            shrink = BlurMatrix.MEAN()(shrink, planes, passes=2)

            # Final mask building #

            # Previous mask may be a bit weak on the pure edge side, so we ensure
            # that the main edges are really excluded. We do not want them to be
            # smoothed by the halo removal.
            shr_med = combine([strong, shrink], ExprOp.MAX, planes=planes) if exclude else strong

            # Subtracts masks and amplifies the difference to be sure we get 255
            # on the areas to be processed.
            mask = norm_expr([large, shr_med], "x y - 2 *", planes, func=func)

            # If edge processing is required, adds the edgemask
            if edgeproc > 0:
                mask = norm_expr([mask, strong], f"x y {edgeproc} 0.66 * * +", planes, func=func)

            # Smooth again and amplify to grow the mask a bit, otherwise the halo
            # parts sticking to the edges could be missed.
            # Also clamp to legal ranges
            mask = BlurMatrix.MEAN()(mask, planes)

            mask = norm_expr(mask, f"x 2 * {ExprOp.clamp(0, peak)}", planes, func=func)

            self._edges = edges
            self._strong = strong
            self._large = large
            self._light = light
            self._shrink = shrink
            self._shr_med = shr_med
            self._main = mask

        def __getitem__(self, index: str) -> ConstantFormatVideoNode:
            index = index.upper()

            if index in self._names:
                return getattr(self, index)

            raise KeyError

        def __iter__(self) -> Iterator[str]:
            yield from self._names

        def __len__(self) -> int:
            return len(self._names)

        @property
        def EDGES(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._edges

        @property
        def SHARP_EDGES(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._strong

        @property
        def LARGE_EDGES(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._large

        @property
        def IGNORE_DETAILS(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._light

        @property
        def SHRINK(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._shrink

        @property
        def SHRINK_EDGES_EXCL(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._shr_med

        @property
        def MAIN(self) -> ConstantFormatVideoNode:  # noqa: N802
            return self._main

        def __vs_del__(self, core_id: int) -> None:
            del self._edges
            del self._strong
            del self._large
            del self._light
            del self._shrink
            del self._shr_med
            del self._main


@FineDehalo
def fine_dehalo(
    clip: vs.VideoNode,
    # Blur params + fine_dehalo mask
    rx: IterArr[float] = 2.0,
    ry: IterArr[float] | None = None,
    blur_func: IterArr[VSFunctionPlanesArgs | None] = None,
    # dehalo_alpha mask params
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    # dehalo_alpha supersampling minmax params
    ss: IterArr[float] = 1.5,
    # dehalo_alpha limiting params
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    # fine_dehalo mask specific params
    edgemask: EdgeDetectT = Robinson3,
    thmi: int = 80,
    thma: int = 128,
    thlimi: int = 50,
    thlima: int = 100,
    exclude: bool = True,
    edgeproc: float = 0.0,
    # Final post processing
    contra: int | float | bool = 0.0,
    # Misc params
    planes: PlanesT = 0,
    *,
    attach_masks: bool = False,
    func: FuncExceptT | None = None,
) -> vs.VideoNode:
    """
    Halo removal function based on `dehalo_alpha`, enhanced with additional masking and optional contra-sharpening
    to better preserve important line detail while effectively reducing halos.

    The parameters `rx`, `ry`, `lowsens`, `highsens`, `ss`, `darkstr`, and `brightstr`
    can be configured per plane and per iteration. You can specify:

        - A single value: applies to all iterations and all planes.
        - A tuple of values: interpreted as iteration-wise.
        - A list inside the tuple: interpreted as per-plane for a specific iteration.

    For example:
        `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` implies 3 iterations:
            - 1st: 2.0 for all planes
            - 2nd: 2.0 for luma, 2.4 for both chroma planes
            - 3rd: 2.2 for luma, 2.0 for U, 2.1 for V

    **Note:** Only the first value of `rx` and `ry` is used for mask generation.

    Example usage:
        ```py
        dehalo = fine_dehalo(clip, ...)
        # Getting the masks of the last fine_dehalo call:
        dehalo_mask = fine_dehalo.masks.MAIN
        ```

    Args:
        clip: Source clip.
        rx: Horizontal radius for halo removal.
        ry: Vertical radius for halo removal. Defaults to `rx` if not set.
        blur_func: Optional custom blurring function to use in place of the default `dehalo_alpha` implementation.
        lowsens: Lower sensitivity threshold — dehalo is fully applied below this value.
        highsens: Upper sensitivity threshold — dehalo is completely skipped above this value.
        ss: Supersampling factor to reduce aliasing artifacts.
        darkstr: Strength factor for suppressing dark halos.
        brightstr: Strength factor for suppressing bright halos.
        edgemask: Edge detection object to use. Defaults to `Robinson3`.
        thmi: Minimum threshold for sharp edge selection; isolates only the strongest (line-like) edges.
        thma: Maximum threshold for sharp edge selection; filters out weaker edges.
        thlimi: Minimum threshold for including edges that were previously ignored.
        thlima: Maximum threshold for the inclusion of additional, less distinct edges.
        exclude: Whether to exclude edges that are too close together.
        edgeproc: If greater than 0, adds the edge mask into the final processing. Defaults to 0.0.
        contra: Contra-sharpening amount.
               - If `True` or `int`, uses [contrasharpening][vsdehalo.contrasharpening]
               - if `float`, uses [contrasharpening_dehalo][vsdehalo.contrasharpening_dehalo] with specified level.
        planes: Planes to process.
        attach_masks: Stores the masks as frame properties in the output clip.
            The prop names are `FineDehaloMask` + the masking step.
        func: An optional function to use for error handling.

    Returns:
        Dehaloed clip.
    """
    func_util = FunctionUtil(clip, func or fine_dehalo, planes, (vs.GRAY, vs.YUV))

    assert check_progressive(clip, func_util.func)

    rx_i = cround(to_arr(to_arr(rx)[0])[0])
    ry_i = cround(to_arr(to_arr(rx if ry is None else ry)[0])[0])

    fine_dehalo.masks = fine_dehalo.Masks(
        func_util.work_clip, rx_i, ry_i, edgemask, thmi, thma, thlimi, thlima, exclude, edgeproc, planes, func
    )

    dehaloed = dehalo_alpha(
        func_util.work_clip,
        rx,
        ry,
        blur_func,
        lowsens,
        highsens,
        ss,
        darkstr,
        brightstr,
        planes,
        func,
    )

    if contra:
        if isinstance(contra, float):
            dehaloed = contrasharpening_dehalo(dehaloed, func_util.work_clip, contra, planes=planes)
        else:
            dehaloed = contrasharpening(dehaloed, func_util.work_clip, int(contra), planes=planes)

    y_merge = func_util.work_clip.std.MaskedMerge(dehaloed, fine_dehalo.masks.MAIN, planes)

    out = func_util.return_clip(y_merge)

    if attach_masks:
        for k, v in fine_dehalo.masks.items():
            out = out.std.ClipToProp(v, "FineDehaloMask" + "".join(w.title() for w in k.split("_")))

    return out


def fine_dehalo2(
    clip: vs.VideoNode,
    mode: OneDimConvModeT = ConvMode.HV,
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

    assert check_variable(clip, func)
    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    work_clip, *chroma = split(clip)

    mask_h = mask_v = None

    if mode in {ConvMode.HV, ConvMode.VERTICAL}:
        mask_h = BlurMatrixBase([1, 2, 1, 0, 0, 0, -1, -2, -1], ConvMode.V)(work_clip, divisor=4, saturate=False)

    if mode in {ConvMode.HV, ConvMode.HORIZONTAL}:
        mask_v = BlurMatrixBase([1, 0, -1, 2, 0, -2, 1, 0, -1], ConvMode.H)(work_clip, divisor=4, saturate=False)

    if mask_h and mask_v:
        mask_h2 = norm_expr([mask_h, mask_v], ["x 3 * y -", ExprOp.clamp()], func=func)
        mask_v2 = norm_expr([mask_v, mask_h], ["x 3 * y -", ExprOp.clamp()], func=func)
        mask_h, mask_v = mask_h2, mask_v2
    elif mask_h:
        mask_h = norm_expr(mask_h, ["x 3 *", ExprOp.clamp()], func=func)
    elif mask_v:
        mask_v = norm_expr(mask_v, ["x 3 *", ExprOp.clamp()], func=func)

    if mask_h:
        mask_h = grow_mask(mask_h, mask_radius, coord=[0, 1, 0, 0, 0, 0, 1, 0], multiply=1.8, func=func)
    if mask_v:
        mask_v = grow_mask(mask_v, mask_radius, coord=[0, 0, 0, 1, 1, 0, 0, 0], multiply=1.8, func=func)

    if clip.format.sample_type == vs.FLOAT:
        mask_h = mask_h and limiter(mask_h, func=func)
        mask_v = mask_v and limiter(mask_v, func=func)

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
        dehaloed = combine([work_clip, dehaloed], ExprOp.MAX if dark else ExprOp.MIN)

    if darkstr != brightstr != 1.0:
        dehaloed = _limit_dehalo(work_clip, dehaloed, darkstr, brightstr, 0)

    out = dehaloed if not chroma else join([dehaloed, *chroma])

    return out if not attach_masks else out.std.SetFrameProps(FineDehalo2MaskV=mask_v, FineDehalo2MaskH=mask_h)
