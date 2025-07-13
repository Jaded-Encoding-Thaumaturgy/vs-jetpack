from __future__ import annotations

from math import sqrt
from typing import Callable, Generic

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
    ConvMode,
    CustomIntEnum,
    CustomValueError,
    FuncExceptT,
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
)

from .alpha import IterArr, _limit_dehalo, dehalo_alpha

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

    def __init__(self, fine_dehalo: Callable[P, R]) -> None:
        self._func = fine_dehalo

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Masks(CustomIntEnum):
        MAIN = 1
        EDGES = 3
        SHARP_EDGES = 4
        LARGE_EDGES = 6
        IGNORE_DETAILS = 5
        SHRINK = 2
        SHRINK_EDGES_EXCL = 7

    def mask(
        self,
        clip: vs.VideoNode,
        dehaloed: vs.VideoNode | None = None,
        rx: int = 1,
        ry: int | None = None,
        thmi: int = 50,
        thma: int = 100,
        thlimi: int = 50,
        thlima: int = 100,
        exclude: bool = True,
        edgeproc: float = 0.0,
        edgemask: EdgeDetect = Robinson3(),
        show_mask: int | FineDehalo.Masks = 1,
        planes: PlanesT = 0,
        first_plane: bool = False,
        func: FuncExceptT | None = None,
    ) -> vs.VideoNode:
        """
        The fine_dehalo mask.

        Args:
            clip: Source clip.
            dehaloed: Optional dehaloed clip to mask. If this is specified, instead of returning the mask, the function
                will return the maskedmerged clip to this.
            rx: Horizontal radius for halo removal.
            ry: Vertical radius for halo removal.
            thmi: Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
            thma: Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
            thlimi: Minimum limiting threshold; include more edges than previously ignored details.
            thlima: Maximum limiting threshold; include more edges than previously ignored details.
            exclude: If True, add an addionnal step to exclude edges close to each other
            edgeproc: If > 0, it will add the edgemask to the processing, defaults to 0.0
            edgemask: Internal mask used for detecting the edges, defaults to Robinson3()
            show_mask: Whether to show the computed halo mask. 1-7 values to select intermediate masks.
            planes: Planes to process.
            first_plane: Whether to mask chroma planes with luma mask.
            func: Function from where this function was called.

        Returns:
            Mask or masked clip.
        """
        func = func or self.mask

        dehalo_mask = fine_dehalo(
            get_y(clip),
            rx,
            ry,
            thmi=thmi,
            thma=thma,
            thlimi=thlimi,
            thlima=thlima,
            exclude=exclude,
            edgeproc=edgeproc,
            edgemask=edgemask,
            planes=planes,
            show_mask=show_mask,
            func=func,
        )

        if dehaloed:
            return clip.std.MaskedMerge(dehaloed, dehalo_mask, planes, first_plane)

        return dehalo_mask


@FineDehalo
def fine_dehalo(
    clip: vs.VideoNode,
    rx: IterArr[float] = 2.0,
    ry: IterArr[float] | None = None,
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    thmi: int = 80,
    thma: int = 128,
    thlimi: int = 50,
    thlima: int = 100,
    ss: IterArr[float] = 1.5,
    contra: int | float | bool = 0.0,
    exclude: bool = True,
    edgeproc: float = 0.0,
    edgemask: EdgeDetectT = Robinson3,
    planes: PlanesT = 0,
    show_mask: int | FineDehalo.Masks | bool = False,
    func: FuncExceptT | None = None,
) -> vs.VideoNode:
    """
    Halo removal script that uses ``dehalo_alpha`` with a few masks and optional contra-sharpening
    to try removing halos without nuking important details like line edges.

    **For ``rx``, ``ry``, only the first value will be used for calculating the mask.**

    ``rx``, ``ry``, ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all
    configurable per plane and iteration. `tuple` means iteration, `list` plane.

    `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` means three iterations.
    * 1st => 2.0 for all planes
    * 2nd => 2.0 for luma, 2.4 for chroma
    * 3rd => 2.2 for luma, 2.0 for u, 2.1 for v

    Args:
        clip: Source clip.
        rx: Horizontal radius for halo removal.
        ry: Vertical radius for halo removal.
        darkstr: Strength factor for dark halos.
        brightstr: Strength factor for bright halos.
        lowsens: Sensitivity setting for how weak the dehalo has to be to get fully accepted.
        highsens: Sensitivity setting for how strong the dehalo has to be to get fully discarded.
        thmi: Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
        thma: Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
        thlimi: Minimum limiting threshold; include more edges than previously ignored details.
        thlima: Maximum limiting threshold; include more edges than previously ignored details.
        ss: Supersampling factor, to avoid creation of aliasing.
        contra: Contrasharpening. If True or int, will use [contrasharpening][vsdehalo.contrasharpening] otherwise uses
            [contrasharpening_fine_dehalo][vsdehalo.contrasharpening_fine_dehalo] with specified level.
        exclude: If True, add an addionnal step to exclude edges close to each other
        edgeproc: If > 0, it will add the edgemask to the processing, defaults to 0.0
        edgemask: Internal mask used for detecting the edges, defaults to Robinson3()
        planes: Planes to process.
        show_mask: Whether to show the computed halo mask. 1-7 values to select intermediate masks.
        func: Function from where this function was called.

    Returns:
        Dehaloed clip.
    """
    func = func or fine_dehalo

    assert check_variable(clip, func)
    assert check_progressive(clip, func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    if show_mask is not False and not (0 < int(show_mask) <= 7):
        raise CustomValueError("valid values for show_mask are 1-7!", func)

    thmif, thmaf, thlimif, thlimaf = [scale_mask(x, 8, clip) for x in [thmi, thma, thlimi, thlima]]

    peak = get_peak_value(clip)
    planes = normalize_planes(clip, planes)

    rx_i = cround(to_arr(to_arr(rx)[0])[0])
    ry_i = cround(to_arr(to_arr(rx if ry is None else ry)[0])[0])

    work_clip, *chroma = split(clip) if planes == [0] else (clip,)

    # Main edges #
    # Basic edge detection, thresholding will be applied later.
    edges = EdgeDetect.ensure_obj(edgemask, func).edgemask(work_clip)

    # Keeps only the sharpest edges (line edges)
    strong = norm_expr(edges, f"x {thmif} - {thmaf - thmif} / {peak} *", planes, func=func)

    # Extends them to include the potential halos
    large = Morpho.expand(strong, rx_i, ry_i, planes=planes, func=func)

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
    shrink = Morpho.expand(light, rx_i, ry_i, XxpandMode.ELLIPSE, planes=planes, func=func)

    # At this point, because the mask was made of a shades of grey, we may
    # end up with large areas of dark grey after shrinking. To avoid this,
    # we amplify and saturate the mask here (actually we could even
    # binarize it).
    shrink = norm_expr(shrink, "x 4 *", planes, func=func)
    shrink = Morpho.inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE, planes=planes, func=func)

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

    # Masking #
    if show_mask:
        return [mask, shrink, edges, strong, light, large, shr_med][int(show_mask) - 1]

    dehaloed = dehalo_alpha(
        work_clip,
        rx,
        ry,
        None,
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
            dehaloed = contrasharpening_dehalo(dehaloed, work_clip, contra, planes=planes)
        else:
            dehaloed = contrasharpening(dehaloed, work_clip, int(contra), planes=planes)

    y_merge = work_clip.std.MaskedMerge(dehaloed, mask, planes)

    if chroma:
        return join([y_merge, *chroma], clip.format.color_family)

    return y_merge


def fine_dehalo2(
    clip: vs.VideoNode,
    mode: OneDimConvModeT = ConvMode.HV,
    radius: int = 2,
    mask_radius: int = 2,
    brightstr: float = 1.0,
    darkstr: float = 1.0,
    dark: bool | None = True,
    show_mask: bool = False,
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
        show_mask: Whether to return the computed mask.

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

    if show_mask:
        if mask_h and mask_v:
            return combine([mask_h, mask_v], ExprOp.MAX)

        assert (ret_mask := mask_h or mask_v)  # noqa: RUF018

        return ret_mask

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

    if not chroma:
        return dehaloed

    return join([dehaloed, *chroma], clip.format.color_family)
