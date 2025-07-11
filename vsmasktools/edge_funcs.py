from __future__ import annotations

from typing import Any, Literal, Sequence, cast, overload

from jetpytools import CustomEnum, CustomNotImplementedError

from vsexprtools import ExprOp, ExprToken, norm_expr
from vsrgtools import BlurMatrix, gauss_blur
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    ConvMode,
    check_variable,
    depth,
    get_peak_value,
    get_y,
    limiter,
    plane,
    scale_delta,
    scale_mask,
    scale_value,
    vs,
)

from .details import multi_detail_mask
from .edge import FDoGTCanny, Kirsch, MagDirection, Prewitt, PrewittTCanny
from .morpho import Morpho
from .spat_funcs import retinex
from .types import Coordinates, GenericMaskT
from .utils import normalize_mask

__all__ = ["dre_edgemask", "limited_linemask", "luma_credit_mask", "luma_mask", "ringing_mask", "tcanny_retinex"]


def ringing_mask(
    clip: vs.VideoNode,
    rad: int = 2,
    brz: float = 0.35,
    thmi: float = 0.315,
    thma: float = 0.5,
    thlimi: float = 0.195,
    thlima: float = 0.392,
    credit_mask: GenericMaskT = Prewitt,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    assert check_variable(clip, ringing_mask)

    thmi, thma, thlimi, thlima = (scale_mask(t, 32, clip) for t in [thmi, thma, thlimi, thlima])

    blur_kernel = BlurMatrix.BINOMIAL(1, mode=ConvMode.SQUARE)

    edgemask = normalize_mask(credit_mask, plane(clip, 0), **kwargs)
    edgemask = limiter(edgemask)

    light = norm_expr(edgemask, f"x {thlimi} - {thma - thmi} / {ExprToken.RangeMax} *", func=ringing_mask)

    shrink = Morpho.dilation(light, rad)
    shrink = Morpho.binarize(shrink, brz)
    shrink = Morpho.erosion(shrink, 2)
    shrink = blur_kernel(shrink, passes=2)

    strong = norm_expr(edgemask, f"x {thmi} - {thlima - thlimi} / {ExprToken.RangeMax} *", func=ringing_mask)
    expand = Morpho.dilation(strong, iterations=rad)

    mask = norm_expr([expand, strong, shrink], "x y z max -", func=ringing_mask)

    return ExprOp.convolution("x", blur_kernel, premultiply=2, multiply=2, clamp=True)(mask)


def luma_mask(clip: vs.VideoNode, thr_lo: float, thr_hi: float, invert: bool = True) -> ConstantFormatVideoNode:
    peak = get_peak_value(clip)

    lo, hi = (peak, 0) if invert else (0, peak)
    inv_pre, inv_post = (peak, "-") if invert else ("", "")

    thr_lo = scale_value(thr_lo, 32, clip)
    thr_hi = scale_value(thr_hi, 32, clip)

    return norm_expr(
        get_y(clip),
        f"x {thr_lo} < {lo} x {thr_hi} > {hi} {inv_pre} x {thr_lo} - {thr_lo} {thr_hi} - / {peak} * {inv_post} ? ?",
        func=ringing_mask,
    )


def luma_credit_mask(
    clip: vs.VideoNode, thr: float = 0.9, edgemask: GenericMaskT = FDoGTCanny, draft: bool = False, **kwargs: Any
) -> ConstantFormatVideoNode:
    y = plane(clip, 0)

    edge_mask = normalize_mask(edgemask, y, **kwargs)

    credit_mask = norm_expr([edge_mask, y], f"y {scale_mask(thr, 32, y)} > y 0 ? x min", func=ringing_mask)

    if not draft:
        credit_mask = Morpho.maximum(credit_mask, iterations=4)
        credit_mask = Morpho.inflate(credit_mask, iterations=2)

    return credit_mask


def tcanny_retinex(
    clip: vs.VideoNode, thr: float, sigma: Sequence[float] = [50, 200, 350], blur_sigma: float = 1.0
) -> ConstantFormatVideoNode:
    blur = gauss_blur(clip, blur_sigma)

    msrcp = retinex(blur, sigma, upper_thr=thr, fast=True, func=tcanny_retinex)

    tcunnied = msrcp.tcanny.TCanny(mode=1, sigma=1)

    return Morpho.minimum(tcunnied, coords=Coordinates.CORNERS)


def limited_linemask(
    clip: vs.VideoNode,
    sigmas: list[float] = [0.000125, 0.0025, 0.0055],
    detail_sigmas: list[float] = [0.011, 0.013],
    edgemasks: Sequence[GenericMaskT] = [Kirsch],
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    clip_y = plane(clip, 0)

    return ExprOp.ADD(
        (normalize_mask(edge, clip_y, **kwargs) for edge in edgemasks),
        (tcanny_retinex(clip_y, s) for s in sigmas),
        (multi_detail_mask(clip_y, s) for s in detail_sigmas),
    )


class dre_edgemask(CustomEnum):  # noqa: N801
    """
    Edgemask with dynamic range enhancement prefiltering.
    """

    RETINEX = cast("dre_edgemask", object())
    """
    Retinex-based edgemask.
    """

    CLAHE = cast("dre_edgemask", object())
    """
    Contrast Limited Adaptive Histogram Equalization.
    Based on the [OpenCV implementation](https://docs.opencv.org/5.x/d5/daf/tutorial_py_histogram_equalization.html)
    """

    def _prefilter(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        if self is dre_edgemask.RETINEX:
            sigmas = kwargs.get("sigmas", [50, 200, 350])

            return retinex(clip, sigmas, 0.001, 0.005)

        if self is dre_edgemask.CLAHE:
            limit, tile = kwargs.get("limit", 0.0305), kwargs.get("tile", 5)

            return depth(depth(clip, 16).vszip.CLAHE(int(scale_delta(limit, 32, 16)), tile), clip)

        raise CustomNotImplementedError

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[dre_edgemask.RETINEX],
        clip: vs.VideoNode,
        tsigma: float = 1,
        brz: float = 0.122,
        *,
        sigmas: Sequence[float] = [50, 200, 350],
    ) -> ConstantFormatVideoNode: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[dre_edgemask.CLAHE],
        clip: vs.VideoNode,
        tsigma: float = 1,
        brz: float = 0.122,
        *,
        limit: float = 0.0305,
        tile: int = 5,
    ) -> ConstantFormatVideoNode: ...

    @overload
    def __call__(
        self, clip: vs.VideoNode, tsigma: float = 1, brz: float = 0.122, **kwargs: Any
    ) -> ConstantFormatVideoNode: ...

    def __call__(
        self, clip: vs.VideoNode, tsigma: float = 1, brz: float = 0.122, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Creates an edgemask with dynamic range enhancement (DRE) prefiltering.

        This function serves as a wrapper around the `retinex` and `vszip.CLAHE` functions,
        applying one of them as a prefilter before generating the edgemask.

        Args:
            clip: Source clip.
            tsigma: Sigma value for TCanny edge detection. Defaults to 1.
            brz: Binarization threshold (32-bit float scale). Defaults to 0.122.
            sigmas: Sigma values for the retinex prefilter. Defaults to [50, 200, 350].
            limit: Limit for CLAHE prefilter. Defaults to 0.0305.
            tile: Tile size for CLAHE prefilter. Defaults to 5.

        Returns:
            Edgemask clip with applied DRE prefiltering.
        """
        luma = get_y(clip)

        dreluma = self._prefilter(luma, **kwargs)

        tcanny = PrewittTCanny.edgemask(dreluma, sigma=tsigma, scale=1)
        tcanny = Morpho.minimum(tcanny, coords=Coordinates.CORNERS)

        kirsch = Kirsch(MagDirection.N | MagDirection.EAST).edgemask(luma)

        add_clip = ExprOp.ADD(tcanny, kirsch)

        if brz > 0:
            add_clip = Morpho.binarize(add_clip, brz)

        return ColorRange.FULL.apply(add_clip)
