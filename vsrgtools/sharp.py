from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from jetpytools import FuncExcept, fallback

from vsexprtools import norm_expr

if TYPE_CHECKING:
    from vsmasktools import MaskLike

from vsjetpack import require_jet_dependency
from vstools import (
    ConvMode,
    FunctionUtil,
    Planes,
    VSFunctionNoArgs,
    VSFunctionPlanesArgs,
    check_ref_clip,
    get_peak_value,
    limiter,
    scale_delta,
    scale_mask,
    scale_value,
    vs,
)

from .blur import box_blur, gauss_blur, median_blur
from .enum import BlurMatrix
from .rgtools import repair

__all__ = ["awarpsharp", "fast_line_darken", "fine_sharp", "soothe", "unsharpen"]


def unsharpen(
    clip: vs.VideoNode,
    strength: float = 1.0,
    blur: vs.VideoNode | VSFunctionNoArgs = partial(gauss_blur, sigma=1.5),
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Apply an unsharp mask to a clip.

    This filter sharpens the input by subtracting a blurred version of the clip
    from the original, scaling the difference by the given `strength`, and
    adding it back to the original image. Conceptually:

        result = clip + (clip - blur(clip)) * strength

    Args:
        clip: Input clip.
        strength: Sharpening strength. Defaults to 1.0.
        blur: Either a blurred reference clip or a callable that takes the source clip
            and returns a blurred version (e.g., a Gaussian blur).
        planes: Which plane to process. Default to all.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A sharpened clip.
    """
    func = func or unsharpen

    if callable(blur):
        blur = blur(clip)

    check_ref_clip(clip, blur, func)

    return norm_expr([clip, blur], f"x y - {strength} * x +", planes, func=func)


def awarpsharp(
    clip: vs.VideoNode,
    mask: MaskLike | None = None,
    thresh: float = 128,
    blur: int | VSFunctionPlanesArgs | Literal[False] = 3,
    depth_h: int | Sequence[int] | None = None,
    depth_v: int | Sequence[int] | None = None,
    mask_first_plane: bool | None = None,
    planes: Planes = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Sharpens edges by warping them.

    Args:
        clip: Clip to process. Must be either the same size as mask, or four times the size of mask in each dimension.
            The latter can be useful if better subpixel interpolation is desired. If clip is upscaled to four times the
            original size, it must be top-left aligned.
        mask: Edge mask. Default is [Sobel][vsmasktools.Sobel].
            [ASobel][vsmasktools.ASobel] is what you need if you want the original implementation.
        thresh: No pixel in the edge mask will have a value greater than thresh. Decrease for weaker sharpening.
        blur: Specifies the blur applied to the edge mask.

               - If an `int`, it sets the number of passes for the default `box_blur` filter.
               - If a callable, a custom blur function will be used instead.
               - If `False`, no blur will be applied.
        depth_h: Controls how far to warp horizontally.
            Negative values warp in the other direction, i.e. will blur the image instead of sharpening.
        depth_v: Controls how far to warp vertically.
            Negative values warp in the other direction, i.e. will blur the image instead of sharpening.
        mask_first_plane: Controls the chroma handling method.
            None defaults to True for YUV color family, False otherwise.
            True will use the edge mask from the luma to warp the chroma.
            False will create an edge mask from each chroma channel and use those to warp each chroma channel
            individually.
        planes: Planes to process. Defaults to all.
        **kwargs: Additional arguments forwarded to the [normalize_mask][vsmasktools.normalize_mask] function.

    Returns:
        Warp-sharpened clip.
    """
    from vsmasktools import Sobel, normalize_mask

    func = FunctionUtil(clip, awarpsharp, planes)

    thresh = scale_mask(thresh, 8, func.work_clip)
    mask_first_plane = fallback(mask_first_plane, func.work_clip.format.color_family == vs.YUV)
    mask_planes = 0 if mask_first_plane else planes

    if mask is None:
        mask = Sobel

    kwargs = {"clamp": (0, thresh)} | kwargs

    mask = normalize_mask(mask, func.work_clip, func.work_clip, func=func.func, planes=mask_planes, **kwargs)

    if blur is not False:
        blur_fn = partial(box_blur, radius=2, passes=blur, planes=planes) if isinstance(blur, int) else blur
        mask = blur_fn(mask, planes=mask_planes)

    warp = func.work_clip.awarp.AWarp(mask, depth_h, depth_v, mask_first_plane, planes)

    return func.return_clip(warp)


@require_jet_dependency("scipy")
def fine_sharp(
    clip: vs.VideoNode,
    mode: int = 0,
    sstr: float = 2.0,
    cstr: float | None = None,
    xstr: float = 0.19,
    lstr: float = 1.49,
    pstr: float = 1.272,
    ldmp: float | None = None,
    hdmp: float = 0.01,
    planes: Planes = 0,
) -> vs.VideoNode:
    """
    Original author: Didée (https://forum.doom9.org/showthread.php?t=166082)
    Small and relatively fast realtime-sharpening function, for 1080p,
    or after scaling 720p → 1080p during playback.
    (to make 720p look more like being 1080p)
    It's a generic sharpener. Only for good quality sources!
    (If the source is crap, FineSharp will happily sharpen the crap) :)
    Noise/grain will be enhanced, too. The method is GENERIC.

    Modus operandi: A basic nonlinear sharpening method is performed,
    then the *blurred* sharp-difference gets subtracted again.

    Args:
        clip: Clip to process.
        mode: 0 or 1, weakest to strongest.
        sstr: strength of sharpening.
        cstr: strength of equalisation.
        xstr: strength of XSharpen-style final sharpening.
        lstr: modifier for non-linear sharpening.
        pstr: exponent for non-linear sharpening.
        ldmp: "low damp" - Avoid over-enhancing very small differences.
        hdmp: "high damp" - Avoid over-enhancing very large differences.

    Returns:
        Sharpened clip.
    """
    func = FunctionUtil(clip, fine_sharp, planes)

    if cstr is None:
        from numpy import asarray
        from scipy import interpolate

        cs = interpolate.CubicSpline(
            (0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 255.0), (0, 0.1, 0.6, 0.9, 1.0, 1.09, 1.15, 1.19, 1.249, 1.5)
        )
        cstr = float(cs(asarray(sstr)))

    if ldmp is None:
        ldmp = sstr + 0.1

    if mode == 0:
        blurred = median_blur(BlurMatrix.BINOMIAL()(func.work_clip, planes), planes=planes)
    elif mode == 1:
        blurred = BlurMatrix.BINOMIAL()(median_blur(func.work_clip, planes=planes), planes)
    else:
        raise NotImplementedError

    sharp = func.work_clip

    if sstr:
        sharp = norm_expr(
            [func.work_clip, blurred],
            "x y = x dup x y - range_size 256 / / dup dup dup abs {lstr} / {pstr} pow "
            "swap3 abs {hdmp} + / swap dup * dup {ldmp} + / * * {sstr} * + ?",
            planes,
            lstr=lstr,
            pstr=1 / pstr,
            sstr=scale_delta(sstr, 8, clip),
            ldmp=ldmp,
            hdmp=hdmp,
            func=func.func,
        )

    if cstr:
        diff = norm_expr([func.work_clip, sharp], "x y - {cstr} * neutral +", planes, cstr=cstr, func=func.func)
        sharp = sharp.std.MergeDiff(BlurMatrix.BINOMIAL()(diff, planes))

    if xstr:
        xysharp = norm_expr([sharp, box_blur(sharp, planes=planes)], "x x y - 9.9 * +", planes, func=func.func)
        rpsharp = repair(xysharp, sharp, 12, planes)
        sharp = sharp.std.Merge(rpsharp, func.norm_seq(xstr, 0))

    return func.return_clip(sharp)


def soothe(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    spatial_strength: float = 0.0,
    temporal_strength: float = 0.75,
    spatial_radius: int = 1,
    temporal_radius: int = 1,
    scenechange: bool = False,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    Lessens the temporal instability and aliasing caused by sharpening, by comparing the original and sharpened clip,
    leaving a smoother and slightly softer output. Soothe is a small postprocessor function for sharpening filters.
    The goal is temporal stabilization of clips that have been sharpened before.

    Args:
        flt: Filtered clip.
        src: Source clip.
        spatial_strength: Spatial soothing strength.
        temporal_strength: Temporal soothing strength.
        spatial_radius: Spatial soothing radius.
        temporal_radius: Temporal soothing radius.
        scenechange: Avoid applying temporal soothing across scene changes.
        planes: Planes to process.

    Returns:
        Soothed clip.
    """
    sharp_diff = src.std.MakeDiff(flt, planes)

    expr = (
        "x neutral - X! y neutral - Y! X@ Y@ xor X@ {strength} * neutral + X@ abs Y@ abs > x y - {strength} * y + x ? ?"
    )

    if spatial_strength:
        blurred = box_blur(sharp_diff, spatial_radius, planes=planes)
        sharp_diff = norm_expr([sharp_diff, blurred], expr, strength=1.0 - spatial_strength, planes=planes, func=soothe)

    if temporal_strength:
        blurred = box_blur(sharp_diff, temporal_radius, 1, ConvMode.TEMPORAL, planes, scenechange=scenechange)
        sharp_diff = norm_expr(
            [sharp_diff, blurred], expr, strength=1.0 - temporal_strength, planes=planes, func=soothe
        )

    return src.std.MakeDiff(sharp_diff, planes)


def fast_line_darken(
    clip: vs.VideoNode,
    strength: float = 48,
    protection: float = 5,
    luma_cap: float = 191,
    threshold: float = 4,
    thinning: float = 0,
) -> vs.VideoNode:
    """
    Sharpens by darkening lines.

    Args:
        clip: Clip to process.
        strength: Line darkening amount. Represents the maximum amount that the luma will be reduced by, weaker
            lines will be reduced by proportionately less.
        protection: Prevents the darkest lines from being darkened. Protection acts as a threshold. Values range from 0
            (no prot) to ~50 (protect everything).
        luma_cap: Value from 0 (black) to 255 (white), used to stop the darkening determination from being 'blinded' by
            bright pixels, and to stop grey lines on white backgrounds being darkened. Any pixels brighter than luma_cap
            are treated as only being as bright as luma_cap. Lowering luma_cap tends to reduce line darkening. 255
            disables capping.
        threshold: Any pixels that were going to be darkened by an amount less than threshold will not be touched.
            Setting this to 0 will disable it, setting it to 4 (default) is recommended, since often a lot of random
            pixels are marked for very slight darkening and a threshold of about 4 should fix them. Note if you set
            threshold too high, some lines will not be darkened.
        thinning: Optional line thinning amount. Setting this to 0 will disable it, which gives a big speed
            increase. Note that thinning the lines will inherently darken the remaining pixels in each line a little.

    Returns:
        Line-darkened clip.
    """
    from vsmasktools import Morpho

    func = FunctionUtil(clip, fast_line_darken, 0, vs.YUV)

    strength /= 128
    cap = scale_value(luma_cap, 8, func.work_clip)
    thr = scale_delta(threshold, 8, func.work_clip)
    thinning /= 16

    max_thr = scale_delta(get_peak_value(func.work_clip) / (protection + 1), func.work_clip, 32)

    closing = limiter(Morpho.minimum(Morpho.maximum(func.work_clip, max_thr)), max_val=cap)
    thick = norm_expr([func.work_clip, closing], "y x {thr} + > x y - {strength} * x + x ?", thr=thr, strength=strength)

    if not thinning:
        return func.return_clip(thick)

    diff = norm_expr([func.work_clip, closing], "y x {thr} + > x y - neutral + neutral ?", thr=thr)
    linemask = BlurMatrix.MEAN()(norm_expr(Morpho.minimum(diff), "x neutral - {thn} * plane_max +", thn=thinning))
    thin = norm_expr([Morpho.maximum(func.work_clip), diff], "x y neutral - {strength} 1 + * +", strength=strength)
    return func.return_clip(thin.std.MaskedMerge(thick, linemask))
