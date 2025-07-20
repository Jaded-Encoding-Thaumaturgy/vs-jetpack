"""
This module implements functions based on the famous dehalo_alpha.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence, TypeAlias

from jetpytools import T, to_arr

from vsexprtools import norm_expr
from vskernels import BSpline, Lanczos, Mitchell, Scaler, ScalerLike
from vsmasktools import Morpho
from vsrgtools import repair
from vstools import (
    ConstantFormatVideoNode,
    CustomIndexError,
    FuncExceptT,
    InvalidColorFamilyError,
    PlanesT,
    check_progressive,
    check_variable,
    core,
    get_peak_value,
    join,
    mod4,
    normalize_planes,
    normalize_seq,
    split,
    vs,
)
from vstools import VSFunctionPlanesArgs as GenericVSFunctionPlanesArgs

__all__ = ["dehalo_alpha"]


IterArr: TypeAlias = T | list[T] | tuple[T | list[T], ...]
VSFunctionPlanesArgs: TypeAlias = GenericVSFunctionPlanesArgs[vs.VideoNode, vs.VideoNode]


def dehalo_alpha(
    clip: vs.VideoNode,
    # Blur params
    rx: IterArr[float] = 2.0,
    ry: IterArr[float] | None = None,
    blur_func: IterArr[VSFunctionPlanesArgs | None] = None,
    # Mask params
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    # Supersampling minmax params
    ss: IterArr[float] = 1.5,
    # Limiting params
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    # Misc params
    planes: PlanesT = 0,
    *,
    attach_masks: bool = False,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Reduce halo artifacts by aggressively processing the edges and their surroundings.

    The parameters `rx`, `ry`, `lowsens`, `highsens`, `ss`, `darkstr`, `brightstr` and `blur_func`
    can be configured per plane and per iteration. You can specify:

        - A single value: applies to all iterations and all planes.
        - A tuple of values: interpreted as iteration-wise.
        - A list inside the tuple: interpreted as per-plane for a specific iteration.

    For example:
        `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` implies 3 iterations:
            - 1st: 2.0 for all planes
            - 2nd: 2.0 for luma, 2.4 for both chroma planes
            - 3rd: 2.2 for luma, 2.0 for U, 2.1 for V

    Args:
        clip: Source clip.
        rx: Horizontal radius for halo removal.
        ry: Vertical radius for halo removal. Defaults to `rx` if not set.
        blur_func: Optional custom blurring function to use in place of the default implementation.
        lowsens: Lower sensitivity threshold — dehalo is fully applied below this value.
            Setting both `lowsens` and `highsens` to `-1` disables mask-based processing entirely.
        highsens: Upper sensitivity threshold — dehalo is completely skipped above this value.
            Setting both `lowsens` and `highsens` to `-1` disables mask-based processing entirely.
        ss: Supersampling factor to reduce aliasing artifacts.
        darkstr: Strength factor for suppressing dark halos.
        brightstr: Strength factor for suppressing bright halos.
        planes: Planes to process. Default to 0.
        attach_masks: Stores generated masks as frame properties in the output clip.
            The prop name is `DehaloAlphaMask_{i}`, where `i` is the iteration index.
        func: An optional function to use for error handling.
        **kwargs: Additionnal advanced parameters.

    Raises:
        CustomIndexError: If `ss`, `rx` or `ry` are lower than 1.0.
        CustomIndexError: If `brightstr` or `darkstr` are not between 0.0 and 1.0 (inclusive).
        CustomIndexError: If `lowsens` or `highsens` are not beween 0 and 100 (inclusive).

    Returns:
        Dehaloed clip.
    """
    func = func or dehalo_alpha

    assert check_variable(clip, func)
    assert check_progressive(clip, func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    if ry is None:
        ry = rx

    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip,)

    values, blur_funcs = _normalize_iter_arr_t(rx, ry, darkstr, brightstr, lowsens, highsens, ss, blur_func=blur_func)
    values_kwargs, _ = _normalize_iter_arr_t(
        kwargs.get("downscaler", Mitchell),
        kwargs.get("upscaler", BSpline),
        kwargs.get("supersampler", Lanczos),
        kwargs.get("supersampler_ref", Mitchell),
        blur_func=None,
    )

    masks_to_prop = list[ConstantFormatVideoNode]()

    for (
        (rx_i, ry_i, darkstr_i, brightstr_i, lowsens_i, highsens_i, ss_i),
        blur_func_i,
        (downs_i, ups_i, sser_i, sser_ref_i),
    ) in zip(values, blur_funcs, values_kwargs):
        if any(x < 1 for x in (*ss_i, *rx_i, *ry_i)):
            raise CustomIndexError("ss, rx, and ry must all be bigger than 1.0!", func)

        if not all(0 <= x <= 1 for x in (*brightstr_i, *darkstr_i)):
            raise CustomIndexError("brightstr and darkstr must be between 0.0 and 1.0!", func)

        # Process without splitting the planes
        # if the radius are the same for all planes or the blur_func is the same
        # or if luma only
        # or clip format is GRAY
        if (
            any(
                [
                    len(set(rx_i)) == len(set(ry_i)) == len(set(blur_func_i)) == 1,
                    planes == [0],
                    work_clip.format.num_planes == 1,
                ]
            )
            and len(set(downs_i)) == len(set(ups_i)) == 1
        ):
            dehalo = (
                blur_func_i[0](work_clip, planes=planes)
                if blur_func_i[0]
                else _dehalo_alpha_blur_func(work_clip, rx_i[0], ry_i[0], downs_i[0], ups_i[0], func)
            )
        else:
            dehalo = join(
                [
                    blur_func_p(p, planes=0)
                    if blur_func_p
                    else _dehalo_alpha_blur_func(p, rx_p, ry_p, downs_p, ups_p, func)
                    for p, rx_p, ry_p, blur_func_p, downs_p, ups_p in zip(
                        split(work_clip), rx_i, ry_i, blur_func_i, downs_i, ups_i
                    )
                ]
            )

        if all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            mask = _dehalo_alpha_mask(work_clip, dehalo, lowsens_i, highsens_i, planes, func)
        elif lowsens_i.count(-1) == len(lowsens_i) and highsens_i.count(-1) == len(highsens_i):
            mask = None
        else:
            raise CustomIndexError("lowsens and highsens must be between 0 and 100!", func)

        if attach_masks and mask:
            masks_to_prop.append(core.std.SetFrameProps(mask, lowsens=lowsens_i, highsens=highsens_i))

        dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes) if mask else dehalo

        dehalo = _dehalo_supersample_minmax(work_clip, dehalo, ss_i, sser_i, sser_ref_i, planes, func)

        work_clip = dehalo = _limit_dehalo(work_clip, dehalo, darkstr_i, brightstr_i, planes, func)

    out = dehalo if not chroma else join([dehalo, *chroma])

    for i, mask in enumerate(masks_to_prop):
        out = out.std.ClipToProp(mask, f"DehaloAlphaMask_{i}")

    return out


# HELPER FUNCTIONS BELOW #


def _normalize_iter_arr_t(
    *values: IterArr[T], blur_func: IterArr[VSFunctionPlanesArgs | None]
) -> tuple[Iterator[tuple[list[T], ...]], Iterator[list[VSFunctionPlanesArgs | None]]]:
    max_len = max((len(x) if isinstance(x, tuple) else 1) for x in (*values, blur_func))

    broadcasted: list[tuple[T | list[T] | VSFunctionPlanesArgs | list[VSFunctionPlanesArgs | None] | None, ...]] = [
        val + (val[-1],) * (max_len - len(val)) if isinstance(val, tuple) else (val,) * max_len
        for val in (*values, blur_func)
    ]

    normalized = list[list[list[T | VSFunctionPlanesArgs] | None]]()

    for subgroup in broadcasted:
        sublist = list[list[T | VSFunctionPlanesArgs | None]]()

        for item in subgroup:
            group = normalize_seq(item)

            sublist.append(group)

        normalized.append(sublist)  # type: ignore[arg-type]

    return (zip(*normalized[:-1]), iter(normalized[-1]))  # type: ignore[arg-type]


def _dehalo_alpha_blur_func(
    clip: ConstantFormatVideoNode,
    rx: float,
    ry: float | None = None,
    downscaler: ScalerLike = Mitchell,
    upscaler: ScalerLike = BSpline,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Default gaussian approximation used in the original dehalo_alpha implementation.
    """
    downscaler = Scaler.ensure_obj(downscaler, func)
    upscaler = Scaler.ensure_obj(upscaler, func)

    if ry is None:
        ry = rx

    return upscaler.scale(  # type: ignore[return-value]
        downscaler.scale(clip, mod4(clip.width / rx), mod4(clip.height / ry)), clip.width, clip.height
    )


def _dehalo_alpha_mask(
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    lowsens: list[float],
    highsens: list[float],
    planes: PlanesT,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    func = func or _dehalo_alpha_mask

    mask = norm_expr(
        [
            Morpho.gradient(clip, planes=planes, func=func),
            Morpho.gradient(ref, planes=planes, func=func),
        ],
        "x x y - x / 0 ? {lowsens} - x {peak} / 256 255 / + 512 255 / / {highsens} + * 0 max 1 min {peak} *",
        planes,
        peak=get_peak_value(clip),
        lowsens=[lo / 255 for lo in lowsens],
        highsens=[hi / 100 for hi in highsens],
        func=func,
    )

    return mask


def _dehalo_supersample_minmax(
    clip: ConstantFormatVideoNode,
    ref: vs.VideoNode,
    ss: Sequence[float],
    supersampler: ScalerLike | Sequence[ScalerLike] = Lanczos,
    supersampler_ref: ScalerLike | Sequence[ScalerLike] = Mitchell,
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    func = func or _dehalo_supersample_minmax

    supersampler_obj = [Scaler.ensure_obj(scaler, func) for scaler in to_arr(supersampler)]  # type: ignore[arg-type]
    supersampler_ref_obj = [Scaler.ensure_obj(scaler, func) for scaler in to_arr(supersampler_ref)]  # type: ignore[arg-type]

    def _supersample(
        work_clip: vs.VideoNode,
        dehalo: vs.VideoNode,
        ss: float,
        supersampler: Scaler,
        supersampler_ref: Scaler,
        planes: PlanesT,
    ) -> ConstantFormatVideoNode:
        if ss <= 1.0:
            return repair(work_clip, dehalo, 1, planes)

        w, h = mod4(work_clip.width * ss), mod4(work_clip.height * ss)
        ss_clip = norm_expr(
            [
                supersampler.scale(work_clip, w, h),
                supersampler_ref.scale(dehalo.std.Maximum(), w, h),
                supersampler_ref.scale(dehalo.std.Minimum(), w, h),
            ],
            "x y min z max",
            planes,
            func=func,
        )

        return supersampler.scale(ss_clip, work_clip.width, work_clip.height)  # type: ignore[return-value]

    if (
        (len(set(ss)) == 1 or planes == [0] or clip.format.num_planes == 1)
        and len(set(supersampler_obj)) == 1
        and len(set(supersampler_ref_obj)) == 1
    ):
        return _supersample(clip, ref, ss[0], supersampler_obj[0], supersampler_ref_obj[0], planes)

    return join(
        [
            _supersample(wplane, dplane, ssp, sss_p, sssref_p, 0)
            for wplane, dplane, ssp, sss_p, sssref_p in zip(
                split(clip), split(ref), ss, supersampler_obj, supersampler_ref_obj
            )
        ]
    )


def _limit_dehalo(
    clip: ConstantFormatVideoNode,
    ref: ConstantFormatVideoNode,
    darkstr: float | list[float],
    brightstr: float | list[float],
    planes: PlanesT = None,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    func = func or _limit_dehalo

    return norm_expr(
        [clip, ref],
        "x y - D! x x y < D@ {darkstr} * D@ {brightstr} * ? -",
        planes,
        darkstr=darkstr,
        brightstr=brightstr,
        func=func,
    )
