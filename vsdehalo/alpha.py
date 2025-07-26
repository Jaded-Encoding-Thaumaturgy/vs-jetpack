"""
This module implements functions based on the famous dehalo_alpha.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence, TypeAlias, TypeGuard
from warnings import deprecated

from jetpytools import T, mod_x

from vsaa import NNEDI3
from vsdenoise import Prefilter
from vsexprtools import norm_expr
from vskernels import BSpline, Lanczos, Mitchell, Point, Scaler, ScalerLike
from vsmasktools import Morpho
from vsrgtools import MeanMode, gauss_blur, repair
from vsscale import pre_ss as pre_supersampling
from vstools import (
    ConstantFormatVideoNode,
    CustomIndexError,
    FuncExceptT,
    FunctionUtil,
    PlanesT,
    check_progressive,
    check_variable_format,
    core,
    join,
    limiter,
    normalize_planes,
    normalize_seq,
    scale_delta,
    split,
    vs,
)
from vstools import VSFunctionPlanesArgs as GenericVSFunctionPlanesArgs

__all__ = ["AlphaBlur", "dehalo_alpha", "dehalo_omega", "dehalo_sigma"]


IterArr: TypeAlias = T | list[T] | tuple[T | list[T], ...]
VSFunctionPlanesArgs: TypeAlias = GenericVSFunctionPlanesArgs[vs.VideoNode, vs.VideoNode]


def dehalo_omega(
    clip: vs.VideoNode,
    # Blur param
    blur: IterArr[float]
    | VSFunctionPlanesArgs
    | tuple[float | list[float] | VSFunctionPlanesArgs, ...] = Prefilter.GAUSS(sigma=1.4),
    # Mask params
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    # Supersampling clamp params
    ss: float | tuple[float, ...] = 1.5,
    # Limiting params
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    # Misc params
    planes: PlanesT = 0,
    attach_masks: bool = False,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Reduce halo artifacts by aggressively processing the edges and their surroundings.

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

    Args:
        clip: Source clip.
        blur: Standard deviation of the Gaussian kernel if float or custom blurring function
            to use in place of the default implementation.
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
        CustomIndexError: If `lowsens` or `highsens` are not beween 0 and 100 (inclusive).

    Returns:
        Dehaloed clip.
    """
    util = FunctionUtil(clip, func or dehalo_omega, planes)

    assert check_progressive(clip, util.func)

    values = _normalize_iter_arr_t(
        blur,
        lowsens,
        highsens,
        ss,
        darkstr,
        brightstr,
        kwargs.get("supersampler", Lanczos()),
        kwargs.get("supersampler_ref", Mitchell()),
    )
    masks_to_prop = list[ConstantFormatVideoNode]()

    work_clip = util.work_clip

    for (
        blur_i,
        lowsens_i,
        highsens_i,
        (ss_i, *_),
        darkstr_i,
        brightstr_i,
        (sser_i, *_),
        (sser_ref_i, *_),
    ) in values:
        # Applying the blur function
        dehalo = (
            blur_i[0](work_clip, planes=planes)
            if _is_callable(blur_i[0])
            else gauss_blur(work_clip, blur_i, planes=planes)
        )

        # Building the mask
        if all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            mask = norm_expr(
                [Morpho.gradient(work_clip, planes=planes), Morpho.gradient(dehalo, planes=planes)],
                "x 0 = x y - dup x / ? range_max * {lowsens} - x range_size + range_size 2 * / {highsens} + *",
                planes,
                func=util.func,
                lowsens=(scale_delta(x, 8, clip) for x in lowsens_i),
                highsens=(x / 100 for x in highsens_i),
            )

            if attach_masks:
                masks_to_prop.append(core.std.SetFrameProps(mask, lowsens=lowsens_i, highsens=highsens_i))

            dehalo = core.std.MaskedMerge(dehalo, work_clip, limiter(mask, planes=planes, func=util.func), planes)

        elif lowsens_i.count(-1) == len(lowsens_i) and highsens_i.count(-1) == len(highsens_i):
            pass
        else:
            raise CustomIndexError("lowsens and highsens must be between 0 and 100!", func)

        # Clamping with supersampling clips to reduce aliasing
        if ss_i == 1:
            dehalo = repair.Mode.MINMAX_SQUARE1(work_clip, dehalo, planes)
        else:
            ss_width = mod_x(work_clip.width * ss_i, 2**work_clip.format.subsampling_w)
            ss_height = mod_x(work_clip.height * ss_i, 2**work_clip.format.subsampling_h)

            sser_i = Scaler.ensure_obj(sser_i)
            sser_ref_i = Scaler.ensure_obj(sser_ref_i)

            clip_ss = sser_i.scale(work_clip, ss_width, ss_height)
            inpand = sser_ref_i.scale(Morpho.minimum(dehalo, planes=planes), ss_width, ss_height)
            expand = sser_ref_i.scale(Morpho.maximum(dehalo, planes=planes), ss_width, ss_height)
            dehalo = sser_i.scale(
                MeanMode.MEDIAN(clip_ss, inpand, expand, planes=planes), work_clip.width, work_clip.height
            )

        # Limiting the dehalo clip to control the bright and dark halos
        work_clip = dehalo = norm_expr(
            [work_clip, dehalo],
            "x y - D! x x y < D@ {darkstr} * D@ {brightstr} * ? -",
            planes,
            func=util.func,
            darkstr=darkstr_i,
            brightstr=brightstr_i,
        )

    out = util.return_clip(dehalo)

    for i, mask in enumerate(masks_to_prop):
        out = out.std.ClipToProp(mask, f"DehaloAlphaMask_{i}")

    return out


class AlphaBlur:
    """
    A Gaussian blur approximation inspired by Dehalo_Alpha.
    """

    __slots__ = ("downscaler", "func", "rx", "ry", "upscaler")

    def __init__(
        self,
        rx: float | Sequence[float] = 2.0,
        ry: float | Sequence[float] | None = None,
        func: FuncExceptT | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an AlphaBlur instance.

        The blur radius roughly corresponds to a Gaussian sigma as follows:
            - Radius 1.5 ≈ sigma 1.0
            - Radius 2.0 ≈ sigma 1.4
            - Radius 3.0 ≈ sigma 2.0
            - Radius 4.0 ≈ sigma 2.75

        Args:
            rx: Horizontal radius for halo removal.
            ry: Vertical radius for halo removal. Defaults to `rx` if not set.
            func: An optional function to use for error handling.
            **kwargs: Optional keyword arguments:

               - downscaler: Custom downscaler Scaler object.
               - upscaler: Custom upscaler Scaler object.
        """

        self.rx = rx
        self.ry = self.rx if ry is None else ry
        self.func = func or self
        self.downscaler = Scaler.ensure_obj(kwargs.get("downscaler", Mitchell()), self.func)
        self.upscaler = Scaler.ensure_obj(kwargs.get("upscaler", BSpline()), self.func)

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, **kwargs: Any) -> Any:
        """
        Applies the Gaussian blur approximation to the input clip.

        Args:
            clip: Source clip.
            planes: Which planes to process. Default to all.

        Raises:
            CustomIndexError: If any of the radius values (`rx` or `ry`) are less than 1.0.

        Returns:
            Blurred clip.
        """
        assert check_variable_format(clip, self.func)

        planes = normalize_planes(clip, planes)

        work_clip, *chroma = split(clip) if planes == [0] else (clip,)

        rxs = normalize_seq(self.rx, work_clip.format.num_planes)
        rys = normalize_seq(self.ry, work_clip.format.num_planes)

        if any(x < 1 for x in (*rxs, *rys)):
            raise CustomIndexError("rx, and ry must all be greater than 1.0!", self.func)

        if (len(set(rxs)) == len(set(rys)) == 1) or planes == [0] or work_clip.format.num_planes == 1:
            processed = self._function(clip, rxs[0], rys[0])

            if not chroma:
                return processed

            return join([processed, *chroma], clip.format.color_family)

        return join([self._function(*values) for values in zip(split(work_clip), rxs, rys)])

    def _function(
        self,
        clip: ConstantFormatVideoNode,
        rx: float,
        ry: float,
    ) -> vs.VideoNode:
        return self.upscaler.scale(
            self.downscaler.scale(
                clip,
                mod_x(clip.width / rx, 2**clip.format.subsampling_w),
                mod_x(clip.height / ry, 2**clip.format.subsampling_h),
            ),
            clip.width,
            clip.height,
        )


@deprecated("dehalo_alpha is deprecated, use dehalo_omega instead.", category=DeprecationWarning)
def dehalo_alpha(
    clip: vs.VideoNode,
    rx: IterArr[float] = 2.0,
    ry: IterArr[float] | None = None,
    darkstr: IterArr[float] = 0.0,
    brightstr: IterArr[float] = 1.0,
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    sigma_mask: None = None,
    ss: float | tuple[float, ...] = 1.5,
    planes: PlanesT = 0,
    show_mask: None = None,
    mask_radius: None = None,
    downscaler: ScalerLike = Mitchell,
    upscaler: ScalerLike = BSpline,
    supersampler: ScalerLike = Lanczos(3),
    supersampler_ref: ScalerLike = Mitchell,
    pre_ss: float = 1.0,
    pre_supersampler: ScalerLike = NNEDI3(noshift=(True, False)),
    pre_downscaler: ScalerLike = Point,
    mask_coords: None = None,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    dehalo_alpha is deprecated, use dehalo_omega instead.
    """
    func = func or dehalo_alpha

    if pre_ss > 1:
        return pre_supersampling(
            clip,
            lambda clip: dehalo_alpha(
                clip,
                rx,
                ry,
                darkstr,
                brightstr,
                lowsens,
                highsens,
                None,
                ss,
                planes,
                None,
                None,
                downscaler,
                upscaler,
                supersampler,
                supersampler_ref,
                func=func,
                **kwargs,
            ),
            pre_ss,
            pre_supersampler,
            pre_downscaler,
            func=func,
        )

    if ry is None:
        ry = rx

    return dehalo_omega(
        clip,
        tuple(
            AlphaBlur(rx_i, ry_i, func, downscaler=downscaler, upscaler=upscaler)
            for rx_i, ry_i in _normalize_iter_arr_t(rx, ry)
        ),
        lowsens,
        highsens,
        ss,
        darkstr,
        brightstr,
        planes,
        func=func,
        supersampler=supersampler,
        supersampler_ref=supersampler_ref,
        **kwargs,
    )


@deprecated("dehalo_sigma is deprecated, use dehalo_sigma instead.", category=DeprecationWarning)
def dehalo_sigma(
    clip: vs.VideoNode,
    brightstr: IterArr[float] = 1.0,
    darkstr: IterArr[float] = 0.0,
    lowsens: IterArr[float] = 50.0,
    highsens: IterArr[float] = 50.0,
    ss: float | tuple[float, ...] = 1.5,
    blur_func: Prefilter = Prefilter.GAUSS,
    planes: PlanesT = 0,
    supersampler: ScalerLike = Lanczos(3),
    supersampler_ref: ScalerLike = Mitchell,
    pre_ss: float = 1.0,
    pre_supersampler: ScalerLike = NNEDI3(noshift=(True, False)),
    pre_downscaler: ScalerLike = Point,
    mask_radius: None = None,
    sigma_mask: None = None,
    mask_coords: None = None,
    show_mask: None = None,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """dehalo_sigma is deprecated, use dehalo_sigma instead."""

    func = func or dehalo_sigma

    if pre_ss > 1:
        return pre_supersampling(
            clip,
            lambda clip: dehalo_sigma(
                clip,
                brightstr,
                darkstr,
                lowsens,
                highsens,
                ss,
                blur_func,
                planes,
                supersampler,
                supersampler_ref,
                func=func,
                **kwargs,
            ),
            pre_ss,
            pre_supersampler,
            pre_downscaler,
            func=func,
        )

    return dehalo_omega(
        clip,
        blur_func,
        lowsens,
        highsens,
        ss,
        darkstr,
        brightstr,
        planes,
        func=func,
        supersampler=supersampler,
        supersampler_ref=supersampler_ref,
        **kwargs,
    )


# HELPER FUNCTIONS BELOW #


def _is_callable(obj: Any) -> TypeGuard[VSFunctionPlanesArgs]:
    return callable(obj)


def _normalize_iter_arr_t(*values: IterArr[Any]) -> Iterator[tuple[list[Any], ...]]:
    max_len = max((len(x) if isinstance(x, tuple) else 1) for x in values)

    broadcasted: list[tuple[Any | list[Any], ...]] = [
        val + (val[-1],) * (max_len - len(val)) if isinstance(val, tuple) else (val,) * max_len for val in values
    ]

    normalized = list[list[list[Any]]]()

    for subgroup in broadcasted:
        sublist = list[list[Any]]()

        for item in subgroup:
            group = normalize_seq(item)

            sublist.append(group)

        normalized.append(sublist)

    return zip(*normalized)
