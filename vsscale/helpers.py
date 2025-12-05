from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from math import ceil, floor
from types import NoneType
from typing import Any, NamedTuple, Self, overload

from jetpytools import CustomTypeError, FuncExcept, mod2, mod_x

from vskernels import MixedScalerProcess, SampleGridModel, Scaler, ScalerLike, is_scaler_like
from vstools import FunctionUtil, Planes, Resolution, VSFunctionNoArgs, get_w, vs

from .various import ComplexSuperSamplerProcess

__all__ = ["CropAbs", "CropRel", "ScalingArgs", "pre_ss", "scale_var_clip"]


def scale_var_clip(
    clip: vs.VideoNode,
    scaler: Scaler | Callable[[Resolution], Scaler] | Callable[[tuple[int, int]], Scaler],
    width: int | Callable[[Resolution], int] | Callable[[tuple[int, int]], int] | None,
    height: int | Callable[[Resolution], int] | Callable[[tuple[int, int]], int],
    shift: tuple[float, float] | Callable[[tuple[int, int]], tuple[float, float]] = (0, 0),
    debug: bool = False,
) -> vs.VideoNode:
    """
    Scale a variable clip to constant or variable resolution.

    Args:
        clip: Source clip.
        scaler: A scaler instance or a callable that returns a scaler instance.
        width: A width integer or a callable that returns the width. If None, it will be calculated from the height and
            the aspect ratio of the clip.
        height: A height integer or a callable that returns the height.
        shift: Optional top shift, left shift tuple or a callable that returns the shifts. Defaults to no shift.
        debug: If True, the `var_width` and `var_height` props will be added to the clip.

    Returns:
        Scaled clip.
    """
    cached_clips = dict[str, vs.VideoNode]()

    no_accepts_var = list[Scaler]()

    def _eval_scale(f: vs.VideoFrame, n: int) -> vs.VideoNode:
        key = f"{f.width}_{f.height}"

        if key not in cached_clips:
            res = Resolution(f.width, f.height)

            norm_scaler = scaler(res) if callable(scaler) else scaler
            norm_shift = shift(res) if callable(shift) else shift
            norm_height = height(res) if callable(height) else height

            if width is None:
                norm_width = get_w(norm_height, res.width / res.height)
            else:
                norm_width = width(res) if callable(width) else width

            part_scaler = partial(norm_scaler.scale, width=norm_width, height=norm_height, shift=norm_shift)

            scaled = clip
            if (scaled.width, scaled.height) != (norm_width, norm_height):
                if norm_scaler not in no_accepts_var:
                    try:
                        scaled = part_scaler(clip)
                    except BaseException:
                        no_accepts_var.append(norm_scaler)

                if norm_scaler in no_accepts_var:
                    const_clip = clip.resize.Point(res.width, res.height)

                    scaled = part_scaler(const_clip)

            if debug:
                scaled = scaled.std.SetFrameProps(var_width=res.width, var_height=res.height)

            cached_clips[key] = scaled

        return cached_clips[key]

    out_clip = clip if callable(width) or callable(height) else clip.std.BlankClip(width, height)

    return out_clip.std.FrameEval(_eval_scale, clip, clip)


type LeftCrop = int
type RightCrop = int
type TopCrop = int
type BottomCrop = int


class CropRel(NamedTuple):
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


class CropAbs(NamedTuple):
    width: int
    height: int
    left: int = 0
    top: int = 0

    def to_rel(self, base_clip: vs.VideoNode) -> CropRel:
        return CropRel(
            self.left, base_clip.width - self.width - self.left, self.top, base_clip.height - self.height - self.top
        )


@dataclass
class ScalingArgs:
    width: int
    height: int
    src_width: float
    src_height: float
    src_top: float
    src_left: float
    mode: str = "hw"

    def _do(self) -> tuple[bool, bool]:
        return "h" in self.mode.lower(), "w" in self.mode.lower()

    def _up_rate(self, clip: vs.VideoNode | None = None) -> tuple[float, float]:
        if clip is None:
            return 1.0, 1.0

        do_h, do_w = self._do()

        return ((clip.height / self.height) if do_h else 1.0, (clip.width / self.width) if do_w else 1.0)

    def kwargs(self, clip_or_rate: vs.VideoNode | float | None = None, /) -> dict[str, Any]:
        kwargs = dict[str, Any]()

        do_h, do_w = self._do()

        if isinstance(clip_or_rate, (vs.VideoNode, NoneType)):
            up_rate_h, up_rate_w = self._up_rate(clip_or_rate)
        else:
            up_rate_h, up_rate_w = clip_or_rate, clip_or_rate

        if do_h:
            kwargs.update(src_height=self.src_height * up_rate_h, src_top=self.src_top * up_rate_h)

        if do_w:
            kwargs.update(src_width=self.src_width * up_rate_w, src_left=self.src_left * up_rate_w)

        return kwargs

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int,
        width: int | None = None,
        *,
        src_top: float = ...,
        src_left: float = ...,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        mode: str = "hw",
    ) -> Self: ...

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: float,
        width: float | None = ...,
        base_height: int | None = ...,
        base_width: int | None = ...,
        src_top: float = ...,
        src_left: float = ...,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs | None = ...,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        mode: str = "hw",
    ) -> Self: ...

    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int | float,  # noqa: PYI041
        width: int | float | None = None,  # noqa: PYI041
        base_height: int | None = None,
        base_width: int | None = None,
        src_top: float = 0,
        src_left: float = 0,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs | None = None,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        mode: str = "hw",
    ) -> Self:
        """
        Get (de)scaling arguments for integer scaling.

        Args:
            base_clip: Source clip.
            height: Target (de)scaling height. Casting to float will ensure fractional calculations.
            width: Target (de)scaling width. Casting to float will ensure fractional calculations. If None, it will be
                calculated from the height and the aspect ratio of the base_clip.
            base_height: The height from which to contain the clip. If None, it will be calculated from the height.
            base_width: The width from which to contain the clip. If None, it will be calculated from the width.
            src_top: Vertical offset.
            src_left: Horizontal offset.
            crop: Tuple of cropping values, or relative/absolute crop specification.
            mode: Scaling mode:

                   - "w" means only the width is calculated.
                   - "h" means only the height is calculated.
                   - "hw" or "wh" mean both width and height are calculated.

            sample_grid_model: Model used to align sampling grid.

        Returns:
            ScalingArgs object suitable for scaling functions.
        """
        if crop:
            if isinstance(crop, CropAbs):
                crop = crop.to_rel(base_clip)
            elif not isinstance(crop, CropRel):
                crop = CropRel(*crop)
        else:
            crop = CropRel()

        ratio_height = height / base_clip.height

        if width is None:
            # Integer scaling -> compute width from aspect ratio
            # Fractional scaling -> proportional scaling
            width = get_w(height, base_clip, 2) if isinstance(height, int) else ratio_height * base_clip.width

        ratio_width = width / base_clip.width

        if all(
            [
                isinstance(height, int),
                isinstance(width, int),
                base_height is None,
                base_width is None,
                crop == (0, 0, 0, 0),
                not sample_grid_model,
            ]
        ):
            return cls(int(width), int(height), int(width), int(height), src_top, src_left, mode)

        if base_height is None:
            base_height = mod2(ceil(height))

        if base_width is None:
            base_width = mod2(ceil(width))

        # half of (container - target), plus cropped portion
        margin_left = (base_width - width) / 2 + ratio_width * crop.left
        margin_right = (base_width - width) / 2 + ratio_width * crop.right
        # Cropped output = container minus floored margins
        cropped_width = base_width - floor(margin_left) - floor(margin_right)

        margin_top = (base_height - height) / 2 + ratio_height * crop.top
        margin_bottom = (base_height - height) / 2 + ratio_height * crop.bottom
        cropped_height = base_height - floor(margin_top) - floor(margin_bottom)

        # Compute src width/height after crop and scaling

        if isinstance(width, int) and crop.left == crop.right == 0:
            # Fully integer width + no crop = source width equals cropped container width
            cropped_src_width = float(cropped_width)
        else:
            # Otherwise scale from cropped source region
            cropped_src_width = ratio_width * (base_clip.width - crop.left - crop.right)

        # Horizontal source offset: fractional remainder + user offset
        cropped_src_left = margin_left - floor(margin_left) + src_left

        if isinstance(height, int) and crop.top == crop.bottom == 0:
            cropped_src_height = float(cropped_height)
        else:
            cropped_src_height = ratio_height * (base_clip.height - crop.top - crop.bottom)

        cropped_src_top = margin_top - floor(margin_top) + src_top

        if sample_grid_model:
            sgm = SampleGridModel.from_param(sample_grid_model, cls.from_args)

            # sample_grid_model() adjusts sampling grid and may modify src_top/left and src dimensions
            kw, (cropped_src_top, cropped_src_left) = sgm(
                cropped_src_width,
                cropped_src_height,
                base_clip.width - crop.left - crop.right,
                base_clip.height - crop.top - crop.bottom,
                (cropped_src_top, cropped_src_left),
                {},
            )
            cropped_src_width, cropped_src_height = kw["src_width"], kw["src_height"]

        return cls(
            cropped_width,
            cropped_height,
            cropped_src_width,
            cropped_src_height,
            cropped_src_top,
            cropped_src_left,
            mode,
        )


@overload
def pre_ss(
    clip: vs.VideoNode,
    function: VSFunctionNoArgs,
    rfactor: float = 2.0,
    sp: type[MixedScalerProcess[Any, Any]] = ComplexSuperSamplerProcess,
    *,
    mod: int = 4,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode: ...


@overload
def pre_ss(
    clip: vs.VideoNode,
    *,
    rfactor: float = 2.0,
    sp: MixedScalerProcess[Any, Any],
    mod: int = 4,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode: ...


@overload
def pre_ss(
    clip: vs.VideoNode,
    function: VSFunctionNoArgs,
    rfactor: float = 2.0,
    *,
    supersampler: ScalerLike | Callable[[vs.VideoNode, int, int], vs.VideoNode],
    downscaler: ScalerLike | Callable[[vs.VideoNode, int, int], vs.VideoNode],
    mod: int = 4,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode: ...


def pre_ss(
    clip: vs.VideoNode,
    function: VSFunctionNoArgs | None = None,
    rfactor: float = 2.0,
    sp: type[MixedScalerProcess[Any, Any]] | MixedScalerProcess[Any, Any] = ComplexSuperSamplerProcess,
    supersampler: ScalerLike | Callable[[vs.VideoNode, int, int], vs.VideoNode] | None = None,
    downscaler: ScalerLike | Callable[[vs.VideoNode, int, int], vs.VideoNode] | None = None,
    mod: int = 4,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Supersamples the input clip, applies a given function to the higher-resolution version, and then downscales it back to the original resolution.

    This function generalizes the behavior of
    [SuperSamplerProcess][vsaa.deinterlacers.SuperSamplerProcess] and
    [ComplexSuperSamplerProcess][vsscale.various.ComplexSuperSamplerProcess].

    - Examples:
        ```py
        out = pre_ss(clip, lambda clip: cool_function(clip, ...), planes=0)
        ```

        - Passing NNEDI3 as a supersampler:
        ```py
        from vsaa import SuperSamplerProcess

        out = pre_ss(clip, lambda clip: cool_function(clip, ...), SuperSamplerProcess)
        ```
        - This works too:
        ```py
        from vsaa import SuperSamplerProcess

        out = pre_ss(clip, sp=SuperSamplerProcess(function=lambda clip: cool_function(clip, ...)))
        ```

        - Specifying `supersampler` and `downscaler`:
        ```py
        from vskernels import Point

        out = pre_ss(clip, lambda clip: cool_function(clip, ...), supersampler=Point, downscaler=Point, planes=0)
        ```

    Args:
        clip: Source clip.
        function: A function to apply on the supersampled clip.
        rfactor: Scaling factor for supersampling. Defaults to 2.
        sp: A [MixedScalerProcess][vskernels.MixedScalerProcess] instance or class.
            Default is [ComplexSuperSamplerProcess[Lanczos]][vsscale.ComplexSuperSamplerProcess].
            It upscales with [Lanczos][vskernels.Lanczos] and downscales with [Point][vskernels.Point].
        supersampler: Scaler used to upscale the input clip if `sp` is not specified.
        downscaler: Downscaler used for undoing the upscaling done by the supersampler if `sp` is not specified.
        mod: Ensures the supersampled resolution is a multiple of this value. Defaults to 4.
        planes: Which planes to process.
        func: An optional function to use for error handling.

    Returns:
        A clip with the given function applied at higher resolution, then downscaled back.
    """  # noqa: E501
    func_util = FunctionUtil(clip, func or pre_ss, planes)

    args = (
        func_util.work_clip,
        mod_x(func_util.work_clip.width * rfactor, mod),
        mod_x(func_util.work_clip.height * rfactor, mod),
    )

    if isinstance(sp, MixedScalerProcess):
        return func_util.return_clip(sp.scale(*args))

    if supersampler and downscaler and function:
        ss = (
            Scaler.ensure_obj(supersampler, func_util.func).scale(*args)
            if is_scaler_like(supersampler)
            else supersampler(*args)
        )

        processed = function(ss)

        args = processed, clip.width, clip.height
        down = (
            Scaler.ensure_obj(downscaler, func_util.func).scale(*args)
            if is_scaler_like(downscaler)
            else downscaler(*args)
        )

        return func_util.return_clip(down)

    if function:
        return pre_ss(clip, sp=sp(function=function), rfactor=rfactor, mod=mod, planes=planes, func=func)

    raise CustomTypeError
