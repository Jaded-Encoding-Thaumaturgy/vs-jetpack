from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import ceil, floor
from types import NoneType
from typing import Any, Callable, NamedTuple, Protocol, Self, TypeAlias, overload

from vskernels import Catrom, Kernel, KernelT, Scaler, ScalerT
from vstools import KwargsT, MatrixT, Resolution, fallback, get_w, mod2, plane, vs


__all__ = [
    'GenericScaler',
    'scale_var_clip',
    'fdescale_args',
    'descale_args',

    'CropRel',
    'CropAbs',
    'ScalingArgs'
]

__abstract__ = [
    'GenericScaler'
]


class _GeneriScaleNoShift(Protocol):
    def __call__(self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwds: Any) -> vs.VideoNode:
        ...


class _GeneriScaleWithShift(Protocol):
    def __call__(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        *args: Any, **kwds: Any
    ) -> vs.VideoNode:
        ...


@dataclass
class GenericScaler(Scaler):
    """
    Generic Scaler base class.
    Inherit from this to create more complex scalers with built-in utils.
    Instantiate with a callable taking at least a VideoNode, width, and height
    to use that as a Scaler in functions taking that.
    """

    kernel: KernelT | None = field(default=None, kw_only=True)
    """
    Base kernel to be used for certain scaling/shifting/resampling operations.
    Must be specified and defaults to catrom
    """

    scaler: ScalerT | None = field(default=None, kw_only=True)
    """Scaler used for scaling operations. Defaults to kernel."""

    shifter: KernelT | None = field(default=None, kw_only=True)
    """Kernel used for shifting operations. Defaults to kernel."""

    def __post_init__(self) -> None:
        self._kernel = Kernel.ensure_obj(self.kernel or Catrom, self.__class__)
        self._scaler = Scaler.ensure_obj(self.scaler or self._kernel, self.__class__)
        self._shifter = Kernel.ensure_obj(
            self.shifter or (self._scaler if isinstance(self._scaler, Kernel) else Catrom), self.__class__
        )

    def __init__(
        self, func: _GeneriScaleNoShift | _GeneriScaleWithShift | Callable[..., vs.VideoNode], **kwargs: Any
    ) -> None:
        self.func = func
        self.kwargs = kwargs

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        width, height = self._wh_norm(clip, width, height)

        kwargs = self.kwargs | kwargs

        output = None

        if shift != (0, 0):
            try:
                output = self.func(clip, width, height, shift, **kwargs)
            except BaseException:
                try:
                    output = self.func(clip, width=width, height=height, shift=shift, **kwargs)
                except BaseException:
                    pass

        if output is None:
            try:
                output = self.func(clip, width, height, **kwargs)
            except BaseException:
                output = self.func(clip, width=width, height=height, **kwargs)

        return self._finish_scale(output, clip, width, height, shift)

    def _finish_scale(
        self, clip: vs.VideoNode, input_clip: vs.VideoNode, width: int, height: int,
        shift: tuple[float, float] = (0, 0), matrix: MatrixT | None = None,
        copy_props: bool = False
    ) -> vs.VideoNode:
        assert input_clip.format

        if input_clip.format.num_planes == 1:
            clip = plane(clip, 0)

        if (clip.width, clip.height) != (width, height):
            clip = self._scaler.scale(clip, width, height)

        if shift != (0, 0):
            clip = self._shifter.shift(clip, shift)  # type: ignore

        assert clip.format

        if clip.format.id != input_clip.format.id:
            clip = self._kernel.resample(clip, input_clip, matrix)

        if copy_props:
            return clip.std.CopyFrameProps(input_clip)

        return clip

    def ensure_scaler(self, scaler: ScalerT) -> Scaler:
        from dataclasses import is_dataclass, replace

        scaler_obj = Scaler.ensure_obj(scaler, self.__class__)

        if is_dataclass(scaler_obj):
            from inspect import Signature  #type: ignore[unreachable]

            kwargs = dict[str, ScalerT]()

            init_keys = Signature.from_callable(scaler_obj.__init__).parameters.keys()

            if 'kernel' in init_keys:
                kwargs.update(kernel=self.kernel or scaler_obj.kernel)

            if 'scaler' in init_keys:
                kwargs.update(scaler=self.scaler or scaler_obj.scaler)

            if 'shifter' in init_keys:
                kwargs.update(shifter=self.shifter or scaler_obj.shifter)

            if kwargs:
                scaler_obj = replace(scaler_obj, **kwargs)

        return scaler_obj


def scale_var_clip(
    clip: vs.VideoNode, scaler: Scaler | Callable[[Resolution], Scaler],
    width: int | Callable[[Resolution], int] | None, height: int | Callable[[Resolution], int],
    shift: tuple[float, float] | Callable[[Resolution], tuple[float, float]] = (0, 0), debug: bool = False
) -> vs.VideoNode:
    """Scale a variable clip to constant or variable resolution."""
    if not debug:
        try:
            return scaler.scale(clip, width, height, shift)  # type: ignore
        except BaseException:
            pass

    _cached_clips = dict[str, vs.VideoNode]()

    no_accepts_var = list[Scaler]()

    def _eval_scale(f: vs.VideoFrame, n: int) -> vs.VideoNode:
        key = f'{f.width}_{f.height}'

        if key not in _cached_clips:
            res = Resolution(f.width, f.height)

            norm_scaler = scaler(res) if callable(scaler) else scaler
            norm_shift = shift(res) if callable(shift) else shift
            norm_height = height(res) if callable(height) else height

            if width is None:
                norm_width = get_w(norm_height, res.width / res.height)
            else:
                norm_width = width(res) if callable(width) else width

            part_scaler = partial(  #type: ignore[misc]
                norm_scaler.scale, width=norm_width, height=norm_height, shift=norm_shift
            )

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

            _cached_clips[key] = scaled

        return _cached_clips[key]

    if callable(width) or callable(height):
        out_clip = clip
    else:
        out_clip = clip.std.BlankClip(width, height)

    return out_clip.std.FrameEval(_eval_scale, clip, clip)


LeftCrop: TypeAlias = int
RightCrop: TypeAlias = int
TopCrop: TypeAlias = int
BottomCrop: TypeAlias = int


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
            self.left,
            base_clip.width - self.width - self.left,
            self.top,
            base_clip.height - self.height - self.top
        )


@dataclass
class ScalingArgs:
    width: int
    height: int
    src_width: float
    src_height: float
    src_top: float
    src_left: float
    mode: str = 'hw'

    def _do(self) -> tuple[bool, bool]:
        return 'h' in self.mode.lower(), 'w' in self.mode.lower()

    def _up_rate(self, clip: vs.VideoNode | None = None) -> tuple[float, float]:
        if clip is None:
            return 1.0, 1.0

        do_h, do_w = self._do()

        return (
            (clip.height / self.height) if do_h else 1.0,
            (clip.width / self.width) if do_w else 1.0
        )

    def kwargs(self, clip_or_rate: vs.VideoNode | float | None = None, /) -> KwargsT:
        kwargs = dict[str, Any]()

        do_h, do_w = self._do()

        if isinstance(clip_or_rate, (vs.VideoNode, NoneType)):
            up_rate_h, up_rate_w = self._up_rate(clip_or_rate)
        else:
            up_rate_h, up_rate_w = clip_or_rate, clip_or_rate

        if do_h:
            kwargs.update(
                src_height=self.src_height * up_rate_h,
                src_top=self.src_top * up_rate_h
            )

        if do_w:
            kwargs.update(
                src_width=self.src_width * up_rate_w,
                src_left=self.src_left * up_rate_w
            )

        return kwargs

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int, width: int | None = None,
        /,
        *,
        src_top: float = ..., src_left: float = ...,
        mode: str = 'hw'
    ) -> Self:
        ...

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: float, width: float | None = ...,
        /,
        base_height: int | None = ..., base_width: int | None = ...,
        src_top: float = ..., src_left: float = ...,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs = ...,
        mode: str = 'hw'
    ) -> Self:
        ...

    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int | float, width: int | float | None = None,
        base_height: int | None = None, base_width: int | None = None,
        src_top: float = 0, src_left: float = 0,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs | None = None,
        mode: str = 'hw'
    ) -> Self:
        if crop:
            if isinstance(crop, CropAbs):
                crop = crop.to_rel(base_clip)
            elif isinstance(crop, CropRel):
                pass
            else:
                crop = CropRel(*crop)
        else:
            crop = CropRel()

        ratio = height / base_clip.height

        if width is None:
            if isinstance(height, int):
                width = get_w(height, base_clip, 2)
            else:
                width = ratio * base_clip.width

        if all([
            isinstance(height, int),
            isinstance(width, int),
            base_height is None,
            base_width is None,
            crop == (0, 0, 0, 0)
        ]):
            return cls(int(width), int(height), int(width), int(height), src_top, src_left, mode)

        if base_height is None:
            base_height = mod2(ceil(height))

        if base_width is None:
            base_width = mod2(ceil(width))

        margin_left = (base_width - width) / 2 + ratio * crop.left
        margin_right = (base_width - width) / 2 + ratio * crop.right
        cropped_width = base_width - floor(margin_left) - floor(margin_right)

        margin_top = (base_height - height) / 2 + ratio * crop.top
        margin_bottom = (base_height - height) / 2 + ratio * crop.bottom
        cropped_height = base_height - floor(margin_top) - floor(margin_bottom)

        if isinstance(width, int) and crop.left == crop.right == 0:
            cropped_src_width = float(cropped_width)
        else:
            cropped_src_width = ratio * (base_clip.width - crop.left - crop.right)

        cropped_src_left = margin_left - floor(margin_left) + src_left

        if isinstance(height, int) and crop.top == crop.bottom == 0:
            cropped_src_height = float(cropped_height)
        else:
            cropped_src_height = ratio * (base_clip.height - crop.top - crop.bottom)

        cropped_src_top = margin_top - floor(margin_top) + src_top

        return cls(
            cropped_width, cropped_height,
            cropped_src_width, cropped_src_height,
            cropped_src_top, cropped_src_left,
            mode
        )


def descale_args(
    clip: vs.VideoNode,
    src_height: float, src_width: float | None = None,
    base_height: int | None = None, base_width: int | None = None,
    crop_top: int = 0, crop_bottom: int = 0,
    crop_left: int = 0, crop_right: int = 0,
    mode: str = 'hw'
) -> ScalingArgs:
    # warnings
    return ScalingArgs.from_args(
        clip.std.AddBorders(crop_left, crop_right, crop_top, crop_bottom),
        src_height, src_width,
        base_height, base_width,
        0, 0,
        CropRel(crop_left, crop_right, crop_top, crop_bottom),
        mode
    )


def fdescale_args(
    clip: vs.VideoNode, src_height: float,
    base_height: int | None = None, base_width: int | None = None,
    src_top: float | None = None, src_left: float | None = None,
    src_width: float | None = None, mode: str = 'hw', up_rate: float = 2.0
) -> tuple[KwargsT, KwargsT]:
    base_height = fallback(base_height, mod2(ceil(src_height)))
    base_width = fallback(base_width, get_w(base_height, clip, 2))

    src_width = fallback(src_width, src_height * clip.width / clip.height)

    cropped_width = base_width - 2 * floor((base_width - src_width) / 2)
    cropped_height = base_height - 2 * floor((base_height - src_height) / 2)

    do_h, do_w = 'h' in mode.lower(), 'w' in mode.lower()

    de_args = dict[str, Any](
        width=cropped_width if do_w else clip.width,
        height=cropped_height if do_h else clip.height
    )

    up_args = dict[str, Any]()

    src_top = fallback(src_top, (cropped_height - src_height) / 2)
    src_left = fallback(src_left, (cropped_width - src_width) / 2)

    if do_h:
        de_args.update(src_height=src_height, src_top=src_top)
        up_args.update(src_height=src_height * up_rate, src_top=src_top * up_rate)

    if do_w:
        de_args.update(src_width=src_width, src_left=src_left)
        up_args.update(src_width=src_width * up_rate, src_left=src_left * up_rate)

    return de_args, up_args
