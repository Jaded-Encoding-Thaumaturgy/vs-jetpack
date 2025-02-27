from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar, Union, cast

from jetpytools import inject_kwargs_params
from vstools import (
    Dar, KwargsT, Resolution, Sar, VSFunctionAllArgs, check_correct_subsampling, fallback, inject_self, vs
)

from ..types import BorderHandling, Center, LeftShift, SampleGridModel, Slope, TopShift
from .abstract import Descaler, Kernel, Resampler, Scaler
from .custom import CustomKernel

__all__ = [
    'LinearScaler', 'LinearDescaler',

    'KeepArScaler',

    'ComplexScaler', 'ComplexScalerT',
    'ComplexKernel', 'ComplexKernelT',

    'CustomComplexKernel',
    'CustomComplexTapsKernel'
]

XarT = TypeVar('XarT', Sar, Dar)


def _from_param(cls: type[XarT], value: XarT | bool | float | None, fallback: XarT) -> XarT | None:
    if value is False:
        return fallback

    if value is True:
        return None

    if isinstance(value, cls):
        return value

    if isinstance(value, SupportsFloat):
        return cls.from_float(float(value))

    return None


class _BaseLinearOperation:
    @staticmethod
    def _linear_op(op_name: str) -> Any:
        @inject_kwargs_params
        def func(
            self: _BaseLinearOperation, clip: vs.VideoNode, width: int | None = None, height: int | None = None,

            shift: tuple[TopShift, LeftShift] = (0, 0), *,
            linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
        ) -> vs.VideoNode:
            from ..util import LinearLight

            has_custom_op = hasattr(self, f'_linear_{op_name}')
            operation = cast(
                VSFunctionAllArgs,
                getattr(self, f'_linear_{op_name}') if has_custom_op else getattr(super(), op_name)
            )

            if sigmoid:
                linear = True

            if not linear and not has_custom_op:
                return operation(clip, width, height, shift, **kwargs)

            resampler: Resampler | None = self if isinstance(self, Resampler) else None

            with LinearLight(clip, linear, sigmoid, resampler, kwargs.pop('format', None)) as ll:
                ll.linear = operation(ll.linear, width, height, shift, **kwargs)  # type: ignore

            return ll.out

        return func


class LinearScaler(_BaseLinearOperation, Scaler):
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def scale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *, linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
        ) -> vs.VideoNode:
            ...
    else:
        scale = inject_self.cached(_BaseLinearOperation._linear_op('scale'))


class LinearDescaler(_BaseLinearOperation, Descaler):
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def descale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *, linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
        ) -> vs.VideoNode:
            ...
    else:
        descale = inject_self.cached(_BaseLinearOperation._linear_op('descale'))


class KeepArScaler(Scaler):
    def _get_kwargs_keep_ar(
        self, sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
        dar_in: Dar | float | bool | None = None, keep_ar: bool | None = None, **kwargs: Any
    ) -> KwargsT:
        kwargs = KwargsT(keep_ar=keep_ar, sar=sar, dar=dar, dar_in=dar_in) | kwargs

        if keep_ar is not None:
            if None not in set(kwargs.get(x) for x in ('keep_ar', 'sar', 'dar', 'dar_in')):
                print(UserWarning(
                    f'{self.__class__.__name__}.scale: "keep_ar" set '
                    'with non-None values set in "sar", "dar" and "dar_in" won\'t do anything!'
                ))
        else:
            kwargs['keep_ar'] = False

        default_val = kwargs.pop('keep_ar')

        for key in ('sar', 'dar', 'dar_in'):
            if kwargs[key] is None:
                kwargs[key] = default_val

        return kwargs

    def _handle_crop_resize_kwargs(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[TopShift, LeftShift],
        sar: Sar | bool | float | None, dar: Dar | bool | float | None, dar_in: Dar | bool | float | None,
        **kwargs: Any
    ) -> tuple[KwargsT, tuple[TopShift, LeftShift], Sar | None]:
        kwargs.setdefault('src_top', kwargs.pop('sy', shift[0]))
        kwargs.setdefault('src_left', kwargs.pop('sx', shift[1]))
        kwargs.setdefault('src_width', kwargs.pop('sw', clip.width))
        kwargs.setdefault('src_height', kwargs.pop('sh', clip.height))

        src_res = Resolution(kwargs['src_width'], kwargs['src_height'])

        src_sar = float(_from_param(Sar, sar, Sar(1, 1)) or Sar.from_clip(clip))
        out_sar = None

        out_dar = float(_from_param(Dar, dar, Dar(0)) or Dar.from_res(width, height))
        src_dar = float(fallback(_from_param(Dar, dar_in, Dar(out_dar)), Dar.from_clip(clip, False)))

        if src_sar not in {0.0, 1.0}:
            if src_sar > 1.0:
                out_dar = (width / src_sar) / height
            else:
                out_dar = width / (height * src_sar)

            out_sar = Sar(1, 1)

        if src_dar != out_dar:
            if src_dar > out_dar:
                src_shift, src_window = 'src_left', 'src_width'

                fix_crop = src_res.width - (src_res.height * out_dar)
            else:
                src_shift, src_window = 'src_top', 'src_height'

                fix_crop = src_res.height - (src_res.width / out_dar)

            fix_shift = fix_crop / 2

            kwargs[src_shift] += fix_shift
            kwargs[src_window] -= fix_crop

        out_shift = (kwargs.pop('src_top'), kwargs.pop('src_left'))

        return kwargs, out_shift, out_sar

    @inject_self.cached
    @inject_kwargs_params
    def scale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0), *,
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None, keep_ar: bool | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        width, height = Scaler._wh_norm(clip, width, height)

        check_correct_subsampling(clip, width, height)

        const_size = 0 not in (clip.width, clip.height)

        if const_size:
            kwargs = self._get_kwargs_keep_ar(sar, dar, dar_in, keep_ar, **kwargs)

            kwargs, shift, out_sar = self._handle_crop_resize_kwargs(clip, width, height, shift, **kwargs)

            kwargs, shift = sample_grid_model.for_dst(clip, width, height, shift, **kwargs)

            border_handling = BorderHandling.from_param(border_handling, self.scale)
            padded = border_handling.prepare_clip(clip, self.kernel_radius)

            shift, clip = tuple(
                s + ((p - c) // 2) for s, c, p in zip(shift, *((x.height, x.width) for x in (clip, padded)))
            ), padded

        clip = Scaler.scale(self, clip, width, height, shift, **kwargs)

        if const_size and out_sar:
            clip = out_sar.apply(clip)

        return clip


class ComplexScaler(LinearScaler, KeepArScaler):
    @inject_self.cached
    @inject_kwargs_params
    def scale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        sar: Sar | bool | float | None = None, dar: Dar | bool | float | None = None, keep_ar: bool | None = None,
        linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False,
        **kwargs: Any
    ) -> vs.VideoNode:
        width, height = Scaler._wh_norm(clip, width, height)
        return super().scale(
            clip, width, height, shift, sar=sar, dar=dar, keep_ar=keep_ar,
            linear=linear, sigmoid=sigmoid, border_handling=border_handling,
            sample_grid_model=sample_grid_model, **kwargs
        )


class ComplexKernel(Kernel, LinearDescaler, ComplexScaler):  # type: ignore
    ...


class CustomComplexKernel(CustomKernel, ComplexKernel):  # type: ignore
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def descale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int, height: int, shift: tuple[TopShift, LeftShift] = (0, 0),
            *, blur: float = 1.0, border_handling: BorderHandling,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            ignore_mask: vs.VideoNode | None = None, linear: bool = False,
            sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
        ) -> vs.VideoNode:
            ...


class CustomComplexTapsKernel(CustomComplexKernel):
    def __init__(self, taps: float, **kwargs: Any) -> None:
        self.taps = taps
        super().__init__(**kwargs)

    @inject_self.cached.property
    def kernel_radius(self) -> int:  # type: ignore
        return ceil(self.taps)


ComplexScalerT = Union[str, type[ComplexScaler], ComplexScaler]
ComplexKernelT = Union[str, type[ComplexKernel], ComplexKernel]
