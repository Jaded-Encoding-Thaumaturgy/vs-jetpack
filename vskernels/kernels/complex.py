from __future__ import annotations

from functools import partial
from math import ceil
from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar, Union

from jetpytools import CustomValueError, FuncExceptT

from vstools import (
    ConstantFormatVideoNode, Dar, FieldBased, KwargsT, Resolution, Sar, VideoNodeT, check_correct_subsampling, fallback, vs
)

from ..types import BorderHandling, Center, LeftShift, SampleGridModel, ShiftT, Slope, TopShift
from .abstract import BaseScaler, Descaler, Kernel, Resampler, Scaler
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


class _BaseLinear(BaseScaler):
    def _linearize(
        self,
        clip: vs.VideoNode,
        linear: bool | None,
        sigmoid: bool | tuple[Slope, Center],
        op_partial: partial[VideoNodeT],
        func: FuncExceptT,
        **kwargs: Any
    ) -> VideoNodeT:

        if linear is False and sigmoid is not False:
            raise CustomValueError(
                "If sigmoid is not False, linear can't be False as well!",
                func, (linear, sigmoid)
            )

        # if a _linear_scale or _linear_descale method is specified in the class,
        # use this method instead of the super().scale or super().descale method.
        # args and keywords are also forwarded.
        if hasattr(self, f'_linear_{op_partial.func.__name__}'):
            op_partial = partial(
                getattr(self, f'_linear_{op_partial.func.__name__}'),
                *op_partial.args,
                **op_partial.keywords
            )

        if sigmoid or linear:
            from ..util import LinearLight

            fmt = self.kwargs.pop("format", kwargs.pop('format', None))

            with LinearLight(
                clip, linear, sigmoid, self if isinstance(self, Resampler) else None, fmt
            ) as ll:
                ll.linear = op_partial(ll.linear, **kwargs)

            return ll.out  # type: ignore[return-value]

        return op_partial(clip, **kwargs)


class LinearScaler(_BaseLinear, Scaler):
    if TYPE_CHECKING:
        def _linear_scale(
            self, clip: vs.VideoNode, width: int | None, height: int | None, shift: tuple[TopShift, LeftShift], **kwargs: Any
        ) -> vs.VideoNode:
            ...

    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # LinearScaler adds `linear` and `sigmoid` parameters
        linear: bool | None = None, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        return self._linearize(
            clip, linear, sigmoid, partial(super().scale, width=width, height=height, shift=shift),
            self.scale, **kwargs
        )


class LinearDescaler(_BaseLinear, Descaler):
    if TYPE_CHECKING:
        def _linear_descale(
            self, clip: vs.VideoNode, width: int | None, height: int | None, shift: ShiftT, **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

    def descale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: ShiftT = (0, 0),
        *,
        # `border_handling`, `sample_grid_model` and `field_based` from Descaler
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        field_based: FieldBased | None = None,
        # LinearDescaler adds `linear` and `sigmoid` parameters
        linear: bool | None = None, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return self._linearize(
            clip, linear, sigmoid, partial(super().descale, width=width, height=height, shift=shift),
            self.descale, **kwargs
        )


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

    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # KeepArScaler adds `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar`
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None, keep_ar: bool | None = None,
        **kwargs: Any
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        width, height = self._wh_norm(clip, width, height)

        check_correct_subsampling(clip, width, height)

        const_size = 0 not in (clip.width, clip.height)

        if const_size:
            kwargs = self._get_kwargs_keep_ar(sar, dar, dar_in, keep_ar, **kwargs)

            kwargs, shift, out_sar = self._handle_crop_resize_kwargs(clip, width, height, shift, **kwargs)

            kwargs, shift = sample_grid_model.for_dst(clip, width, height, shift, **kwargs)

            if (bh := BorderHandling.from_param(border_handling, self.scale)) is not None:
                border_handling = bh
            else:
                raise TypeError

            padded = border_handling.prepare_clip(clip, self.kernel_radius)

            shift, clip = tuple(
                s + ((p - c) // 2) for s, c, p in zip(shift, *((x.height, x.width) for x in (clip, padded)))
            ), padded

        clip = super().scale(clip, width, height, shift, **kwargs)

        if const_size and out_sar:
            clip = out_sar.apply(clip)

        return clip


class ComplexScaler(KeepArScaler, LinearScaler):
    if TYPE_CHECKING:
        def scale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *,
            # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
            dar_in: Dar | bool | float | None = None, keep_ar: bool | None = None,
            # `linear` and `sigmoid` from LinearScaler
            linear: bool | None = False, sigmoid: bool | tuple[Slope, Center] = False,
            **kwargs: Any
        ) -> vs.VideoNode | ConstantFormatVideoNode:
            ...


class ComplexKernel(Kernel, LinearDescaler, ComplexScaler):
    ...


class CustomComplexKernel(CustomKernel, ComplexKernel):
    if TYPE_CHECKING:
        def descale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: ShiftT = (0, 0),
            *,
            # `border_handling`, `sample_grid_model` and `field_based` from Descaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            field_based: FieldBased | None = None,
            # `linear` and `sigmoid` parameters from LinearDescaler
            linear: bool | None = None, sigmoid: bool | tuple[Slope, Center] = False,
            # `blur` and `ignore_mask` from CustomKernel
            blur: float = 1.0, ignore_mask: vs.VideoNode | None = None,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...


class CustomComplexTapsKernel(CustomComplexKernel):
    def __init__(self, taps: float, **kwargs: Any) -> None:
        self.taps = taps
        super().__init__(**kwargs)

    @Scaler.cached_property
    def kernel_radius(self) -> int:
        return ceil(self.taps)

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**dict(taps=self.taps) | attrs)


ComplexScalerT = Union[str, type[ComplexScaler], ComplexScaler]
ComplexKernelT = Union[str, type[ComplexKernel], ComplexKernel]
