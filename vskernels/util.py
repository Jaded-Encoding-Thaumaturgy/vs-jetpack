from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING, Any, ClassVar, cast

from jetpytools import inject_kwargs_params
from typing_extensions import Self

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode, CustomRuntimeError, CustomValueError, HoldsVideoFormatT, InvalidTransferError, Matrix,
    MatrixT, Transfer, cachedproperty, depth, get_video_format, inject_self, vs
)

if TYPE_CHECKING:
    from .kernels.abstract import BaseScaler
else:
    BaseScaler = Any

from .kernels import Bicubic, Catrom, Kernel, KernelT, Point, Resampler, ResamplerT, Scaler
from .types import Center, LeftShift, Slope, TopShift

__all__ = [
    'NoShift', 'NoScale',

    'LinearLight',

    'resample_to'
]


class NoShiftBase(Kernel):
    def get_scale_args(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().get_scale_args(clip, (0, 0), *(args and args[1:]), **kwargs)

    def get_descale_args(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().get_descale_args(clip, (0, 0), *(args and args[1:]), **kwargs)


class NoShift(Bicubic, NoShiftBase):
    """
    Class util used to always pass shift=(0, 0)\n
    By default it inherits from :py:class:`vskernels.Bicubic`,
    this behaviour can be changed with :py:attr:`Noshift.from_kernel`\n

    Use case, for example vsaa's ZNedi3:
    ```
    test = ...  # some clip, 480x480
    doubled_no_shift = Znedi3(field=0, nsize=4, nns=3, shifter=NoShift).scale(test, 960, 960)
    down = Point.scale(double, 480, 480)
    ```
    """

    def __class_getitem__(cls, kernel: KernelT) -> type[Kernel]:
        return cls.from_kernel(kernel)

    @staticmethod
    def from_kernel(kernel: KernelT) -> type[Kernel]:
        """
        Function or decorator for making a kernel not shift.

        As example, in vsaa:
        ```
        doubled_no_shift = Znedi3(..., shifter=NoShift.from_kernel('lanczos')).scale(...)

        # which in *this case* can also be written as this
        doubled_no_shift = Znedi3(..., shifter=NoShift, scaler=Lanczos).scale(...)
        ```

        Or for some other code:
        ```
        @NoShift.from_kernel
        class CustomCatromWithoutShift(Catrom):
            # some cool code
            ...
        ```
        """

        kernel_t = Kernel.from_param(kernel)

        class inner_no_shift(NoShiftBase, kernel_t):  # type: ignore
            ...

        return inner_no_shift


class NoScaleBase(Scaler):
    @inject_self.cached
    @inject_kwargs_params
    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> vs.VideoNode:
        try:
            width, height = Scaler._wh_norm(clip, width, height)
            return super().scale(clip, clip.width, clip.height, shift, **kwargs)
        except Exception:
            return clip


class NoScale(NoScaleBase, Bicubic):
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def scale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            **kwargs: Any
        ) -> vs.VideoNode:
            ...

    def __class_getitem__(cls, kernel: KernelT) -> type[Kernel]:
        return cls.from_kernel(kernel)

    @staticmethod
    def from_kernel(kernel: KernelT) -> type[Kernel]:
        kernel_t = Kernel.from_param(kernel)

        class inner_no_scale(kernel_t, NoScaleBase):  # type: ignore
            ...

        return inner_no_scale


@dataclass
class LinearLight:
    clip: vs.VideoNode

    linear: bool = True
    sigmoid: bool | tuple[Slope, Center] = False

    resampler: ResamplerT | None = Catrom

    out_fmt: vs.VideoFormat | None = None

    _linear: ClassVar[vs.VideoNode]

    @dataclass
    class LinearLightProcessing(cachedproperty.baseclass):
        ll: LinearLight

        def get_linear(self) -> vs.VideoNode:
            wclip: vs.VideoNode = self.ll._wclip

            if self.ll._wclip.format.color_family is vs.YUV:
                wclip = self.ll._resampler.resample(wclip, vs.RGBS, None, self.ll._matrix)
            else:
                wclip = depth(wclip, 32)

            if self.ll.linear:
                wclip = Point.scale_function(wclip, transfer_in=self.ll._curve, transfer=Transfer.LINEAR)

            if self.ll.sigmoid:
                if Transfer.from_video(wclip, func=self.__class__) in (Transfer.ST2084, Transfer.STD_B67):
                    raise InvalidTransferError(
                        self.__class__, Transfer.from_video(wclip, func=self.__class__),
                        'Sigmoid scaling is not supported with HDR!'
                    )

                wclip = norm_expr(
                    wclip,
                    '{center} 1 {slope} / 1 x 0 max 1 min {scale} * {offset} + / 1 - log * -',
                    center=self.ll._scenter, slope=self.ll._sslope,
                    scale=self.ll._sscale, offset=self.ll._soffset,
                    func=self.__class__
                )

            return wclip

        def set_linear(self, processed: vs.VideoNode) -> None:
            if self.ll._exited:
                raise CustomRuntimeError(
                    'You can\'t set .linear after going out of the context manager!', func=self.__class__
                )
            self._linear = processed

        linear = cachedproperty[[Self], vs.VideoNode, Self, vs.VideoNode, ...](get_linear, set_linear)

        @cachedproperty
        def out(self) -> vs.VideoNode:
            if not self.ll._exited:
                raise CustomRuntimeError(
                    'You can\'t get .out while still inside of the context manager!', func=self.__class__
                )

            if not hasattr(self, '_linear'):
                raise CustomValueError('You need to set .linear before getting .out!', self.__class__)

            processed = self._linear

            if self.ll.sigmoid:
                processed = norm_expr(
                    processed,
                    '1 1 {slope} {center} x 0 max 1 min - * exp + / {offset} - {scale} /',
                    slope=self.ll._sslope, center=self.ll._scenter,
                    offset=self.ll._soffset, scale=self.ll._sscale,
                    func=self.__class__
                )

            if self.ll.linear:
                processed = Point.scale_function(processed, transfer_in=Transfer.LINEAR, transfer=self.ll._curve)

            return resample_to(processed, self.ll._fmt, self.ll._matrix, self.ll._resampler)

    def __enter__(self) -> LinearLightProcessing:
        self.linear = self.linear or not not self.sigmoid

        if self.sigmoid is not False:
            if self.sigmoid is True:
                self.sigmoid = (6.5, 0.75)

            self._sslope, self._scenter = self.sigmoid

            if 1.0 > self._sslope or self._sslope > 20.0:
                raise CustomValueError('sigmoid slope has to be in range 1.0-20.0 (inclusive).', self.__class__)

            if 0.0 > self._scenter or self._scenter > 1.0:
                raise CustomValueError('sigmoid center has to be in range 0.0-1.0 (inclusive).', self.__class__)

            self._soffset = 1.0 / (1 + exp(self._sslope * self._scenter))
            self._sscale = 1.0 / (1 + exp(self._sslope * (self._scenter - 1))) - self._soffset

        _fmt = self.out_fmt or self.clip.format
        assert _fmt
        self._fmt = _fmt

        self._wclip = cast(ConstantFormatVideoNode, depth(self.clip, 32) if self.sigmoid else self.clip)
        self._curve = Transfer.from_video(self.clip)
        self._matrix = Matrix.from_video(self.clip)
        self._resampler = Catrom.ensure_obj(self.resampler)

        self._exited = False

        return self.LinearLightProcessing(self)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._exited = True


def resample_to(
    clip: vs.VideoNode, out_fmt: HoldsVideoFormatT, matrix: MatrixT | None = None, resampler: ResamplerT = Catrom
) -> vs.VideoNode:
    out_fmt = get_video_format(out_fmt)
    assert clip.format

    resampler = Resampler.from_param(resampler)

    if out_fmt == clip.format:
        return clip

    if out_fmt.color_family is clip.format.color_family:
        return depth(clip, out_fmt)

    if out_fmt.subsampling_w == out_fmt.subsampling_h == 0:
        return Point.resample(clip, out_fmt, matrix)

    return resampler.resample(clip, out_fmt, matrix)
