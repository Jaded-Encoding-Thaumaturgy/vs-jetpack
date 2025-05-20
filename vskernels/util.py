from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial, wraps
from math import exp
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Generic, TypeVar, Union, overload

from jetpytools import P
from typing_extensions import Self

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode, CustomRuntimeError, CustomValueError, HoldsVideoFormatT, Matrix, MatrixT, Transfer,
    VideoFormatT, cachedproperty, check_variable_format, depth, get_video_format, vs, vs_object
)

from .abstract import Resampler, ResamplerT, Scaler, ScalerT
from .abstract.base import BaseScalerMeta
from .kernels import Catrom, Point
from .types import Center, LeftShift, Slope, TopShift

__all__ = [
    'LinearLight',
    'NoScale',

    'resample_to'
]

_ScalerT = TypeVar("_ScalerT", bound=Scaler)

class NoScale(Scaler, Generic[_ScalerT], partial_abstract=True):
    """
    A utility scaler class that performs no scaling on the input clip.

    If used without a specified scaler, it defaults to inheriting from `Catrom`.
    """

    if not TYPE_CHECKING:
        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            if cls.__bases__ == (Scaler, Generic):
                return cls.__class_getitem__(Catrom)()

            return super().__new__(cls, *args, **kwargs)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        """
        Return the input clip unscaled, validating that the dimensions are consistent.

        :param clip:                The source clip.
        :param width:               Optional width to validate against the clip's width.
        :param height:              Optional height to validate against the clip's height.
        :param shift:               Subpixel shift (top, left).
        :param kwargs:              Additional arguments forwarded to the scale function.
        :raises CustomValueError:   If `width` or `height` differ from the clip's dimensions.
        """
        width, height = self._wh_norm(clip, width, height)

        if width != clip.width or height != clip.height:
            raise CustomValueError(
                "When using NoScale, `width` and `height` must match the clip's dimensions.",
                self.__class__, (width, height)
            )

        if shift == (0, 0) and not kwargs and not self.kwargs:
            return clip

        return super().scale(clip, width, height, shift, **kwargs)

    def __class_getitem__(cls, scaler: type[_ScalerT]) -> type[NoScale[_ScalerT]]:
        """
        Specialize NoScale with a given scaler kernel.

        Example:
        ```py
        NoScale[Bicubic]
        ```

        :param scaler:  A Scaler type used to specialize NoScale.
        :return:        A new subclass of NoScale using the provided kernel.
        """
        return BaseScalerMeta(  # type: ignore[return-value]
            cls.__name__, (scaler, cls), cls.__dict__.copy(), partial_abstract=True
        )

    @classmethod
    def from_scaler(cls, scaler: ScalerT) -> type[NoScale[Scaler]]:
        """
        Create a specialized NoScale class using a specific scaler.

        :param scaler:  A Scaler instance, type or string used as a base for specialization.
        :return:        A dynamically created NoScale subclass based on the given scaler.
        """
        return NoScale.__class_getitem__(Scaler.from_param(scaler))


@dataclass
class LinearLightProcessing(cachedproperty.baseclass):
    ll: LinearLight

    def get_linear(self) -> vs.VideoNode:
        """Getter for `linear` cached property."""
        wclip = self.ll._resampler.resample(
            self.ll._wclip,
            vs.RGBS if self.ll._wclip.format.color_family in (vs.YUV, vs.RGB) else vs.GRAYS,
            matrix_in=self.ll._matrix,
            transfer_in=self.ll._curve,
            transfer=Transfer.LINEAR
        )

        if self.ll.sigmoid:
            wclip = norm_expr(
                wclip,
                '{center} 1 {slope} / 1 x 0 max 1 min {scale} * {offset} + / 1 - log * -',
                center=self.ll._scenter, slope=self.ll._sslope,
                scale=self.ll._sscale, offset=self.ll._soffset,
                func=self.__class__
            )

        return wclip

    def set_linear(self, processed: vs.VideoNode) -> None:
        """Setter for `linear` cached property."""
        if self.ll._exited:
            raise CustomRuntimeError(
                'You can\'t set .linear after going out of the context manager!', func=self.__class__
            )
        self._linear = processed

    linear = cachedproperty[[Self], vs.VideoNode, Self, vs.VideoNode, ...](get_linear, set_linear)
    """Cached property to use for linear light processing."""

    @cachedproperty
    def out(self) -> vs.VideoNode:
        if not self.ll._exited:
            raise CustomRuntimeError(
                'You can\'t get .out while still inside of the context manager!', func=self.__class__
            )

        if not hasattr(self, '_linear'):
            raise CustomValueError('You need to set .linear before getting .out!', self.__class__)

        if self.ll.sigmoid:
            processed = norm_expr(
                self._linear,
                '1 1 {slope} {center} x 0 max 1 min - * exp + / {offset} - {scale} /',
                slope=self.ll._sslope, center=self.ll._scenter,
                offset=self.ll._soffset, scale=self.ll._sscale,
                func=self.__class__
            )
        else:
            processed = self._linear

        processed = vs.core.resize2.Point(processed, transfer_in=Transfer.LINEAR, transfer=self.ll._curve)

        return resample_to(processed, self.ll._fmt, self.ll._matrix, self.ll._resampler)


@dataclass
class LinearLight(AbstractContextManager[LinearLightProcessing], vs_object):
    """
    Utility class for processing a clip in linear format.

    Usage:
        ```py
        with LinearLight(clip, ...) as ll:
            ll.linear = function(ll.linear, ...)
        out = ll.out
        ```
    """

    clip: vs.VideoNode
    """Input clip."""

    sigmoid: bool | tuple[Slope, Center] = False
    """
    Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
    `True` applies the defaults values (6.5, 0.75).
    Keep in mind sigmoid slope has to be in range 1.0-20.0. (inclusive)
    and sigmoid center has to be in range 0.0-1.0 (inclusive).
    """

    resampler: ResamplerT | None = Catrom
    """Resampler for converting to linear format and converting back to input clip format."""

    out_fmt: int | VideoFormatT | HoldsVideoFormatT | None = None
    """Optional output format."""

    @overload
    @classmethod
    def from_func(
        cls,
        func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
        /,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerT | None = Catrom,
        out_fmt: int | VideoFormatT | HoldsVideoFormatT | None = None
    ) -> Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]:
        """
        Example:
            ``` py
            @LinearLight.from_func
            def decorated_function(clip: vs.VideoNode, ...) -> vs.VideoNode:
                ...
            ```
        """

    @overload
    @classmethod
    def from_func(
        cls,
        /,
        *,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerT | None = Catrom,
        out_fmt: int | VideoFormatT | HoldsVideoFormatT | None = None
    ) -> Callable[
        [Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]],
        Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]
    ]:
        """
        Example:
            ``` py
            @LinearLight.from_func(sigmoid=(6.5, 0.75))
            def decorated_function(clip: vs.VideoNode, ...) -> vs.VideoNode:
                ...
            ```
        """

    @classmethod
    def from_func(
        cls,
        func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode] | None = None,
        /,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerT | None = Catrom,
        out_fmt: int | VideoFormatT | HoldsVideoFormatT | None = None
    ) -> Union[
        Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
        Callable[
            [Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]],
            Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]
        ]
    ]:
        """Decorator version of LinearLight."""

        if func is None:
            return partial(cls.from_func, sigmoid=sigmoid, resampler=resampler, out_fmt=out_fmt)

        @wraps(func)
        def _wrapped(clip: vs.VideoNode, *args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
            with cls(clip, sigmoid, resampler, out_fmt) as ll:
                ll.linear = func(clip, *args, **kwargs)
            return ll.out

        return _wrapped

    def __enter__(self) -> LinearLightProcessing:
        assert check_variable_format(self.clip, self.__class__)

        if self.sigmoid is not False:
            if self.sigmoid is True:
                self.sigmoid = (6.5, 0.75)

            self._sslope, self._scenter = self.sigmoid

            if not 1.0 <= self._sslope <= 20.0:
                raise CustomValueError('sigmoid slope has to be in range 1.0-20.0 (inclusive).', self.__class__)

            if not 0.0 <= self._scenter <= 1.0:
                raise CustomValueError('sigmoid center has to be in range 0.0-1.0 (inclusive).', self.__class__)

            self._soffset = 1.0 / (1 + exp(self._sslope * self._scenter))
            self._sscale = 1.0 / (1 + exp(self._sslope * (self._scenter - 1))) - self._soffset

        self._fmt = self.out_fmt or self.clip.format

        self._wclip = self.clip
        self._curve = Transfer.from_video(self.clip)
        self._matrix = Matrix.from_video(self.clip)
        self._resampler = Resampler.ensure_obj(self.resampler or Catrom)

        self._exited = False

        return LinearLightProcessing(self)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._exited = True

    def __vs_del__(self, core_id: int) -> None:
        if not TYPE_CHECKING:
            self.clip = None
            self.out_fmt = None
            self._fmt = None
            self._wclip = None


def resample_to(
    clip: vs.VideoNode,
    out_fmt: int | VideoFormatT | HoldsVideoFormatT,
    matrix: MatrixT | None = None,
    resampler: ResamplerT = Catrom
) -> vs.VideoNode:
    out_fmt = get_video_format(out_fmt)
    assert clip.format

    resampler = Resampler.from_param(resampler)

    if out_fmt == clip.format:
        return clip

    if out_fmt.color_family is clip.format.color_family:
        return depth(clip, out_fmt)

    if out_fmt.subsampling_w == out_fmt.subsampling_h == 0:
        return Point().resample(clip, out_fmt, matrix)

    return resampler().resample(clip, out_fmt, matrix)
