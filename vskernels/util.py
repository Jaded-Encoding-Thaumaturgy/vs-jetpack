from __future__ import annotations

from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from functools import partial, wraps
from math import exp
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Concatenate, Self, TypeAlias, get_origin, overload

from jetpytools import CustomRuntimeError, CustomValueError, cachedproperty, classproperty

from vsjetpack import TypeIs, TypeVar
from vstools import (
    HoldsVideoFormat,
    Matrix,
    MatrixLike,
    Transfer,
    VideoFormatLike,
    VSFunctionNoArgs,
    VSObject,
    VSObjectABC,
    depth,
    get_video_format,
    vs,
)

from .abstract import (
    Bobber,
    BobberLike,
    ComplexDescaler,
    ComplexDescalerLike,
    ComplexKernel,
    ComplexKernelLike,
    ComplexScaler,
    ComplexScalerLike,
    CustomComplexKernel,
    CustomComplexKernelLike,
    Descaler,
    DescalerLike,
    Kernel,
    KernelLike,
    Resampler,
    ResamplerLike,
    Scaler,
    ScalerLike,
)
from .abstract.base import BaseScaler, BaseScalerMeta, abstract_kernels, partial_abstract_kernels
from .kernels import Catrom, Point
from .types import Center, LeftShift, Slope, TopShift

__all__ = [
    "BaseMixedScaler",
    "BaseScalerSpecializer",
    "LinearLight",
    "MixedScalerProcess",
    "NoScale",
    "NoScaleLike",
    "ScalerSpecializer",
    "is_complex_descaler_like",
    "is_complex_kernel_like",
    "is_complex_scaler_like",
    "is_custom_complex_kernel_like",
    "is_descaler_like",
    "is_kernel_like",
    "is_noscale_like",
    "is_resampler_like",
    "is_scaler_like",
    "resample_to",
]


class BaseScalerSpecializerMeta(BaseScalerMeta):
    """
    Meta class for BaseScalerSpecializer to handle specialization logic.
    """

    __specializer__: type[BaseScaler]

    def __new__[MetaSelf: BaseScalerSpecializerMeta](  # noqa: PYI019
        mcls: type[MetaSelf],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        specializer: type[BaseScaler] | None = None,
        **kwargs: Any,
    ) -> MetaSelf:
        if specializer:
            bases = (*bases, specializer)
            namespace["__orig_bases__"] = (*namespace["__orig_bases__"], specializer)

            name += f"[{specializer.__name__}]"

            with suppress(KeyError):
                del namespace["kernel_radius"]

        obj = super().__new__(mcls, name, bases, namespace, **kwargs)

        if specializer:
            obj.__specializer__ = specializer

        return obj

    @property
    def __isspecialized__(self) -> bool:
        return hasattr(self, "__specializer__")


class BaseScalerSpecializer[DefaultScalerT: BaseScaler](BaseScaler, metaclass=BaseScalerSpecializerMeta, abstract=True):
    """
    An abstract base class to provide specialization logic for BaseScaler-like classes.
    """

    default_scaler: ClassVar[type[BaseScaler]]

    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            if not cls.__isspecialized__:
                cls = get_origin(cls[cls.default_scaler])

            return super().__new__(cls, *args, **kwargs)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        for k in kwargs.copy():
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

        self.kwargs = kwargs

    def __class_getitem__(cls, base_scaler: Any) -> GenericAlias:
        """
        Specialize this class with a given scaler kernel.

        Args:
            base_scaler: A BaseScaler type used to specialize this class.

        Returns:
            A new subclass using the provided kernel.
        """
        if isinstance(base_scaler, TypeVar):
            return GenericAlias(cls, (base_scaler,))

        specialized_scaler = BaseScalerSpecializerMeta(
            cls.__name__,
            (cls,),
            cls.__dict__.copy(),
            specializer=base_scaler,
            partial_abstract=base_scaler in partial_abstract_kernels,
            abstract=base_scaler in abstract_kernels,
        )

        return GenericAlias(specialized_scaler, (base_scaler,))

    @property
    def specializer(self) -> DefaultScalerT:
        """
        Returns the effective specializer.

        Returns:
            The effective specializer.
        """
        return self.__class__.__specializer__  # type: ignore[return-value]


class ScalerSpecializer[DefaultScalerT: Scaler](BaseScalerSpecializer[DefaultScalerT], Scaler, abstract=True):
    """
    An abstract base class to provide specialization logic for Scaler-like classes.
    """


_ScalerWithCatromDefaultT = TypeVar("_ScalerWithCatromDefaultT", bound=Scaler, default=Catrom)


class NoScale(ScalerSpecializer[_ScalerWithCatromDefaultT]):
    """
    A utility scaler class that performs no scaling on the input clip.

    If used without a specified scaler, it defaults to inheriting from `Catrom`.
    """

    default_scaler = Catrom

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Return the input clip unscaled, validating that the dimensions are consistent.

        Args:
            clip: The source clip.
            width: Optional width to validate against the clip's width.
            height: Optional height to validate against the clip's height.
            shift: Subpixel shift (top, left).
            **kwargs: Additional arguments forwarded to the scale function.

        Raises:
            CustomValueError: If `width` or `height` differ from the clip's dimensions.
        """
        width, height = self._wh_norm(clip, width, height)

        if width != clip.width or height != clip.height:
            raise CustomValueError(
                "When using NoScale, `width` and `height` must match the clip's dimensions.",
                self.__class__,
                (width, height),
            )

        if shift == (0, 0) and not kwargs and not self.kwargs:
            return clip

        return super().scale(clip, width, height, shift, **kwargs)

    @classmethod
    def from_scaler(cls, scaler: ScalerLike) -> type[NoScale[Scaler]]:
        """
        Create a specialized NoScale class using a specific scaler.

        Args:
            scaler: A Scaler instance, type or string used as a base for specialization.

        Returns:
            A dynamically created NoScale subclass based on the given scaler.
        """
        return NoScale[Scaler.from_param(scaler)]  # type: ignore[return-value,misc]


_ScalerWithScalerDefaultT = TypeVar("_ScalerWithScalerDefaultT", bound=Scaler, default=Scaler)

# TODO: type NoScaleLike[_ScalerT: Scaler = Scaler] = str | type[NoScale[_ScalerT]] | NoScale[_ScalerT]
NoScaleLike: TypeAlias = str | type[NoScale[_ScalerWithScalerDefaultT]] | NoScale[_ScalerWithScalerDefaultT]
"""
Type alias for anything that can resolve to a NoScale scaler.

This includes:

- A string identifier.
- A class type subclassing [NoScale][vskernels.NoScale].
- An instance of [NoScale][vskernels.NoScale].
"""


class BaseMixedScalerMeta[*_BaseScalerTs](BaseScalerSpecializerMeta):
    """
    Meta class for BaseMixedScaler to handle mixed scaling logic.
    """

    __others__: tuple[*_BaseScalerTs]

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *others: *_BaseScalerTs,
        specializer: type[BaseScaler] | None = None,
        **kwargs: Any,
    ) -> BaseMixedScalerMeta[*_BaseScalerTs]:
        obj = super().__new__(mcls, name, bases, namespace, specializer=specializer, **kwargs)

        if others:
            obj.__others__ = others
        else:
            for t in namespace["__orig_bases__"]:
                if (os := getattr(t, "__others__", ())) or (
                    isinstance(t, GenericAlias) and (os := getattr(t.__origin__, "__others__", ()))
                ):
                    obj.__others__ = os
                    break
            else:
                obj.__others__ = ()  # type: ignore[assignment]

        return obj


class BaseMixedScaler[DefaultScalerT: BaseScaler, *_BaseScalerTs](
    BaseScalerSpecializer[DefaultScalerT], metaclass=BaseMixedScalerMeta, abstract=True
):
    """
    An abstract base class to provide mixed or chained scaling for BaseScaler-like classes.
    """

    @classproperty
    @classmethod
    def _others(cls) -> tuple[*_BaseScalerTs]:
        # Workaround as we can't specify the bound values of a TypeVarTuple yet
        return tuple(o() for o in cls.__others__)  # type: ignore[operator]

    def __class_getitem__(cls, scalers: Any) -> GenericAlias:
        """
        Specialize this class with a given scaler kernel.

        Args:
            base_scaler: A BaseScaler type(s) used to specialize this class.

        Returns:
            A new subclass using the provided kernel.
        """
        if isinstance(scalers, tuple):
            specializer, *others = scalers
        else:
            specializer = scalers
            others = ()

        if isinstance(specializer, TypeVar):
            cls.__others__ = tuple(others) or cls.__others__
            return GenericAlias(cls, (specializer, *cls.__others__))

        mixed_spe = BaseMixedScalerMeta(
            cls.__name__,
            (cls,),
            cls.__dict__.copy(),
            *others,
            specializer=specializer,
            partial_abstract=specializer in partial_abstract_kernels,
            abstract=specializer in abstract_kernels,
        )

        return GenericAlias(mixed_spe, (specializer, *others))


class MixedScalerProcess[DefaultScalerT: Scaler, *_BaseScalerTs](
    BaseMixedScaler[DefaultScalerT, *_BaseScalerTs], Scaler, abstract=True
):
    """
    An abstract class for chained scaling with an additional processing step.
    """

    def __init__(self, *, function: VSFunctionNoArgs, **kwargs: Any) -> None:
        """
        Initialize the MixedScalerProcess.

        Args:
            function: A function to apply on the scaling pipeline.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(**kwargs)

        self.function = function


@dataclass
class LinearLightProcessing(VSObject):
    ll: LinearLight

    def get_linear(self) -> vs.VideoNode:
        """
        Getter for `linear` cached property.
        """
        if (
            self.ll._wclip.format.sample_type,
            self.ll._wclip.format.subsampling_w,
            self.ll._wclip.format.subsampling_h,
        ) != (vs.FLOAT, 0, 0):
            fmt = vs.RGBS if self.ll._wclip.format.color_family in (vs.YUV, vs.RGB) else vs.GRAYS
        else:
            fmt = self.ll._wclip

        wclip = self.ll._resampler.resample(
            self.ll._wclip, fmt, matrix_in=self.ll._matrix, transfer_in=self.ll._curve, transfer=Transfer.LINEAR
        )

        if self.ll.sigmoid:
            from vsexprtools import norm_expr

            wclip = norm_expr(
                wclip,
                "{center} 1 {slope} / 1 x 0 max 1 min {scale} * {offset} + / 1 - log * -",
                center=self.ll._scenter,
                slope=self.ll._sslope,
                scale=self.ll._sscale,
                offset=self.ll._soffset,
                func=self.__class__,
            )

        return wclip

    def set_linear(self, processed: vs.VideoNode) -> None:
        """
        Setter for `linear` cached property.
        """
        if self.ll._exited:
            raise CustomRuntimeError(
                "You can't set .linear after going out of the context manager!", func=self.__class__
            )
        self._linear = processed

    linear = cachedproperty[vs.VideoNode](get_linear, set_linear)
    """
    Cached property to use for linear light processing.
    """

    @cachedproperty
    def out(self) -> vs.VideoNode:
        if not self.ll._exited:
            raise CustomRuntimeError(
                "You can't get .out while still inside of the context manager!", func=self.__class__
            )

        if not hasattr(self, "_linear"):
            raise CustomValueError("You need to set .linear before getting .out!", self.__class__)

        if self.ll.sigmoid:
            from vsexprtools import norm_expr

            processed = norm_expr(
                self._linear,
                "1 1 {slope} {center} x 0 max 1 min - * exp + / {offset} - {scale} /",
                slope=self.ll._sslope,
                center=self.ll._scenter,
                offset=self.ll._soffset,
                scale=self.ll._sscale,
                func=self.__class__,
            )
        else:
            processed = self._linear

        processed = vs.core.resize2.Point(processed, transfer_in=Transfer.LINEAR, transfer=self.ll._curve)

        return resample_to(processed, self.ll._fmt, self.ll._matrix, self.ll._resampler)


@dataclass
class LinearLight(VSObjectABC, AbstractContextManager[LinearLightProcessing]):
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

    resampler: ResamplerLike = Catrom
    """Resampler for converting to linear format and converting back to input clip format."""

    out_fmt: int | VideoFormatLike | HoldsVideoFormat | None = None
    """Optional output format."""

    @overload
    @classmethod
    def from_func[**P](
        cls,
        func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
        /,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerLike = Catrom,
        out_fmt: int | VideoFormatLike | HoldsVideoFormat | None = None,
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
    def from_func[**P](
        cls,
        /,
        *,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerLike = Catrom,
        out_fmt: int | VideoFormatLike | HoldsVideoFormat | None = None,
    ) -> Callable[
        [Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]], Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]
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
    def from_func[**P](
        cls,
        func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode] | None = None,
        /,
        sigmoid: bool | tuple[Slope, Center] = False,
        resampler: ResamplerLike = Catrom,
        out_fmt: int | VideoFormatLike | HoldsVideoFormat | None = None,
    ) -> (
        Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]
        | Callable[
            [Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]], Callable[Concatenate[vs.VideoNode, P], vs.VideoNode]
        ]
    ):
        """
        Decorator version of LinearLight.
        """

        if func is None:
            return partial(cls.from_func, sigmoid=sigmoid, resampler=resampler, out_fmt=out_fmt)

        @wraps(func)
        def _wrapped(clip: vs.VideoNode, *args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
            with cls(clip, sigmoid, resampler, out_fmt) as ll:
                ll.linear = func(clip, *args, **kwargs)
            return ll.out

        return _wrapped

    def __enter__(self) -> LinearLightProcessing:
        if self.sigmoid is not False:
            if self.sigmoid is True:
                self.sigmoid = (6.5, 0.75)

            self._sslope, self._scenter = self.sigmoid

            if not 1.0 <= self._sslope <= 20.0:
                raise CustomValueError("sigmoid slope has to be in range 1.0-20.0 (inclusive).", self.__class__)

            if not 0.0 <= self._scenter <= 1.0:
                raise CustomValueError("sigmoid center has to be in range 0.0-1.0 (inclusive).", self.__class__)

            self._soffset = 1.0 / (1 + exp(self._sslope * self._scenter))
            self._sscale = 1.0 / (1 + exp(self._sslope * (self._scenter - 1))) - self._soffset

        self._fmt = self.out_fmt or self.clip.format

        self._wclip = self.clip
        self._curve = Transfer.from_video(self.clip)
        self._matrix = Matrix.from_video(self.clip)
        self._resampler = Resampler.ensure_obj(self.resampler)

        self._exited = False

        return LinearLightProcessing(self)

    def __exit__(self, *args: object, **kwargs: Any) -> None:
        self._exited = True


def resample_to(
    clip: vs.VideoNode,
    out_fmt: int | VideoFormatLike | HoldsVideoFormat,
    matrix: MatrixLike | None = None,
    resampler: ResamplerLike = Catrom,
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


def _is_base_scaler_like(obj: Any, base_scaler: type[BaseScaler]) -> bool:
    if isinstance(obj, base_scaler) or (isinstance(obj, BaseScalerMeta) and issubclass(obj, base_scaler)):
        return True

    if isinstance(obj, str):
        try:
            base_scaler.from_param(obj)
            return True
        except base_scaler._err_class:
            pass

    return False


def is_scaler_like(obj: Any) -> TypeIs[ScalerLike]:
    """Returns true if obj is a ScalerLike."""
    return _is_base_scaler_like(obj, Scaler)


def is_descaler_like(obj: Any) -> TypeIs[DescalerLike]:
    """Returns true if obj is a DescalerLike."""
    return _is_base_scaler_like(obj, Descaler)


def is_resampler_like(obj: Any) -> TypeIs[ResamplerLike]:
    """Returns true if obj is a ResamplerLike."""
    return _is_base_scaler_like(obj, Resampler)


def is_kernel_like(obj: Any) -> TypeIs[KernelLike]:
    """Returns true if obj is a KernelLike."""
    return _is_base_scaler_like(obj, Kernel)


def is_bobber_like(obj: Any) -> TypeIs[BobberLike]:
    """Returns true if obj is a BobberLike."""
    return _is_base_scaler_like(obj, Bobber)


def is_complex_scaler_like(obj: Any) -> TypeIs[ComplexScalerLike]:
    """Returns true if obj is a ComplexScalerLike."""
    return _is_base_scaler_like(obj, ComplexScaler)


def is_complex_descaler_like(obj: Any) -> TypeIs[ComplexDescalerLike]:
    """Returns true if obj is a ComplexDescalerLike."""
    return _is_base_scaler_like(obj, ComplexDescaler)


def is_complex_kernel_like(obj: Any) -> TypeIs[ComplexKernelLike]:
    """Returns true if obj is a ComplexKernelLike."""
    return _is_base_scaler_like(obj, ComplexKernel)


def is_custom_complex_kernel_like(obj: Any) -> TypeIs[CustomComplexKernelLike]:
    """Returns true if obj is a CustomComplexKernelLike."""
    return _is_base_scaler_like(obj, CustomComplexKernel)


def is_noscale_like[_ScalerT: Scaler](obj: Any, specializer: type[_ScalerT] = Scaler) -> TypeIs[NoScaleLike[_ScalerT]]:  # type: ignore[assignment]
    """
    Returns true if obj is a NoScaleLike.
    """
    if isinstance(obj, NoScale):
        return isinstance(obj.specializer, specializer)

    if isinstance(obj, str):
        try:
            NoScale.from_param(obj)
            return True
        except NoScale._err_class:
            return False

    if isinstance(obj, GenericAlias):
        obj = get_origin(obj)

    if isinstance(obj, BaseScalerSpecializerMeta) and issubclass(obj, NoScale):
        return (obj.__isspecialized__ and issubclass(obj.__specializer__, specializer)) or issubclass(
            obj.default_scaler, specializer
        )

    return False
