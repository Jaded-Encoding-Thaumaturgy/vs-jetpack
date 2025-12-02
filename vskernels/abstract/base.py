"""
This module defines the base abstract interfaces for general scaling operations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, wraps
from inspect import isabstract
from math import ceil
from types import NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    NoReturn,
    Self,
    SupportsInt,
    get_origin,
    overload,
)

from jetpytools import (
    CustomNotImplementedError,
    CustomRuntimeError,
    CustomValueError,
    FuncExcept,
    classproperty,
    fallback,
    get_subclasses,
    normalize_seq,
)
from jetpytools import cachedproperty as jetpytools_cachedproperty

from vsjetpack import TypeIs
from vstools import (
    ChromaLocation,
    ColorRange,
    FieldBased,
    FieldBasedLike,
    HoldsVideoFormat,
    Matrix,
    MatrixLike,
    Primaries,
    PrimariesLike,
    PropEnum,
    Transfer,
    TransferLike,
    VideoFormatLike,
    VSObjectABC,
    VSObjectABCMeta,
    check_variable_resolution,
    core,
    get_video_format,
    split,
    vs,
)
from vstools.enums.color import _norm_props_enums

from ..exceptions import (
    UnknownDescalerError,
    UnknownKernelError,
    UnknownResamplerError,
    UnknownScalerError,
    _UnknownBaseScalerError,
)
from ..types import LeftShift, TopShift

__all__ = [
    "BaseScaler",
    "Bobber",
    "BobberLike",
    "Descaler",
    "DescalerLike",
    "Kernel",
    "KernelLike",
    "Resampler",
    "ResamplerLike",
    "Scaler",
    "ScalerLike",
]


def _add_init_kwargs[BaseScalerT: BaseScaler, **P, R](
    method: Callable[Concatenate[BaseScalerT, P], R],
) -> Callable[Concatenate[BaseScalerT, P], R]:
    @wraps(method)
    def _wrapped(self: BaseScalerT, *args: P.args, **kwargs: P.kwargs) -> R:
        init_kwargs = {k: self.kwargs.pop(k) for k in self.kwargs.keys() & method.__annotations__.keys()}

        returned = method(self, *args, **init_kwargs | kwargs)

        self.kwargs |= init_kwargs

        return returned

    setattr(_wrapped, "__has_init_kwargs__", True)

    return _wrapped


def _base_from_param[BaseScalerT: BaseScaler](
    cls: type[BaseScalerT],
    value: str | type[BaseScalerT] | BaseScalerT | None,
    exception_cls: type[_UnknownBaseScalerError],
    func_except: FuncExcept | None,
) -> type[BaseScalerT]:
    # If value is an instance returns the class
    if isinstance(value, cls):
        return value.__class__

    # If value is a type and a subclass of the caller returns the value itself
    if isinstance(value, BaseScalerMeta) and issubclass(value, cls):
        return value

    # The value could be a GenericAlias
    if origin := get_origin(value):
        return _base_from_param(cls, origin, exception_cls, func_except)

    # Search for the subclasses of the caller and the caller itself
    if isinstance(value, str):
        all_scalers = {s.__name__.lower(): s for s in [*get_subclasses(cls), cls]}

        try:
            return all_scalers[value.lower().strip()]
        except KeyError:
            raise exception_cls(func_except or cls.from_param, value)

    if value is None:
        return cls

    raise exception_cls(func_except or cls.from_param, str(value))


def _base_ensure_obj[BaseScalerT: BaseScaler](
    cls: type[BaseScalerT], value: str | type[BaseScalerT] | BaseScalerT | None, func_except: FuncExcept | None
) -> BaseScalerT:
    if isinstance(value, cls):
        return value

    return cls.from_param(value, func_except)()


@cache
def _is_base_scaler_abstract(cls: type[BaseScaler]) -> bool:
    return cls in abstract_kernels or not hasattr(cls, "kernel_radius") or isabstract(cls)


def _is_format_resolver(
    value: Any,
) -> TypeIs[Callable[[vs.VideoNode], SupportsInt | VideoFormatLike | HoldsVideoFormat]]:
    return callable(value)


def _resolve_video_spec_args(clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
    if _is_format_resolver(fmt := kwargs.get("format")):
        kwargs["format"] = get_video_format(fmt(clip))

    spec_map: dict[str, type[PropEnum]] = {
        "matrix": Matrix,
        "matrix_in": Matrix,
        "transfer": Transfer,
        "transfer_in": Transfer,
        "primaries": Primaries,
        "primaries_in": Primaries,
        "range": ColorRange,
        "range_in": ColorRange,
        "chromaloc": ChromaLocation,
        "chromaloc_in": ChromaLocation,
    }

    for name, prop_enum in spec_map.items():
        if callable(resolver := kwargs.get(name)):
            kwargs[name] = prop_enum.from_param_with_fallback(resolver(clip))

    return kwargs


abstract_kernels: list[BaseScalerMeta] = []
"""
List of fully abstract kernel classes.

Used internally to track kernel base classes that should not be used directly.
"""

partial_abstract_kernels: list[BaseScalerMeta] = []
"""
List of partially abstract kernel classes.

These may implement some but not all kernel functionality.
"""


class BaseScalerMeta(VSObjectABCMeta):
    """
    Metaclass for scaler classes.

    This metaclass can be used to enforce abstraction rules by specifying
    `abstract` or `partial_abstract` as keyword arguments in the class definition.

    - If ``abstract=True``: The class is marked as fully abstract and added to
      the ``abstract_kernels`` registry. It should not be instantiated.
    - If ``partial_abstract=True``: The class is considered partially abstract,
      meaning it may lack certain implementations (e.g., kernel radius) but is
      still allowed to be instantiated. It is added to ``partial_abstract_kernels``.
    """

    class cachedproperty[R](jetpytools_cachedproperty[R]):  # noqa: N801
        """
        Read only version of jetpytools.cachedproperty.
        """

        if TYPE_CHECKING:

            def __init__[BaseScalerT: BaseScaler, **P](
                self, func: Callable[Concatenate[BaseScalerT, P], R]
            ) -> None: ...

        def __set__(self, instance: None, value: Any) -> NoReturn:
            """
            Raise an error when attempting to set a cached property.
            """
            raise AttributeError("Can't set attribute")

    cached_property = cachedproperty

    def __new__[MetaSelf: BaseScalerMeta](  # noqa: PYI019
        mcls: type[MetaSelf],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        abstract: bool = False,
        partial_abstract: bool = False,
        **kwargs: Any,
    ) -> MetaSelf:
        """
        Makes a new BaseScalerMeta type class.

        Args:
            abstract: If True, the class is treated as fully abstract and added to the ``abstract_kernels`` list.
            partial_abstract: If True, the class is considered partially abstract and added to the
                ``partial_abstract_kernels`` list.
        """

        obj = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Decorate the implemented_funcs with `_add_init_kwargs` to add the init kwargs to the funcs.
        for impl_func_name in getattr(obj, "_implemented_funcs"):
            func = getattr(obj, impl_func_name)

            if not getattr(func, "__has_init_kwargs__", False):
                setattr(obj, impl_func_name, _add_init_kwargs(func))

        if abstract:
            abstract_kernels.append(obj)
            return obj

        if partial_abstract:
            partial_abstract_kernels.append(obj)

            # If partial_abstract is True, add kernel_radius property
            # if it not implemented by _static_kernel_radius or kernel_radius
            if not hasattr(obj, "_static_kernel_radius") and not hasattr(obj, "kernel_radius"):
                setattr(obj, "kernel_radius", _partial_abstract_kernel_radius)

        # If a _static_kernel_radius attr is implemented, check if kernel_radius property is there
        if hasattr(obj, "_static_kernel_radius") and not hasattr(obj, "kernel_radius"):
            setattr(obj, "kernel_radius", _static_kernel_radius_property)

        return obj


@BaseScalerMeta.cachedproperty
def _partial_abstract_kernel_radius(self: BaseScaler) -> int:
    raise CustomNotImplementedError("kernel_radius is not implemented!", self.__class__)


setattr(_partial_abstract_kernel_radius, "__name__", "kernel_radius")


@BaseScalerMeta.cachedproperty
def _static_kernel_radius_property(self: BaseScaler) -> int:
    return ceil(self._static_kernel_radius)


setattr(_static_kernel_radius_property, "__name__", "kernel_radius")


class BaseScaler(VSObjectABC, metaclass=BaseScalerMeta, abstract=True):
    """
    Base abstract scaling interface for VapourSynth scalers.
    """

    kwargs: dict[str, Any]
    """Arguments passed to the implemented funcs or internal scale function."""

    _static_kernel_radius: ClassVar[int]
    """Optional fixed kernel radius for the scaler."""

    _err_class: ClassVar[type[_UnknownBaseScalerError]]
    """Custom error class used for validation failures."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ()
    """
    Tuple of function names that are implemented in the current class.

    These functions determine which keyword arguments will be extracted from the `__init__` method.
    """

    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            """
            Create a new instance of the scaler, validating kernel radius if applicable.
            """
            if not _is_base_scaler_abstract(cls):
                return super().__new__(cls)

            if cls in abstract_kernels:
                raise CustomRuntimeError(f"Can't instantiate abstract class {cls.__name__}.", cls)

            raise CustomRuntimeError(
                "When inheriting from BaseScaler, you must implement the kernel radius by either adding "
                "the `kernel_radius` property or setting the class variable `_static_kernel_radius`.",
                cls,
            )

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional keyword arguments.

        These keyword arguments are automatically forwarded to
        the [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only
        if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.kwargs = kwargs

    def __str__(self) -> str:
        """
        Return the human-readable string representation of the scaler.

        Returns:
            Pretty-printed string with class name and arguments.
        """
        return self.pretty_string

    @staticmethod
    def _wh_norm(
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[int, int]:
        """
        Normalize width and height to fall back to the clip's dimensions if not provided.

        Args:
            clip: Input video clip.
            width: Optional width value.
            height: Optional height value.

        Returns:
            Tuple of resolved (width, height).
        """
        return (fallback(width, clip.width), fallback(height, clip.height))

    @classmethod
    def from_param(
        cls,
        scaler: str | type[Self] | Self | None = None,
        /,
        func_except: FuncExcept | None = None,
    ) -> type[Self]:
        """
        Resolve and return a scaler type from a given input (string, type, or instance).

        Args:
            scaler: Scaler identifier (string, class, or instance).
            func_except: Function returned for custom error handling.

        Returns:
            Resolved scaler type.
        """
        return _base_from_param(cls, scaler, cls._err_class, func_except)

    @classmethod
    def ensure_obj(
        cls,
        scaler: str | type[Self] | Self | None = None,
        /,
        func_except: FuncExcept | None = None,
    ) -> Self:
        """
        Ensure that the input is a scaler instance, resolving it if necessary.

        Args:
            scaler: Scaler identifier (string, class, or instance).
            func_except: Function returned for custom error handling.

        Returns:
            Scaler instance.
        """
        return _base_ensure_obj(cls, scaler, func_except)

    if TYPE_CHECKING:

        @BaseScalerMeta.cachedproperty
        def kernel_radius(self) -> int:
            """
            Return the effective kernel radius for the scaler.

            Raises:
                CustomNotImplementedError: If no kernel radius is defined.

            Returns:
                Kernel radius.
            """
            ...

    def _pretty_string(self, **attrs: Any) -> str:
        """
        Build a formatted string representation including class name and arguments.

        Args:
            **attrs: Additional attributes to include.

        Returns:
            String representation of the object.
        """
        return (
            f"{self.__class__.__name__}" + "(" + ", ".join(f"{k}={v}" for k, v in (attrs | self.kwargs).items()) + ")"
        )

    @property
    def pretty_string(self) -> str:
        """
        Cached property returning a user-friendly string representation.

        Returns:
            Pretty-printed string with arguments.
        """
        return self._pretty_string()

    @classproperty
    @classmethod
    def is_abstract(cls) -> bool:
        """Return True if this class can't be instantiated."""
        return _is_base_scaler_abstract(cls)

    @classproperty.cached
    @classmethod
    def implemented_funcs(cls) -> frozenset[str]:
        """
        Returns a set of function names that are implemented in the current class and the parent classes.

        These functions determine which keyword arguments will be extracted from the `__init__` method.

        Returns:
            Frozen set of function names.
        """
        return frozenset(func for klass in cls.mro() for func in getattr(klass, "_implemented_funcs", ()))


class Scaler(BaseScaler):
    """
    Abstract scaling interface.

    Subclasses should define a `scale_function` to perform the actual scaling logic.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownScalerError

    scale_function: Callable[..., vs.VideoNode]
    """Scale function called internally when performing scaling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("scale", "supersample")

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale a clip to a specified resolution.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.
            **kwargs: Additional arguments forwarded to the scale function.

        Returns:
            The scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)

        return self.scale_function(clip, **_norm_props_enums(self.get_scale_args(clip, shift, width, height, **kwargs)))

    def supersample(
        self, clip: vs.VideoNode, rfactor: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Supersample a clip by a given scaling factor.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            rfactor: Scaling factor for supersampling.
            shift: Subpixel shift (top, left) applied during scaling.
            **kwargs: Additional arguments forwarded to the scale function.

        Raises:
            CustomValueError: If resulting resolution is non-positive.

        Returns:
            The supersampled clip.
        """
        assert check_variable_resolution(clip, self.supersample)

        dst_width, dst_height = ceil(clip.width * rfactor), ceil(clip.height * rfactor)

        if max(dst_width, dst_height) <= 0.0:
            raise CustomValueError(
                'Multiplying the resolution by "rfactor" must result in a positive resolution!',
                self.supersample,
                rfactor,
            )

        return self.scale(clip, dst_width, dst_height, shift, **kwargs)

    def get_scale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate the keyword arguments used for scaling.

        Args:
            clip: The source clip.
            shift: Subpixel shift (top, left).
            width: Target width.
            height: Target height.
            **kwargs: Extra parameters to merge.

        Returns:
            Final dictionary of keyword arguments for the scale function.
        """
        return {"width": width, "height": height, "src_top": shift[0], "src_left": shift[1]} | self.kwargs | kwargs


class Descaler(BaseScaler):
    """
    Abstract descaling interface.

    Subclasses must define the `descale_function` used to perform the descaling.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownDescalerError

    descale_function: Callable[..., vs.VideoNode]
    """Descale function called internally when performing descaling operations."""

    rescale_function: Callable[..., vs.VideoNode]
    """Rescale function called internally when performing upscaling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("descale", "rescale")

    def descale(
        self,
        clip: vs.VideoNode,
        width: int | None,
        height: int | None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Descale a clip to the given resolution.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target descaled width (defaults to clip width if None).
            height: Target descaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.

        Returns:
            The descaled clip.
        """
        width, height = self._wh_norm(clip, width, height)

        return self.descale_function(
            clip, **_norm_props_enums(self.get_descale_args(clip, shift, width, height, **kwargs))
        )

    def rescale(
        self,
        clip: vs.VideoNode,
        width: int | None,
        height: int | None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Rescale a clip to the given resolution from a previously descaled clip.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target scaled width (defaults to clip width if None).
            height: Target scaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.

        Returns:
            The scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)

        return self.rescale_function(
            clip, **_norm_props_enums(self.get_rescale_args(clip, shift, width, height, **kwargs))
        )

    def get_descale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct the argument dictionary used for descaling.

        Args:
            clip: The source clip.
            shift: Subpixel shift (top, left).
            width: Target width for descaling.
            height: Target height for descaling.
            **kwargs: Extra keyword arguments to merge.

        Returns:
            Combined keyword argument dictionary.
        """
        return {"width": width, "height": height, "src_top": shift[0], "src_left": shift[1]} | self.kwargs | kwargs

    def get_rescale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct the argument dictionary used for upscaling.

        Args:
            clip: The source clip.
            shift: Subpixel shift (top, left).
            width: Target width for upscaling.
            height: Target height for upscaling.
            **kwargs: Extra keyword arguments to merge.

        Returns:
            Combined keyword argument dictionary.
        """
        return {"width": width, "height": height, "src_top": shift[0], "src_left": shift[1]} | self.kwargs | kwargs


class Resampler(BaseScaler):
    """
    Abstract resampling interface.

    Subclasses must define the `resample_function` used to perform the resampling.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownResamplerError

    resample_function: Callable[..., vs.VideoNode]
    """Resample function called internally when performing resampling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("resample",)

    def resample(
        self,
        clip: vs.VideoNode,
        format: SupportsInt | VideoFormatLike | HoldsVideoFormat,
        matrix: MatrixLike | None = None,
        matrix_in: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        transfer_in: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        primaries_in: PrimariesLike | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Resample a video clip to the given format.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            format: The target video format, which can either be:

                   - an integer format ID,
                   - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                   - or a source from which a valid `VideoFormat` can be extracted.
            matrix: An optional color transformation matrix to apply.
            matrix_in: An optional input matrix for color transformations.
            transfer: An optional color transformation transfer to apply.
            transfer_in: An optional input transfer for color transformations.
            primaries: Optional color transformation primaries to apply.
            primaries_in: Optional input primaries for color transformations.
            **kwargs: Additional keyword arguments passed to the `resample_function`.

        Returns:
            The resampled clip.
        """
        return self.resample_function(
            clip,
            **_norm_props_enums(
                self.get_resample_args(
                    clip, format, matrix, matrix_in, transfer, transfer_in, primaries, primaries_in, **kwargs
                )
            ),
        )

    def get_resample_args(
        self,
        clip: vs.VideoNode,
        format: SupportsInt | VideoFormatLike | HoldsVideoFormat,
        matrix: MatrixLike | None,
        matrix_in: MatrixLike | None,
        transfer: TransferLike | None = None,
        transfer_in: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        primaries_in: PrimariesLike | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct the argument dictionary used for resampling.

        Args:
            clip: The source clip.
            format: The target video format, which can either be:

                   - an integer format ID,
                   - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                   - or a source from which a valid `VideoFormat` can be extracted.
            matrix: The matrix for color transformation.
            matrix_in: The input matrix for color transformation.
            transfer: The transfer for color transformation.
            transfer_in: The input transfer for color transformation.
            primaries: The primaries for color transformation.
            primaries_in: The input primaries for color transformation.
            **kwargs: Additional keyword arguments for resampling.

        Returns:
            A dictionary containing the resampling arguments.
        """
        return _resolve_video_spec_args(
            clip,
            **(
                {
                    "format": get_video_format(format).id,
                    "matrix": Matrix.from_param_with_fallback(matrix),
                    "matrix_in": Matrix.from_param_with_fallback(matrix_in),
                    "transfer": Transfer.from_param_with_fallback(transfer),
                    "transfer_in": Transfer.from_param_with_fallback(transfer_in),
                    "primaries": Primaries.from_param_with_fallback(primaries),
                    "primaries_in": Primaries.from_param_with_fallback(primaries_in),
                }
                | self.kwargs
                | kwargs
            ),
        )


class Kernel(Scaler, Descaler, Resampler):
    """
    Abstract kernel interface combining scaling, descaling, resampling, and shifting functionality.

    Subclasses are expected to implement the actual transformation logic by overriding the methods or
    providing the respective `*_function` callables (`scale_function`, `descale_function`, `resample_function`).

    This class is abstract and should not be used directly.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownKernelError

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("shift",)

    @overload
    def shift(self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift], /, **kwargs: Any) -> vs.VideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            shift: A (top, left) tuple values for shift.
            **kwargs: Additional arguments passed to the internal `scale` call.

        Returns:
            A new clip with the applied shift.

        Raises:
            VariableFormatError: If the input clip has variable format.
        """

    @overload
    def shift(
        self, clip: vs.VideoNode, shift_top: float | list[float], shift_left: float | list[float], /, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            shift_top: Vertical shift or list of Vertical shifts.
            shift_left: Horizontal shift or list of horizontal shifts.
            **kwargs: Additional arguments passed to the internal `scale` call.

        Returns:
            A new clip with the applied shift.

        Raises:
            VariableFormatError: If the input clip has variable format.
        """

    def shift(
        self,
        clip: vs.VideoNode,
        shifts_or_top: float | tuple[float, float] | list[float],
        shift_left: float | list[float] | None = None,
        /,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        If a single float or tuple is provided, it is used uniformly.
        If a list is given, the shift is applied per plane.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            shifts_or_top: Either a single vertical shift, a (top, left) tuple, or a list of vertical shifts.
            shift_left: Horizontal shift or list of horizontal shifts. Ignored if `shifts_or_top` is a tuple.
            **kwargs: Additional arguments passed to the internal `scale` call.

        Returns:
            A new clip with the applied shift.

        Raises:
            VariableFormatError: If the input clip has variable format.
            CustomValueError: If the input clip is GRAY but lists of shift has been passed.
        """

        def _shift(src: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0)) -> vs.VideoNode:
            return self.scale(src, shift=shift, **kwargs)

        if isinstance(shifts_or_top, tuple):
            return _shift(clip, shifts_or_top)

        if isinstance(shifts_or_top, (int, float)) and isinstance(shift_left, (int, float, NoneType)):
            return _shift(clip, (shifts_or_top, shift_left or 0))

        if shift_left is None:
            shift_left = 0.0

        shifts_top = normalize_seq(shifts_or_top, clip.format.num_planes)
        shifts_left = normalize_seq(shift_left, clip.format.num_planes)

        if clip.format.num_planes == 1:
            return _shift(clip, (shifts_top[0], shifts_left[0]))

        shifted_planes = [
            plane if top == left == 0 else _shift(plane, (top, left))
            for plane, top, left in zip(split(clip), shifts_top, shifts_left)
        ]

        return core.std.ShufflePlanes(shifted_planes, [0, 0, 0], clip.format.color_family, clip)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Generate a base set of parameters to pass for scaling, descaling, or resampling.

        Args:
            is_descale: Whether this is for a descale operation.
            clip: The source clip.
            width: Target width.
            height: Target height.
            **kwargs: Additional keyword arguments to include.

        Returns:
            Dictionary of combined parameters.
        """
        return _resolve_video_spec_args(clip, **{"width": width, "height": height} | self.kwargs | kwargs)

    def get_scale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and normalize argument dictionary for a scale operation.

        Args:
            clip: The source clip.
            shift: Vertical and horizontal shift to apply.
            width: Target width.
            height: Target height.
            **kwargs: Additional arguments to pass to the scale function.

        Returns:
            Dictionary of keyword arguments for the scale function.
        """
        return {"src_top": shift[0], "src_left": shift[1]} | self.get_params_args(False, clip, width, height, **kwargs)

    def get_descale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and normalize argument dictionary for a descale operation.

        Args:
            clip: The source clip.
            shift: Vertical and horizontal shift to apply.
            width: Target width.
            height: Target height.
            **kwargs: Additional arguments to pass to the descale function.

        Returns:
            Dictionary of keyword arguments for the descale function.
        """
        return {"src_top": shift[0], "src_left": shift[1]} | self.get_params_args(True, clip, width, height, **kwargs)

    def get_rescale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and normalize argument dictionary for a rescale operation.

        Args:
            clip: The source clip.
            shift: Vertical and horizontal shift to apply.
            width: Target width.
            height: Target height.
            **kwargs: Additional arguments to pass to the rescale function.

        Returns:
            Dictionary of keyword arguments for the rescale function.
        """
        return {"src_top": shift[0], "src_left": shift[1]} | self.get_params_args(True, clip, width, height, **kwargs)

    def get_resample_args(
        self,
        clip: vs.VideoNode,
        format: SupportsInt | VideoFormatLike | HoldsVideoFormat,
        matrix: MatrixLike | None,
        matrix_in: MatrixLike | None,
        transfer: TransferLike | None = None,
        transfer_in: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        primaries_in: PrimariesLike | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and normalize argument dictionary for a resample operation.

        Args:
            clip: The source clip.
            format: The target video format, which can either be:

                   - an integer format ID,
                   - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                   - or a source from which a valid `VideoFormat` can be extracted.
            matrix: Target color matrix.
            matrix_in: Source color matrix.
            transfer: Target color transfer.
            transfer_in: Source color transfer.
            primaries: Target color primaries.
            primaries_in: Source color primaries.
            **kwargs: Additional arguments to pass to the resample function.

        Returns:
            Dictionary of keyword arguments for the resample function.
        """
        return {
            "format": get_video_format(format).id,
            "matrix": Matrix.from_param_with_fallback(matrix),
            "matrix_in": Matrix.from_param_with_fallback(matrix_in),
            "transfer": Transfer.from_param_with_fallback(transfer),
            "transfer_in": Transfer.from_param_with_fallback(transfer_in),
            "primaries": Primaries.from_param_with_fallback(primaries),
            "primaries_in": Primaries.from_param_with_fallback(primaries_in),
        } | self.get_params_args(False, clip, **kwargs)


class Bobber(BaseScaler):
    """
    Abstract scaler class that applies bob deinterlacing.
    """

    bob_function: Callable[..., vs.VideoNode]
    """Bob function called internally when performing bobbing operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("bob", "deinterlace")

    def bob(self, clip: vs.VideoNode, *, tff: FieldBasedLike | bool | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Apply bob deinterlacing to a given clip using the selected resizer.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip
            tff: Field order of the clip.

        Returns:
            The bobbed clip.
        """
        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        return self.bob_function(clip, **self.get_bob_args(clip, tff=clip_fieldbased.is_tff, **kwargs))

    def deinterlace(
        self, clip: vs.VideoNode, *, tff: FieldBasedLike | bool | None = None, double_rate: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Apply deinterlacing to a given clip using the selected resizer.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip
            tff: Field order of the clip.
            double_rate: Whether to double the frame rate (True) or retain the original rate (False).

        Returns:
            The bobbed clip.
        """
        bobbed = self.bob(clip, tff=tff, **kwargs)

        if not double_rate:
            return bobbed[::2]

        return bobbed

    def get_bob_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> dict[str, Any]:
        """
        Generate the keyword arguments used for bobbing.

        Args:
            clip: The source clip.
            shift: Subpixel shift (top, left).
            **kwargs: Extra parameters to merge.

        Returns:
            Final dictionary of keyword arguments for the bob function.
        """
        return self.kwargs | kwargs


type ScalerLike = str | type[Scaler] | Scaler
"""
Type alias for anything that can resolve to a Scaler.

This includes:

- A string identifier.
- A class type subclassing [Scaler][vskernels.Scaler].
- An instance of a [Scaler][vskernels.Scaler].
"""

type DescalerLike = str | type[Descaler] | Descaler
"""
Type alias for anything that can resolve to a Descaler.

This includes:

- A string identifier.
- A class type subclassing [Descaler][vskernels.Descaler].
- An instance of a [Descaler][vskernels.Descaler].
"""

type ResamplerLike = str | type[Resampler] | Resampler
"""
Type alias for anything that can resolve to a Resampler.

This includes:

- A string identifier.
- A class type subclassing [Resampler][vskernels.Resampler].
- An instance of a [Resampler][vskernels.Resampler].
"""

type KernelLike = str | type[Kernel] | Kernel
"""
Type alias for anything that can resolve to a Kernel.

This includes:

- A string identifier.
- A class type subclassing [Kernel][vskernels.Kernel].
- An instance of a [Kernel][vskernels.Kernel].
"""

type BobberLike = str | type[Bobber] | Bobber
"""
Type alias for anything that can resolve to a Bobber.

This includes:

- A string identifier.
- A class type subclassing [Bobber][vskernels.Bobber].
- An instance of a [Bobber][vskernels.Bobber].
"""
