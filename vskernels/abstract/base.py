"""
This module defines the base abstract interfaces for general scaling operations.
"""

from __future__ import annotations

import functools

from abc import ABC, ABCMeta
from inspect import Signature
from math import ceil
from types import NoneType
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Concatenate, Literal, NoReturn, TypeVar, Union, cast, get_origin, overload
)

from jetpytools import P, R, T_co
from typing_extensions import Self, deprecated

from vstools import (
    ConstantFormatVideoNode, CustomNotImplementedError, CustomRuntimeError, CustomValueError, FuncExceptT,
    HoldsVideoFormatT, Matrix, MatrixT, VideoFormatT, VideoNodeT, check_correct_subsampling, check_variable_format,
    check_variable_resolution, core, fallback, get_subclasses, get_video_format, normalize_seq, split, vs, vs_object
)
from vstools.enums.color import _norm_props_enums

from ..exceptions import (
    UnknownDescalerError, UnknownKernelError, UnknownResamplerError, UnknownScalerError, _UnknownBaseScalerError
)
from ..types import LeftShift, TopShift

__all__ = [
    "Scaler",
    "Descaler",
    "Resampler",
    "Kernel",

    "ScalerLike",
    "DescalerLike",
    "ResamplerLike",
    "KernelLike"
]


def _add_init_kwargs(method: Callable[Concatenate[_BaseScalerT, P], R]) -> Callable[Concatenate[_BaseScalerT, P], R]:
    signature = Signature.from_callable(method)

    @functools.wraps(method)
    def _wrapped(self: _BaseScalerT, *args: P.args, **kwargs: P.kwargs) -> R:
        # TODO: remove this
        if not TYPE_CHECKING:
            if isinstance(self, vs.VideoNode):
                import inspect
                import pathlib
                import re
                import warnings

                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    f"The `{method.__name__}` must be called on an instance, not the class. "
                    "For example, use: Bicubic().scale(...) instead of Bicubic.scale(...)",
                    DeprecationWarning,
                    2,
                    skip_file_prefixes=(str(pathlib.Path(__file__).resolve()),)
                )

                frame_infos = inspect.stack()
                frame_info = frame_infos[1]
                f0 = inspect.currentframe()
                f1 = f0.f_back  # pyright: ignore

                try:
                    if (code := frame_info.code_context):
                        match = re.search(rf'(\w+)\.{method.__name__}', code[0])
                        if match:
                            clip = self
                            self = eval(match.group(1), f1.f_globals, f1.f_locals)()  # pyright: ignore
                            args = (clip, ) + args  # pyright: ignore
                finally:
                    frame_infos.clear()
                    del frame_info, f0, f1

        init_kwargs = {k: self.kwargs.pop(k) for k in self.kwargs.keys() & signature.parameters.keys()}

        returned = method(self, *args, **init_kwargs | kwargs)

        self.kwargs |= init_kwargs

        return returned

    setattr(_wrapped, "__has_init_kwargs__", True)

    return _wrapped


def _base_from_param(
    cls: type[_BaseScalerT],
    value: str | type[BaseScaler] | BaseScaler | None,
    exception_cls: type[_UnknownBaseScalerError],
    func_except: FuncExceptT | None,
) -> type[_BaseScalerT]:
    if isinstance(value, str):
        all_scalers = {s.__name__.lower(): s for s in get_subclasses(BaseScaler)}

        try:
            return cast(type[_BaseScalerT], all_scalers[value.lower().strip()])
        except KeyError:
            raise exception_cls(func_except or cls.from_param, value)

    if isinstance(value, type) or isinstance(get_origin(value), type):
        return cast(type[_BaseScalerT], value)

    if isinstance(value, cls):
        return value.__class__

    return cls


def _base_ensure_obj(
    cls: type[_BaseScalerT],
    value: str | type[BaseScaler] | BaseScaler | None,
    func_except: FuncExceptT | None,
) -> _BaseScalerT:
    if value is None:
        return cls()

    if isinstance(value, cls):
        return value

    return cls.from_param(value, func_except)()  # type: ignore[arg-type]


@functools.cache
def _check_kernel_radius(cls: type[BaseScaler]) -> Literal[True]:
    if cls in abstract_kernels:
        raise CustomRuntimeError(f"Can't instantiate abstract class {cls.__name__}!", cls)

    if "kernel_radius" in set((attr for sub_cls in cls.__mro__ for attr in sub_cls.__dict__.keys())):
        return True

    raise CustomRuntimeError(
        "When inheriting from BaseScaler, you must implement the kernel radius by either adding "
        "the `kernel_radius` property or setting the class variable `_static_kernel_radius`.",
        cls,
)


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


class BaseScalerMeta(ABCMeta):
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

    def __new__(
        mcls: type[_BaseScalerMetaT],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        abstract: bool = False,
        partial_abstract: bool = False,
        **kwargs: Any,
    ) -> _BaseScalerMetaT:
        """
        Makes a new BaseScalerMeta type class.

        :param abstract:            If True, the class is treated as fully abstract
                                    and added to the ``abstract_kernels`` list.
        :param partial_abstract:    If True, the class is considered partially abstract
                                    and added to the ``partial_abstract_kernels`` list.
        """

        obj = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Decorate the `_implemented_funcs` with `_add_init_kwargs` to add the init kwargs to the funcs.
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
                @BaseScaler.cached_property
                def _partial_abstract_kernel_radius(self: BaseScaler) -> int:
                    raise CustomNotImplementedError("kernel_radius is not implemented!", self.__class__)

                _partial_abstract_kernel_radius.attrname = "kernel_radius"

                setattr(obj, "kernel_radius", _partial_abstract_kernel_radius)

        # If a _static_kernel_radius attr is implemented, check if kernel_radius property is there
        if hasattr(obj, "_static_kernel_radius") and not hasattr(obj, "kernel_radius"):
            @BaseScaler.cached_property
            def _static_kernel_radius_property(self: BaseScaler) -> int:
                return ceil(self._static_kernel_radius)

            _static_kernel_radius_property.attrname = "kernel_radius"

            setattr(obj, "kernel_radius", _static_kernel_radius_property)

        return obj


_BaseScalerMetaT = TypeVar("_BaseScalerMetaT", bound=BaseScalerMeta)


class BaseScaler(vs_object, ABC, metaclass=BaseScalerMeta, abstract=True):
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
    Tuple of function names that are implemented.

    These functions determine which keyword arguments will be extracted from the __init__ method.
    """

    class cached_property(functools.cached_property[T_co]):
        """Read only version of functools.cached_property."""

        if TYPE_CHECKING:
            def __init__(self, func: Callable[Concatenate[_BaseScalerT, P], T_co]) -> None: ...

        def __set__(self, instance: None, value: Any) -> NoReturn:  # type: ignore[override]
            """Raise an error when attempting to set a cached property."""
            raise AttributeError("Can't set attribute")

    if not TYPE_CHECKING:
        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            """
            Create a new instance of the scaler, validating kernel radius if applicable.
            """
            if _check_kernel_radius(cls):
                obj = super().__new__(cls)
                obj.kwargs = {}
                return obj

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        :param kwargs:  Keyword arguments that configure the internal scaling behavior.
        """
        self.kwargs = kwargs

    def __str__(self) -> str:
        """
        Return the human-readable string representation of the scaler.

        :return: Pretty-printed string with class name and arguments.
        """
        return self.pretty_string

    @staticmethod
    def _wh_norm(clip: vs.VideoNode, width: int | None = None, height: int | None = None) -> tuple[int, int]:
        """
        Normalize width and height to fall back to the clip's dimensions if not provided.

        :param clip:    Input video clip.
        :param width:   Optional width value.
        :param height:  Optional height value.
        :return:        Tuple of resolved (width, height).
        """
        return (fallback(width, clip.width), fallback(height, clip.height))

    @classmethod
    def from_param(
        cls,
        scaler: str | type[Self] | Self | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> type[Self]:
        """
        Resolve and return a scaler type from a given input (string, type, or instance).

        :param scaler:          Scaler identifier (string, class, or instance).
        :param func_except:     Function returned for custom error handling.
        :return:                Resolved scaler type.
        """
        return _base_from_param(cls, scaler, cls._err_class, func_except)

    @classmethod
    def ensure_obj(
        cls,
        scaler: str | type[Self] | Self | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> Self:
        """
        Ensure that the input is a scaler instance, resolving it if necessary.

        :param scaler:          Scaler identifier (string, class, or instance).
        :param func_except:     Function returned for custom error handling.
        :return:                Scaler instance.
        """
        return _base_ensure_obj(cls, scaler, func_except)

    if TYPE_CHECKING:
        @cached_property
        def kernel_radius(self) -> int:
            """
            Return the effective kernel radius for the scaler.

            :raises CustomNotImplementedError:  If no kernel radius is defined.
            :return:                            Kernel radius.
            """
            ...

    def _pretty_string(self, **attrs: Any) -> str:
        """
        Build a formatted string representation including class name and arguments.

        :param attrs:   Additional attributes to include.
        :return:        String representation of the object.
        """
        return (
            f"{self.__class__.__name__}" + "(" + ", ".join(f"{k}={v}" for k, v in (attrs | self.kwargs).items()) + ")"
        )

    @cached_property
    def pretty_string(self) -> str:
        """
        Cached property returning a user-friendly string representation.

        :return: Pretty-printed string with arguments.
        """
        return self._pretty_string()

    def __vs_del__(self, core_id: int) -> None:
        self.kwargs.clear()


_BaseScalerT = TypeVar("_BaseScalerT", bound=BaseScaler)


class Scaler(BaseScaler):
    """
    Abstract scaling interface.

    Subclasses should define a `scale_function` to perform the actual scaling logic.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownScalerError

    scale_function: Callable[..., vs.VideoNode]
    """Scale function called internally when performing scaling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("scale",)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        """
        Scale a clip to a specified resolution.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:        The source clip.
        :param width:       Target width (defaults to clip width if None).
        :param height:      Target height (defaults to clip height if None).
        :param shift:       Subpixel shift (top, left) applied during scaling.
        :param kwargs:      Additional arguments forwarded to the scale function.
        :return:            The scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

        return self.scale_function(clip, **_norm_props_enums(self.get_scale_args(clip, shift, width, height, **kwargs)))

    def supersample(
        self, clip: VideoNodeT, rfactor: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> VideoNodeT:
        """
        Supersample a clip by a given scaling factor.

        :param clip:                The source clip.
        :param rfactor:             Scaling factor for supersampling.
        :param shift:               Subpixel shift (top, left) applied during scaling.
        :param kwargs:              Additional arguments forwarded to the scale function.
        :raises CustomValueError:   If resulting resolution is non-positive.
        :return:                    The supersampled clip.
        """
        assert check_variable_resolution(clip, self.supersample)

        dst_width, dst_height = ceil(clip.width * rfactor), ceil(clip.height * rfactor)

        if max(dst_width, dst_height) <= 0.0:
            raise CustomValueError(
                'Multiplying the resolution by "rfactor" must result in a positive resolution!',
                self.supersample,
                rfactor,
            )

        return self.scale(clip, dst_width, dst_height, shift, **kwargs)  # type: ignore[return-value]

    @deprecated('The "multi" method is deprecated. Use "supersample" instead.', category=DeprecationWarning)
    def multi(
        self, clip: VideoNodeT, multi: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> VideoNodeT:
        """
        Deprecated alias for `supersample`.

        :param clip:    The source clip.
        :param multi:   Supersampling factor.
        :param shift:   Subpixel shift (top, left) applied during scaling.
        :param kwargs:  Additional arguments forwarded to the scale function.
        :return:        The supersampled clip.
        """
        return self.supersample(clip, multi, shift, **kwargs)

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

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param width:   Target width.
        :param height:  Target height.
        :param kwargs:  Extra parameters to merge.
        :return:        Final dictionary of keyword arguments for the scale function.
        """
        return dict(width=width, height=height, src_top=shift[0], src_left=shift[1]) | self.kwargs | kwargs


_ScalerT = TypeVar("_ScalerT", bound=Scaler)


class Descaler(BaseScaler):
    """
    Abstract descaling interface.

    Subclasses must define the `descale_function` used to perform the descaling.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownDescalerError

    descale_function: Callable[..., ConstantFormatVideoNode]
    """Descale function called internally when performing descaling operations."""

    rescale_function: Callable[..., ConstantFormatVideoNode]
    """Rescale function called internally when performing upscaling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("descale", "rescale")

    def descale(
        self,
        clip: vs.VideoNode,
        width: int | None,
        height: int | None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Descale a clip to the given resolution.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:    The source clip.
        :param width:   Target descaled width (defaults to clip width if None).
        :param height:  Target descaled height (defaults to clip height if None).
        :param shift:   Subpixel shift (top, left) applied during scaling.
        :return:        The descaled clip.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

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
    ) -> ConstantFormatVideoNode:
        """
        Rescale a clip to the given resolution from a previously descaled clip.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:    The source clip.
        :param width:   Target scaled width (defaults to clip width if None).
        :param height:  Target scaled height (defaults to clip height if None).
        :param shift:   Subpixel shift (top, left) applied during scaling.
        :return:        The scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

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

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param width:   Target width for descaling.
        :param height:  Target height for descaling.
        :param kwargs:  Extra keyword arguments to merge.
        :return:        Combined keyword argument dictionary.
        """
        return dict(width=width, height=height, src_top=shift[0], src_left=shift[1]) | self.kwargs | kwargs

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

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param width:   Target width for upscaling.
        :param height:  Target height for upscaling.
        :param kwargs:  Extra keyword arguments to merge.
        :return:        Combined keyword argument dictionary.
        """
        return dict(width=width, height=height, src_top=shift[0], src_left=shift[1]) | self.kwargs | kwargs


class Resampler(BaseScaler):
    """
    Abstract resampling interface.

    Subclasses must define the `resample_function` used to perform the resampling.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownResamplerError

    resample_function: Callable[..., ConstantFormatVideoNode]
    """Resample function called internally when performing resampling operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("resample",)

    def resample(
        self,
        clip: vs.VideoNode,
        format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None = None,
        matrix_in: MatrixT | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Resample a video clip to the given format.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:        The source clip.
        :param format:      The target video format, which can either be:
                                - an integer format ID,
                                - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                                - or a source from which a valid `VideoFormat` can be extracted.
        :param matrix:      An optional color transformation matrix to apply.
        :param matrix_in:   An optional input matrix for color transformations.
        :param kwargs:      Additional keyword arguments passed to the `resample_function`.
        :return:            The resampled clip.
        """
        return self.resample_function(
            clip, **_norm_props_enums(self.get_resample_args(clip, format, matrix, matrix_in, **kwargs))
        )

    def get_resample_args(
        self,
        clip: vs.VideoNode,
        format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None,
        matrix_in: MatrixT | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct the argument dictionary used for resampling.

        :param clip:        The source clip.
        :param format:      The target video format, which can either be:
                                - an integer format ID,
                                - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                                - or a source from which a valid `VideoFormat` can be extracted.
        :param matrix:      The matrix for color transformation.
        :param matrix_in:   The input matrix for color transformation.
        :param kwargs:      Additional keyword arguments for resampling.
        :return:            A dictionary containing the resampling arguments.
        """
        return (
            dict(
                format=get_video_format(format).id,
                matrix=Matrix.from_param(matrix),
                matrix_in=Matrix.from_param(matrix_in),
            )
            | self.kwargs
            | kwargs
        )


class Kernel(Scaler, Descaler, Resampler):
    """
    Abstract kernel interface combining scaling, descaling, resampling, and shifting functionality.

    Subclasses are expected to implement the actual transformation logic by overriding the methods or
    providing the respective `*_function` callables (`scale_function`, `descale_function`, `resample_function`).

    This class is abstract and should not be used directly.
    """

    _err_class: ClassVar[type[_UnknownBaseScalerError]] = UnknownKernelError

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("scale", "descale", "rescale", "resample", "shift")

    @overload
    def shift(self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift], /, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:                    The source clip.
        :param shift:                   A (top, left) tuple values for shift.
        :param kwargs:                  Additional arguments passed to the internal `scale` call.

        :return:                        A new clip with the applied shift.
        :raises VariableFormatError:    If the input clip has variable format.
        """

    @overload
    def shift(
        self, clip: vs.VideoNode, shift_top: float | list[float], shift_left: float | list[float], /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:                    The source clip.
        :param shift_top:               Vertical shift or list of Vertical shifts.
        :param shift_left:              Horizontal shift or list of horizontal shifts.
        :param kwargs:                  Additional arguments passed to the internal `scale` call.

        :return:                        A new clip with the applied shift.
        :raises VariableFormatError:    If the input clip has variable format.
        """

    def shift(
        self,
        clip: vs.VideoNode,
        shifts_or_top: float | tuple[float, float] | list[float],
        shift_left: float | list[float] | None = None,
        /,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        If a single float or tuple is provided, it is used uniformly.
        If a list is given, the shift is applied per plane.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        :param clip:                    The source clip.
        :param shifts_or_top:           Either a single vertical shift, a (top, left) tuple, or a list of vertical shifts.
        :param shift_left:              Horizontal shift or list of horizontal shifts. Ignored if `shifts_or_top` is a tuple.
        :param kwargs:                  Additional arguments passed to the internal `scale` call.

        :return:                        A new clip with the applied shift.
        :raises VariableFormatError:    If the input clip has variable format.
        :raises CustomValueError:       If the input clip is GRAY but lists of shift has been passed.
        """
        assert check_variable_format(clip, self.shift)

        n_planes = clip.format.num_planes

        def _shift(src: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0)) -> ConstantFormatVideoNode:
            return self.scale(src, shift=shift, **kwargs)  # type: ignore[return-value]

        if isinstance(shifts_or_top, tuple):
            return _shift(clip, shifts_or_top)

        if isinstance(shifts_or_top, (int, float)) and isinstance(shift_left, (int, float, NoneType)):
            return _shift(clip, (shifts_or_top, shift_left or 0))

        if shift_left is None:
            shift_left = 0.0

        shifts_top = normalize_seq(shifts_or_top, n_planes)
        shifts_left = normalize_seq(shift_left, n_planes)

        if n_planes == 1:
            if len(set(shifts_top)) > 1 or len(set(shifts_left)) > 1:
                raise CustomValueError(
                    "Inconsistent shift values detected for a single plane. "
                    "All shift values must be identical when passing a GRAY clip.",
                    self.shift,
                    (shifts_top, shifts_left),
                )

            return _shift(clip, (shifts_top[0], shifts_left[0]))

        shifted_planes = [
            plane if top == left == 0 else _shift(plane, (top, left))
            for plane, top, left in zip(split(clip), shifts_top, shifts_left)
        ]

        return core.std.ShufflePlanes(shifted_planes, [0, 0, 0], clip.format.color_family)

    @classmethod
    def from_param(cls, kernel: KernelLike | None = None, /, func_except: FuncExceptT | None = None) -> type[Self]:
        """
        Resolve and return a kernel class from a string name, class type, or instance.

        :param kernel:              Kernel identifier as a string, class type, or instance. If None, defaults to the current class.
        :param func_except:         Function returned for custom error handling.

        :return:                    The resolved kernel class.
        :raises UnknownKernelError: If the kernel could not be identified.
        """
        return _base_from_param(cls, kernel, cls._err_class, func_except)

    @classmethod
    def ensure_obj(cls, kernel: KernelLike | None = None, /, func_except: FuncExceptT | None = None) -> Self:
        """
        Ensure that the given kernel input is returned as a kernel instance.

        :param kernel:              Kernel name, class, or instance. Defaults to current class if None.
        :param func_except:         Function returned for custom error handling.

        :return:                    The resolved and instantiated kernel.
        """
        return _base_ensure_obj(cls, kernel, func_except)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Generate a base set of parameters to pass for scaling, descaling, or resampling.

        :param is_descale:  Whether this is for a descale operation.
        :param clip:        The source clip.
        :param width:       Target width.
        :param height:      Target height.
        :param kwargs:      Additional keyword arguments to include.

        :return:            Dictionary of combined parameters.
        """
        return dict(width=width, height=height) | self.kwargs | kwargs

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

        :param clip:    The source clip.
        :param shift:   Vertical and horizontal shift to apply.
        :param width:   Target width.
        :param height:  Target height.
        :param kwargs:  Additional arguments to pass to the scale function.

        :return:        Dictionary of keyword arguments for the scale function.
        """
        return dict(src_top=shift[0], src_left=shift[1]) | self.get_params_args(False, clip, width, height, **kwargs)

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

        :param clip:    The source clip.
        :param shift:   Vertical and horizontal shift to apply.
        :param width:   Target width.
        :param height:  Target height.
        :param kwargs:  Additional arguments to pass to the descale function.

        :return:        Dictionary of keyword arguments for the descale function.
        """
        return dict(src_top=shift[0], src_left=shift[1]) | self.get_params_args(True, clip, width, height, **kwargs)

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

        :param clip:    The source clip.
        :param shift:   Vertical and horizontal shift to apply.
        :param width:   Target width.
        :param height:  Target height.
        :param kwargs:  Additional arguments to pass to the rescale function.

        :return:        Dictionary of keyword arguments for the rescale function.
        """
        return dict(src_top=shift[0], src_left=shift[1]) | self.get_params_args(True, clip, width, height, **kwargs)

    def get_resample_args(
        self,
        clip: vs.VideoNode,
        format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None,
        matrix_in: MatrixT | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and normalize argument dictionary for a resample operation.

        :param clip:        The source clip.
        :param format:      The target video format, which can either be:
                                - an integer format ID,
                                - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                                - or a source from which a valid `VideoFormat` can be extracted.
        :param matrix:      Target color matrix.
        :param matrix_in:   Source color matrix.
        :param kwargs:      Additional arguments to pass to the resample function.

        :return:            Dictionary of keyword arguments for the resample function.
        """
        return dict(
            format=get_video_format(format).id, matrix=Matrix.from_param(matrix), matrix_in=Matrix.from_param(matrix_in)
        ) | self.get_params_args(False, clip, **kwargs)


ScalerLike = Union[str, type[Scaler], Scaler]
"""
Type alias for anything that can resolve to a Scaler.

This includes:
- A string identifier.
- A class type subclassing `Scaler`.
- An instance of a `Scaler`.
"""

DescalerLike = Union[str, type[Descaler], Descaler]
"""
Type alias for anything that can resolve to a Descaler.

This includes:
- A string identifier.
- A class type subclassing `Descaler`.
- An instance of a `Descaler`.
"""

ResamplerLike = Union[str, type[Resampler], Resampler]
"""
Type alias for anything that can resolve to a Resampler.

This includes:
- A string identifier.
- A class type subclassing `Resampler`.
- An instance of a `Resampler`.
"""

KernelLike = Union[str, type[Kernel], Kernel]
"""
Type alias for anything that can resolve to a Kernel.

This includes:
- A string identifier.
- A class type subclassing `Kernel`.
- An instance of a `Kernel`.
"""
