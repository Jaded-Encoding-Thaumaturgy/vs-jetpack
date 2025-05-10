from __future__ import annotations

from abc import ABC, ABCMeta
from functools import lru_cache
from inspect import Signature
from math import ceil
from types import NoneType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Protocol, Sequence, TypeVar, Union, cast, overload

from jetpytools import T_co, inject_kwargs_params
from typing_extensions import Self

from vstools import (
    ConstantFormatVideoNode, CustomIndexError, CustomNotImplementedError, CustomRuntimeError, CustomValueError,
    FieldBased, FuncExceptT, HoldsVideoFormatT, KwargsT, Matrix, MatrixT, VideoFormatT, check_correct_subsampling,
    check_variable_format, check_variable_resolution, core, depth, expect_bits, fallback, get_subclasses,
    get_video_format, inject_self, normalize_seq, split, vs, vs_object
)
from vstools.enums.color import _norm_props_enums

from ..exceptions import UnknownDescalerError, UnknownKernelError, UnknownResamplerError, UnknownScalerError
from ..types import (
    BorderHandling, BotFieldLeftShift, BotFieldTopShift, LeftShift, SampleGridModel, ShiftT, TopFieldLeftShift,
    TopFieldTopShift, TopShift
)

__all__ = [
    'Scaler', 'ScalerT',
    'Descaler', 'DescalerT',
    'Resampler', 'ResamplerT',
    'Kernel', 'KernelT'
]


@lru_cache
def _get_keywords(_methods: tuple[Callable[..., Any] | None, ...], self: Any) -> set[str]:
    methods_list = list(_methods)

    for cls in self.__class__.mro():
        if hasattr(cls, 'get_implemented_funcs'):
            methods_list.extend(cls.get_implemented_funcs(self))

    methods = {m for m in methods_list if m}

    keywords = set[str]()

    for method in methods:
        try:
            signature = method.__signature__  # type: ignore[attr-defined]
        except Exception:
            signature = Signature.from_callable(method)

        keywords.update(signature.parameters.keys())

    return keywords


def _clean_self_kwargs(methods: tuple[Callable[..., Any] | None, ...], self: Any) -> KwargsT:
    return {k: v for k, v in self.kwargs.items() if k not in _get_keywords(methods, self)}


def _base_from_param(
    cls: type[BaseScalerT],
    basecls: type[BaseScalerT],
    value: str | type[BaseScalerT] | BaseScalerT | None,
    exception_cls: type[CustomValueError],
    excluded: Sequence[type] = [],
    func_except: FuncExceptT | None = None
) -> type[BaseScalerT]:
    if isinstance(value, str):
        all_scalers = get_subclasses(BaseScaler, excluded)
        search_str = value.lower().strip()

        for scaler_cls in all_scalers:
            if scaler_cls.__name__.lower() == search_str:
                return cast(type[BaseScalerT], scaler_cls)

        raise exception_cls(func_except or cls.from_param, value)

    if isinstance(value, type) and issubclass(value, basecls):
        return value

    if isinstance(value, cls):
        return value.__class__

    return cls


def _base_ensure_obj(
    cls: type[BaseScalerT],
    basecls: type[BaseScalerT],
    value: str | type[BaseScalerT] | BaseScalerT | None,
    exception_cls: type[CustomValueError],
    excluded: Sequence[type] = [],
    func_except: FuncExceptT | None = None
) -> BaseScalerT:
    if value is None:
        new_scaler = cls()
    elif isinstance(value, cls) or isinstance(value, basecls):
        new_scaler = value
    else:
        new_scaler = cls.from_param(value, func_except)()

    if new_scaler.__class__ in excluded:
        raise exception_cls(
            func_except or cls.ensure_obj, new_scaler.__class__,
            'This {cls_name} can\'t be instantiated to be used!',
            cls_name=new_scaler.__class__
        )

    return new_scaler


def _check_kernel_radius(cls: BaseScalerMeta, obj: BaseScaler) -> BaseScaler:
    if cls in partial_abstract_kernels:
        return obj

    if cls in abstract_kernels:
        raise CustomRuntimeError(f"Can't instantiate abstract class {cls.__name__}!", cls)

    mro = set(cls.mro())
    mro.discard(BaseScaler)

    for sub_cls in mro:
        if any(v in sub_cls.__dict__.keys() for v in ('_static_kernel_radius', 'kernel_radius')):
            return obj

    raise CustomRuntimeError(
        'When inheriting from BaseScaler, you must implement the kernel radius by either adding '
        'the `kernel_radius` property or setting the class variable `_static_kernel_radius`.',
        reason=cls
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


_BaseScalerMetaT = TypeVar("_BaseScalerMetaT", bound="BaseScalerMeta")


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
        **kwargs: Any
    ) -> _BaseScalerMetaT:
        """
        :param abstract:            If True, the class is treated as fully abstract
                                    and added to the ``abstract_kernels`` list.
        :param partial_abstract:    If True, the class is considered partially abstract
                                    and added to the ``partial_abstract_kernels`` list.
        """

        obj = super().__new__(mcls, name, bases, namespace, **kwargs)

        if abstract:
            abstract_kernels.append(obj)
        elif partial_abstract:
            partial_abstract_kernels.append(obj)

        return obj


class BaseScaler(vs_object, ABC, metaclass=BaseScalerMeta, abstract=True):
    """
    Base abstract scaling interface for VapourSynth scalers.
    """

    kwargs: KwargsT
    """Arguments passed to the internal scale function."""

    _static_kernel_radius: ClassVar[int]
    """Optional fixed kernel radius for the scaler."""

    _err_class: ClassVar[type[CustomValueError]]
    """Custom error class used for validation failures."""

    if not TYPE_CHECKING:
        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            """
            Create a new instance of the scaler, validating kernel radius if applicable.
            """
            return _check_kernel_radius(cls, super().__new__(cls))

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional keyword arguments.

        :param kwargs:      Parameters to pass to the internal scale function.
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
        cls: type[BaseScalerT], scaler: str | type[BaseScalerT] | BaseScalerT | None = None, /,
        func_except: FuncExceptT | None = None
    ) -> type[BaseScalerT]:
        """
        Resolve and return a scaler type from a given input (string, type, or instance).

        :param scaler:          Scaler identifier (string, class, or instance).
        :param func_except:     Function returned for custom error handling.
        :return:                Resolved scaler type.
        """
        return _base_from_param(
            cls, (mro := cls.mro())[mro.index(BaseScaler) - 1], scaler, cls._err_class, [], func_except
        )

    @classmethod
    def ensure_obj(
        cls: type[BaseScalerT], scaler: str | type[BaseScalerT] | BaseScalerT | None = None, /,
        func_except: FuncExceptT | None = None
    ) -> BaseScalerT:
        """
        Ensure that the input is a scaler instance, resolving it if necessary.

        :param scaler:          Scaler identifier (string, class, or instance).
        :param func_except:     Function returned for custom error handling.
        :return:                Scaler instance.
        """
        return _base_ensure_obj(
            cls, (mro := cls.mro())[mro.index(BaseScaler) - 1], scaler, cls._err_class, [], func_except
        )

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        """
        Return the effective kernel radius for the scaler.

        :raises CustomNotImplementedError:  If no kernel radius is defined.
        :return:                            Kernel radius.
        """
        if hasattr(self, '_static_kernel_radius'):
            return ceil(self._static_kernel_radius)
        raise CustomNotImplementedError('kernel_radius is not implemented!', self.__class__)

    def get_clean_kwargs(self, *funcs: Callable[..., Any] | None) -> KwargsT:
        """
        Filter and return clean kwargs applicable to the given functions.

        :param funcs:   Functions to match kwargs against.
        :return:        Filtered kwargs dictionary.
        """
        return _clean_self_kwargs(funcs, self)

    def _pretty_string(self, **attrs: Any) -> str:
        """
        Build a formatted string representation including class name and arguments.

        :param attrs:   Additional attributes to include.
        :return:        String representation of the object.
        """
        return (
            f"{self.__class__.__name__}"
            + '(' + ', '.join(f'{k}={v}' for k, v in (attrs | self.kwargs).items()) + ')'
        )

    @inject_self.cached.property
    def pretty_string(self) -> str:
        """
        Cached property returning a user-friendly string representation.

        :return: Pretty-printed string with arguments.
        """
        return self._pretty_string()


BaseScalerT = TypeVar('BaseScalerT', bound=BaseScaler)


class Scaler(BaseScaler):
    """
    Abstract scaling interface.

    Subclasses should define a `scale_function` to perform the actual scaling logic.
    """

    _err_class = UnknownScalerError

    scale_function: Callable[..., vs.VideoNode]
    """Scale function called internally when performing scaling operations."""

    @inject_self.cached
    @inject_kwargs_params
    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Scale a clip to a specified resolution.

        :param clip:        The source clip.
        :param width:       Target width (defaults to clip width if None).
        :param height:      Target height (defaults to clip height if None).
        :param shift:       Subpixel shift (top, left) applied during scaling.
        :param kwargs:      Additional arguments forwarded to the scale function.
        :return:            Scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

        return self.scale_function(
            clip, **_norm_props_enums(self.get_scale_args(clip, shift, width, height, **kwargs))
        )

    @inject_self.cached
    def supersample(
        self, clip: vs.VideoNode, rfactor: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Supersample a clip by a given scaling factor.

        :param clip:                The source clip.
        :param rfactor:             Scaling factor for supersampling.
        :param shift:               Subpixel shift (top, left) applied during scaling.
        :param kwargs:              Additional arguments forwarded to the scale function.
        :raises CustomValueError:   If resulting resolution is non-positive.
        :return:                    Supersampled video clip.
        """
        assert check_variable_resolution(clip, self.multi)

        dst_width, dst_height = ceil(clip.width * rfactor), ceil(clip.height * rfactor)

        if max(dst_width, dst_height) <= 0.0:
            raise CustomValueError(
                'Multiplying the resolution by "rfactor" must result in a positive resolution!', self.supersample, rfactor
            )

        return self.scale(clip, dst_width, dst_height, shift, **kwargs)

    @inject_self.cached
    def multi(
        self, clip: vs.VideoNode, multi: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Deprecated alias for `supersample`.

        :param clip:    The source clip.
        :param multi:   Supersampling factor.
        :param shift:   Subpixel shift (top, left) applied during scaling.
        :param kwargs:  Additional arguments forwarded to the scale function.
        :return:        Supersampled video clip.
        """
        import warnings

        warnings.warn('The "multi" method is deprecated. Use "supersample" instead.', DeprecationWarning)

        return self.supersample(clip, multi, shift, **kwargs)

    @inject_kwargs_params
    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Generate the keyword arguments used for scaling.

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param width:   Target width.
        :param height:  Target height.
        :param funcs:   Optional callables to match applicable kwargs.
        :param kwargs:  Extra parameters to merge.
        :return:        Final dictionary of keyword arguments for the scale function.
        """
        return (
            dict(
                src_top=shift[0],
                src_left=shift[1]
            )
            | self.get_clean_kwargs(*funcs)
            | dict(width=width, height=height)
            | kwargs
        )

    def get_implemented_funcs(self) -> tuple[Callable[..., Any], ...]:
        """
        Return a tuple of implemented scaler functions.

        :return: Tuple containing implemented scaling methods (e.g., scale, supersample).
        """
        return (self.scale, self.supersample)


class Descaler(BaseScaler):
    """
    Abstract descaling interface.

    Subclasses must define the `descale_function` used to perform the descaling.
    """

    _err_class = UnknownDescalerError

    descale_function: Callable[..., ConstantFormatVideoNode]
    """Descale function called internally when performing descaling operations."""

    @inject_self.cached
    @inject_kwargs_params
    def descale(
        self, clip: vs.VideoNode, width: int | None, height: int | None,
        shift: ShiftT = (0, 0),
        *,
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        field_based: FieldBased | None = None,
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Descale a clip to the given resolution.

        Supports both progressive and interlaced sources. When interlaced, it will separate fields,
        perform per-field descaling, and weave them back.

        :param clip:                The source clip.
        :param width:               Target descaled width.
        :param height:              Target descaled height.
        :param shift:               Subpixel shift (top, left) or per-field shifts.
        :param border_handling:     Method for handling image borders during sampling.
        :param sample_grid_model:   Model used to align sampling grid.
        :param field_based:         Field-based processing mode (interlaced or progressive).
        :param kwargs:              Additional arguments passed to `descale_function`.
        :raises CustomIndexError:   If trying to descale to an odd height in interlaced mode.
        :raises CustomValueError:   If invalid shift is passed for progressive mode.
        :return:                    Descaled video node.
        """
        width, height = self._wh_norm(clip, width, height)

        check_correct_subsampling(clip, width, height)

        field_based = FieldBased.from_param_or_video(field_based, clip)

        clip, bits = expect_bits(clip, 32)

        de_base_args = (width, height // (1 + field_based.is_inter))
        kwargs |= dict(border_handling=BorderHandling.from_param(border_handling, self.descale))

        if field_based.is_inter:
            shift_y, shift_x = self._shift_norm(shift, False, self.descale)

            kwargs_tf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[0], shift_x[0]), **kwargs)
            kwargs_bf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[1], shift_x[1]), **kwargs)

            de_kwargs_tf = self.get_descale_args(clip, (shift_y[0], shift_x[0]), *de_base_args, **kwargs_tf)
            de_kwargs_bf = self.get_descale_args(clip, (shift_y[1], shift_x[1]), *de_base_args, **kwargs_bf)

            if height % 2:
                raise CustomIndexError('You can\'t descale to odd resolution when crossconverted!', self.descale)

            field_shift = 0.125 * height / clip.height

            fields = clip.std.SeparateFields(field_based.is_tff)

            interleaved = core.std.Interleave([
                self.descale_function(fields[offset::2], **_norm_props_enums(
                    de_kwargs | dict(src_top=de_kwargs.get('src_top', 0.0) + (field_shift * mult))
                ))
                for offset, mult, de_kwargs in [(0, 1, de_kwargs_tf), (1, -1, de_kwargs_bf)]
            ])

            descaled = interleaved.std.DoubleWeave(field_based.is_tff)[::2]
        else:
            shift = self._shift_norm(shift, True, self.descale)

            kwargs, shift = sample_grid_model.for_src(clip, width, height, shift, **kwargs)

            de_kwargs = self.get_descale_args(clip, shift, *de_base_args, **kwargs)

            descaled = self.descale_function(clip, **_norm_props_enums(de_kwargs))

        return depth(descaled, bits)

    @overload
    def _shift_norm(
        self,
        shift: ShiftT,
        assume_progressive: Literal[True] = ...,
        func: FuncExceptT | None = None
        ) -> tuple[TopShift, LeftShift]:
        ...

    @overload
    def _shift_norm(
        self,
        shift: ShiftT,
        assume_progressive: Literal[False] = ...,
        func: FuncExceptT | None = None
        ) -> tuple[
            tuple[TopFieldTopShift, BotFieldTopShift],
            tuple[TopFieldLeftShift, BotFieldLeftShift]
        ]:
        ...

    def _shift_norm(
        self,
        shift: ShiftT,
        assume_progressive: bool = True,
        func: FuncExceptT | None = None
        ) -> Any:
        """
        Normalize shift values depending on field-based status.

        :param shift:               Shift values (single or per-field).
        :param assume_progressive:  Whether to assume the input is progressive.
        :param func:                Function returned for custom error handling.
        :raises CustomValueError:   If per-field shift is used in progressive mode.
        :return:                    Normalized shift values.
        """
        if assume_progressive:
            if any(isinstance(sh, tuple) for sh in shift):
                raise CustomValueError("You can't descale per-field when the input is progressive!", func, shift)
        else:
            shift_y, shift_x = tuple[tuple[float, float], ...](
                sh if isinstance(sh, tuple) else (sh, sh) for sh in shift
            )
            shift = shift_y, shift_x

        return shift

    @inject_kwargs_params
    def get_descale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Construct the argument dictionary used for descaling.

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param width:   Target width for descaling.
        :param height:  Target height for descaling.
        :param funcs:   Optional callables used to filter kwargs.
        :param kwargs:  Extra keyword arguments to merge.
        :return:        Combined keyword argument dictionary.
        """
        return (
            dict(
                src_top=shift[0],
                src_left=shift[1]
            )
            | self.get_clean_kwargs(*funcs)
            | dict(width=width, height=height)
            | kwargs
        )

    def get_implemented_funcs(self) -> tuple[Callable[..., Any], ...]:
        """
        Return a tuple of implemented descaler methods.

        :return: Tuple containing the `descale` method.
        """
        return (self.descale, )


class Resampler(BaseScaler):
    """
    Abstract resampling interface.

    Subclasses must define the `resample_function` used to perform the resampling.
    """

    _err_class = UnknownResamplerError

    resample_function: Callable[..., ConstantFormatVideoNode]
    """Resample function called internally when performing resampling operations."""

    @inject_self.cached
    @inject_kwargs_params
    def resample(
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None = None, matrix_in: MatrixT | None = None, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Resample a video clip to the given format.

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
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None, matrix_in: MatrixT | None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Construct the argument dictionary used for resampling.

        :param clip:        The source clip.
        :param format:      The target video format, which can either be:
                                - an integer format ID,
                                - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                                - or a source from which a valid `VideoFormat` can be extracted.
        :param matrix:      The matrix for color transformation.
        :param matrix_in:   The input matrix for color transformation.
        :param funcs:       Optional functions used to filter or modify the keyword arguments.
        :param kwargs:      Additional keyword arguments for resampling.
        :return:            A dictionary containing the resampling arguments.
        """
        return (
            dict(
                format=get_video_format(format).id,
                matrix=Matrix.from_param(matrix),
                matrix_in=Matrix.from_param(matrix_in)
            )
            | self.get_clean_kwargs(*funcs)
            | kwargs
        )

    def get_implemented_funcs(self) -> tuple[Callable[..., Any], ...]:
        """
        Return the implemented resampling methods.

        :return: A tuple containing the `resample` method.
        """
        return (self.resample, )


if TYPE_CHECKING:
    class _kernel_shift_injected_self_func(Protocol[T_co]):  # pyright: ignore[reportInvalidTypeVarUse]
        @overload
        @staticmethod
        def __call__(
            clip: vs.VideoNode, shift: tuple[TopShift, LeftShift], /, **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift:                   A (top, left) tuple values for shift.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """

        @overload
        @staticmethod
        def __call__(
            clip: vs.VideoNode,
            shift_top: float | list[float],
            shift_left: float | list[float],
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift_top:               Vertical shift or list of Vertical shifts.
            :param shift_left:              Horizontal shift or list of horizontal shifts.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """

        @overload
        @staticmethod
        def __call__(
            self: T_co, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift], /, **kwargs: Any  # type: ignore[misc]
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift:                   A (top, left) tuple values for shift.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """

        @overload
        @staticmethod
        def __call__(
            self: T_co,  # type: ignore[misc]
            clip: vs.VideoNode,
            shift_top: float | list[float],
            shift_left: float | list[float],
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift_top:               Vertical shift or list of Vertical shifts.
            :param shift_left:              Horizontal shift or list of horizontal shifts.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """

        @overload
        @staticmethod
        def __call__(
            cls: type[T_co],  # pyright: ignore
            clip: vs.VideoNode,
            shift: tuple[TopShift, LeftShift],
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift:                   A (top, left) tuple values for shift.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """

        @overload
        @staticmethod
        def __call__(
            cls: type[T_co],  # pyright: ignore
            clip: vs.VideoNode,
            shift_top: float | list[float],
            shift_left: float | list[float],
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Apply a subpixel shift to the clip using the kernel's scaling logic.

            :param clip:                    The source clip.
            :param shift_top:               Vertical shift or list of Vertical shifts.
            :param shift_left:              Horizontal shift or list of horizontal shifts.
            :param kwargs:                  Additional arguments passed to the internal `scale` call.

            :return:                        A new clip with the applied shift.
            :raises VariableFormatError:    If the input clip has variable format.
            """


    class _inject_self_cached_shift(inject_self.cached[T_co, ..., ConstantFormatVideoNode]):
        def __get__(self, class_obj: Any, class_type: Any) -> _kernel_shift_injected_self_func[T_co]:
            ...
else:
    _inject_self_cached_shift = inject_self.cached


class Kernel(Scaler, Descaler, Resampler):
    """
    Abstract kernel interface combining scaling, descaling, resampling, and shifting functionality.

    Subclasses are expected to implement the actual transformation logic by overriding the methods or
    providing the respective `*_function` callables (`scale_function`, `descale_function`, `resample_function`).

    This class is abstract and should not be used directly.
    """

    _err_class = UnknownKernelError  # type: ignore[assignment]

    @_inject_self_cached_shift
    @inject_kwargs_params
    def shift(
        self,
        clip: vs.VideoNode,
        shifts_or_top: float | tuple[float, float] | list[float],
        shift_left: float | list[float] | None = None,
        /,
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Apply a subpixel shift to the clip using the kernel's scaling logic.

        If a single float or tuple is provided, it is used uniformly.
        If a list is given, the shift is applied per plane.

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

        if isinstance(shifts_or_top, float) and isinstance(shift_left, (float, NoneType)):
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
                    self.shift, (shifts_top, shifts_left)
                )

            return _shift(clip, (shifts_top[0], shifts_left[0]))

        shifted_planes = [
            plane if top == left == 0 else _shift(plane, (top, left))
            for plane, top, left in zip(split(clip), shifts_top, shifts_left)
        ]

        return core.std.ShufflePlanes(shifted_planes, [0, 0, 0], clip.format.color_family)

    @overload
    @classmethod
    def from_param(
        cls: type[Kernel], kernel: KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> type[Kernel]:
        ...

    @overload
    @classmethod
    def from_param(
        cls: type[Kernel], kernel: ScalerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> type[Scaler]:
        ...

    @overload
    @classmethod
    def from_param(
        cls: type[Kernel], kernel: DescalerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> type[Descaler]:
        ...

    @overload
    @classmethod
    def from_param(
        cls: type[Kernel], kernel: ResamplerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> type[Resampler]:
        ...

    @classmethod
    def from_param(
        cls, kernel: str | type[BaseScaler] | BaseScaler | None = None, /, func_except: FuncExceptT | None = None
    ) -> type[BaseScaler]:
        """
        Resolve and return a kernel class from a string name, class type, or instance.

        :param kernel:              Kernel identifier as a string, class type, or instance. If None, defaults to the current class.
        :param func_except:         Function returned for custom error handling.

        :return:                    The resolved kernel class.
        :raises UnknownKernelError: If the kernel could not be identified.
        """
        return _base_from_param(
            cls, Kernel, kernel, UnknownKernelError, abstract_kernels, func_except
        )

    @overload
    @classmethod
    def ensure_obj(
        cls: type[Kernel], kernel: KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> Kernel:
        ...

    @overload
    @classmethod
    def ensure_obj(
        cls: type[Kernel], kernel: ScalerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> Scaler:
        ...

    @overload
    @classmethod
    def ensure_obj(
        cls: type[Kernel], kernel: DescalerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> Descaler:
        ...

    @overload
    @classmethod
    def ensure_obj(
        cls: type[Kernel], kernel: ResamplerT | KernelT | None = ..., /, func_except: FuncExceptT | None = None
    ) -> Resampler:
        ...

    @classmethod
    def ensure_obj(
        cls: type[Kernel], kernel: str | type[BaseScaler] | BaseScaler | None = None, /,
        func_except: FuncExceptT | None = None
    ) -> BaseScaler:
        """
        Ensure that the given kernel input is returned as a kernel instance.

        :param kernel:              Kernel name, class, or instance. Defaults to current class if None.
        :param func_except:         Function returned for custom error handling.

        :return:                    The resolved and instantiated kernel.
        :raises UnknownKernelError: If the kernel is unknown or cannot be instantiated.
        """
        return _base_ensure_obj(
            cls, Kernel, kernel, UnknownKernelError, abstract_kernels, func_except
        )

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> KwargsT:
        """
        Generate a base set of parameters to pass for scaling, descaling, or resampling.

        :param is_descale:  Whether this is for a descale operation.
        :param clip:        The source clip.
        :param width:       Target width.
        :param height:      Target height.
        :param kwargs:      Additional keyword arguments to include.

        :return:            Dictionary of combined parameters.
        """
        return dict(width=width, height=height) | kwargs

    @inject_kwargs_params
    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Generate and normalize argument dictionary for a scale operation.

        :param clip:    The source clip.
        :param shift:   Vertical and horizontal shift to apply.
        :param width:   Target width.
        :param height:  Target height.
        :param funcs:   Callable functions whose keyword arguments will be cleaned and included.
        :param kwargs:  Additional arguments to pass to the scale function.

        :return:        Dictionary of keyword arguments for the scale function.
        """
        return (
            dict(src_top=shift[0], src_left=shift[1])
            | self.get_clean_kwargs(*funcs)
            | self.get_params_args(False, clip, width, height, **kwargs)
        )

    @inject_kwargs_params
    def get_descale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Generate and normalize argument dictionary for a descale operation.

        :param clip:    The source clip.
        :param shift:   Vertical and horizontal shift to apply.
        :param width:   Target width.
        :param height:  Target height.
        :param funcs:   Callable functions whose keyword arguments will be cleaned and included.
        :param kwargs:  Additional arguments to pass to the descale function.

        :return:        Dictionary of keyword arguments for the descale function.
        """
        return (
            dict(src_top=shift[0], src_left=shift[1])
            | self.get_clean_kwargs(*funcs)
            | self.get_params_args(True, clip, width, height, **kwargs)
        )

    @inject_kwargs_params
    def get_resample_args(
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None, matrix_in: MatrixT | None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
        """
        Generate and normalize argument dictionary for a resample operation.

        :param clip:        The source clip.
        :param format:      The target video format, which can either be:
                                - an integer format ID,
                                - a `vs.PresetVideoFormat` or `vs.VideoFormat`,
                                - or a source from which a valid `VideoFormat` can be extracted.
        :param matrix:      Target color matrix.
        :param matrix_in:   Source color matrix.
        :param funcs:       Callable functions whose keyword arguments will be cleaned and included.
        :param kwargs:      Additional arguments to pass to the resample function.

        :return:            Dictionary of keyword arguments for the resample function.
        """
        return (
            dict(
                format=get_video_format(format).id,
                matrix=Matrix.from_param(matrix),
                matrix_in=Matrix.from_param(matrix_in)
            )
            | self.get_clean_kwargs(*funcs)
            | self.get_params_args(False, clip, **kwargs)
        )

    def get_implemented_funcs(self) -> tuple[Callable[..., Any], ...]:
        """
        Return the implemented Kernel methods.

        :return: A tuple containing the `shift` method.
        """
        return (self.shift, )


ScalerT = Union[str, type[Scaler], Scaler]
"""Type alias for anything that can resolve to a Scaler.

This includes:
- A string identifier (typically used for dynamic resolution).
- A class type subclassing `Scaler`.
- An instance of a `Scaler`.
"""

DescalerT = Union[str, type[Descaler], Descaler]
"""Type alias for anything that can resolve to a Descaler.

This includes:
- A string identifier.
- A class type subclassing `Descaler`.
- An instance of a `Descaler`.
"""

ResamplerT = Union[str, type[Resampler], Resampler]
"""Type alias for anything that can resolve to a Resampler.

This includes:
- A string identifier.
- A class type subclassing `Resampler`.
- An instance of a `Resampler`.
"""

KernelT = Union[str, type[Kernel], Kernel]
"""Type alias for anything that can resolve to a Kernel.

This includes:
- A string identifier.
- A class type subclassing `Kernel`.
- An instance of a `Kernel`.
"""
