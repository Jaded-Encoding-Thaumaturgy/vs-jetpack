from __future__ import annotations

from abc import ABC, ABCMeta
from functools import lru_cache
from inspect import Signature
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Sequence, TypeVar, Union, cast, overload

from typing_extensions import Self

from jetpytools import inject_kwargs_params

from vstools import (
    ConstantFormatVideoNode, CustomIndexError, CustomNotImplementedError, CustomRuntimeError, CustomValueError,
    FieldBased, FuncExceptT, HoldsVideoFormatT, KwargsT, Matrix, MatrixT, VideoFormatT, VideoNodeT,
    check_correct_subsampling, check_variable_format, check_variable_resolution, core, depth, expect_bits, fallback,
    get_subclasses, get_video_format, inject_self, split, vs, vs_object
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
partial_abstract_kernels: list[BaseScalerMeta] = []


class BaseScalerMeta(ABCMeta):
    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        abstract: bool = False,
        partial_abstract: bool = False,
        **kwargs: Any
    ) -> BaseScalerMeta:
        """
        Adding `abstract=True` to the class declaration will force the class to be abstract and not be instantiated.
        Adding `partial_abstract=True` to the class declaration will allow the class to be instantiated
        but will likely raise an error if the kernel radius is not implemented.
        """

        obj = super().__new__(mcls, name, bases, namespace, **kwargs)

        if abstract:
            abstract_kernels.append(obj)
        elif partial_abstract:
            partial_abstract_kernels.append(obj)

        return obj


class BaseScaler(vs_object, ABC, metaclass=BaseScalerMeta, abstract=True):
    """
    Base abstract scaling interface.
    """

    kwargs: KwargsT
    """Arguments passed to the internal scale function"""

    _static_kernel_radius: ClassVar[int]
    _err_class: ClassVar[type[CustomValueError]]

    if not TYPE_CHECKING:
        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            return _check_kernel_radius(cls, super().__new__(cls))

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.pretty_string

    @staticmethod
    def _wh_norm(clip: vs.VideoNode, width: int | None = None, height: int | None = None) -> tuple[int, int]:
        return (fallback(width, clip.width), fallback(height, clip.height))

    @classmethod
    def from_param(
        cls: type[BaseScalerT], scaler: str | type[BaseScalerT] | BaseScalerT | None = None, /,
        func_except: FuncExceptT | None = None
    ) -> type[BaseScalerT]:
        return _base_from_param(
            cls, (mro := cls.mro())[mro.index(BaseScaler) - 1], scaler, cls._err_class, [], func_except
        )

    @classmethod
    def ensure_obj(
        cls: type[BaseScalerT], scaler: str | type[BaseScalerT] | BaseScalerT | None = None, /,
        func_except: FuncExceptT | None = None
    ) -> BaseScalerT:
        return _base_ensure_obj(
            cls, (mro := cls.mro())[mro.index(BaseScaler) - 1], scaler, cls._err_class, [], func_except
        )

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        if hasattr(self, '_static_kernel_radius'):
            return ceil(self._static_kernel_radius)
        raise CustomNotImplementedError('kernel_radius is not implemented!', self.__class__)

    def get_clean_kwargs(self, *funcs: Callable[..., Any] | None) -> KwargsT:
        return _clean_self_kwargs(funcs, self)

    def _pretty_string(self, **attrs: Any) -> str:
        return (
            f"{self.__class__.__name__}"
            + '(' + ', '.join(f'{k}={v}' for k, v in (attrs | self.kwargs).items()) + ')'
        )

    @inject_self.cached.property
    def pretty_string(self) -> str:
        return self._pretty_string()


BaseScalerT = TypeVar('BaseScalerT', bound=BaseScaler)


class Scaler(BaseScaler):
    """
    Abstract scaling interface.
    """

    _err_class = UnknownScalerError

    scale_function: Callable[..., vs.VideoNode]
    """Scale function called internally when scaling"""

    @inject_self.cached
    @inject_kwargs_params
    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> vs.VideoNode:
        width, height = Scaler._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

        return self.scale_function(
            clip, **_norm_props_enums(self.get_scale_args(clip, shift, width, height, **kwargs))
        )

    @inject_self.cached
    def multi(
        self, clip: vs.VideoNode, multi: float = 2, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable_resolution(clip, self.multi)

        dst_width, dst_height = ceil(clip.width * multi), ceil(clip.height * multi)

        if max(dst_width, dst_height) <= 0.0:
            raise CustomValueError(
                'Multiplying the resolution by "multi" must result in a positive resolution!', self.multi, multi
            )

        return self.scale(clip, dst_width, dst_height, shift, **kwargs)

    @inject_kwargs_params
    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
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
        return (self.scale, self.multi)


class Descaler(BaseScaler):
    """
    Abstract descaling interface.
    """

    _err_class = UnknownDescalerError

    descale_function: Callable[..., ConstantFormatVideoNode]
    """Descale function called internally when descaling"""

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
        return (self.descale, )


class Resampler(BaseScaler):
    """
    Abstract resampling interface.
    """

    _err_class = UnknownResamplerError

    resample_function: Callable[..., ConstantFormatVideoNode]
    """Resample function called internally when resampling"""

    @inject_self.cached
    @inject_kwargs_params
    def resample(
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None = None, matrix_in: MatrixT | None = None, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return self.resample_function(
            clip, **_norm_props_enums(self.get_resample_args(clip, format, matrix, matrix_in, **kwargs))
        )

    def get_resample_args(
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None, matrix_in: MatrixT | None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
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
        return (self.resample, )


class Kernel(Scaler, Descaler, Resampler):
    """
    Abstract kernel interface.
    """

    _err_class = UnknownKernelError  # type: ignore[assignment]

    if TYPE_CHECKING:
        @overload  # type: ignore[misc]
        @inject_kwargs_params
        @staticmethod
        def shift(
            clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0), /, **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

        @overload
        @inject_kwargs_params
        def shift(
            self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0), /, **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

        @overload
        @inject_kwargs_params
        @staticmethod
        def shift(
            clip: vs.VideoNode,
            shift_top: float | list[float] = 0.0,
            shift_left: float | list[float] = 0.0,
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

        @overload
        @inject_kwargs_params
        def shift(
            self,
            clip: vs.VideoNode,
            shift_top: float | list[float] = 0.0,
            shift_left: float | list[float] = 0.0,
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

        @overload
        @inject_kwargs_params
        def shift(self, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
            ...

        @inject_kwargs_params
        def shift(*args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
            ...
    else:
        @inject_self.cached
        @inject_kwargs_params
        def shift(
            self,
            clip: vs.VideoNode,
            shifts_or_top: float | tuple[float, float] | list[float] | None = None,
            shift_left: float | list[float] | None = None,
            /,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            assert check_variable_format(clip, self.shift)

            n_planes = clip.format.num_planes

            def _shift(src: VideoNodeT, shift: tuple[TopShift, LeftShift] = (0, 0)) -> VideoNodeT:
                return self.scale(src, src.width, src.height, shift, **kwargs)

            if not shifts_or_top and not shift_left:
                return _shift(clip)

            if isinstance(shifts_or_top, tuple):
                return _shift(clip, shifts_or_top)

            if isinstance(shifts_or_top, float) and isinstance(shift_left, float):
                return _shift(clip, (shifts_or_top, shift_left))

            if shifts_or_top is None:
                shifts_or_top = 0.0
            if shift_left is None:
                shift_left = 0.0

            shifts_top = shifts_or_top if isinstance(shifts_or_top, list) else [shifts_or_top]
            shifts_left = shift_left if isinstance(shift_left, list) else [shift_left]

            if not shifts_top:
                shifts_top = [0.0] * n_planes
            elif (ltop := len(shifts_top)) > n_planes:
                shifts_top = shifts_top[:n_planes]
            else:
                shifts_top += shifts_top[-1:] * (n_planes - ltop)

            if not shifts_left:
                shifts_left = [0.0] * n_planes
            elif (lleft := len(shifts_left)) > n_planes:
                shifts_left = shifts_left[:n_planes]
            else:
                shifts_left += shifts_left[-1:] * (n_planes - lleft)

            if len(set(shifts_top)) == len(set(shifts_left)) == 1 or n_planes == 1:
                return _shift(clip, (shifts_top[0], shifts_left[0]))

            planes = split(clip)

            shifted_planes = [
                plane if top == left == 0 else _shift(plane, (top, left))
                for plane, top, left in zip(planes, shifts_top, shifts_left)
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
        return _base_ensure_obj(
            cls, Kernel, kernel, UnknownKernelError, abstract_kernels, func_except
        )

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> KwargsT:
        return dict(width=width, height=height) | kwargs

    @inject_kwargs_params
    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> KwargsT:
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
        return (self.shift, )


ScalerT = Union[str, type[Scaler], Scaler]
DescalerT = Union[str, type[Descaler], Descaler]
ResamplerT = Union[str, type[Resampler], Resampler]
KernelT = Union[str, type[Kernel], Kernel]
