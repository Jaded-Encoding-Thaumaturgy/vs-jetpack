from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NoReturn,
    SupportsIndex,
    TypeAlias,
    cast,
    overload,
)

from typing_extensions import Self

from vstools import (
    ByteData,
    ColorRangeT,
    get_depth,
    get_lowest_value,
    get_neutral_value,
    get_peak_value,
    get_plane_sizes,
    scale_value,
    vs,
)

from .operators import BaseOperator, ExprOperators

if TYPE_CHECKING:
    from .manager import inline_expr


__all__ = ["ClipPropsVar", "ClipVar", "ComplexVar", "ComputedVar", "ExprOtherT", "ExprVar", "LiteralVar", "resolverT"]


class ExprVar(int):
    parent_expr: inline_expr | None

    def __new__(cls, x: ByteData, __parent_expr: inline_expr | None = None, /, *args: Any, **kwargs: Any) -> Self:
        return super().__new__(cls, 0)

    def __add__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.ADD(self, other)

    def __iadd__(self, other: ExprOtherT) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.ADD(self, other)

    def __radd__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.ADD(other, self)

    def __sub__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.SUB(self, other)

    def __isub__(self, other: ExprOtherT) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.SUB(self, other)

    def __rsub__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.SUB(other, self)

    def __mul__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.MUL(self, other)

    def __imul__(self, other: ExprOtherT) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.MUL(self, other)

    def __rmul__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.MUL(other, self)

    def __truediv__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.DIV(self, other)

    def __rtruediv__(self, other: ExprOtherT) -> ComputedVar:  # type: ignore
        return ExprOperators.DIV(other, self)

    def __itruediv__(self, other: ExprOtherT) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.DIV(self, other)

    def __floordiv__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.FLOOR(ExprOperators.DIV(self, other))

    def __ifloordiv__(self, other: ExprOtherT) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.FLOOR(ExprOperators.DIV(self, other))

    def __rfloordiv__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.FLOOR(ExprOperators.DIV(other, self))

    def __pow__(self, other: ExprOtherT, module: int | None = None) -> ComputedVar:  # type: ignore
        if module is not None:
            raise NotImplementedError
        return ExprOperators.POW(self, other)

    def __rpow__(self, other: ExprOtherT) -> ComputedVar:  # type: ignore
        return ExprOperators.POW(other, self)

    def __exp__(self) -> ComputedVar:
        return ExprOperators.EXP(self)

    def __log__(self) -> ComputedVar:
        return ExprOperators.LOG(self)

    def __sqrt__(self) -> ComputedVar:
        return ExprOperators.SQRT(self)

    def __round__(self, ndigits: SupportsIndex | None = None) -> ComputedVar:
        if ndigits is not None:
            raise NotImplementedError
        return ExprOperators.ROUND(self)

    def __trunc__(self) -> ComputedVar:
        return ExprOperators.TRUNC(self)

    def __ceil__(self) -> ComputedVar:
        return ExprOperators.FLOOR(ExprOperators.ADD(self, 0.5))

    def __floor__(self) -> ComputedVar:
        return ExprOperators.FLOOR(self)

    def __neg__(self) -> ComputedVar:
        return ExprOperators.MUL(ExprOperators.ABS(self), -1)

    def __pos__(self) -> ComputedVar:
        return ExprOperators.ABS(self)

    def __invert__(self) -> NoReturn:
        raise NotImplementedError

    def __int__(self) -> ComputedVar:
        return ExprOperators.TRUNC(self)

    def __float__(self) -> ComputedVar:
        return ComputedVar([self])

    def __abs__(self) -> ComputedVar:
        return ExprOperators.ABS(self)

    def __mod__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.MOD(self, other)

    def __rmod__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.MOD(other, self)

    def __divmod__(self, _: ExprOtherT) -> NoReturn:
        raise NotImplementedError

    def __rdivmod__(self, _: ExprOtherT) -> NoReturn:
        raise NotImplementedError

    def __lt__(self, other: ExprOtherT) -> ComputedVar:  # type: ignore
        return ExprOperators.LT(self, other)

    def __lte__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.LTE(self, other)

    def __gt__(self, other: ExprOtherT) -> ComputedVar:  # type: ignore
        return ExprOperators.GT(self, other)

    def __gte__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.GTE(self, other)

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __and__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.AND(self, other)

    def __rand__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.AND(self, other)

    def __or__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.OR(self, other)

    def __ror__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.OR(other, self)

    def __xor__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.XOR(self, other)

    def __rxor__(self, other: ExprOtherT) -> ComputedVar:
        return ExprOperators.XOR(self, other)

    def __iter__(self) -> Iterator[ComputedVar]:
        return iter([self])  # type: ignore

    def __getitem__(self, item: Any) -> NoReturn:
        raise RuntimeError("You can only access offsetted pixels on constant clips variables")

    def to_str(self, **kwargs: Any) -> str:
        return str(self)

    def assert_in_context(self) -> Literal[True]:
        if not self.parent_expr:
            return True

        if not self.parent_expr._in_context:
            raise ValueError("You can only access this variable in context!")

        return True

    def as_var(self) -> ComputedVar:
        if isinstance(self, ComputedVar):
            return self
        return ComputedVar([self])


class LiteralVar(ExprVar):
    parent_expr: None

    def __init__(self, value: int | float | str):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


ExprOtherT: TypeAlias = ExprVar | LiteralVar | int | float


class ComputedVar(ExprVar):
    def __init__(self, operations: Iterable[BaseOperator | ExprVar]) -> None:
        self.operations = list(operations)
        self.parent_expr = next(
            (x.parent_expr if isinstance(x, (ComputedVar, ClipVar)) else None for x in self.operations), None
        )

    def to_str(self, **kwargs: Any) -> str:
        return " ".join([x.to_str(**kwargs) for x in self.operations])

    def __str__(self) -> str:
        return " ".join([str(x) for x in self.operations])


resolverT: TypeAlias = Callable[..., LiteralVar]  # noqa


@dataclass
class ComplexVar(LiteralVar):
    value: int | float | str
    resolve: resolverT

    def to_str(self, **kwargs: Any) -> str:
        return str(self.resolve(**kwargs))

    @overload
    @staticmethod
    def resolver() -> Callable[[Callable[..., Any]], resolverT]: ...

    @overload
    @staticmethod
    def resolver(function: Callable[..., Any] | None = None) -> resolverT: ...

    @staticmethod
    def resolver(function: Callable[..., Any] | None = None) -> Callable[[Callable[..., Any]], resolverT] | resolverT:
        if function is None:
            return cast(Callable[[Callable[..., Any]], resolverT], ComplexVar.resolver)

        @wraps(function)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return LiteralVar(function(*args, **kwargs))

        return cast(resolverT, _wrapper)


class ClipPropsVar:
    # Some commonly used props
    PlaneStatsMin: ComputedVar
    PlaneStatsMax: ComputedVar
    PlaneStatsAverage: ComputedVar

    def __init__(self, clip_var: ClipVar) -> None:
        self.clip_var = clip_var

    def __getattribute__(self, name: str) -> ComputedVar:
        clip_var: ClipVar = object.__getattribute__(self, "clip_var")
        return ComputedVar([LiteralVar(f"{clip_var.char}.{name}")])


class ClipVar(ExprVar):
    parent_expr: inline_expr

    def __init__(self, char: str, clip: vs.VideoNode, parent_expr: inline_expr) -> None:
        self.char = char
        self.clip = clip
        self.parent_expr = parent_expr
        self.props = ClipPropsVar(self)

    def __str__(self) -> str:
        return self.char

    # Pixel Access
    _IdxType: TypeAlias = int | ExprVar

    def __getitem__(self, index: _IdxType | tuple[_IdxType, _IdxType] | slice) -> ComputedVar:  # type: ignore
        if isinstance(index, tuple):
            x, y = index
            if isinstance(x, ExprVar) or isinstance(y, ExprVar):
                return ExprOperators.ABS_PIX(self.char, x, y)
            else:
                return ExprOperators.REL_PIX(self.char, x, y)
        elif isinstance(index, slice):
            print(index.start, index.stop, index.step)  # TODO
        else:
            print(str(index))  # TODO

        return ComputedVar([self])

    # Helper properties
    @property
    def peak(self) -> LiteralVar:
        return LiteralVar(get_peak_value(self.clip))

    @property
    def peak_chroma(self) -> LiteralVar:
        return LiteralVar(get_peak_value(self.clip, True))

    @property
    def neutral(self) -> LiteralVar:
        return LiteralVar(get_neutral_value(self.clip))

    @property
    def neutral_chroma(self) -> LiteralVar:
        return LiteralVar(get_neutral_value(self.clip))

    @property
    def lowest(self) -> LiteralVar:
        return LiteralVar(get_lowest_value(self.clip))

    @property
    def lowest_chroma(self) -> LiteralVar:
        return LiteralVar(get_lowest_value(self.clip, True))

    @property
    def width(self) -> LiteralVar:
        return LiteralVar("width")

    @property
    def width_luma(self) -> LiteralVar:
        return LiteralVar(self.clip.width)

    @property
    def width_chroma(self) -> LiteralVar:
        return LiteralVar(get_plane_sizes(self.clip, 1)[0])

    @property
    def height(self) -> LiteralVar:
        return LiteralVar("height")

    @property
    def height_luma(self) -> LiteralVar:
        return LiteralVar(self.clip.height)

    @property
    def height_chroma(self) -> LiteralVar:
        return LiteralVar(get_plane_sizes(self.clip, 1)[1])

    @property
    def depth(self) -> LiteralVar:
        return LiteralVar(get_depth(self.clip))

    # Helper functions
    def scale(
        self,
        value: float,
        input_depth: int = 8,
        range_in: ColorRangeT | None = None,
        range_out: ColorRangeT | None = None,
        scale_offsets: bool = True,
        family: vs.ColorFamily | None = None,
    ) -> ComplexVar:
        @ComplexVar.resolver
        def _resolve(plane: int = 0, **kwargs: Any) -> Any:
            return scale_value(
                value, input_depth, get_depth(self.clip), range_in, range_out, scale_offsets, plane in {1, 2}, family
            )

        return ComplexVar(f"{self.char}.scale({value})", _resolve)
