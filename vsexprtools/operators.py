from __future__ import annotations

import math
import operator as op
from copy import copy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Sequence,
    SupportsAbs,
    SupportsIndex,
    SupportsRound,
    TypeAlias,
    Union,
    cast,
    overload,
)

from vstools import R, SupportsFloatOrIndex, SupportsRichComparison, SupportsTrunc, T

from .exprop import ExprOp

if TYPE_CHECKING:
    from .variables import ComputedVar, ExprOtherT, ExprVar
else:
    ExprOtherT = Any
    ExprVar = Any


__all__ = [
    "BaseOperator",
    "BinaryBaseOperator",
    "BinaryBoolOperator",
    "BinaryMathOperator",
    "BinaryOperator",
    "ExprOperators",
    "TernaryBaseOperator",
    "TernaryCompOperator",
    "TernaryIfOperator",
    "TernaryOperator",
    "TernaryPixelAccessOperator",
    "UnaryBaseOperator",
    "UnaryBoolOperator",
    "UnaryMathOperator",
    "UnaryOperator",
]

SuppRC: TypeAlias = SupportsRichComparison


def _norm_lit(arg: str | ExprOtherT | BaseOperator) -> ExprVar | BaseOperator:
    from .variables import ExprVar, LiteralVar

    if isinstance(arg, (ExprVar, BaseOperator)):
        return arg

    return LiteralVar(arg)


def _normalize_args(*args: str | ExprOtherT | BaseOperator) -> Iterable[ExprVar | BaseOperator]:
    for arg in args:
        yield _norm_lit(arg)


@dataclass
class BaseOperator:
    rpn_name: ExprOp

    def to_str(self, **kwargs: Any) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.rpn_name


class UnaryBaseOperator(BaseOperator):
    def __call__(self, arg0: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(arg0, self))


class BinaryBaseOperator(BaseOperator):
    def __call__(self, arg0: ExprOtherT, arg1: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(arg0, arg1, self))


class TernaryBaseOperator(BaseOperator):
    def __call__(self, arg0: ExprOtherT, arg1: ExprOtherT, arg2: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(arg0, arg1, arg2, self))


@dataclass
class UnaryOperator(Generic[T], UnaryBaseOperator):
    function: Callable[[T], T]


@dataclass
class UnaryMathOperator(Generic[T, R], UnaryBaseOperator):
    function: Callable[[T], R]


@dataclass
class UnaryBoolOperator(UnaryBaseOperator):
    function: Callable[[object], bool]


@dataclass
class BinaryOperator(Generic[T, R], BinaryBaseOperator):
    function: Callable[[T, R], T | R]


@dataclass
class BinaryMathOperator(Generic[T, R], BinaryBaseOperator):
    function: Callable[[T, T], R]


@dataclass
class BinaryBoolOperator(BinaryBaseOperator):
    function: Callable[[Any, Any], bool]


@dataclass
class TernaryOperator(Generic[T, R], TernaryBaseOperator):
    function: Callable[[bool, T, R], T | R]


class TernaryIfOperator(TernaryOperator[ExprOtherT, ExprOtherT]):
    def __call__(self, cond: ExprOtherT, if_true: ExprOtherT, if_false: ExprOtherT) -> ComputedVar:
        return super().__call__(cond, if_true, if_false)


@dataclass
class TernaryCompOperator(TernaryBaseOperator):
    function: Callable[[SuppRC, SuppRC, SuppRC], SuppRC]


class TernaryPixelAccessOperator(Generic[T], TernaryBaseOperator):
    char: str
    x: T
    y: T

    def __call__(self, char: str, x: T, y: T) -> ComputedVar:  # type: ignore
        from .variables import ComputedVar

        self.set_vars(char, x, y)
        return ComputedVar([copy(self)])

    def set_vars(self, char: str, x: T, y: T) -> None:
        self.char = char
        self.x = x
        self.y = y

    def __str__(self) -> str:
        if not hasattr(self, "char"):
            raise ValueError("TernaryPixelAccessOperator: You have to call set_vars!")

        return self.rpn_name.format(char=str(self.char), x=int(self.x), y=int(self.y))  # type: ignore[call-overload]


class ExprOperators:
    # 1 Argument
    @staticmethod
    def EXP(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates e**x for every element in x."""
        return UnaryMathOperator(ExprOp.EXP, math.exp)(x)

    @staticmethod
    def LOG(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the natural logarithm for every element in x."""
        return UnaryMathOperator(ExprOp.LOG, math.log)(x)

    @staticmethod
    def SQRT(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the square root for every element in x."""
        return UnaryMathOperator(ExprOp.SQRT, math.sqrt)(x)

    @staticmethod
    def SIN(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the sine for every element in x."""
        return UnaryMathOperator(ExprOp.SIN, math.sin)(x)

    @staticmethod
    def COS(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the cosine for every element in x."""
        return UnaryMathOperator(ExprOp.COS, math.cos)(x)

    @staticmethod
    def ABS(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the absolute value for every element in x."""
        return UnaryMathOperator[SupportsAbs[SupportsIndex], SupportsIndex](ExprOp.ABS, abs)(x)

    @staticmethod
    def NOT(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical NOT on every element in x."""
        return UnaryBoolOperator(ExprOp.NOT, op.not_)(x)

    @staticmethod
    def DUP(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Duplicates the top of the stack."""
        return UnaryBaseOperator(ExprOp.DUP)(x)

    @staticmethod
    def DUPN(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Duplicates the nth element from the top of the stack."""
        return UnaryBaseOperator(ExprOp.DUPN)(x)

    @staticmethod
    def TRUNC(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Truncates every element in x."""
        return UnaryMathOperator[SupportsTrunc, int](ExprOp.TRUNC, math.trunc)(x)

    @staticmethod
    def ROUND(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Rounds every element in x."""
        return UnaryMathOperator[SupportsRound[int], int](ExprOp.ROUND, lambda val: round(val))(x)

    @staticmethod
    def FLOOR(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Floors every element in x."""
        return UnaryMathOperator[SupportsFloatOrIndex, int](ExprOp.FLOOR, math.floor)(x)

    # 2 Arguments
    @staticmethod
    def MAX(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the maximum of a and b."""
        return BinaryMathOperator[SuppRC, SuppRC](ExprOp.MAX, max)(a, b)

    @staticmethod
    def MIN(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the minimum of a and b."""
        return BinaryMathOperator[SuppRC, SuppRC](ExprOp.MIN, min)(a, b)

    @staticmethod
    def ADD(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs addition of two elements (a + b)."""
        return BinaryOperator(ExprOp.ADD, op.add)(a, b)

    @staticmethod
    def SUB(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs subtraction of two elements (a - b)."""
        return BinaryOperator(ExprOp.SUB, op.sub)(a, b)

    @staticmethod
    def MUL(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs multiplication of two elements (a * b)."""
        return BinaryOperator(ExprOp.MUL, op.mul)(a, b)

    @staticmethod
    def DIV(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs division of two elements (a / b)."""
        return BinaryOperator(ExprOp.DIV, op.truediv)(a, b)

    @staticmethod
    def POW(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a to the power of b (a ** b)."""
        return BinaryOperator(ExprOp.POW, op.pow)(a, b)

    @staticmethod
    def GT(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a > b."""
        return BinaryBoolOperator(ExprOp.GT, op.gt)(a, b)

    @staticmethod
    def LT(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a < b."""
        return BinaryBoolOperator(ExprOp.LT, op.lt)(a, b)

    @staticmethod
    def EQ(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a == b."""
        return BinaryBoolOperator(ExprOp.EQ, op.eq)(a, b)

    @staticmethod
    def GTE(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a >= b."""
        return BinaryBoolOperator(ExprOp.GTE, op.ge)(a, b)

    @staticmethod
    def LTE(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a <= b."""
        return BinaryBoolOperator(ExprOp.LTE, op.le)(a, b)

    @staticmethod
    def AND(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical AND."""
        return BinaryBoolOperator(ExprOp.AND, op.and_)(a, b)

    @staticmethod
    def OR(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical OR."""
        return BinaryBoolOperator(ExprOp.OR, op.or_)(a, b)

    @staticmethod
    def XOR(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical XOR."""
        return BinaryOperator(ExprOp.XOR, op.xor)(a, b)

    @staticmethod
    def SWAP(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Swaps the top two elements of the stack."""
        return BinaryBaseOperator(ExprOp.SWAP)(a, b)

    @staticmethod
    def SWAPN(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Swaps the top element with the nth element from the top of the stack."""
        return BinaryBaseOperator(ExprOp.SWAPN)(a, b)

    @staticmethod
    def MOD(a: ExprOtherT, b: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a % b."""
        return BinaryOperator(ExprOp.MOD, op.mod)(a, b)

    # 3 Arguments
    @staticmethod
    def TERN(cond: ExprOtherT, if_true: ExprOtherT, if_false: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Ternary operator (if cond then if_true else if_false)."""
        return TernaryIfOperator(ExprOp.TERN, lambda x, y, z: (x if z else y))(cond, if_true, if_false)

    @staticmethod
    def CLAMP(value: ExprOtherT, min_val: ExprOtherT, max_val: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Clamps a value between a min and a max."""
        return TernaryCompOperator(ExprOp.CLAMP, lambda x, y, z: max(y, min(x, z)))(value, min_val, max_val)

    # Aliases
    IF = TERN

    # Special Operators
    @staticmethod
    def REL_PIX(char: str, x: int, y: int) -> ComputedVar:  # noqa: N802
        """Relative pixel access."""
        return TernaryPixelAccessOperator[int](ExprOp.REL_PIX)(char, x, y)

    @staticmethod
    def ABS_PIX(char: str, x: Union[int, "ExprVar"], y: Union[int, "ExprVar"]) -> ComputedVar:  # noqa: N802
        """Absolute pixel access."""
        return TernaryPixelAccessOperator[Union[int, "ExprVar"]](ExprOp.ABS_PIX)(char, x, y)

    # Helper Functions

    @overload
    @classmethod
    def as_var(cls, arg0: ExprOtherT) -> ComputedVar:
        pass

    @overload
    @classmethod
    def as_var(cls, arg0: Sequence[ExprOtherT]) -> list[ComputedVar]:
        pass

    @classmethod
    def as_var(cls, arg0: ExprOtherT | Sequence[ExprOtherT]) -> ComputedVar | list[ComputedVar]:
        from .variables import ComputedVar

        if isinstance(arg0, Sequence):
            return cast(list[ComputedVar], list(arg0))
        return cast(ComputedVar, arg0)
