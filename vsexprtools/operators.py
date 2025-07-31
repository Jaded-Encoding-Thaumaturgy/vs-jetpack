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
    def __call__(self, x: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, self))  # pyright: ignore[reportArgumentType]


class BinaryBaseOperator(BaseOperator):
    def __call__(self, x: ExprOtherT, y: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, y, self))  # pyright: ignore[reportArgumentType]


class TernaryBaseOperator(BaseOperator):
    def __call__(self, x: ExprOtherT, y: ExprOtherT, z: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, y, z, self))  # pyright: ignore[reportArgumentType]


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


@dataclass
class TernaryIfOperator(TernaryOperator["ExprOtherT", "ExprOtherT"]):
    def __call__(self, cond: ExprOtherT, if_true: ExprOtherT, if_false: ExprOtherT) -> ComputedVar:
        return super().__call__(cond, if_true, if_false)


@dataclass
class TernaryCompOperator(TernaryBaseOperator):
    function: Callable[[SuppRC, SuppRC, SuppRC], SuppRC]


@dataclass
class TernaryClampOperator(TernaryCompOperator):
    def __call__(self, x: ExprOtherT, min: ExprOtherT, max: ExprOtherT) -> ComputedVar:
        return super().__call__(x, min, max)


class TernaryPixelAccessOperator(Generic[T], TernaryBaseOperator):
    char: str
    x: T
    y: T

    def __call__(self, char: str, x: T, y: T) -> ComputedVar:  # type: ignore
        from .variables import ComputedVar

        self.set_vars(char, x, y)
        return ComputedVar([copy(self)])  # pyright: ignore[reportArgumentType]

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
    EXP = UnaryMathOperator(ExprOp.EXP, math.exp)
    """Exponential function (e^x)."""

    LOG = UnaryMathOperator(ExprOp.LOG, math.log)
    """Natural logarithm of x."""

    SQRT = UnaryMathOperator(ExprOp.SQRT, math.sqrt)
    """Square root of x."""

    SIN = UnaryMathOperator(ExprOp.SIN, math.sin)
    """Sine (radians) of x."""

    COS = UnaryMathOperator(ExprOp.COS, math.cos)
    """Cosine (radians) of x."""

    ABS = UnaryMathOperator[SupportsAbs[SupportsIndex], SupportsIndex](ExprOp.ABS, abs)
    """Absolute value of x."""

    NOT = UnaryBoolOperator(ExprOp.NOT, op.not_)
    """Logical NOT of x."""

    DUP = BaseOperator(ExprOp.DUP)
    """Duplicate the top of the stack."""

    DUPN = BaseOperator(ExprOp.DUPN)
    """Duplicates the nth element from the top of the stack."""

    TRUNC = UnaryMathOperator[SupportsTrunc, int](ExprOp.TRUNC, math.trunc)
    """Truncate x to integer (toward zero)."""

    ROUND = UnaryMathOperator[SupportsRound[int], int](ExprOp.ROUND, lambda x: round(x))
    """Round x to nearest integer."""

    FLOOR = UnaryMathOperator[SupportsFloatOrIndex, int](ExprOp.FLOOR, math.floor)
    """Round down x to nearest integer."""

    # DROP / DROPN / SORTN / VAR_STORE / VAR_PUSH ??

    # 2 Arguments
    MAX = BinaryMathOperator[SuppRC, SuppRC](ExprOp.MAX, max)
    """Calculates the maximum of x and y."""

    MIN = BinaryMathOperator[SuppRC, SuppRC](ExprOp.MIN, min)
    """Calculates the minimum of x and y."""

    ADD = BinaryOperator(ExprOp.ADD, op.add)
    """Performs addition of two elements (x + y)."""

    SUB = BinaryOperator(ExprOp.SUB, op.sub)
    """Performs subtraction of two elements (x - y)."""

    MUL = BinaryOperator(ExprOp.MUL, op.mul)
    """Performs multiplication of two elements (x * y)."""

    DIV = BinaryOperator(ExprOp.DIV, op.truediv)
    """Performs division of two elements (x / y)."""

    POW = BinaryOperator(ExprOp.POW, op.pow)
    """Performs x to the power of y (x ** y)."""

    GT = BinaryBoolOperator(ExprOp.GT, op.gt)
    """Performs x > y."""

    LT = BinaryBoolOperator(ExprOp.LT, op.lt)
    """Performs x < y."""

    EQ = BinaryBoolOperator(ExprOp.EQ, op.eq)
    """Performs x == y."""

    GTE = BinaryBoolOperator(ExprOp.GTE, op.ge)
    """Performs x >= y."""

    LTE = BinaryBoolOperator(ExprOp.LTE, op.le)
    """Performs x <= y."""

    AND = BinaryBoolOperator(ExprOp.AND, op.and_)
    """Performs a logical AND."""

    OR = BinaryBoolOperator(ExprOp.OR, op.or_)
    """Performs a logical OR."""

    XOR = BinaryOperator(ExprOp.XOR, op.xor)
    """Performs a logical XOR."""

    SWAP = BinaryBaseOperator(ExprOp.SWAP)
    """Swaps the top two elements of the stack."""

    SWAPN = BinaryBaseOperator(ExprOp.SWAPN)
    """Swaps the top element with the nth element from the top of the stack."""

    MOD = BinaryOperator(ExprOp.MOD, op.mod)
    """Performs x % y."""

    # 3 Arguments
    TERN = TernaryIfOperator(ExprOp.TERN, lambda x, y, z: (x if z else y))
    """Ternary operator (if cond then if_true else if_false)."""

    CLAMP = TernaryClampOperator(ExprOp.CLAMP, lambda x, y, z: max(y, min(x, z)))
    """Clamps a value between a min and a max."""

    # Aliases
    IF = TERN
    """Ternary operator (if cond then if_true else if_false)."""

    # Special Operators
    REL_PIX = TernaryPixelAccessOperator[int](ExprOp.REL_PIX)
    """Relative pixel access."""

    ABS_PIX = TernaryPixelAccessOperator[Union[int, "ExprVar"]](ExprOp.ABS_PIX)
    """Absolute pixel access."""

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
