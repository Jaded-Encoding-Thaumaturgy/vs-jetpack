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
    EXP = UnaryMathOperator(ExprOp.EXP, math.exp)

    LOG = UnaryMathOperator(ExprOp.LOG, math.log)

    SQRT = UnaryMathOperator(ExprOp.SQRT, math.sqrt)

    SIN = UnaryMathOperator(ExprOp.SIN, math.sin)

    COS = UnaryMathOperator(ExprOp.COS, math.cos)

    ABS = UnaryMathOperator[SupportsAbs[SupportsIndex], SupportsIndex](ExprOp.ABS, abs)

    NOT = UnaryBoolOperator(ExprOp.NOT, op.not_)

    DUP = BaseOperator(ExprOp.DUP)

    DUPN = BaseOperator(ExprOp.DUPN)

    TRUNC = UnaryMathOperator[SupportsTrunc, int](ExprOp.TRUNC, math.trunc)

    ROUND = UnaryMathOperator[SupportsRound[int], int](ExprOp.ROUND, lambda x: round(x))

    FLOOR = UnaryMathOperator[SupportsFloatOrIndex, int](ExprOp.FLOOR, math.floor)

    # 2 Arguments
    MAX = BinaryMathOperator[SuppRC, SuppRC](ExprOp.MAX, max)

    MIN = BinaryMathOperator[SuppRC, SuppRC](ExprOp.MIN, min)

    ADD = BinaryOperator(ExprOp.ADD, op.add)

    SUB = BinaryOperator(ExprOp.SUB, op.sub)

    MUL = BinaryOperator(ExprOp.MUL, op.mul)

    DIV = BinaryOperator(ExprOp.DIV, op.truediv)

    POW = BinaryOperator(ExprOp.POW, op.pow)

    GT = BinaryBoolOperator(ExprOp.GT, op.gt)

    LT = BinaryBoolOperator(ExprOp.LT, op.lt)

    EQ = BinaryBoolOperator(ExprOp.EQ, op.eq)

    GTE = BinaryBoolOperator(ExprOp.GTE, op.ge)

    LTE = BinaryBoolOperator(ExprOp.LTE, op.le)

    AND = BinaryBoolOperator(ExprOp.AND, op.and_)

    OR = BinaryBoolOperator(ExprOp.OR, op.or_)

    XOR = BinaryOperator(ExprOp.XOR, op.xor)

    SWAP = BinaryBaseOperator(ExprOp.SWAP)

    SWAPN = BinaryBaseOperator(ExprOp.SWAPN)

    MOD = BinaryOperator(ExprOp.MOD, op.mod)

    # 3 Arguments
    TERN = TernaryIfOperator(ExprOp.TERN, lambda x, y, z: (x if z else y))

    CLAMP = TernaryCompOperator(ExprOp.CLAMP, lambda x, y, z: max(y, min(x, z)))

    # Aliases
    IF = TERN

    # Special Operators
    REL_PIX = TernaryPixelAccessOperator[int](ExprOp.REL_PIX)
    ABS_PIX = TernaryPixelAccessOperator[Union[int, "ExprVar"]](ExprOp.ABS_PIX)

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
