from __future__ import annotations

import math
import operator as op
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NoReturn,
    SupportsAbs,
    SupportsIndex,
    SupportsRound,
)

from jetpytools import R, Singleton, SupportsFloatOrIndex, SupportsRichComparison, SupportsTrunc, T

from ..exprop import ExprOp

if TYPE_CHECKING:
    from .variables import ComputedVar, ExprVarLike

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


@dataclass
class BaseOperator:
    """Base class for all operators used in RPN (Reverse Polish Notation) expressions."""

    rpn_name: ExprOp
    """The RPN name of the operator."""

    def __str__(self) -> NoReturn:
        raise NotImplementedError

    def __call__(self, *args: Any) -> ComputedVar:
        from .variables import ComputedVar, ExprVar, LiteralVar

        return ComputedVar(arg if isinstance(arg, ExprVar) else LiteralVar(arg) for arg in (*args, self.rpn_name))


class UnaryBaseOperator(BaseOperator):
    """Base class for all unary (single-operand) operators."""


class BinaryBaseOperator(BaseOperator):
    """Base class for all binary (two-operand) operators."""


class TernaryBaseOperator(BaseOperator):
    """Base class for all ternary (three-operand) operators."""


@dataclass
class UnaryOperator(Generic[T], UnaryBaseOperator):
    """Unary operator with a single input and output of the same type."""

    function: Callable[[T], T]


@dataclass
class UnaryMathOperator(Generic[T, R], UnaryBaseOperator):
    """Unary math operator that transforms a value of type T to a result of type R."""

    function: Callable[[T], R]


@dataclass
class UnaryBoolOperator(UnaryBaseOperator):
    """Unary operator that returns a boolean value."""

    function: Callable[[object], bool]


@dataclass
class BinaryOperator(Generic[T, R], BinaryBaseOperator):
    """Binary operator with two inputs of types T and R, returning T or R."""

    function: Callable[[T, R], T | R]


@dataclass
class BinaryMathOperator(Generic[T, R], BinaryBaseOperator):
    """Binary math operator combining two T values into an R result."""

    function: Callable[[T, T], R]


@dataclass
class BinaryBoolOperator(BinaryBaseOperator):
    """Binary operator returning a boolean based on two input values."""

    function: Callable[[Any, Any], bool]


@dataclass
class TernaryOperator(Generic[T, R], TernaryBaseOperator):
    """Ternary operator accepting a condition and two values of types T and R."""

    function: Callable[[bool, T, R], T | R]


@dataclass
class TernaryIfOperator(TernaryOperator["ExprVarLike", "ExprVarLike"]):
    """Implements a ternary if-else operator (cond ? if_true : if_false)."""

    if TYPE_CHECKING:

        def __call__(self, cond: ExprVarLike, if_true: ExprVarLike, if_false: ExprVarLike, /) -> ComputedVar: ...


@dataclass
class TernaryCompOperator(TernaryBaseOperator):
    """Ternary comparison operator used for value constraints or selection logic."""

    function: Callable[[SupportsRichComparison, SupportsRichComparison, SupportsRichComparison], SupportsRichComparison]


@dataclass
class TernaryClampOperator(TernaryCompOperator):
    """Clamps a value between a minimum and maximum using rich comparisons."""

    if TYPE_CHECKING:

        def __call__(self, x: ExprVarLike, min: ExprVarLike, max: ExprVarLike, /) -> ComputedVar: ...


class TernaryPixelAccessOperator(TernaryBaseOperator):
    """
    Ternary operator for pixel-level access in a 2D space.
    """

    def __call__(self, char: str, x: int, y: int) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(self.rpn_name.format(char=char, x=x, y=y))


class ExprOperators(Singleton):
    """
    A singleton class that defines the expression operators used in [inline_expr][vsexprtools.inline_expr].
    """

    __slots__ = ()

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
    MAX = BinaryMathOperator[SupportsRichComparison, SupportsRichComparison](ExprOp.MAX, max)
    """Calculates the maximum of x and y."""

    MIN = BinaryMathOperator[SupportsRichComparison, SupportsRichComparison](ExprOp.MIN, min)
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
    TERN = TernaryIfOperator(ExprOp.TERN, lambda cond, if_true, if_false: (if_true if cond else if_false))
    """Ternary operator (if cond then if_true else if_false)."""

    CLAMP = TernaryClampOperator(ExprOp.CLAMP, lambda v, mini, maxi: max(mini, min(v, maxi)))
    """Clamps a value between a min and a max."""

    # Aliases
    IF = TERN
    """Ternary operator (if cond then if_true else if_false)."""

    # Special Operators
    REL_PIX = TernaryPixelAccessOperator(ExprOp.REL_PIX)
    """Relative pixel access."""

    ABS_PIX = TernaryPixelAccessOperator(ExprOp.ABS_PIX)
    """Absolute pixel access."""

    # Helper Functions

    @classmethod
    def as_var(cls, x: ExprVarLike) -> ComputedVar:
        """
        Converts an expression variable to a ComputedVar.

        Args:
            x: A single ExprVarLike.

        Returns:
            A ComputedVar.
        """
        from .variables import ComputedVar

        return ComputedVar(x)
