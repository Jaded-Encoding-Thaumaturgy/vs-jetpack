from __future__ import annotations

from typing import TYPE_CHECKING, Collection, Iterable, Literal

from jetpytools import Singleton, SupportsString
from typing_extensions import Self

from vstools import ConvMode, OnePassConvModeT, flatten

from ..exprop import ExprOp

if TYPE_CHECKING:
    from .variables import ComputedVar, ExprVarLike, LiteralVar

__all__ = ["Operators"]


class Operators(Singleton):
    """
    A singleton class that defines the expression operators used in [inline_expr][vsexprtools.inline_expr].
    """

    __slots__ = ()

    # 1 Argument
    def exp(self, x: ExprVarLike) -> ComputedVar:
        """Exponential function (e^x)."""
        return self.as_var([x, ExprOp.EXP])

    def log(self, x: ExprVarLike) -> ComputedVar:
        """Natural logarithm of x."""
        return self.as_var([x, ExprOp.LOG])

    def sqrt(self, x: ExprVarLike) -> ComputedVar:
        """Square root of x."""
        return self.as_var([x, ExprOp.SQRT])

    def sin(self, x: ExprVarLike) -> ComputedVar:
        """Sine (radians) of x."""
        return self.as_var([x, ExprOp.SIN])

    def cos(self, x: ExprVarLike) -> ComputedVar:
        """Cosine (radians) of x."""
        return self.as_var([x, ExprOp.COS])

    def abs(self, x: ExprVarLike) -> ComputedVar:
        """Absolute value of x."""
        return self.as_var([x, ExprOp.ABS])

    def not_(self, x: ExprVarLike) -> ComputedVar:
        """Logical NOT of x."""
        return self.as_var([x, ExprOp.NOT])

    def trunc(self, x: ExprVarLike) -> ComputedVar:
        """Truncate x to integer (toward zero)."""
        return self.as_var([x, ExprOp.TRUNC])

    def round(self, x: ExprVarLike) -> ComputedVar:
        """Round x to nearest integer."""
        return self.as_var([x, ExprOp.ROUND])

    def floor(self, x: ExprVarLike) -> ComputedVar:
        """Round down x to nearest integer."""
        return self.as_var([x, ExprOp.FLOOR])

    # DROP / DROPN / SORTN / VAR_STORE / VAR_PUSH ??

    # 2 Arguments
    def max(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Calculates the maximum of x and y."""
        return self.as_var([x, y, ExprOp.MAX])

    def min(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Calculates the minimum of x and y."""
        return self.as_var([x, y, ExprOp.MIN])

    def add(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs addition of two elements (x + y)."""
        return self.as_var([x, y, ExprOp.ADD])

    def sub(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs subtraction of two elements (x - y)."""
        return self.as_var([x, y, ExprOp.SUB])

    def mul(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs multiplication of two elements (x * y)."""
        return self.as_var([x, y, ExprOp.MUL])

    def div(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs division of two elements (x / y)."""
        return self.as_var([x, y, ExprOp.DIV])

    def pow(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x to the power of y (x ** y)."""
        return self.as_var([x, y, ExprOp.POW])

    def mod(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x % y."""
        return self.as_var([x, y, ExprOp.MOD])

    def xor(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical XOR."""
        return self.as_var([x, y, ExprOp.XOR])

    def gt(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x > y."""
        return self.as_var([x, y, ExprOp.GT])

    def lt(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x < y."""
        return self.as_var([x, y, ExprOp.LT])

    def eq(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x == y."""
        return self.as_var([x, y, ExprOp.EQ])

    def gte(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x >= y."""
        return self.as_var([x, y, ExprOp.GTE])

    def lte(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x <= y."""
        return self.as_var([x, y, ExprOp.LTE])

    def and_(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical AND."""
        return self.as_var([x, y, ExprOp.AND])

    def or_(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical OR."""
        return self.as_var([x, y, ExprOp.OR])

    # 3 Arguments
    def tern(self, cond: ExprVarLike, if_true: ExprVarLike, if_false: ExprVarLike) -> ComputedVar:
        """Ternary operator (if cond then if_true else if_false)."""
        return self.as_var([cond, if_true, if_false, ExprOp.TERN])

    if_ = tern

    def clamp(self, x: ExprVarLike, min: ExprVarLike, max: ExprVarLike) -> ComputedVar:
        """Clamps a value between a min and a max."""
        return self.as_var([x, min, max, ExprOp.CLAMP])

    # Special Operators
    def rel_pix(self, char: str, x: int, y: int) -> ComputedVar:
        """Relative pixel access."""
        return self.as_var(ExprOp.REL_PIX.format(char=char, x=x, y=y))

    def abs_pix(self, char: str, x: int, y: int) -> ComputedVar:
        """Absolute pixel access."""
        return self.as_var(ExprOp.ABS_PIX.format(char=char, x=x, y=y))

    # Helper Functions
    def __call__(self) -> Self:
        return self

    def matrix(
        self,
        char: SupportsString | Collection[SupportsString],
        radius: int,
        mode: Literal[ConvMode.SQUARE, ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.TEMPORAL],
        exclude: Iterable[tuple[int, int]] | None = None,
    ) -> list[LiteralVar]:
        """
        Convenience method wrapping [ExprOp.matrix][vsexprtools.ExprOp.matrix].

        Args:
            char: The variable representing the central pixel(s).
            radius: The radius of the kernel in pixels (e.g., 1 for 3x3).
            mode: The convolution mode. `HV` is not supported.
            exclude: Optional set of (x, y) coordinates to exclude from the matrix.

        Returns:
            A list of [LiteralVar][vsexprtools.inline.variables.LiteralVar] instances
            representing the matrix of expressions.
        """
        from .variables import LiteralVar

        matrix, *_ = ExprOp.matrix(char, radius, mode, exclude)

        return [LiteralVar(str(m)) for m in matrix]

    def convolution(
        self,
        var: SupportsString | Collection[SupportsString],
        matrix: Iterable[ExprVarLike] | Iterable[Iterable[ExprVarLike]],
        bias: ExprVarLike | None = None,
        divisor: ExprVarLike | bool = True,
        saturate: bool = True,
        mode: OnePassConvModeT = ConvMode.SQUARE,
        premultiply: ExprVarLike | None = None,
        multiply: ExprVarLike | None = None,
        clamp: bool = False,
    ) -> ComputedVar:
        """
        Convenience method wrapping [ExprOp.convolution][vsexprtools.ExprOp.convolution].

        Args:
            var: The variable used as the central value
                or elements proportional to the radius if mode is `Literal[ConvMode.TEMPORAL]`.
            matrix: A flat or 2D iterable representing the convolution weights.
            bias: A constant value to add to the result after convolution (default: None).
            divisor: If True, normalizes by the sum of weights; if False, skips division;
                Otherwise, divides by this value.
            saturate: If False, applies `abs()` to avoid negatives.
            mode: The convolution shape.
            premultiply: Optional scalar to multiply the result before normalization.
            multiply: Optional scalar to multiply the result at the end.
            clamp: If True, clamps the final result to [RangeMin, RangeMax].

        Returns:
            A `ComputedVar` representing the expression-based convolution.
        """
        convo, *_ = ExprOp.convolution(
            var, flatten(matrix), bias, divisor, saturate, mode, premultiply, multiply, clamp
        )
        return self.as_var(convo.to_str())

    @staticmethod
    def as_var(x: ExprVarLike | Iterable[ExprVarLike]) -> ComputedVar:
        """
        Converts an expression variable to a ComputedVar.

        Args:
            x: A single ExprVarLike.

        Returns:
            A ComputedVar.
        """
        from .variables import ComputedVar

        return ComputedVar(x)


op = Operators()
