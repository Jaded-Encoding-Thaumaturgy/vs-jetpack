from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Final,
    Iterable,
    Literal,
    NoReturn,
    SupportsIndex,
    TypeAlias,
    cast,
)

from jetpytools import Singleton, SupportsString, to_arr
from typing_extensions import Self

from vstools import ConvMode, OnePassConvModeT, flatten, vs, vs_object

from ..exprop import ExprOp, ExprToken

__all__ = ["ClipVar", "ComputedVar", "ExprVar", "ExprVarLike", "LiteralVar", "Operators", "Tokens"]


class Operators(Singleton):
    """
    A singleton class that defines the expression operators used in [inline_expr][vsexprtools.inline_expr].
    """

    __slots__ = ()

    # 1 Argument
    def exp(self, x: ExprVarLike) -> ComputedVar:
        """Exponential function (e^x)."""
        return ComputedVar([x, ExprOp.EXP])

    def log(self, x: ExprVarLike) -> ComputedVar:
        """Natural logarithm of x."""
        return ComputedVar([x, ExprOp.LOG])

    def sqrt(self, x: ExprVarLike) -> ComputedVar:
        """Square root of x."""
        return ComputedVar([x, ExprOp.SQRT])

    def sin(self, x: ExprVarLike) -> ComputedVar:
        """Sine (radians) of x."""
        return ComputedVar([x, ExprOp.SIN])

    def asin(self, x: ExprVarLike) -> ComputedVar:
        """Arcsine (inverse sine) of x."""
        return ComputedVar([x, ExprOp.ASIN])

    def cos(self, x: ExprVarLike) -> ComputedVar:
        """Cosine (radians) of x."""
        return ComputedVar([x, ExprOp.COS])

    def acos(self, x: ExprVarLike) -> ComputedVar:
        """Arccosine (inverse cosine) of x."""
        return ComputedVar([x, ExprOp.ACOS])

    def tan(self, x: ExprVarLike) -> ComputedVar:
        """Tangent (radians) of x."""
        return ComputedVar([x, ExprOp.TAN])

    def atan(self, x: ExprVarLike) -> ComputedVar:
        """Arctangent of x"""
        return ComputedVar([x, ExprOp.ATAN])

    def abs(self, x: ExprVarLike) -> ComputedVar:
        """Absolute value of x."""
        return ComputedVar([x, ExprOp.ABS])

    def not_(self, x: ExprVarLike) -> ComputedVar:
        """Logical NOT of x."""
        return ComputedVar([x, ExprOp.NOT])

    def trunc(self, x: ExprVarLike) -> ComputedVar:
        """Truncate x to integer (toward zero)."""
        return ComputedVar([x, ExprOp.TRUNC])

    def round(self, x: ExprVarLike) -> ComputedVar:
        """Round x to nearest integer."""
        return ComputedVar([x, ExprOp.ROUND])

    def floor(self, x: ExprVarLike) -> ComputedVar:
        """Round down x to nearest integer."""
        return ComputedVar([x, ExprOp.FLOOR])

    def ceil(self, x: ExprVarLike) -> ComputedVar:
        """Round up x to nearest integer."""
        return ComputedVar([x, ExprOp.CEIL])

    def bitnot(self, x: ExprVarLike) -> ComputedVar:
        """Performs a bitwise NOT."""
        return ComputedVar([x, ExprOp.BITNOT])

    def sgn(self, x: ExprVarLike) -> ComputedVar:
        """Sign function (-1, 0, or 1) of x."""
        return ComputedVar([x, ExprOp.SGN])

    def neg(self, x: ExprVarLike) -> ComputedVar:
        """Negation (multiply by -1) of x."""
        return ComputedVar([x, ExprOp.NEG])

    # 2 Arguments
    def max(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Calculates the maximum of x and y."""
        return ComputedVar([x, y, ExprOp.MAX])

    def min(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Calculates the minimum of x and y."""
        return ComputedVar([x, y, ExprOp.MIN])

    def add(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs addition of two elements (x + y)."""
        return ComputedVar([x, y, ExprOp.ADD])

    def sub(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs subtraction of two elements (x - y)."""
        return ComputedVar([x, y, ExprOp.SUB])

    def mul(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs multiplication of two elements (x * y)."""
        return ComputedVar([x, y, ExprOp.MUL])

    def div(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs division of two elements (x / y)."""
        return ComputedVar([x, y, ExprOp.DIV])

    def pow(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x to the power of y (x ** y)."""
        return ComputedVar([x, y, ExprOp.POW])

    def gt(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x > y."""
        return ComputedVar([x, y, ExprOp.GT])

    def lt(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x < y."""
        return ComputedVar([x, y, ExprOp.LT])

    def eq(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x == y."""
        return ComputedVar([x, y, ExprOp.EQ])

    def gte(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x >= y."""
        return ComputedVar([x, y, ExprOp.GTE])

    def lte(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x <= y."""
        return ComputedVar([x, y, ExprOp.LTE])

    def and_(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical AND."""
        return ComputedVar([x, y, ExprOp.AND])

    def or_(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical OR."""
        return ComputedVar([x, y, ExprOp.OR])

    def xor(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a logical XOR."""
        return ComputedVar([x, y, ExprOp.XOR])

    def mod(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs x % y."""
        return ComputedVar([x, y, ExprOp.MOD])

    def bitand(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a bitwise AND."""
        return ComputedVar([x, y, ExprOp.BITAND])

    def bitor(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a bitwise OR."""
        return ComputedVar([x, y, ExprOp.BITOR])

    def bitxor(self, x: ExprVarLike, y: ExprVarLike) -> ComputedVar:
        """Performs a bitwise XOR."""
        return ComputedVar([x, y, ExprOp.BITXOR])

    # 3 Arguments
    def tern(self, cond: ExprVarLike, if_true: ExprVarLike, if_false: ExprVarLike) -> ComputedVar:
        """Ternary operator (if cond then if_true else if_false)."""
        return ComputedVar([cond, if_true, if_false, ExprOp.TERN])

    if_ = tern
    """Alias for [tern][vsexprtools.inline.helpers.Operators.tern]."""

    def clamp(self, x: ExprVarLike, min: ExprVarLike, max: ExprVarLike) -> ComputedVar:
        """Clamps a value between a min and a max."""
        return ComputedVar([x, min, max, ExprOp.CLAMP])

    def lerp(self, x: ExprVarLike, y: ExprVarLike, t: ExprVarLike) -> ComputedVar:
        """Performs a linear interpolation of t between x and y."""
        return ComputedVar([x, y, t, ExprOp.LERP])

    # inf
    def polyval(self, x: ExprVarLike, *coeffs: ExprVarLike) -> ComputedVar:
        """Evaluates a polynomial at x using Horner's method."""
        return ComputedVar(ExprOp.polyval(x, *coeffs).to_str())

    # Special Operators
    def rel_pix(self, char: SupportsString, x: int, y: int) -> ComputedVar:
        """Relative pixel access."""
        return ComputedVar(ExprOp.REL_PIX.format(char=char, x=x, y=y))

    def abs_pix(self, char: SupportsString, x: int, y: int) -> ComputedVar:
        """Absolute pixel access."""
        return ComputedVar(ExprOp.ABS_PIX.format(char=char, x=x, y=y))

    def __call__(self) -> Self:
        """
        Returns itself.

        Returns:
            Returns itself.
        """
        return self

    # Helper Functions
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
            A list of [LiteralVar][vsexprtools.inline.helpers.LiteralVar] instances
            representing the matrix of expressions.
        """
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
        return ComputedVar(convo.to_str())


op = Operators()
"""The Operators singleton instance"""


class ExprVar(ABC):
    """Base interface for variables used in RPN expression"""

    def __add__(self, other: ExprVarLike) -> ComputedVar:
        if other == 0:
            return self.as_var()
        return op.add(self, other)

    def __iadd__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        if other == 0:
            return self.as_var()
        return op.add(self, other)

    def __radd__(self, other: ExprVarLike) -> ComputedVar:
        if other == 0:
            return self.as_var()
        return op.add(other, self)

    def __sub__(self, other: ExprVarLike) -> ComputedVar:
        return op.sub(self, other)

    def __isub__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return op.sub(self, other)

    def __rsub__(self, other: ExprVarLike) -> ComputedVar:
        return op.sub(other, self)

    def __mul__(self, other: ExprVarLike) -> ComputedVar:
        return op.mul(self, other)

    def __imul__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return op.mul(self, other)

    def __rmul__(self, other: ExprVarLike) -> ComputedVar:
        return op.mul(other, self)

    def __truediv__(self, other: ExprVarLike) -> ComputedVar:
        return op.div(self, other)

    def __rtruediv__(self, other: ExprVarLike) -> ComputedVar:
        return op.div(other, self)

    def __itruediv__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return op.div(self, other)

    def __floordiv__(self, other: ExprVarLike) -> ComputedVar:
        return op.floor(op.div(self, other))

    def __ifloordiv__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return op.floor(op.div(self, other))

    def __rfloordiv__(self, other: ExprVarLike) -> ComputedVar:
        return op.floor(op.div(other, self))

    def __pow__(self, other: ExprVarLike, module: int | None = None) -> ComputedVar:
        if module is not None:
            raise NotImplementedError
        return op.pow(self, other)

    def __rpow__(self, other: ExprVarLike, module: int | None = None) -> ComputedVar:
        if module is not None:
            raise NotImplementedError
        return op.pow(other, self)

    def __exp__(self) -> ComputedVar:
        return op.exp(self)

    def __log__(self) -> ComputedVar:
        return op.log(self)

    def __sqrt__(self) -> ComputedVar:
        return op.sqrt(self)

    def __round__(self, ndigits: SupportsIndex | None = None) -> ComputedVar:
        if ndigits is not None:
            raise NotImplementedError
        return op.round(self)

    def __trunc__(self) -> ComputedVar:
        return op.trunc(self)

    def __ceil__(self) -> ComputedVar:
        return op.ceil(self)

    def __floor__(self) -> ComputedVar:
        return op.floor(self)

    def __neg__(self) -> ComputedVar:
        return op.neg(self)

    def __pos__(self) -> ComputedVar:
        return op.abs(self)

    def __invert__(self) -> NoReturn:
        raise NotImplementedError

    def __int__(self) -> ComputedVar:
        return op.trunc(self)

    def __float__(self) -> ComputedVar:
        return ComputedVar(self)

    def __abs__(self) -> ComputedVar:
        return op.abs(self)

    def __mod__(self, other: ExprVarLike) -> ComputedVar:
        return op.mod(self, other)

    def __rmod__(self, other: ExprVarLike) -> ComputedVar:
        return op.mod(other, self)

    def __divmod__(self, _: ExprVarLike) -> NoReturn:
        raise NotImplementedError

    def __rdivmod__(self, _: ExprVarLike) -> NoReturn:
        raise NotImplementedError

    def __lt__(self, other: ExprVarLike) -> ComputedVar:
        return op.lt(self, other)

    def __lte__(self, other: ExprVarLike) -> ComputedVar:
        return op.lte(self, other)

    def __gt__(self, other: ExprVarLike) -> ComputedVar:
        return op.gt(self, other)

    def __gte__(self, other: ExprVarLike) -> ComputedVar:
        return op.gte(self, other)

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __and__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitand(self, other)

    def __rand__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitand(self, other)

    def __or__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitor(self, other)

    def __ror__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitor(self, other)

    def __xor__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitxor(self, other)

    def __rxor__(self, other: ExprVarLike) -> ComputedVar:
        return op.bitxor(self, other)

    def to_str(self, **kwargs: Any) -> str:
        """
        Returns the string representation of the expression variable.

        Args:
            **kwargs: Additional keywords arguments.

        Returns:
            The string representation of the expression variable.
        """
        return str(self)

    @abstractmethod
    def __str__(self) -> str: ...

    def __hash__(self) -> int:
        return hash(self)

    def __eq__(self, value: object) -> bool:
        return str(self) == str(value)

    def __format__(self, format_spec: str) -> str:
        return f"{self.__str__():{format_spec}}"

    def as_var(self) -> ComputedVar:
        """
        Converts the expression variable to a ComputedVar.

        Returns:
            A ComputedVar.
        """
        if isinstance(self, ComputedVar):
            return self
        return ComputedVar(self)


ExprVarLike: TypeAlias = int | float | str | ExprVar
"""Type alias representing any expression-compatible variable or literal."""


class LiteralVar(ExprVar):
    """Literal value wrapper for use in RPN expressions."""

    def __init__(self, value: ExprVarLike):
        """
        Initializes a new LiteralVar.

        Args:
            value: An integer, float, string, or ExprVar to wrap.
        """
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


class ComputedVar(ExprVar):
    """
    Represents a fully built RPN expression as a sequence of operations with per-plane operations support.
    """

    def __init__(self, operations: ExprVarLike | Iterable[ExprVarLike]) -> None:
        """
        Initializes a new ComputedVar.

        Args:
            operations: An iterable of operators and/or expression variables that define the computation.
        """
        self._operations_per_plane: list[list[ExprVar]] = [
            [LiteralVar(x) if not isinstance(x, ExprVar) else x for x in to_arr(operations)]  # type: ignore[arg-type]
        ] * 3

    def __str__(self) -> str:
        """
        Returns a string representation of the expression in RPN format for the first plane.

        Raises:
            CustomRuntimeError: If expressions differ between planes.
        """
        return self.to_str()

    def __getitem__(self, index: SupportsIndex) -> Self:
        """
        Returns a ComputedVar for a specific plane.

        Args:
            index: Plane index (0 for Y/R, 1 for U/G, 2 for V/B).

        Returns:
            A ComputedVar corresponding to the selected plane.
        """
        return self.__class__(self._operations_per_plane[index])

    def __setitem__(self, index: SupportsIndex, value: ExprVarLike) -> None:
        """
        Sets the expression for a specific plane.

        Args:
            index: Plane index.
            value: Expression to assign to the plane.
        """
        self._operations_per_plane[index] = ComputedVar(value)._operations_per_plane[index]

    def __delitem__(self, index: SupportsIndex) -> None:
        """Deletes the expression for a specific plane by resetting it to a single variable."""
        self._operations_per_plane[index] = [self.__class__("")]

    @property
    def y(self) -> Self:
        """Returns the Y (luma) plane expression."""
        return self[0]

    @y.setter
    def y(self, value: ExprVarLike) -> None:
        """Sets the Y (luma) plane expression."""
        self[0] = value

    @y.deleter
    def y(self) -> None:
        """Deletes the Y (luma) plane expression."""
        del self[0]

    @property
    def u(self) -> Self:
        """Returns the U (chroma) plane expression."""
        return self[1]

    @u.setter
    def u(self, value: ExprVarLike) -> None:
        """Sets the U (chroma) plane expression."""
        self[1] = value

    @u.deleter
    def u(self) -> None:
        """Deletes the U (chroma) plane expression."""
        del self[1]

    @property
    def v(self) -> Self:
        """Returns the V (chroma) plane expression."""
        return self[2]

    @v.setter
    def v(self, value: ExprVarLike) -> None:
        """Sets the V (chroma) plane expression."""
        self[2] = value

    @v.deleter
    def v(self) -> None:
        """Deletes the V (chroma) plane expression."""
        del self[2]

    @property
    def uv(self) -> tuple[Self, Self]:
        """Returns the U and V (chroma) planes expression."""
        return self[1], self[2]

    @uv.setter
    def uv(self, value: ExprVarLike) -> None:
        """Sets the U and V (chroma) planes expression."""
        self[1], self[2] = value, value

    @uv.deleter
    def uv(self) -> None:
        """Deletes the U and V (chroma) planes expression."""
        del self[1]
        del self[2]

    @property
    def r(self) -> Self:
        """Returns the R (red) plane expression."""
        return self[0]

    @r.setter
    def r(self, value: ExprVarLike) -> None:
        """Sets the R (red) plane expression."""
        self[0] = value

    @r.deleter
    def r(self) -> None:
        """Deletes the R (red) plane expression."""
        del self[0]

    @property
    def g(self) -> Self:
        """Returns the G (green) plane expression."""
        return self[1]

    @g.setter
    def g(self, value: ExprVarLike) -> None:
        """Sets the G (green) plane expression."""
        self[1] = value

    @g.deleter
    def g(self) -> None:
        """Deletes the G (green) plane expression."""
        del self[1]

    @property
    def b(self) -> Self:
        """Returns the B (blue) plane expression."""
        return self[2]

    @b.setter
    def b(self, value: ExprVarLike) -> None:
        """Sets the B (blue) plane expression."""
        self[2] = value

    @b.deleter
    def b(self) -> None:
        """Deletes the B (blue) plane expression."""
        del self[2]

    def to_str_per_plane(self, num_planes: int = 3) -> list[str]:
        """
        Returns string representations of the expression in RPN format for each plane.

        Args:
            num_planes: Optional number of planes to include (defaults to 3).

        Returns:
            A list of strings, one for each plane.
        """
        return [p.to_str(plane=i) for x, i in zip(self._operations_per_plane, range(num_planes)) for p in x]

    def to_str(self, *, plane: int = 0, **kwargs: Any) -> str:
        """
        Returns a string representation of the expression in RPN format for the selected plane.

        Args:
            plane: Optional plane index to select which expression to stringify.
            **kwargs: Additional keyword arguments passed to each expression's to_str method.

        Returns:
            String representation of the expression in RPN format for the selected plane.
        """

        return " ".join(x.to_str(**kwargs) for x in self._operations_per_plane[plane])


class ClipVarProps:
    """Helper class exposing common frame properties of a ClipVar."""

    # Some commonly used props
    PlaneStatsMin: ComputedVar
    PlaneStatsMax: ComputedVar
    PlaneStatsAverage: ComputedVar

    def __init__(self, var: ClipVar) -> None:
        self._var = var

    def __getitem__(self, key: str) -> ComputedVar:
        """Accesses a frame property using [] notation from the clip symbol."""
        return self.__getattr__(key)

    def __getattr__(self, name: str) -> ComputedVar:
        """Accesses a frame property using dot notation from the clip symbol."""
        return ComputedVar(f"{self._var.char}.{name}")


class ClipVar(ExprVar, vs_object):
    """
    Expression variable that wraps a VideoNode and provides symbolic and numeric access.
    """

    if TYPE_CHECKING:
        LumaMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """The minimum luma value in limited range."""

        ChromaMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """The minimum chroma value in limited range."""

        LumaMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """The maximum luma value in limited range."""

        ChromaMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """The maximum chroma value in limited range."""

        Neutral: Final[ComputedVar] = cast(ComputedVar, ...)
        """The neutral value (e.g. 128 for 8-bit limited, 0 for float)."""

        RangeHalf: Final[ComputedVar] = cast(ComputedVar, ...)
        """Half of the full range (e.g. 128.0 for 8-bit full range)."""

        RangeSize: Final[ComputedVar] = cast(ComputedVar, ...)
        """The size of the full range (e.g. 256 for 8-bit, 65536 for 16-bit)."""

        RangeMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Minimum value in full range (chroma-aware)."""

        LumaRangeMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Minimum luma value based on input clip's color range."""

        ChromaRangeMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Minimum chroma value based on input clip's color range."""

        RangeMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Maximum value in full range (chroma-aware)."""

        LumaRangeMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Maximum luma value based on input clip's color range."""

        ChromaRangeMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Maximum chroma value based on input clip's color range."""

        RangeInMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `RangeMin`, but adapts to input `range_in` parameter."""

        LumaRangeInMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `LumaRangeMin`, but adapts to input `range_in`."""

        ChromaRangeInMin: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `ChromaRangeMin`, but adapts to input `range_in`."""

        RangeInMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `RangeMax`, but adapts to input `range_in`."""

        LumaRangeInMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `LumaRangeMax`, but adapts to input `range_in`."""

        ChromaRangeInMax: Final[ComputedVar] = cast(ComputedVar, ...)
        """Like `ChromaRangeMax`, but adapts to input `range_in`."""

    def __init__(self, char: str, node: vs.VideoNode) -> None:
        """
        Initializes a new ClipVar instance.

        Args:
            char: A short symbolic name representing this clip in the RPN expression.
            node: The actual VapourSynth VideoNode.
        """
        self._char = char
        self._node = node

    def __str__(self) -> str:
        return self.char

    def __getitem__(self, index: tuple[int, int]) -> ComputedVar:
        """
        Returns a ComputedVar for a relative pixel access.

        Args:
            index: Tuple of relative pixel coordinates.

        Returns:
            A ComputedVar representing the relative pixel coordinates.
        """
        return op.rel_pix(self.char, *index)

    @property
    @cache
    def props(self) -> ClipVarProps:
        """A helper to access frame properties."""
        return ClipVarProps(self)

    @property
    def char(self) -> str:
        """A short symbolic name representing this clip in the RPN expression."""
        return self._char

    @property
    def node(self) -> vs.VideoNode:
        """The actual VapourSynth VideoNode."""
        return self._node

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> ComputedVar:
            return getattr(tokens, name)(self)

    def __vs_del__(self, core_id: int) -> None:
        del self._node


class Token(LiteralVar):
    """An expression token wrapping [ExprToken][vsexprtools.ExprToken]."""

    def __init__(self, expr_token: ExprToken) -> None:
        """
        Initializes a new Token instance.

        Args:
            expr_token: Token to wrapped.
        """
        super().__init__(expr_token.value)

    def __call__(self, var: ClipVar) -> ComputedVar:
        """
        Returns a version of the token specific to a clip variable.

        Args:
            var: The ClipVar to use.

        Returns:
            A ComputedVar with a var suffix for use in inline expressions.
        """
        return ComputedVar(f"{self.value}_{var.char}")


class Tokens(Singleton):
    """
    A singleton class that defines the expression tokens used in [inline_expr][vsexprtools.inline_expr].
    """

    if TYPE_CHECKING:
        LumaMin: Final[Token] = cast(Token, ...)
        """The minimum luma value in limited range."""

        ChromaMin: Final[Token] = cast(Token, ...)
        """The minimum chroma value in limited range."""

        LumaMax: Final[Token] = cast(Token, ...)
        """The maximum luma value in limited range."""

        ChromaMax: Final[Token] = cast(Token, ...)
        """The maximum chroma value in limited range."""

        Neutral: Final[Token] = cast(Token, ...)
        """The neutral value (e.g. 128 for 8-bit limited, 0 for float)."""

        RangeHalf: Final[Token] = cast(Token, ...)
        """Half of the full range (e.g. 128.0 for 8-bit full range)."""

        RangeSize: Final[Token] = cast(Token, ...)
        """The size of the full range (e.g. 256 for 8-bit, 65536 for 16-bit)."""

        RangeMin: Final[Token] = cast(Token, ...)
        """Minimum value in full range (chroma-aware)."""

        LumaRangeMin: Final[Token] = cast(Token, ...)
        """Minimum luma value based on input clip's color range."""

        ChromaRangeMin: Final[Token] = cast(Token, ...)
        """Minimum chroma value based on input clip's color range."""

        RangeMax: Final[Token] = cast(Token, ...)
        """Maximum value in full range (chroma-aware)."""

        LumaRangeMax: Final[Token] = cast(Token, ...)
        """Maximum luma value based on input clip's color range."""

        ChromaRangeMax: Final[Token] = cast(Token, ...)
        """Maximum chroma value based on input clip's color range."""

        RangeInMin: Final[Token] = cast(Token, ...)
        """Like `RangeMin`, but adapts to input `range_in` parameter."""

        LumaRangeInMin: Final[Token] = cast(Token, ...)
        """Like `LumaRangeMin`, but adapts to input `range_in`."""

        ChromaRangeInMin: Final[Token] = cast(Token, ...)
        """Like `ChromaRangeMin`, but adapts to input `range_in`."""

        RangeInMax: Final[Token] = cast(Token, ...)
        """Like `RangeMax`, but adapts to input `range_in`."""

        LumaRangeInMax: Final[Token] = cast(Token, ...)
        """Like `LumaRangeMax`, but adapts to input `range_in`."""

        ChromaRangeInMax: Final[Token] = cast(Token, ...)
        """Like `ChromaRangeMax`, but adapts to input `range_in`."""

    @cache
    def _get_token(self, name: str) -> Token:
        return Token(ExprToken[name])

    if not TYPE_CHECKING:
        __isabstractmethod__ = False

        def __getattr__(self, name: str) -> Token:
            return self._get_token(name)


tokens = Tokens()
"""The Tokens singleton instance"""
