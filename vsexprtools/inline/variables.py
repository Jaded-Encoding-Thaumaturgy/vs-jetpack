from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, NoReturn, Protocol, SupportsIndex, TypeAlias

from jetpytools import to_arr

from vstools import (
    ColorRangeT,
    get_depth,
    get_lowest_value,
    get_neutral_value,
    get_peak_value,
    get_plane_sizes,
    scale_value,
    vs,
)

from .operators import ExprOperators

__all__ = ["ClipPropsVar", "ClipVar", "ComplexVar", "ComputedVar", "ExprVar", "ExprVarLike", "LiteralVar"]


class ExprVar(ABC):
    """Base interface for variables used in RPN expression"""

    def __add__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.ADD(self, other)

    def __iadd__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.ADD(self, other)

    def __radd__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.ADD(other, self)

    def __sub__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.SUB(self, other)

    def __isub__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.SUB(self, other)

    def __rsub__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.SUB(other, self)

    def __mul__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.MUL(self, other)

    def __imul__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.MUL(self, other)

    def __rmul__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.MUL(other, self)

    def __truediv__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.DIV(self, other)

    def __rtruediv__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.DIV(other, self)

    def __itruediv__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.DIV(self, other)

    def __floordiv__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.FLOOR(ExprOperators.DIV(self, other))

    def __ifloordiv__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return ExprOperators.FLOOR(ExprOperators.DIV(self, other))

    def __rfloordiv__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.FLOOR(ExprOperators.DIV(other, self))

    def __pow__(self, other: ExprVarLike, module: int | None = None) -> ComputedVar:
        if module is not None:
            raise NotImplementedError
        return ExprOperators.POW(self, other)

    def __rpow__(self, other: ExprVarLike, module: int | None = None) -> ComputedVar:
        if module is not None:
            raise NotImplementedError
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
        return ComputedVar(self)

    def __abs__(self) -> ComputedVar:
        return ExprOperators.ABS(self)

    def __mod__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.MOD(self, other)

    def __rmod__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.MOD(other, self)

    def __divmod__(self, _: ExprVarLike) -> NoReturn:
        raise NotImplementedError

    def __rdivmod__(self, _: ExprVarLike) -> NoReturn:
        raise NotImplementedError

    def __lt__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.LT(self, other)

    def __lte__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.LTE(self, other)

    def __gt__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.GT(self, other)

    def __gte__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.GTE(self, other)

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __and__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.AND(self, other)

    def __rand__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.AND(self, other)

    def __or__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.OR(self, other)

    def __ror__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.OR(other, self)

    def __xor__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.XOR(self, other)

    def __rxor__(self, other: ExprVarLike) -> ComputedVar:
        return ExprOperators.XOR(self, other)

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

    def as_var(self) -> ComputedVar:
        """
        Converts the expression variable to a ComputedVar.

        Returns:
            A ComputedVar.
        """
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
    """Represents a fully built RPN expression as a sequence of operations."""

    def __init__(self, operations: ExprVarLike | Iterable[ExprVarLike]) -> None:
        """
        Initializes a new ComputedVar.

        Args:
            operations: An iterable of operators and/or expression variables that define the computation.
        """
        self.operations = tuple(LiteralVar(x) for x in to_arr(operations))  # type: ignore[arg-type]

    def to_str(self, **kwargs: Any) -> str:
        """
        Returns a string representation of the expression in RPN format.

        Args:
            **kwargs: Additional keywords arguments.

        Returns:
            String representation of the expression in RPN format.
        """
        return " ".join(x.to_str(**kwargs) for x in self.operations)

    def __str__(self) -> str:
        return " ".join(str(x) for x in self.operations)


class Resolver(Protocol):
    """Protocol for deferred resolution of expression values."""

    def __call__(self, *, plane: int = ..., **kwargs: Any) -> int | float | str | ExprVar: ...


@dataclass
class ComplexVar(LiteralVar):
    """
    A literal variable that resolves dynamically using a custom function.
    """

    value: int | float | str
    """The symbolic value used in the RPN expression."""
    resolve: Resolver
    """A callable used to compute the actual value during evaluation."""

    def to_str(self, *, plane: int = 0, **kwargs: Any) -> str:
        """
        Returns C

        Args:
            plane: Plane index.
            **kwargs: Additional keywords arguments.

        Returns:
            The resolved string value, optionally using a specified plane.
        """
        return str(self.resolve(plane=plane, **kwargs))


class ClipPropsVar:
    """Helper class exposing common frame properties of a ClipVar."""

    # Some commonly used props
    PlaneStatsMin: ComputedVar
    PlaneStatsMax: ComputedVar
    PlaneStatsAverage: ComputedVar

    def __init__(self, clip_var: ClipVar) -> None:
        self.clip_var = clip_var

    def __getitem__(self, key: str) -> ComputedVar:
        return getattr(self, key)

    def __getattribute__(self, name: str) -> ComputedVar:
        """Accesses a computed property using dot notation from the clip symbol."""

        return ComputedVar(f"{super().__getattribute__('clip_var').char}.{name}")


class ClipVar(ExprVar):
    """
    Expression variable that wraps a VideoNode and provides symbolic and numeric access.
    """

    char: str
    """A short symbolic name representing this clip in the RPN expression."""

    clip: vs.VideoNode
    """The actual VapourSynth VideoNode."""

    props: ClipPropsVar
    """A helper to access frame properties."""

    def __init__(self, char: str, clip: vs.VideoNode) -> None:
        self.char = char
        self.clip = clip
        self.props = ClipPropsVar(self)

    def __str__(self) -> str:
        return self.char

    # Pixel Access
    def __getitem__(self, index: tuple[int, int]) -> ComputedVar:
        """Access a pixel at a specific coordinate using relative addressing."""
        return ExprOperators.REL_PIX(self.char, *index)

    # Helper properties
    @property
    def peak(self) -> LiteralVar:
        """Returns the peak value for the clip's bit depth."""
        return LiteralVar(get_peak_value(self.clip))

    @property
    def peak_chroma(self) -> LiteralVar:
        """Returns the peak chroma value for the clip's bit depth."""
        return LiteralVar(get_peak_value(self.clip, True))

    @property
    def neutral(self) -> LiteralVar:
        """Returns the neutral value for the clip."""
        return LiteralVar(get_neutral_value(self.clip))

    @property
    def neutral_chroma(self) -> LiteralVar:
        """Returns the neutral chroma value."""
        return LiteralVar(get_neutral_value(self.clip))

    @property
    def lowest(self) -> LiteralVar:
        """Returns the lowest possible pixel value"""
        return LiteralVar(get_lowest_value(self.clip))

    @property
    def lowest_chroma(self) -> LiteralVar:
        """Returns the lowest chroma value."""
        return LiteralVar(get_lowest_value(self.clip, True))

    @property
    def width(self) -> LiteralVar:
        """Returns a symbolic 'width' identifier."""
        return LiteralVar("width")

    @property
    def width_luma(self) -> LiteralVar:
        """Returns the actual width of the luma plane."""
        return LiteralVar(self.clip.width)

    @property
    def width_chroma(self) -> LiteralVar:
        """Returns the width of the chroma plane."""
        return LiteralVar(get_plane_sizes(self.clip, 1)[0])

    @property
    def height(self) -> LiteralVar:
        """Returns a symbolic 'height' identifier."""
        return LiteralVar("height")

    @property
    def height_luma(self) -> LiteralVar:
        """Returns the actual height of the luma plane."""
        return LiteralVar(self.clip.height)

    @property
    def height_chroma(self) -> LiteralVar:
        """Returns the height of the chroma plane."""
        return LiteralVar(get_plane_sizes(self.clip, 1)[1])

    @property
    def depth(self) -> LiteralVar:
        """Returns the bit depth of the clip."""
        return LiteralVar(get_depth(self.clip))

    # Helper function for scaled values
    def scale(
        self,
        value: float,
        input_depth: int = 8,
        range_in: ColorRangeT | None = None,
        range_out: ColorRangeT | None = None,
        scale_offsets: bool = True,
        family: vs.ColorFamily | None = None,
    ) -> ComplexVar:
        """
        Returns a scaled version of the given value based on the clip's format.
        """

        def _resolve(plane: int = 0, **kwargs: Any) -> Any:
            return scale_value(
                value, input_depth, get_depth(self.clip), range_in, range_out, scale_offsets, plane in {1, 2}, family
            )

        return ComplexVar(f"{self.char}.scale({value})", _resolve)
