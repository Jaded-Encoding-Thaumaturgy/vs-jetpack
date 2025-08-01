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

from .operators import op

__all__ = ["ClipPropsVar", "ClipVar", "ComplexVar", "ComputedVar", "ExprVar", "ExprVarLike", "LiteralVar"]


class ExprVar(ABC):
    """Base interface for variables used in RPN expression"""

    def __add__(self, other: ExprVarLike) -> ComputedVar:
        return op.add(self, other)

    def __iadd__(self, other: ExprVarLike) -> ComputedVar:  # noqa: PYI034
        return op.add(self, other)

    def __radd__(self, other: ExprVarLike) -> ComputedVar:
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
        return op.floor(op.add(self, 0.5))

    def __floor__(self) -> ComputedVar:
        return op.floor(self)

    def __neg__(self) -> ComputedVar:
        return op.mul(op.abs(self), -1)

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
        return op.and_(self, other)

    def __rand__(self, other: ExprVarLike) -> ComputedVar:
        return op.and_(self, other)

    def __or__(self, other: ExprVarLike) -> ComputedVar:
        return op.or_(self, other)

    def __ror__(self, other: ExprVarLike) -> ComputedVar:
        return op.or_(other, self)

    def __xor__(self, other: ExprVarLike) -> ComputedVar:
        return op.xor(self, other)

    def __rxor__(self, other: ExprVarLike) -> ComputedVar:
        return op.xor(self, other)

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
        self.operations = tuple(LiteralVar(x) if not isinstance(x, ExprVar) else x for x in to_arr(operations))  # type: ignore[arg-type]

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

    node: vs.VideoNode
    """The actual VapourSynth VideoNode."""

    props: ClipPropsVar
    """A helper to access frame properties."""

    def __init__(self, char: str, node: vs.VideoNode) -> None:
        self.char = char
        self.node = node
        self.props = ClipPropsVar(self)

    def __str__(self) -> str:
        return self.char

    # Pixel Access
    def __getitem__(self, index: tuple[int, int]) -> ComputedVar:
        """Access a pixel at a specific coordinate using relative addressing."""
        return op.rel_pix(self.char, *index)

    # Helper properties
    @property
    def peak(self) -> LiteralVar:
        """Returns the peak value for the clip's bit depth."""
        return LiteralVar(get_peak_value(self.node))

    @property
    def peak_chroma(self) -> LiteralVar:
        """Returns the peak chroma value for the clip's bit depth."""
        return LiteralVar(get_peak_value(self.node, True))

    @property
    def neutral(self) -> LiteralVar:
        """Returns the neutral value for the clip."""
        return LiteralVar(get_neutral_value(self.node))

    @property
    def neutral_chroma(self) -> LiteralVar:
        """Returns the neutral chroma value."""
        return LiteralVar(get_neutral_value(self.node))

    @property
    def lowest(self) -> LiteralVar:
        """Returns the lowest possible pixel value"""
        return LiteralVar(get_lowest_value(self.node))

    @property
    def lowest_chroma(self) -> LiteralVar:
        """Returns the lowest chroma value."""
        return LiteralVar(get_lowest_value(self.node, True))

    @property
    def width(self) -> LiteralVar:
        """Returns a symbolic 'width' identifier."""
        return LiteralVar("width")

    @property
    def width_luma(self) -> LiteralVar:
        """Returns the actual width of the luma plane."""
        return LiteralVar(self.node.width)

    @property
    def width_chroma(self) -> LiteralVar:
        """Returns the width of the chroma plane."""
        return LiteralVar(get_plane_sizes(self.node, 1)[0])

    @property
    def height(self) -> LiteralVar:
        """Returns a symbolic 'height' identifier."""
        return LiteralVar("height")

    @property
    def height_luma(self) -> LiteralVar:
        """Returns the actual height of the luma plane."""
        return LiteralVar(self.node.height)

    @property
    def height_chroma(self) -> LiteralVar:
        """Returns the height of the chroma plane."""
        return LiteralVar(get_plane_sizes(self.node, 1)[1])

    @property
    def depth(self) -> LiteralVar:
        """Returns the bit depth of the clip."""
        return LiteralVar(get_depth(self.node))

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
