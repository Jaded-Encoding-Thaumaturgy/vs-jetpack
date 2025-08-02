from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, NoReturn, SupportsIndex, TypeAlias

from jetpytools import CustomRuntimeError, to_arr
from typing_extensions import Self

from vstools import vs, vs_object

from .operators import op

__all__ = ["ClipVar", "ComputedVar", "ExprVar", "ExprVarLike", "LiteralVar"]


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
        self._check_expr_per_planes()

        return " ".join(str(x) for x in self._operations_per_plane[0])

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
        self._operations_per_plane[index] = [op.as_var()]

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

    def _has_expr_per_planes(self) -> bool:
        return len({tuple(str(op) for op in opp) for opp in self._operations_per_plane}) > 1

    def _check_expr_per_planes(self) -> None:
        if self._has_expr_per_planes():
            raise CustomRuntimeError(
                "Cannot generate a unified string representation: the operations differ between planes. "
                "Use `to_str(plane=...)` or `to_str_per_plane()` instead"
            )

    def to_str_per_plane(self, num_planes: int = 3) -> list[str]:
        """
        Returns string representations of the expression in RPN format for each plane.

        Args:
            num_planes: Optional number of planes to include (defaults to 3).

        Returns:
            A list of strings, one for each plane.
        """
        return [p.to_str(plane=i) for x, i in zip(self._operations_per_plane, range(num_planes)) for p in x]

    def to_str(self, *, plane: int | None = None, **kwargs: Any) -> str:
        """
        Returns a string representation of the expression in RPN format.

        Args:
            plane: Optional plane index to select which expression to stringify. If not specified,
                all planes must have identical expressions.
            **kwargs: Additional keyword arguments passed to each expression's to_str method.

        Returns:
            String representation of the expression in RPN format.

        Raises:
            CustomRuntimeError: If plane is None and expressions differ across planes.
        """
        if plane is None:
            self._check_expr_per_planes()
            plane = 0

        return " ".join(x.to_str(plane=plane, **kwargs) for x in self._operations_per_plane[plane])


class ClipVar(ExprVar, vs_object):
    """
    Expression variable that wraps a VideoNode and provides symbolic and numeric access.
    """

    char: str
    """A short symbolic name representing this clip in the RPN expression."""

    node: vs.VideoNode
    """The actual VapourSynth VideoNode."""

    # Some commonly used props
    PlaneStatsMin: ComputedVar
    PlaneStatsMax: ComputedVar
    PlaneStatsAverage: ComputedVar

    def __init__(self, char: str, node: vs.VideoNode) -> None:
        """
        Initializes a new ClipVar instance.

        Args:
            char: A short symbolic name representing this clip in the RPN expression.
            node: The actual VapourSynth VideoNode.
        """
        self.char = char
        self.node = node

    def __str__(self) -> str:
        return self.char

    def __getitem__(self, index: tuple[int, int] | str) -> ComputedVar:
        if isinstance(index, str):
            return getattr(self, index)
        return op.rel_pix(self.char, *index)

    def __getattr__(self, name: str) -> ComputedVar:
        return ComputedVar(f"{self.char}.{name}")

    def __vs_del__(self, core_id: int) -> None:
        del self.node
