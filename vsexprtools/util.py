from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cache
from itertools import count
from typing import TYPE_CHECKING, SupportsIndex, overload

from jetpytools import CustomIndexError, CustomTypeError

if TYPE_CHECKING:
    from vapoursynth import _ReturnDict_akarin_Version

from vstools import EXPR_VARS, HoldsVideoFormat, VideoFormatLike, get_video_format, vs

__all__ = ["ExprVars"]


class ExprVars(Sequence[str], Iterator[str]):
    """
    A helper class for generating variable names used in RPN expressions.
    """

    start: int
    """Starting index for variable generation (inclusive)."""
    stop: int
    """Ending index for variable generation (exclusive)."""
    step: int
    """Step size for iteration."""
    curr: int
    """Current index in iteration."""
    expr_src: bool
    """If True, variables are named as `src0`, `src1`, etc. Otherwise, "x", "y", "z", "a" and so on."""

    @overload
    def __init__(
        self, stop: SupportsIndex | ExprVars | HoldsVideoFormat | VideoFormatLike, /, *, expr_src: bool = False
    ) -> None: ...

    @overload
    def __init__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex | None = 1, /, *, expr_src: bool = False
    ) -> None: ...

    def __init__(
        self,
        start_stop: SupportsIndex | ExprVars | HoldsVideoFormat | VideoFormatLike,
        stop: SupportsIndex | None = None,
        step: SupportsIndex | None = None,
        /,
        *,
        expr_src: bool = False,
    ) -> None:
        """
        Initialize an ExprVars instance.

        Args:
            start_stop: A start index or an object from which to infer the number of variables (e.g., video format).
            stop: Stop index (exclusive). Required only if `start_stop` is a numeric start value.
            step: Step size for iteration. Default to 1.
            expr_src: Whether to use `srcX` naming or use alphabetic variables.

        Raises:
            CustomIndexError: If `start` is negative or `stop` is not greater than `start`.
            CustomTypeError: If invalid types are provided.
        """
        if isinstance(start_stop, ExprVars):
            self.start = start_stop.start
            self.stop = start_stop.stop
            self.step = start_stop.step
            self.curr = start_stop.curr
            self.expr_src = start_stop.expr_src
            return

        if stop is None:
            self.start = 0
            self.stop = (
                get_video_format(start_stop).num_planes
                if isinstance(start_stop, HoldsVideoFormat | VideoFormatLike)
                else start_stop.__index__()
            )
        else:
            if isinstance(start_stop, HoldsVideoFormat | VideoFormatLike):
                raise CustomTypeError(
                    "start cannot be a video format when stop is provided.", self.__class__, start_stop
                )
            self.start = start_stop.__index__()
            self.stop = stop.__index__()

        self.step = (step or 1).__index__()

        if self.start < 0:
            raise CustomIndexError('"start" must be greater than or equal to 0.', self.__class__, self.start)

        if self.stop <= self.start:
            raise CustomIndexError('"stop" must be greater than "start".', self.__class__, (self.start, self.stop))

        self.expr_src = expr_src
        self.curr = self.start

    def __next__(self) -> str:
        """
        Returns the next variable name in the sequence.

        Returns:
            The next variable name.

        Raises:
            StopIteration: When the end of the range is reached.
        """
        if self.curr >= self.stop:
            raise StopIteration

        var = self[self.curr]

        self.curr += self.step

        return var

    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None, SupportsIndex, SupportsIndex | None]
    ) -> str:
        indices = range(self.start, self.stop, self.step)

        if isinstance(index, slice):
            return " ".join(self._format_var(i, self.expr_src) for i in indices[index])

        try:
            return self._format_var(indices[index], self.expr_src)
        except IndexError:
            raise CustomIndexError("Index out of range", self.__class__, index) from None

    def __len__(self) -> int:
        """
        Returns the number of variable names that will be generated.

        Returns:
            The number of elements in the iterable.
        """
        return len(range(self.start, self.stop, self.step))

    @staticmethod
    def _format_var(i: int, expr_src: bool) -> str:
        return f"src{i}" if expr_src or i > 25 else EXPR_VARS[i]

    @classmethod
    def get_var(cls, value: SupportsIndex) -> str:
        """
        Get a variable name for a specific index.

        Args:
            value: Index to convert to variable name.

        Returns:
            The variable name.

        Raises:
            CustomIndexError: If the index is negative.
        """
        value = value.__index__()

        if value < 0:
            raise CustomIndexError('"value" should be bigger than 0!')

        return cls._format_var(value, value > 25)

    def __str__(self) -> str:
        """
        Returns the string representation of the variable sequence.

        Returns:
            Space-separated variable names.
        """
        return " ".join(self)

    @classmethod
    def cycle(cls) -> Iterator[str]:
        """
        An infinite generator of variable names, looping through `EXPR_VARS` then continuing with `srcX` style.

        Returns:
            An infinite iterator of variable names.
        """
        for x in count():
            yield cls.get_var(x)


@cache
def _get_akarin_expr_version() -> _ReturnDict_akarin_Version:
    return vs.core.akarin.Version()
