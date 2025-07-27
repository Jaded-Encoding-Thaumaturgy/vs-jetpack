from __future__ import annotations

import re
from functools import cache
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence, SupportsIndex, overload

from jetpytools import CustomTypeError, SupportsString
from typing_extensions import Self

if TYPE_CHECKING:
    from vapoursynth._nodes import _ReturnDict_akarin_Version  # pyright: ignore[reportMissingModuleSource]

from vstools import (
    EXPR_VARS,
    MISSING,
    ColorRange,
    CustomIndexError,
    FuncExceptT,
    HoldsVideoFormatT,
    MissingT,
    PlanesT,
    VideoFormatT,
    get_video_format,
    normalize_planes,
    normalize_seq,
    vs,
)

__all__ = [
    "ExprVars",
    "bitdepth_aware_tokenize_expr",
    "extra_op_tokenize_expr",
    "norm_expr_planes",
]


class ExprVars(Iterable[str]):
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
        self, stop: SupportsIndex | Self | HoldsVideoFormatT | VideoFormatT, /, *, expr_src: bool = False
    ) -> None: ...

    @overload
    def __init__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, expr_src: bool = False
    ) -> None: ...

    @overload
    def __init__(self, /, *args: Any, **kwargs: Any) -> None: ...

    def __init__(
        self,
        start_stop: SupportsIndex | Self | HoldsVideoFormatT | VideoFormatT,
        stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1,
        /,
        *,
        expr_src: bool = False,
    ) -> None:
        """
        Initialize an ExprVars instance.

        Args:
            start_stop: A start index or an object from which to infer the number of variables (e.g., video format).
            stop: Stop index (exclusive). Required only if `start_stop` is a numeric start value.
            step: Step size for iteration.
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

        if stop is MISSING:
            self.start = 0
            self.stop = (
                get_video_format(start_stop).num_planes
                if isinstance(start_stop, HoldsVideoFormatT | VideoFormatT)
                else start_stop.__index__()
            )
        else:
            if isinstance(start_stop, HoldsVideoFormatT | VideoFormatT):
                raise CustomTypeError("start cannot be a video format when stop is provided.", self, start_stop)
            self.start = start_stop.__index__()
            self.stop = stop.__index__()

        self.step = step.__index__()

        if self.start < 0:
            raise CustomIndexError('"start" must be greater than or equal to 0.', self, self.start)

        if self.stop <= self.start:
            raise CustomIndexError('"stop" must be greater than "start".', self, (self.start, self.stop))

        self.expr_src = expr_src
        self.curr = self.start

    @overload
    def __call__(
        self, stop: SupportsIndex | Self | HoldsVideoFormatT | VideoFormatT, /, *, expr_src: bool = False
    ) -> Self: ...

    @overload
    def __call__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, expr_src: bool = False
    ) -> Self: ...

    def __call__(
        self,
        start_stop: SupportsIndex | Self | HoldsVideoFormatT | VideoFormatT,
        stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1,
        /,
        *,
        expr_src: bool = False,
    ) -> Self:
        """
        Allows an ExprVars instance to be called like a function to create a new instance with new parameters.

        Args:
            start_stop: A start index or an object from which to infer the number of variables (e.g., video format).
            stop: Stop index (exclusive). Required only if `start_stop` is a numeric start value.
            step: Step size for iteration.
            expr_src: Whether to use `srcX` naming or use alphabetic variables.

        Returns:
            A new instance with the specified parameters.
        """

        return self.__class__(start_stop, stop, step, expr_src=expr_src)

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the variable names.

        Returns:
            Iterator yielding variable names (e.g., 'x', 'y', ..., or 'src0', 'src1', ...).
        """

        indices = range(self.start, self.stop, self.step)

        for x in indices:
            if self.expr_src or x > 25:
                yield f"src{x}"
            else:
                yield EXPR_VARS[x]

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

        var = f"src{self.curr}" if self.expr_src or self.curr > 25 else EXPR_VARS[self.curr]

        self.curr += self.step

        return var

    def __len__(self) -> int:
        """
        Returns the number of variable names that will be generated.

        Returns:
            The number of elements in the iterable.
        """
        return self.stop - self.start

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

        return f"src{value}" if value > 25 else EXPR_VARS[value]

    def __str__(self) -> str:
        """
        Returns the string representation of the variable sequence.

        Returns:
            Space-separated variable names.
        """
        return " ".join(iter(self))

    @classmethod
    def cycle(cls) -> Iterator[str]:
        """
        An infinite generator of variable names, looping through `EXPR_VARS` then continuing with `srcX` style.

        Returns:
            An infinite iterator of variable names.
        """
        for x in count():
            yield cls.get_var(x)


def extra_op_tokenize_expr(expr: str) -> str:
    # Workaround for the not implemented op
    from .exprop import ExprOp

    def _replace_polyval(matched: re.Match[str]) -> str:
        degree = int(matched.group(1))
        return ExprOp.POLYVAL.convert_extra(degree)

    for extra_op in ExprOp._extra_op_names_:
        if extra_op.lower() in expr:
            if extra_op == ExprOp.POLYVAL.name:
                expr = re.sub(rf"\b{extra_op.lower()}\b", _replace_polyval, expr)
            else:
                expr = re.sub(rf"\b{extra_op.lower()}\b", getattr(ExprOp, extra_op).convert_extra(), expr)

    return expr


def bitdepth_aware_tokenize_expr(
    clips: Sequence[vs.VideoNode], expr: str, chroma: bool, func: FuncExceptT | None = None
) -> str:
    from .exprop import ExprToken

    func = func or bitdepth_aware_tokenize_expr

    expr = extra_op_tokenize_expr(expr)

    if not expr or len(expr) < 4:
        return expr

    replaces = list[tuple[str, Callable[[vs.VideoNode, bool, ColorRange], float]]]()

    for token in sorted(ExprToken, key=lambda x: len(x), reverse=True):
        if token.value in expr:
            replaces.append((token.value, token.get_value))

        if token.name in expr:
            replaces.append((f"{token.__class__.__name__}.{token.name}", token.get_value))

    if not replaces:
        return expr

    clips = list(clips)
    ranges = [ColorRange.from_video(c, func=func) for c in clips]

    mapped_clips = list(reversed(list(zip(["", *EXPR_VARS], clips[:1] + clips, ranges[:1] + ranges))))

    for mkey, function in replaces:
        if mkey in expr:
            for key, clip, crange in [
                (f"{mkey}_{k}" if k else f"{mkey}", clip, crange) for k, clip, crange in mapped_clips
            ]:
                expr = re.sub(rf"\b{key}\b", str(function(clip, chroma, crange)), expr)

        if re.search(rf"\b{mkey}\b", expr):
            raise CustomIndexError("Parsing error or not enough clips passed!", func, reason=expr)

    return expr


def norm_expr_planes(
    clip: vs.VideoNode,
    expr: str | list[str],
    planes: PlanesT = None,
    **kwargs: Iterable[SupportsString] | SupportsString,
) -> list[str]:
    assert clip.format

    expr_array = normalize_seq(expr, clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    string_args = [(key, normalize_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**({"plane_idx": i} | {key: value[i] for key, value in string_args})) if i in planes else ""
        for i, exp in enumerate(expr_array)
    ]


@cache
def _get_akarin_expr_version() -> _ReturnDict_akarin_Version:
    return vs.core.akarin.Version()
