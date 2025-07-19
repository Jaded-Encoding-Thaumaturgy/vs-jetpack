from __future__ import annotations

import re
from itertools import count
from typing import Callable, Iterable, Iterator, Sequence, SupportsIndex, TypeAlias, overload

from jetpytools import SupportsString

from vstools import (
    EXPR_VARS,
    MISSING,
    ColorRange,
    CustomIndexError,
    CustomNotImplementedError,
    FuncExceptT,
    HoldsVideoFormatT,
    MissingT,
    PlanesT,
    VideoFormatT,
    classproperty,
    fallback,
    get_video_format,
    normalize_planes,
    normalize_seq,
    vs,
)

__all__ = [
    "ExprVarRangeT",
    "ExprVars",
    "ExprVarsT",
    "ExprVarRangeT",
    "extra_op_tokenize_expr",
    "bitdepth_aware_tokenize_expr",
    "norm_expr_planes",
]


class _ExprVars(Iterable[str]):
    start: int
    stop: int
    step: int
    curr: int
    akarin: bool

    @overload
    def __init__(self, stop: SupportsIndex | ExprVarRangeT, /, *, akarin: bool | None = None) -> None: ...

    @overload
    def __init__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> None: ...

    def __init__(
        self,
        start_stop: SupportsIndex | ExprVarRangeT,
        stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1,
        /,
        *,
        akarin: bool | None = None,
    ) -> None:
        if isinstance(start_stop, ExprVarsT):
            self.start = start_stop.start
            self.stop = start_stop.stop
            self.step = start_stop.step
            self.curr = start_stop.curr
            self.akarin = start_stop.akarin
            return

        if stop is MISSING:
            self.start = 0
            if isinstance(start_stop, HoldsVideoFormatT | VideoFormatT):
                self.stop = get_video_format(start_stop).num_planes
            else:
                self.stop = start_stop.__index__()
        else:
            self.start = 0 if start_stop is None else start_stop.__index__()  # type: ignore
            self.stop = 255 if stop is None else stop.__index__()

        self.step = 1 if step is None else step.__index__()

        if self.start < 0:
            raise CustomIndexError('"start" must be bigger or equal than 0!')
        elif self.stop <= self.start:
            raise CustomIndexError('"stop" must be bigger than "start"!')

        self.curr = self.start

    @overload
    def __call__(self, stop: SupportsIndex | ExprVarRangeT, /, *, akarin: bool | None = None) -> _ExprVars: ...

    @overload
    def __call__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> _ExprVars: ...

    def __call__(
        self,
        start_stop: SupportsIndex | ExprVarRangeT,
        stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1,
        /,
        *,
        akarin: bool | None = None,
    ) -> _ExprVars:
        return ExprVars(start_stop, stop, step, akarin=akarin)  # type: ignore

    def __iter__(self) -> Iterator[str]:
        indices = range(self.start, self.stop, self.step)

        if self.akarin:
            return (f"src{x}" for x in indices)

        return (EXPR_VARS[x] for x in indices)

    def __next__(self) -> str:
        if self.curr >= self.stop:
            raise StopIteration

        var = f"src{self.curr}" if self.akarin else EXPR_VARS[self.curr]

        self.curr += self.step

        return var

    def __len__(self) -> int:
        return self.stop - self.start

    @classmethod
    def get_var(cls, value: SupportsIndex, akarin: bool | None = None) -> str:
        value = value.__index__()

        if value < 0:
            raise CustomIndexError('"value" should be bigger than 0!')

        return f"src{value}" if value >= 26 else EXPR_VARS[value]

    @overload
    def __class_getitem__(cls, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str: ...

    @overload
    def __class_getitem__(cls, slice: slice | tuple[slice, bool], /) -> list[str]: ...

    def __class_getitem__(
        cls,
        idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool],
        /,
    ) -> str | list[str]:
        if isinstance(idx_slice, tuple):
            idx_slice, akarin = idx_slice
        else:
            akarin = None

        if isinstance(idx_slice, slice):
            return list(ExprVars(idx_slice.start or 0, fallback(idx_slice.stop, MISSING), fallback(idx_slice.step, 1)))
        elif isinstance(idx_slice, SupportsIndex):
            return ExprVars.get_var(idx_slice.__index__(), akarin)

        raise CustomNotImplementedError

    @overload
    def __getitem__(self, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str: ...

    @overload
    def __getitem__(self, slice: slice | tuple[slice, bool], /) -> list[str]: ...

    def __getitem__(  # type: ignore
        self,
        idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool],
        /,
    ) -> str | list[str]: ...

    def __str__(self) -> str:
        return " ".join(iter(self))

    @classproperty
    @classmethod
    def cycle(cls) -> Iterator[str]:
        for x in count():
            yield cls.get_var(x)


ExprVars: _ExprVars = _ExprVars  # type: ignore
ExprVarsT: TypeAlias = _ExprVars
ExprVarRangeT: TypeAlias = ExprVarsT | HoldsVideoFormatT | VideoFormatT | SupportsIndex


def extra_op_tokenize_expr(expr: str) -> str:
    # Workaround for the not implemented op
    from .exprop import ExprOp

    for extra_op in ExprOp._extra_op_names_:
        expr = re.sub(rf"\b{extra_op.lower()}\b", getattr(ExprOp, extra_op).convert_extra(), expr)

    return expr


def bitdepth_aware_tokenize_expr(
    clips: Sequence[vs.VideoNode], expr: str, chroma: bool, func: FuncExceptT | None = None
) -> str:
    from .exprop import ExprToken

    func = func or bitdepth_aware_tokenize_expr

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

    mapped_clips = reversed(list(zip(["", *EXPR_VARS], clips[:1] + clips, ranges[:1] + ranges)))

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
