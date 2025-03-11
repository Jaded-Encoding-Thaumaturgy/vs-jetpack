from __future__ import annotations

from itertools import count
from typing import Any, Callable, Iterable, Iterator, Sequence, SupportsIndex, TypeAlias, overload

from vstools import (
    EXPR_VARS, MISSING, ColorRange, CustomIndexError, CustomNotImplementedError, CustomRuntimeError, FuncExceptT,
    HoldsVideoFormatT, MissingT, PlanesT, VideoFormatT, classproperty, core, fallback, get_video_format,
    normalize_planes, normalize_seq, vs
)

__all__ = [
    # VS variables
    'complexpr_available',
    # Expr helpers
    'ExprVars', 'ExprVarsT', 'ExprVarRangeT', 'bitdepth_aware_tokenize_expr',
    # VS helpers
    'norm_expr_planes'
]


class _complexpr_available:
    @property
    def fp16(self) -> bool:
        from .funcs import expr_func

        if not hasattr(self, '_fp16_available'):
            try:
                expr_func(core.std.BlankClip(format=vs.GRAYH), 'x dup *')
                self._fp16_available = True
            except Exception:
                self._fp16_available = False

        return self._fp16_available

    def __bool__(self) -> bool:
        try:
            return bool(core.akarin.Expr)
        except AttributeError:
            ...

        return False


complexpr_available = _complexpr_available()


class _ExprVars(Iterable[str]):
    start: int
    stop: int
    step: int
    curr: int
    akarin: bool

    @overload
    def __init__(self, stop: SupportsIndex | ExprVarRangeT, /, *, akarin: bool | None = None) -> None:
        ...

    @overload
    def __init__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> None:
        ...

    def __init__(
        self, start_stop: SupportsIndex | ExprVarRangeT, stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1, /, *, akarin: bool | None = None
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

        self.akarin = self._check_akarin(self.stop, akarin)

        self.curr = self.start

    @overload
    def __call__(self, stop: SupportsIndex | ExprVarRangeT, /, *, akarin: bool | None = None) -> _ExprVars:
        ...

    @overload
    def __call__(
        self, start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> _ExprVars:
        ...

    def __call__(
        self, start_stop: SupportsIndex | ExprVarRangeT, stop: SupportsIndex | MissingT = MISSING,
        step: SupportsIndex = 1, /, *, akarin: bool | None = None
    ) -> _ExprVars:
        return ExprVars(start_stop, stop, step, akarin=akarin)  # type: ignore

    def __iter__(self) -> Iterator[str]:
        indices = range(self.start, self.stop, self.step)

        if self.akarin:
            return (f'src{x}' for x in indices)

        return (EXPR_VARS[x] for x in indices)

    def __next__(self) -> str:
        if self.curr >= self.stop:
            raise StopIteration

        var = f'src{self.curr}' if self.akarin else EXPR_VARS[self.curr]

        self.curr += self.step

        return var

    def __len__(self) -> int:
        return self.stop - self.start

    @classmethod
    def _check_akarin(cls, stop: SupportsIndex, akarin: bool | None = None) -> bool:
        stop = stop.__index__()

        if akarin is None:
            akarin = stop > 26

        if akarin and not complexpr_available:
            raise cls._get_akarin_err(
                'You are trying to get more than 26 variables or srcX vars, you need akarin plugin!'
            )

        return akarin

    @classmethod
    def get_var(cls, value: SupportsIndex, akarin: bool | None = None) -> str:
        value = value.__index__()

        if value < 0:
            raise CustomIndexError('"value" should be bigger than 0!')

        akarin = cls._check_akarin(value + 1, akarin)

        return f'src{value}' if akarin else EXPR_VARS[value]

    @classmethod
    def _get_akarin_err(cls, message: str = 'You need the akarin plugin to run this function!') -> CustomRuntimeError:
        return CustomRuntimeError(f'{message}\nDownload it from https://github.com/AkarinVS/vapoursynth-plugin')

    @overload
    def __class_getitem__(cls, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str:
        ...

    @overload
    def __class_getitem__(cls, slice: slice | tuple[slice, bool], /) -> list[str]:
        ...

    def __class_getitem__(
        cls, idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool], /,
    ) -> str | list[str]:
        if isinstance(idx_slice, tuple):
            idx_slice, akarin = idx_slice
        else:
            akarin = None

        if isinstance(idx_slice, slice):
            return list(ExprVars(  # type: ignore
                idx_slice.start or 0, fallback(idx_slice.stop, MISSING), fallback(idx_slice.step, 1)
            ))
        elif isinstance(idx_slice, SupportsIndex):
            return ExprVars.get_var(idx_slice.__index__(), akarin)

        raise CustomNotImplementedError

    @overload
    def __getitem__(self, index: SupportsIndex | tuple[SupportsIndex, bool], /) -> str:
        ...

    @overload
    def __getitem__(self, slice: slice | tuple[slice, bool], /) -> list[str]:
        ...

    def __getitem__(  # type: ignore
        self, idx_slice: SupportsIndex | slice | tuple[SupportsIndex | slice, bool], /,
    ) -> str | list[str]:
        ...

    def __str__(self) -> str:
        return ' '.join(iter(self))

    @classproperty
    def cycle(cls) -> Iterator[str]:
        for x in count():
            yield cls.get_var(x)


ExprVars: _ExprVars = _ExprVars  # type: ignore
ExprVarsT: TypeAlias = _ExprVars
ExprVarRangeT: TypeAlias = ExprVarsT | HoldsVideoFormatT | VideoFormatT | SupportsIndex


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
            replaces.append(
                (f'{token.__class__.__name__}.{token.name}', token.get_value)
            )

    if not replaces:
        return expr

    clips = list(clips)
    ranges = [ColorRange.from_video(c, func=func) for c in clips]

    mapped_clips = list(reversed(list(zip(['', *EXPR_VARS], clips[:1] + clips, ranges[:1] + ranges))))

    for mkey, function in replaces:
        if mkey in expr:
            for key, clip, crange in [
                (f'{mkey}_{k} ' if k else f'{mkey} ', clip, crange)
                for k, clip, crange in mapped_clips
            ]:
                expr = expr.replace(key, str(function(clip, chroma, crange) * 1.0) + ' ')

        if mkey in expr:
            raise CustomIndexError('Parsing error or not enough clips passed!', func, reason=expr)

    return expr


def norm_expr_planes(
    clip: vs.VideoNode, expr: str | list[str], planes: PlanesT = None, **kwargs: Any
) -> list[str]:
    assert clip.format

    expr_array = normalize_seq(expr, clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    string_args = [(key, normalize_seq(value)) for key, value in kwargs.items()]

    return [
        exp.format(**({'plane_idx': i} | {key: value[i] for key, value in string_args}))
        if i in planes else '' for i, exp in enumerate(expr_array, 0)
    ]
