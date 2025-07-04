from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import NamedTuple, Sequence

from vstools import vs

from .funcs import expr_func
from .operators import ExprOperators
from .polyfills import disable_poly, enable_poly
from .util import ExprVars
from .variables import ClipVar, ComputedVar, ExprVar

__all__ = ["InlineExpr", "inline_expr"]


class InlineExpr(NamedTuple):
    clips: list[ClipVar]
    op: ExprOperators
    out: inline_expr


class inline_expr(AbstractContextManager[InlineExpr]):  # noqa: N801
    _clips: list[vs.VideoNode]
    _in_context: bool
    _final_clip: vs.VideoNode | None
    _final_expr_node: ComputedVar

    def __init__(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> None:
        self._in_context = False

        self._clips = list(clips) if isinstance(clips, Sequence) else [clips]
        self._clips_char_map = [ClipVar(char, clip, self) for char, clip in zip(ExprVars.cycle, self._clips)]  # pyright: ignore[reportArgumentType]

        self._final_clip = None
        self._final_expr_node = self._clips_char_map[0].as_var()

    def __enter__(self) -> InlineExpr:
        self._in_context = True

        enable_poly()

        return InlineExpr(self._clips_char_map, ExprOperators(), self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._final_clip = self._get_clip()

        disable_poly()

        self._in_context = False

        return None

    def _get_clip(self) -> vs.VideoNode:
        fmt = self._clips[0].format
        assert fmt

        return expr_func(self._clips, [self._final_expr_node.to_str(plane=plane) for plane in range(fmt.num_planes)])

    @property
    def out(self) -> ComputedVar:
        return self._final_expr_node

    @out.setter
    def out(self, out_var: ExprVar) -> None:
        self._final_expr_node = ExprOperators.as_var(out_var)

    @property
    def clip(self) -> vs.VideoNode:
        if self._in_context:
            raise ValueError("You can only get the output clip out of the context manager!")

        if self._final_expr_node is None:
            raise ValueError("inline_expr: you need to call `out` with the output node!")

        if self._final_clip is None:
            raise ValueError("inline_expr: can't get output clip if the manager errored!")

        return self._final_clip
