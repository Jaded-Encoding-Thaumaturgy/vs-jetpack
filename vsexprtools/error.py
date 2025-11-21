import sys
from typing import Any, Sequence

from jetpytools import CustomRuntimeError, FuncExcept, SupportsString, norm_func_name, to_arr
from jetpytools.exceptions.base import CustomErrorMeta

from vstools import VSObject, VSObjectMeta, get_video_format, vs

from .util import ExprVars

__all__ = ["CustomExprError"]


def _color_tag(string: str, tag_start: str, tag_end: str = "\033[0m") -> str:
    if sys.stdout and sys.stdout.isatty():
        return f"{tag_start}{string}{tag_end}"
    return string


class _CustomExprErrorMeta(VSObjectMeta, CustomErrorMeta): ...


class CustomExprError(VSObject, CustomRuntimeError, metaclass=_CustomExprErrorMeta):
    """Thrown when a Expr error occurs."""

    def __init__(
        self,
        message: SupportsString,
        func: FuncExcept,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        expr: str | Sequence[str],
        fmt: int | None,
        opt: bool,
        boundary: bool,
        **kwargs: Any,
    ) -> None:
        self.clips = to_arr(clips)
        self.expr = to_arr(expr)
        self.fmt = fmt
        self.opt = opt
        self.boundary = boundary
        super().__init__(message, func, self.expr, **kwargs)

    def __str__(self) -> str:
        func_header = _color_tag(norm_func_name(self.func).strip(), "\033[0;36m")
        func_header = f"({func_header}) "

        clips_info = [
            _color_tag("Clip(s):", "\033[0;33m"),
            *(_color_tag(f"    {var}:", "\033[1;37m") + f" {c!r}" for c, var in zip(self.clips, ExprVars.cycle())),
        ]

        expr_infos = [
            _color_tag("Expression(s):", "\033[0;33m"),
            *(_color_tag(f"    Plane {i}:", "\033[1;37m") + f" {e!r}" for i, e in enumerate(self.expr)),
        ]

        args_infos = [
            _color_tag("Flags:", "\033[0;33m"),
            _color_tag("    Format:", "\033[1;37m")
            + f" {get_video_format(self.fmt) if self.fmt is not None else None!r}",
            _color_tag("    Integer evaluation:", "\033[1;37m") + f" {self.opt}",
            _color_tag("    Boundary type:", "\033[1;37m") + f" {('Clamped edges', 'Mirrored edges')[self.boundary]}",
        ]

        out = (
            f"{func_header}\n    {self.message!s}\n\n"
            + "\n".join(clips_info)
            + "\n\n"
            + "\n".join(expr_infos)
            + "\n\n"
            + "\n".join(args_infos)
        )

        return out
