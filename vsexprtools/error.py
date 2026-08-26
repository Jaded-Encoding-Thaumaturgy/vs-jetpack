import os
import sys
from collections.abc import Sequence
from typing import Any, SupportsInt

from jetpytools import CustomRuntimeError, FuncExcept, SupportsString, norm_func_name, to_arr
from jetpytools.exceptions.base import CustomErrorMeta

from vstools import VSObject, VSObjectMeta, get_video_format, vs

from .util import ExprVars

__all__ = ["CustomExprError"]


def _color_tag(string: str, tag_start: str, tag_end: str = "\033[0m", should_color: bool | None = None) -> str:
    if should_color is None:
        should_color = sys.stdout and sys.stdout.isatty()

    return f"{tag_start}{string}{tag_end}" if should_color else string


class _CustomExprErrorMeta(VSObjectMeta, CustomErrorMeta): ...


class CustomExprError(VSObject, CustomRuntimeError, metaclass=_CustomExprErrorMeta):
    """Thrown when a Expr error occurs."""

    def __init__(
        self,
        message: SupportsString,
        func: FuncExcept,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        expr: str | Sequence[str],
        fmt: SupportsInt | None,
        boundary: bool,
        **kwargs: Any,
    ) -> None:
        self.clips = to_arr(clips)
        self.expr = to_arr(expr)
        self.fmt = fmt
        self.boundary = boundary
        super().__init__(message, func, self.expr, **kwargs)

    def __str__(self) -> str:
        truthy = {"1", "true", "yes", "on"}
        should_color = (
            bool(sys.stdout)
            and sys.stdout.isatty()
            and os.getenv("NO_COLOR", "").lower() not in truthy
            and os.getenv("JETPYTOOLS_NO_COLOR", "").lower() not in truthy
            and os.getenv("VSEXPRTOOLS_NO_COLOR", "").lower() not in truthy
        )

        func_header = _color_tag(norm_func_name(self.func).strip(), "\033[0;36m", should_color=should_color)
        func_header = f"({func_header}) "

        clips_info = [
            _color_tag("Clip(s):", "\033[0;33m", should_color=should_color),
            *(
                _color_tag(f"    {var}:", "\033[1;37m", should_color=should_color) + f" {c!r}"
                for c, var in zip(self.clips, ExprVars.cycle())
            ),
        ]

        expr_infos = [
            _color_tag("Expression(s):", "\033[0;33m", should_color=should_color),
            *(
                _color_tag(f"    Plane {i}:", "\033[1;37m", should_color=should_color) + f" {e!r}"
                for i, e in enumerate(self.expr)
            ),
        ]

        args_infos = [
            _color_tag("Flags:", "\033[0;33m", should_color=should_color),
            _color_tag("    Format:", "\033[1;37m", should_color=should_color)
            + f" {get_video_format(self.fmt) if self.fmt is not None else None!r}",
            _color_tag("    Boundary type:", "\033[1;37m", should_color=should_color)
            + f" {('Clamped edges', 'Mirrored edges')[self.boundary]}",
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
