from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from jetpytools import CustomRuntimeError

from ..error import _color_tag

if TYPE_CHECKING:
    from .manager import InlineExprWrapper


class CustomInlineExprError(CustomRuntimeError):
    """Thrown when a InlineExpr error occurs."""

    def add_stack_infos(self, expr_vars: dict[str, Any], ie: InlineExprWrapper) -> None:
        """
        Adds additional informations on the expr vars used within the with statement of a InlineExprWrapper.

        Args:
            expr_vars: Dictionary containing the current scope's local expr variables.
            ie: The InlineExprWrapper instance.
        """
        from .helpers import ComputedVar
        from .manager import InlineExprWrapper

        self.add_note("\n")
        self.add_note(_color_tag("InlineExpr stack infos:", "\033[0;33m"))

        for k, v in sorted(expr_vars.items()):
            if isinstance(v, InlineExprWrapper):
                continue

            if isinstance(v, ComputedVar):
                v = ("\n    " + (" " * (len(k) + 2))).join(v.to_str_per_plane(ie._format.num_planes))
            elif v is not None:
                with suppress(TypeError):
                    v = str([str(x) for x in iter(v)])

            self.add_note("    " + _color_tag(f"{k}: ", "\033[1;37m") + str(v))
