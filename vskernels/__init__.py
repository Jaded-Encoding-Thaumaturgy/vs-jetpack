"""
Kernels are a collection of wrappers pertaining to (de)scaling, format
conversion, and other related operations, all while providing a consistent and
clean interface. This allows for easy expansion and ease of use for any other
maintainers who wishes to use them in their own functions.

You can create presets for common scaling algorithms or settings, while ensuring
the interface will always remain the same, even across different plugins with
their own settings and expected behavior.
"""

# TODO: remove this
from typing import TYPE_CHECKING, Any
from warnings import simplefilter, warn

from .abstract import *
from .exceptions import *
from .kernels import *
from .types import *
from .util import *

__version__: str


if not TYPE_CHECKING:
    # ruff: noqa: F405
    _alias_map = {
        "ScalerT": ScalerLike,
        "DescalerT": DescalerLike,
        "ResamplerT": ResamplerLike,
        "KernelT": KernelLike,
        "ComplexScalerT": ComplexScalerLike,
        "ComplexDescalerT": ComplexDescalerLike,
        "ComplexKernelT": ComplexKernelLike,
        "CustomComplexKernelT": CustomComplexKernelLike,
        "ZimgComplexKernelT": ZimgComplexKernelLike,
    }

    class _TypeAliasDeprecation(DeprecationWarning): ...

    simplefilter("module", _TypeAliasDeprecation)

    def __getattr__(name: str) -> Any:
        if name in _alias_map:
            from pathlib import Path

            warn(
                f"'{name}' is deprecated and will be removed in a future version. Use '{name[:-1]}Like' instead.",
                _TypeAliasDeprecation,
                stacklevel=2,
                skip_file_prefixes=(str(Path(__file__).resolve()),),
            )

            return _alias_map[name]

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
