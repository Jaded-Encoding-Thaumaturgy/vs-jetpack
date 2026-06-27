"""
Functions and utils related to VapourSynth

This module is a collection of functions, utils, types, type-utils, and more
aimed at helping at having a common ground between VapourSynth packages, and
simplify writing them.
"""

from typing import TYPE_CHECKING, Any

from .enums import *
from .exceptions import *
from .functions import *
from .types import *
from .utils import *
from .vs_proxy import *

if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        import importlib
        import warnings

        try:
            attr = getattr(importlib.import_module("jetpytools"), name)
            warnings.warn(
                "Importing 'jetpytools' symbols via 'vstools' is deprecated and will be removed in a future version. "
                "Import them directly from 'jetpytools' instead.",
                DeprecationWarning,
            )
            return attr
        except AttributeError:
            ...

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
