# TODO: remove this
import warnings
from functools import cache
from pathlib import Path
from typing import Any

from .abstract import *
from .exceptions import *
from .kernels import *
from .types import *
from .util import *

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
warnings.simplefilter("always", DeprecationWarning)


@cache
def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"'{name}' is deprecated and will be removed in a future version. Use '{name[:-1]}Like' instead.",
        DeprecationWarning,
        stacklevel=3,
        skip_file_prefixes=(str(Path(__file__).resolve()),),
    )


def __getattr__(name: str) -> Any:
    if name in _alias_map:
        _warn_deprecated(name)
        return _alias_map[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
