from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Any, Unpack, overload

from jetpytools import CustomValueError, complex_hash

from vstools import core, vs

from ...abstract.base import ResolvableKwargs
from ...types import LeftShift, TopShift
from .abstract import ZimgComplexKernel

__all__ = [
    "AdobeBicubic",
    "AdobeBicubicSharper",
    "AdobeBicubicSmoother",
    "BSpline",
    "Bicubic",
    "BicubicAuto",
    "BicubicSharp",
    "Catrom",
    "FFmpegBicubic",
    "Hermite",
    "Mitchell",
    "Robidoux",
    "RobidouxSharp",
    "RobidouxSoft",
]


class Bicubic(ZimgComplexKernel):
    """
    Bicubic resizer.
    """

    scale_function: Callable[..., vs.VideoNode] = core.lazy.resize2.Bicubic
    resample_function: Callable[..., vs.VideoNode] = core.lazy.resize2.Bicubic
    descale_function: Callable[..., vs.VideoNode] = core.lazy.descale.Debicubic
    rescale_function: Callable[..., vs.VideoNode] = core.lazy.descale.Bicubic

    def __init__(self, b: float = 0, c: float = 0.5, **kwargs: Unpack[ResolvableKwargs]) -> None:
        """
        Initialize the scaler with specific 'b' and 'c' parameters and optional arguments.

        Args:
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def __hash__(self) -> int:
        # Remove the class name
        return complex_hash.hash(*((k, v) for k, v in self.__dict__.items() if not k.startswith("__")))

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)

        if is_descale:
            return {"b": self.b, "c": self.c} | args

        return {"filter_param_a": self.b, "filter_param_b": self.c} | args

    def get_bob_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().get_bob_args(
            clip, shift, filter="bicubic", **{"filter_param_a": self.b, "filter_param_b": self.c} | kwargs
        )

    @ZimgComplexKernel.cachedproperty
    def kernel_radius(self) -> int:
        if (self.b, self.c) == (0, 0):
            return 1
        return 2

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**{"b": round(self.b, 3), "c": round(self.c, 3)} | attrs)


class SpecialBicubic(Bicubic, abstract=True):
    """
    Base class for fixed-parameter bicubic presets.
    """

    b: float
    c: float
    blur: float = 1.0

    def __init__(self, **kwargs: Unpack[ResolvableKwargs]) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(**kwargs | {"b": self.b, "c": self.c} | ({"blur": self.blur} if self.blur != 1.0 else {}))


class BSpline(SpecialBicubic):
    """
    BSpline resizer (b=1, c=0).
    """

    b = 1
    c = 0


class Hermite(SpecialBicubic):
    """
    Hermite resizer (b=0, c=0).
    """

    b = 0
    c = 0


class Mitchell(SpecialBicubic):
    """
    Mitchell resizer (b=1/3, c=1/3).
    """

    b = 1 / 3
    c = 1 / 3


class Catrom(SpecialBicubic):
    """
    Catrom resizer (b=0, c=0.5).
    """

    b = 0
    c = 0.5


class FFmpegBicubic(SpecialBicubic):
    """
    FFmpeg's swscale default resizer (b=0, c=0.6).
    """

    b = 0
    c = 0.6


class AdobeBicubic(SpecialBicubic):
    """
    Adobe's "Bicubic" interpolation preset resizer (b=0, c=0.75).
    """

    b = 0
    c = 0.75


class AdobeBicubicSharper(SpecialBicubic):
    """
    Adobe's "Bicubic Sharper" interpolation preset resizer (b=0, c=1, blur=1.05).
    """

    b = 0
    c = 1
    blur = 1.05


class AdobeBicubicSmoother(SpecialBicubic):
    """
    Adobe's "Bicubic Smoother" interpolation preset resizer (b=0, c=0.625, blur=1.15).
    """

    b = 0
    c = 5 / 8
    blur = 1.15


class BicubicSharp(SpecialBicubic):
    """
    BicubicSharp resizer (b=0, c=1).
    """

    b = 0
    c = 1


class RobidouxSoft(SpecialBicubic):
    """
    RobidouxSoft resizer (b=0.67962, c=0.16019).
    """

    b = (9 - 3 * sqrt(2)) / 7
    c = (1 - b) / 2


class Robidoux(SpecialBicubic):
    """
    Robidoux resizer (b=0.37822, c=0.31089).
    """

    b = 12 / (19 + 9 * sqrt(2))
    c = 113 / (58 + 216 * sqrt(2))


class RobidouxSharp(SpecialBicubic):
    """
    RobidouxSharp resizer (b=0.26201, c=0.36899).
    """

    b = 6 / (13 + 7 * sqrt(2))
    c = 7 / (2 + 12 * sqrt(2))


class BicubicAuto(Bicubic):
    """
    Bicubic resizer that follows the rule of `b + 2c = ...`
    """

    @overload
    def __init__(self, b: float = ..., c: None = None, **kwargs: Unpack[ResolvableKwargs]) -> None: ...

    @overload
    def __init__(self, b: None = None, c: float = ..., **kwargs: Unpack[ResolvableKwargs]) -> None: ...

    def __init__(self, b: float | None = None, c: float | None = None, **kwargs: Unpack[ResolvableKwargs]) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            **kwargs: Keyword arguments that configure the internal scaling behavior.

        Raises:
            CustomValueError: If both 'b' and 'c' are specified
        """
        if None not in {b, c}:
            raise CustomValueError("You can't specify both b and c!", self.__class__)

        super().__init__(*self._get_bc_args(b, c), **kwargs)

    def _get_bc_args(self, b: float | None, c: float | None) -> tuple[float, float]:
        autob = 0.0 if b is None else b
        autoc = 0.5 if c is None else c

        if c is not None and b is None:
            autob = 1.0 - 2 * c
        elif c is None and b is not None:
            autoc = (1.0 - b) / 2

        return autob, autoc
