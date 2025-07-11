from __future__ import annotations

from math import sqrt
from typing import Any, Callable, overload

from vstools import ConstantFormatVideoNode, CustomValueError, core, vs

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
    resample_function: Callable[..., ConstantFormatVideoNode] = core.lazy.resize2.Bicubic
    descale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Debicubic
    rescale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Bicubic

    def __init__(self, b: float = 0, c: float = 0.5, **kwargs: Any) -> None:
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

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)
        if is_descale:
            return args | {"b": self.b, "c": self.c}
        return args | {"filter_param_a": self.b, "filter_param_b": self.c}

    def get_bob_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().get_bob_args(
            clip, shift, filter="bicubic", filter_param_a=self.b, filter_param_b=self.c, **kwargs
        )

    @ZimgComplexKernel.cached_property
    def kernel_radius(self) -> int:
        if (self.b, self.c) == (0, 0):
            return 1
        return 2

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**{"b": self.b, "c": self.c} | attrs)


class BSpline(Bicubic):
    """
    BSpline resizer (b=1, c=0).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=1, c=0, **kwargs)


class Hermite(Bicubic):
    """
    Hermite resizer (b=0, c=0).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=0, **kwargs)


class Mitchell(Bicubic):
    """
    Mitchell resizer (b=1/3, c=1/3).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=1 / 3, c=1 / 3, **kwargs)


class Catrom(Bicubic):
    """
    Catrom resizer (b=0, c=0.5).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=0.5, **kwargs)


class FFmpegBicubic(Bicubic):
    """
    FFmpeg's swscale default resizer (b=0, c=0.6).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=0.6, **kwargs)


class AdobeBicubic(Bicubic):
    """
    Adobe's "Bicubic" interpolation preset resizer (b=0, c=0.75).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=0.75, **kwargs)


class AdobeBicubicSharper(Bicubic):
    """
    Adobe's "Bicubic Sharper" interpolation preset resizer (b=0, c=1, blur=1.05).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=1, blur=1.05, **kwargs)


class AdobeBicubicSmoother(Bicubic):
    """
    Adobe's "Bicubic Smoother" interpolation preset resizer (b=0, c=0.625, blur=1.15).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=5 / 8, blur=1.15, **kwargs)


class BicubicSharp(Bicubic):
    """
    BicubicSharp resizer (b=0, c=1).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(b=0, c=1, **kwargs)


class RobidouxSoft(Bicubic):
    """
    RobidouxSoft resizer (b=0.67962, c=0.16019).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        b = (9 - 3 * sqrt(2)) / 7
        c = (1 - b) / 2
        super().__init__(b=b, c=c, **kwargs)


class Robidoux(Bicubic):
    """
    Robidoux resizer (b=0.37822, c=0.31089).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        b = 12 / (19 + 9 * sqrt(2))
        c = 113 / (58 + 216 * sqrt(2))

        super().__init__(b=b, c=c, **kwargs)


class RobidouxSharp(Bicubic):
    """
    RobidouxSharp resizer (b=0.26201, c=0.36899).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        b = 6 / (13 + 7 * sqrt(2))
        c = 7 / (2 + 12 * sqrt(2))

        super().__init__(b=b, c=c, **kwargs)


class BicubicAuto(Bicubic):
    """
    Bicubic resizer that follows the rule of `b + 2c = ...`
    """

    @overload
    def __init__(self, b: float = ..., c: None = None, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, b: None = None, c: float = ..., **kwargs: Any) -> None: ...

    def __init__(self, b: float | None = None, c: float | None = None, **kwargs: Any) -> None:
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
