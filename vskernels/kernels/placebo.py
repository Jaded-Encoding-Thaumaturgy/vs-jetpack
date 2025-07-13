from __future__ import annotations

from math import ceil
from typing import Any, ClassVar

from vstools import core, fallback, vs

from ..abstract import ComplexScaler
from ..types import LeftShift, TopShift

__all__ = [
    "EwaBicubic",
    "EwaGinseng",
    "EwaHann",
    "EwaJinc",
    "EwaLanczos",
    "EwaLanczos4Sharpest",
    "EwaLanczosSharp",
    "EwaRobidoux",
    "EwaRobidouxSharp",
    "Placebo",
]


class Placebo(ComplexScaler, abstract=True):
    """
    Abstract Placebo scaler class.

    This class and its subclasses depend on [vs-placebo](https://github.com/sgt0/vs-placebo)
    """

    _kernel: ClassVar[str]
    """Name of the Placebo kernel"""

    scale_function = core.lazy.placebo.Resample

    def __init__(
        self,
        radius: float | None = None,
        b: float | None = None,
        c: float | None = None,
        clamp: float = 0.0,
        blur: float = 0.0,
        taper: float = 0.0,
        antiring: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the scaler with optional arguments.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            clamp: Represents an extra weighting/clamping coefficient for negative weights. A value of 0.0 represents no
                clamping. A value of 1.0 represents full clamping, i.e. all negative lobes will be removed.
            blur: Additional blur coefficient. This effectively stretches the kernel, without changing the effective
                radius of the filter radius.
            taper: Additional taper coefficient. This essentially flattens the function's center.
            antiring: Antiringing strength.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.radius = radius
        self.b = b
        self.c = c
        self.clamp = clamp
        self.blur = blur
        self.taper = taper
        self.antiring = antiring
        super().__init__(**kwargs)

    def get_scale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("format", None)

        return (
            {
                "sx": kwargs.pop("src_left", shift[1]),
                "sy": kwargs.pop("src_top", shift[0]),
                "width": width,
                "height": height,
                "filter": self._kernel,
                "radius": self.radius,
                "param1": self.b,
                "param2": self.c,
                "clamp": self.clamp,
                "taper": self.taper,
                "blur": self.blur,
                "antiring": self.antiring,
            }
            | self.kwargs
            | kwargs
            | {"linearize": False, "sigmoidize": False}
        )

    @ComplexScaler.cached_property
    def kernel_radius(self) -> int:
        if self.radius:
            return ceil(self.radius)

        if self.b or self.c:
            b, c = fallback(self.b, 0), fallback(self.c, 0.5)

            if (b, c) == (0, 0):
                return 1

        return 2


class EwaBicubic(Placebo):
    """
    Ewa Bicubic resizer.
    """

    _kernel = "ewa_robidoux"

    def __init__(self, b: float = 0.0, c: float = 0.5, radius: int | None = None, **kwargs: Any) -> None:
        """
        Initialize the scaler with specific 'b' and 'c' parameters and optional arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            radius: Overrides the filter kernel radius.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        if radius is None:
            radius = 1 if (b, c) == (0, 0) else 2

        super().__init__(radius, b, c, **kwargs)


class EwaLanczos(Placebo):
    """
    Ewa Lanczos resizer.
    """

    _kernel = "ewa_lanczos"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaLanczosSharp(Placebo):
    """Ewa Lanczos resizer."""

    _kernel = "ewa_lanczossharp"

    def __init__(
        self, radius: float = 3.2383154841662362076499, blur: float = 0.98125058372237073562493, **kwargs: Any
    ) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            blur: Additional blur coefficient. This effectively stretches the kernel,
                without changing the effective radius of the filter radius.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, blur=blur, **kwargs)


class EwaLanczos4Sharpest(Placebo):
    """Ewa Lanczos resizer."""

    _kernel = "ewa_lanczos4sharpest"

    def __init__(
        self,
        radius: float = 4.2410628637960698819573,
        blur: float = 0.88451209326050047745788,
        antiring: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            blur: Additional blur coefficient. This effectively stretches the kernel,
                without changing the effective radius of the filter radius.
            antiring: Antiringing strength.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, blur=blur, antiring=antiring, **kwargs)


class EwaJinc(Placebo):
    """
    Ewa Jinc resizer.
    """

    _kernel = "ewa_jinc"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaGinseng(Placebo):
    """
    Ewa Ginseng resizer.
    """

    _kernel = "ewa_ginseng"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaHann(Placebo):
    """
    Ewa Hann resizer.
    """

    _kernel = "ewa_hann"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaRobidoux(Placebo):
    """
    Ewa Robidoux resizer.
    """

    _kernel = "ewa_robidoux"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the kernel with optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(None, None, None, **kwargs)


class EwaRobidouxSharp(Placebo):
    """
    Ewa Robidoux Sharp resizer.
    """

    _kernel = "ewa_robidouxsharp"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the kernel with optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(None, None, None, **kwargs)
