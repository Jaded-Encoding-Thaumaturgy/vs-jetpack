from __future__ import annotations

from collections.abc import Callable
from math import cos, exp, log, pi, sin, sqrt
from typing import Any

from vstools import vs

from ...abstract import CustomComplexKernel, CustomComplexTapsKernel
from ..zimg import Bilinear, Lanczos, Point

__all__ = [
    "BlackMan",
    "BlackManMinLobe",
    "Bohman",
    "Box",
    "Cosine",
    "CustomBilinear",
    "CustomLanczos",
    "CustomPoint",
    "Gaussian",
    "Hamming",
    "Hann",
    "Sinc",
    "Welch",
    "WindowedSinc",
]


def sinc(x: float) -> float:
    return 1.0 if x == 0.0 else sin(x * pi) / (x * pi)


class CustomPoint(CustomComplexKernel, Point):
    """
    Point resizer using the [CustomKernel][vskernels.CustomKernel] class.
    """

    _static_kernel_radius = 0

    def kernel(self, *, x: float) -> float:
        return 1.0


class CustomBilinear(CustomComplexKernel, Bilinear):
    """
    Bilinear resizer using the [CustomKernel][vskernels.CustomKernel] class.
    """

    _static_kernel_radius = 1

    def kernel(self, *, x: float) -> float:
        return max(1.0 - abs(x), 0.0)


class CustomLanczos(CustomComplexTapsKernel, Lanczos):
    """
    Lanczos resizer using the [CustomKernel][vskernels.CustomKernel] class.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

    def kernel(self, *, x: float) -> float:
        x, taps = abs(x), self.kernel_radius

        return sinc(x) * sinc(x / taps) if x < taps else 0.0

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return CustomComplexTapsKernel.get_params_args(self, is_descale, clip, width, height, **kwargs)


class Gaussian(CustomComplexTapsKernel):
    """
    Gaussian resizer.
    """

    class Sigma(float):
        """
        A class for Gaussian sigma scaling transformations.
        """

        def from_fmtc(self, curve: float) -> float:
            """
            Converts a curve value from fmtc to the Gaussian sigma.
            """
            if not curve:
                return 0.0
            return sqrt(1.0 / (2.0 * (curve / 10.0) * log(2)))

        def to_fmtc(self, sigma: float) -> float:
            """
            Converts a Gaussian sigma to fmtc's curve value.
            """
            if not sigma:
                return 0.0
            return 10 / (2 * log(2) * (sigma**2))

        def from_libplacebo(self, sigma: float) -> float:
            """
            Converts a sigma value from libplacebo to the Gaussian sigma.
            """
            if not sigma:
                return 0.0
            return sqrt(sigma / 4)

        def to_libplacebo(self, sigma: float) -> float:
            """
            Converts a Gaussian sigma to libplacebo's sigma value.
            """
            if not sigma:
                return 0.0
            return 4 * (sigma**2)

    def __init__(self, sigma: float = 0.5, taps: float = 2, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific sigma and number of taps.

        Args:
            sigma: The standard deviation (sigma) of the Gaussian function. It is the same as imagemagick's sigma
                scaling.
            taps: Determines the radius of the kernel.
        """

        self._sigma = sigma

        super().__init__(taps, **kwargs)

    @property
    def sigma(self) -> Sigma:
        return self.Sigma(self._sigma)

    def kernel(self, *, x: float) -> float:
        return 1 / (self._sigma * sqrt(2 * pi)) * exp(-(x**2) / (2 * self._sigma**2))


class Box(CustomComplexKernel):
    """
    Box resizer.
    """

    _static_kernel_radius = 1

    def kernel(self, *, x: float) -> float:
        return 1.0 if x >= -0.5 and x < 0.5 else 0.0


def noop_window(x: float, radius: int) -> float:
    return 1.0


def blackman_window(x: float, radius: int) -> float:
    w_x = x * (pi / radius)
    return 0.42 + 0.50 * cos(w_x) + 0.08 * cos(w_x * 2)


def blackman_minlobe_window(x: float, radius: int) -> float:
    w_x = x * (pi / radius)
    return 0.355768 + 0.487396 * cos(w_x) + 0.144232 * cos(w_x * 2) + 0.012604 * cos(w_x * 3)


def hann_window(x: float, radius: int) -> float:
    return 0.5 + 0.5 * cos(pi * x / radius)


def hamming_window(x: float, radius: int) -> float:
    return 0.54 + 0.46 * cos(pi * x / radius)


def welch_window(x: float, radius: int) -> float:
    x /= radius
    return 1.0 - x * x


def cosine_window(x: float, radius: int) -> float:
    cosine = cos(pi * x / radius)
    return 0.34 + cosine * (0.5 + cosine * 0.16)


def bohman_window(x: float, radius: int) -> float:
    x /= radius

    cosine = cos(pi * x)
    sine = sqrt(1.0 - cosine * cosine)

    return (1.0 - x) * cosine + (1.0 / pi) * sine


class WindowedSinc(CustomComplexTapsKernel):
    """Generic windowed-sinc kernel."""

    def __init__(
        self,
        taps: float = 3,
        window: Callable[[float, int], float] | None = None,
        cutoff: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Evaluates as `sinc(x) * window(abs(x), kernel_radius)` inside the kernel radius and `0` outside it.

        Passing `window=None` produces a plain truncated sinc.

        Args:
            taps: Determines the radius of the kernel.
            window: A function `(abs(distance), radius) -> coefficient` to shape the sinc wave.
                Defaults to a plain truncated sinc if omitted.
            cutoff: Optional float support bound.
                Samples beyond this evaluate to 0, even if they fall within the `taps` radius.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.window = window or noop_window
        self.cutoff = cutoff
        super().__init__(taps, **kwargs)

    def kernel(self, *, x: float) -> float:
        x = abs(x)
        if x >= self.kernel_radius or (self.cutoff is not None and x >= self.cutoff):
            return 0.0

        return sinc(x) * self.window(x, self.kernel_radius)


class BlackMan(WindowedSinc):
    """
    Blackman windowed-sinc resizer.

    This is equivalent to `WindowedSinc(taps, window=blackman_window)`.
    """

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, blackman_window, **kwargs)


class BlackManMinLobe(WindowedSinc):
    """
    Blackmanminlobe windowed-sinc resizer.

    This is equivalent to `WindowedSinc(taps, window=blackman_minlobe_window)`.
    """

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, blackman_minlobe_window, **kwargs)


class Sinc(WindowedSinc):
    """
    Plain truncated sinc resizer.

    This is equivalent to `WindowedSinc(taps)` with the default no-op window.
    """

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, noop_window, **kwargs)


class Hann(WindowedSinc):
    """
    Hann windowed-sinc kernel.

    This is equivalent to `WindowedSinc(taps, window=hann_window)`.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, hann_window, **kwargs)


class Hamming(WindowedSinc):
    """
    Hamming windowed-sinc kernel.

    This is equivalent to `WindowedSinc(taps, window=hamming_window)`.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, hamming_window, **kwargs)


class Welch(WindowedSinc):
    """
    Welch windowed-sinc kernel.

    This is equivalent to `WindowedSinc(taps, window=welch_window)`.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, welch_window, **kwargs)


class Cosine(WindowedSinc):
    """
    Cosine windowed-sinc kernel.

    This is equivalent to `WindowedSinc(taps, window=cosine_window)`.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, cosine_window, **kwargs)


class Bohman(WindowedSinc):
    """
    Bohman windowed-sinc kernel.

    This is equivalent to `WindowedSinc(taps, window=bohman_window)`.
    """

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        super().__init__(taps, bohman_window, **kwargs)
