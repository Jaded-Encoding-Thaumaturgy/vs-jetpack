from __future__ import annotations

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
]


def sinc(x: float) -> float:
    return 1.0 if x == 0.0 else sin(x * pi) / (x * pi)


class CustomPoint(CustomComplexKernel, Point):
    """
    Point resizer using the `CustomKernel` class.
    """

    _static_kernel_radius = 0

    def kernel(self, *, x: float) -> float:
        return 1.0


class CustomBilinear(CustomComplexKernel, Bilinear):
    """
    Bilinear resizer using the `CustomKernel` class.
    """

    _static_kernel_radius = 1

    def kernel(self, *, x: float) -> float:
        return max(1.0 - abs(x), 0.0)


class CustomLanczos(CustomComplexTapsKernel, Lanczos):
    """
    Lanczos resizer using the `CustomKernel` class.
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


class BlackMan(CustomComplexTapsKernel):
    """
    Blackman resizer.
    """

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

    def _win_coef(self, x: float) -> float:
        w_x = x * (pi / self.kernel_radius)

        return 0.42 + 0.50 * cos(w_x) + 0.08 * cos(w_x * 2)

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return sinc(x) * self._win_coef(x)


class BlackManMinLobe(BlackMan):
    """
    Blackmanminlobe resizer.
    """

    def _win_coef(self, x: float) -> float:
        w_x = x * (pi / self.kernel_radius)

        return 0.355768 + 0.487396 * cos(w_x) + 0.144232 * cos(w_x * 2) + 0.012604 * cos(w_x * 3)


class Sinc(CustomComplexTapsKernel):
    """
    Sinc resizer.
    """

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return sinc(x)


class Hann(CustomComplexTapsKernel):
    """
    Hann kernel.
    """

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return 0.5 + 0.5 * cos(pi * x)


class Hamming(CustomComplexTapsKernel):
    """
    Hamming kernel.
    """

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return 0.54 + 0.46 * cos(pi * x)


class Welch(CustomComplexTapsKernel):
    """
    Welch kernel.
    """

    def kernel(self, *, x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0

        return 1.0 - x * x


class Cosine(CustomComplexTapsKernel):
    """
    Cosine kernel.
    """

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        cosine = cos(pi * x)

        return 0.34 + cosine * (0.5 + cosine * 0.16)


class Bohman(CustomComplexTapsKernel):
    """
    Bohman kernel.
    """

    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        cosine = cos(pi * x)
        sine = sqrt(1.0 - cosine * cosine)

        return (1.0 - x) * cosine + (1.0 / pi) * sine
