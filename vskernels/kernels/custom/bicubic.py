from typing import Any

from ...abstract import CustomComplexKernel
from ..zimg import Bicubic
from .helpers import poly3

__all__ = [
    "CustomBicubic",
]


class CustomBicubic(CustomComplexKernel, Bicubic):
    """
    Bicubic resizer using the `CustomKernel` class
    """

    class bic_vals:  # noqa: N801
        @staticmethod
        def p0(b: float, c: float) -> float:
            return (6.0 - 2.0 * b) / 6.0

        @staticmethod
        def p2(b: float, c: float) -> float:
            return (-18.0 + 12.0 * b + 6.0 * c) / 6.0

        @staticmethod
        def p3(b: float, c: float) -> float:
            return (12.0 - 9.0 * b - 6.0 * c) / 6.0

        @staticmethod
        def q0(b: float, c: float) -> float:
            return (8.0 * b + 24.0 * c) / 6.0

        @staticmethod
        def q1(b: float, c: float) -> float:
            return (-12.0 * b - 48.0 * c) / 6.0

        @staticmethod
        def q2(b: float, c: float) -> float:
            return (6.0 * b + 30.0 * c) / 6.0

        @staticmethod
        def q3(b: float, c: float) -> float:
            return (-b - 6.0 * c) / 6.0

    def __init__(self, b: float = 0, c: float = 0.5, **kwargs: Any) -> None:
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def kernel(self, *, x: float) -> float:
        x, b, c = abs(x), self.b, self.c

        if x < 1.0:
            return poly3(x, self.bic_vals.p0(b, c), 0.0, self.bic_vals.p2(b, c), self.bic_vals.p3(b, c))

        if x < 2.0:
            return poly3(
                x, self.bic_vals.q0(b, c), self.bic_vals.q1(b, c), self.bic_vals.q2(b, c), self.bic_vals.q3(b, c)
            )

        return 0.0
