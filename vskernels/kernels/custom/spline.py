from __future__ import annotations

from math import comb
from typing import Any

from ...abstract import CustomComplexTapsKernel
from .helpers import poly3

__all__ = ["Spline"]

# ruff: noqa: N802


class Spline(CustomComplexTapsKernel):
    """
    Spline resizer with an arbitrary number of taps.
    """

    def __init__(self, taps: float = 2, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)
        self._coefs = self._splineKernelCoeff()

    def _naturalCubicSpline(self, values: list[int]) -> list[Any]:
        import numpy as np

        n = len(values) - 1

        rhs = values[:-1] + values[1:] + [0] * (2 * n)

        eqns = list[list[int]]()
        # left value = sample
        eqns += [[0] * (4 * i) + [i**3, i**2, i, 1] + [0] * (4 * (n - i - 1)) for i in range(n)]
        # right value = sample
        eqns += [[0] * (4 * i) + [(i + 1) ** 3, (i + 1) ** 2, i + 1, 1] + [0] * (4 * (n - i - 1)) for i in range(n)]
        # derivatives match
        eqns += [
            (
                [0] * (4 * i)
                + [3 * (i + 1) ** 2, 2 * (i + 1), 1, 0]
                + [-3 * (i + 1) ** 2, -2 * (i + 1), -1, 0]
                + [0] * (4 * (n - i - 2))
            )
            for i in range(n - 1)
        ]
        # second derivatives match
        eqns += [
            [0] * (4 * i) + [6 * (i + 1), 2, 0, 0] + [-6 * (i + 1), -2, 0, 0] + [0] * (4 * (n - i - 2))
            for i in range(n - 1)
        ]
        eqns += [[0, 2, 0, 0] + [0] * (4 * (n - 1))]
        eqns += [[0] * (4 * (n - 1)) + [6 * n**2, 2 * n, 0, 0]]

        assert len(rhs) == len(eqns)

        return list(np.linalg.solve(np.array(eqns), np.array(rhs)))

    def _splineKernelCoeff(self) -> list[Any]:
        import numpy as np

        taps = self.kernel_radius

        coeffs = list[np.float64]()

        def _shiftPolynomial(coeffs: list[np.float64], shift: float) -> list[Any]:
            return [
                sum(c * comb(k, m) * (-shift) ** max(0, k - m) for k, c in enumerate(coeffs[::-1]))
                for m in range(len(coeffs))
            ][::-1]

        for i in range(taps):
            samplept = taps - i - 1
            samples = [0] * samplept + [1] + [0] * (2 * taps - samplept - 1)

            assert len(samples) == 2 * taps

            coeffs += _shiftPolynomial(self._naturalCubicSpline(samples)[4 * taps - 4 : 4 * taps], -(taps - 1) + i)

        return coeffs

    def kernel(self, *, x: float) -> float:
        x, taps = abs(x), self.kernel_radius

        if x >= taps:
            return 0.0

        tap = int(x)

        a, b, c, d = self._coefs[4 * tap : 4 * tap + 4]

        return poly3(x, d, c, b, a)
