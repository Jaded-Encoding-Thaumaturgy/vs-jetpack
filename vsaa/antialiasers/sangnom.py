"""
This module implements wrappers for the SangNom single field deinterlacer using edge-directed interpolation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from vstools import ConstantFormatVideoNode, core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, Interpolater

__all__ = [
    "SANGNOM",

    "SangNomSS", "SangNomDR", "SangNomDR",

    'SangNom',
]


@dataclass
class SANGNOM(Interpolater):
    """Base class for SANGNOM interpolating methods."""

    aa_strength: int | Sequence[int] = 48
    """
    The strength of luma anti-aliasing, applied to an 8-bit clip.
    Must be an integer between 0 and 128, inclusive.
    """

    double_fps: bool = False
    """
    Whether to double the frame rate of the clip.
    """

    # Class Variable
    _shift = -0.5

    def _preprocess_clip(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        if self.double_fps:
            return clip.std.SeparateFields(self.field).std.DoubleWeave(self.field)
        return super()._preprocess_clip(clip)

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(aa=self.aa_strength, order=0 if self.double_fps else self.field + 1) | kwargs

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        interpolated = core.sangnom.SangNom(
            clip, dh=double_y or not self.drop_fields, **self.get_aa_args(clip, **kwargs) | kwargs
        )

        return self.shift_interpolate(clip, interpolated, double_y)


class SangNomSS(SANGNOM, SuperSampler):
    """Concrete implementation of SANGNOM used as a supersampler."""

    _static_kernel_radius = 2


class SangNomSR(SANGNOM, SingleRater):
    """Concrete implementation of SANGNOM used as a single-rater."""


class SangNomDR(SangNomSR, DoubleRater):
    """Concrete implementation of SANGNOM used as a double-rater."""


class SangNom(SangNomDR, SangNomSS, Antialiaser):
    """Full implementation of the SANGNOM anti-aliaser"""
