"""
This module implements wrappers for the Neural Network Edge Directed Interpolation (3rd gen.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import ConstantFormatVideoNode, inject_self, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, Interpolater, _FullInterpolate

__all__ = [
    "NNEDI3",

    "Nnedi3SS", "Nnedi3DR", "Nnedi3DR",

    'Nnedi3',
]


@dataclass
class NNEDI3(_FullInterpolate, Interpolater):
    """Base class for NNEDI3 interpolating methods."""

    nsize: int = 0
    """
    Size of the local neighbourhood around each pixel used by the predictor neural network.
    Possible settings:
    - 0: 8x6
    - 1: 16x6
    - 2: 32x6
    - 3: 48x6
    - 4: 8x4
    - 5: 16x4
    - 6: 32x4
    """

    nns: int = 4
    """
    Number of neurons in the predictor neural network. Possible values:
    - 0: 16
    - 1: 32
    - 2: 64
    - 3: 128
    - 4: 256
    """

    qual: int = 2
    """
    The number of different neural network predictions that are blended together to compute the final output value.
    Each neural network was trained on a different set of training data.
    Blending the results of these different networks improves generalisation to unseen data.
    Possible values are 1 and 2.
    """

    etype: int = 0
    """
    The set of weights used in the predictor neural network. Possible values:
    - 0: Weights trained to minimise absolute error.
    - 1: Weights trained to minimise squared error.
    """

    pscrn: int = 1
    """
    The prescreener used to decide which pixels should be processed by the predictor neural network,
    and which can be handled by simple cubic interpolation.
    Since most pixels can be handled by cubic interpolation, using the prescreener
    generally results in much faster processing. Possible values:
    - 0: No prescreening. No pixels will be processed with cubic interpolation. This is really slow.
    - 1: Old prescreener.
    - 2: New prescreener level 0.
    - 3: New prescreener level 1.
    - 4: New prescreener level 2.

    The new prescreener is not available with float input.
    """

    opencl: bool = False
    """
    Enables the use of the OpenCL variant.
    Note that this will only work if full interpolation can be performed.
    """

    # Class Variable
    _shift = 0.5

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.opencl and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        assert clip.format

        if (pscrn := self.pscrn) > 1 and clip.format.sample_type == vs.FLOAT:
            from warnings import warn
            warn(
                f"{self.__class__.__name__}: The new prescreener {self.pscrn} is not available with float input. "
                "Falling back to old prescreener...",
                Warning
            )
            pscrn = 1

        return dict(nsize=self.nsize, nns=self.nns, qual=self.qual, etype=self.etype, pscrn=pscrn) | kwargs

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        interpolated = clip.znedi3.nnedi3(
            self.field, double_y or not self.drop_fields, **self.get_aa_args(clip) | kwargs
        )
        return self.shift_interpolate(clip, interpolated, double_y)

    def full_interpolate(
        self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return clip.sneedif.NNEDI3(
            self.field, double_y, double_x, transpose_first=self.transpose_first, **self.get_aa_args(clip) | kwargs
        )


class Nnedi3SS(NNEDI3, SuperSampler):
    """Concrete implementation of NNEDI3 used as a supersampler."""
    
    @inject_self.cached.property
    def kernel_radius(self) -> int:
        match self.nsize:
            case 1 | 5:
                return 16
            case 2 | 6:
                return 32
            case 3:
                return 48
            case _:
                return 8


class Nnedi3SR(NNEDI3, SingleRater):
    """Concrete implementation of NNEDI3 used as a single-rater."""


class Nnedi3DR(Nnedi3SR, DoubleRater):
    """Concrete implementation of NNEDI3 used as a double-rater."""


class Nnedi3(Nnedi3DR, Nnedi3SS, Antialiaser):
    """Full implementation of the NNEDI3 anti-aliaser"""
