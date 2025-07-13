from __future__ import annotations

from math import ceil
from typing import Any

from jetpytools import SPath, SPathLike

from vskernels import Catrom, KernelLike, ScalerLike
from vstools import ConstantFormatVideoNode, check_variable, core, depth, join, vs

from .generic import BaseGenericScaler

__all__ = [
    "PlaceboShader",
]


class PlaceboShader(BaseGenericScaler):
    """
    Placebo shader class.
    """

    _static_kernel_radius = 2

    def __init__(
        self,
        shader: str | SPathLike,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

        self.shader = SPath(shader).resolve().to_str()

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        kwargs = self.kwargs | kwargs

        output = depth(clip, 16)

        # Add fake chroma planes
        if output.format.num_planes == 1:
            if width > output.width or height > output.height:
                output = output.resize.Point(format=vs.YUV444P16)
            else:
                for div in (4, 2):
                    if width % div == 0 and height % div == 0:
                        blank = core.std.BlankClip(output, output.width // div, output.height // div, vs.GRAY16)
                        break
                else:
                    blank = core.std.BlankClip(output, format=vs.GRAY16)

                output = join(output, blank, blank)

        # Configure filter param mainly used for chroma planes if input clip is GRAY. Box was slightly faster.
        if "filter" not in kwargs:
            kwargs["filter"] = "box" if output.format.num_planes == 1 else "ewa_lanczos"

        output = core.placebo.Shader(
            output,
            self.shader,
            output.width * ceil(width / output.width),
            output.height * ceil(height / output.height),
            **kwargs,
        )

        return self._finish_scale(output, clip, width, height, shift)
