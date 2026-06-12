from __future__ import annotations

from math import ceil
from typing import Any

from jetpytools import CustomRuntimeError, SPath, SPathLike

from vskernels import Catrom, KernelLike, ScalerLike
from vstools import core, join, vs

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
    ) -> vs.VideoNode:
        width, height = self._wh_norm(clip, width, height)
        kwargs = self.kwargs | kwargs

        if clip.format.bits_per_sample != 16 or clip.format.sample_type != vs.INTEGER:
            raise CustomRuntimeError("The input clip must be a YUVXXXP16 formats.")

        # Add fake chroma planes
        if clip.format.num_planes == 1:
            if width > clip.width or height > clip.height:
                clip = clip.resize.Point(format=vs.YUV444P16)
            else:
                for div in (4, 2):
                    if width % div == 0 and height % div == 0:
                        blank = core.std.BlankClip(clip, clip.width // div, clip.height // div, vs.GRAY16)
                        break
                else:
                    blank = core.std.BlankClip(clip, format=vs.GRAY16)

                clip = join(clip, blank, blank)

        # Configure filter param mainly used for chroma planes if input clip is GRAY. Box was slightly faster.
        if "filter" not in kwargs:
            kwargs["filter"] = "box" if clip.format.num_planes == 1 else "ewa_lanczos"

        output = core.placebo.Shader(
            clip,
            self.shader,
            clip.width * ceil(width / clip.width),
            clip.height * ceil(height / clip.height),
            **kwargs,
        )

        return self._finish_scale(output, clip, width, height, shift)
