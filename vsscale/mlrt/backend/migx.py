import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from jetpytools import copy_signature

from vstools import vs

from .base import Backend


@dataclass(kw_only=True, frozen=True)
class MIGX(Backend):
    """MIGraphX backend for AMD GPUs."""

    supports_onnx_serialization: ClassVar[bool] = False

    # Hardware & Runtime Execution
    device_id: int = 0
    num_streams: int = 1

    # Model Precision & Data Types
    fp16: bool = False
    bf16: bool = False

    # Input Shapes & Optimization Profiles
    opt_shapes: tuple[int, int] | None = None

    # Builder Tuning & Optimization Levels
    fast_math: bool = True
    exhaustive_tune: bool = False
    short_path: bool | None = None

    # Miscellaneous & Custom Settings
    custom_env: Mapping[str, str] = field(default_factory=dict)
    custom_args: Sequence[str] = field(default_factory=list)

    @copy_signature(Backend.inference)
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: bool = False,
        **kwargs: Any,
    ) -> vs.VideoNode | list[vs.VideoNode]:
        network_path = Path(network_path)
        # channels = sum(clip.format.num_planes for clip in to_arr(clips))
        program_path = self.build_program(network_path, tilesize)

        return super().inference(clips, program_path, overlap, tilesize, flexible=flexible, **kwargs)

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"device_id": self.device_id, "num_streams": self.num_streams}

    def build_program(self, *args: Any, **kwargs: Any) -> Path:
        raise NotImplementedError
