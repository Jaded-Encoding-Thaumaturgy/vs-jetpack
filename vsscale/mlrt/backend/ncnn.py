from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from vstools import core, vs

from .base import BackendAutoConvertFloat


@dataclass(kw_only=True, frozen=True)
class NCNN(BackendAutoConvertFloat):
    """NCNN Vulkan backend."""

    plugin = core.lazy.ncnn

    # Hardware & Runtime Execution
    device_id: int = 0
    """Vulkan device index used by NCNN."""
    num_streams: int = 1
    """Number of parallel NCNN inference streams."""

    # Model Precision & Data Types
    fp16: bool | None = True
    """Enable NCNN FP16 storage/arithmetic where supported."""
    fp16_blacklist_ops: Collection[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return super().get_args(clips) | {
            "fp16": self.fp16,
            "device_id": self.device_id,
            "num_streams": self.num_streams,
        }
