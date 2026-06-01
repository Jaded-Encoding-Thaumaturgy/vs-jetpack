from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jetpytools import fallback

from vstools import vs

from .base import Backend


@dataclass(kw_only=True, frozen=True)
class NCNN(Backend):
    """Base class for NCNN-backed inference configurations."""

    if TYPE_CHECKING:
        from .ncnn import VK


@dataclass(kw_only=True, frozen=True)
class VK(NCNN):
    """NCNN Vulkan backend."""

    # Hardware & Runtime Execution
    device_id: int = 0
    """Vulkan device index used by NCNN."""
    num_streams: int = 1
    """Number of parallel NCNN inference streams."""

    # Model Precision & Data Types
    fp16: bool = True
    """Enable NCNN FP16 storage/arithmetic where supported."""
    fp16_blacklist_ops: Sequence[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""
    output_format: Backend.OutputFormat | None = None
    """Requested output precision. Defaults to FP16 when `fp16` is enabled, otherwise FP32."""

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {
            "fp16": self.fp16,
            "output_format": int(fallback(self.output_format, self.fp16)),
            "device_id": self.device_id,
            "num_streams": self.num_streams,
        }


if not TYPE_CHECKING:
    NCNN.VK = VK
