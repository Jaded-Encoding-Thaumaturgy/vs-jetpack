from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jetpytools import fallback

from vstools import vs

from .base import Backend


@dataclass(kw_only=True, frozen=True)
class NCNN(Backend): ...


@dataclass(kw_only=True, init=False, frozen=True)
class VK(NCNN):
    """NCNN Vulkan backend."""

    # Hardware & Runtime Execution
    device_id: int = 0
    num_streams: int = 1

    # Model Precision & Data Types
    fp16: bool
    fp16_blacklist_ops: Sequence[str] | None
    output_format: Backend.OutputFormat | None = None

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {
            "fp16": self.fp16,
            "output_format": int(fallback(self.output_format, self.fp16)),
            "device_id": self.device_id,
            "num_streams": self.num_streams,
        }
