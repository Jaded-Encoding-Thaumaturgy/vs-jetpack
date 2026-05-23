from dataclasses import dataclass

from .base import Backend


@dataclass(kw_only=True, frozen=True)
class NCNN(Backend): ...


@dataclass(kw_only=True, init=False, frozen=True)
class VK(NCNN):
    """NCNN Vulkan backend."""

    # Hardware & Runtime Execution
    device_id: int
    num_streams: int

    # Model Precision & Data Types
    fp16: bool

    def __init__(
        self,
        *,
        fp16: bool = False,
        device_id: int = 0,
        num_streams: int = 1,
    ) -> None:
        object.__setattr__(self, "fp16", fp16)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "num_streams", num_streams)
        object.__setattr__(self, "fp16", fp16)
