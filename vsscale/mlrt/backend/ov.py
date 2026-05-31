from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vstools import core, vs

from .base import Backend

type Shape = tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class OV(Backend):
    """Base OpenVINO backend configuration."""

    plugin = core.lazy.ov

    custom_config: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """
    Extra OpenVINO runtime configuration keys merged into the device configuration passed to `core.ov.Model`.

    https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpu__prop__cpp__api.html
    https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpp__prop__api.html
    """

    @property
    def device(self) -> str:
        if type(self) is OV:
            raise NotImplementedError("OV is an abstract class")
        return self.__class__.__name__

    @property
    def config(self) -> Mapping[str, Any]:
        return self.custom_config

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"device": self.device, "fp16": False, "config": lambda: self.config}


@dataclass(kw_only=True, frozen=True)
class CPU(OV):
    """OpenVINO CPU backend."""

    # Hardware & Runtime Execution
    num_streams: int = 1
    """Number of OpenVINO inference streams."""
    num_threads: int = 0
    """Maximum CPU inference threads. `0` lets OpenVINO choose."""
    bind_thread: bool = True
    """Enable OpenVINO CPU thread pinning."""

    # Model Precision & Data Types
    fp16: bool = False
    """Request FP16 inference precision."""
    bf16: bool = False
    """Request BF16 inference precision."""
    fp16_blacklist_ops: Sequence[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""

    def __post_init__(self) -> None:
        if self.fp16 and self.bf16:
            raise ValueError("ORT CPU does not support both fp16 and bf16")

    @property
    def config(self) -> Mapping[str, Any]:
        return dict(super().config) | {
            "NUM_STREAMS": self.num_streams,
            "INFERENCE_NUM_THREADS": self.num_streams,
            "ENABLE_CPU_PINNING": {False: "NO", True: "YES"}[self.bind_thread],
            "INFERENCE_PRECISION_HINT": "f16" if self.fp16 else "bf16" if self.bf16 else "f32",
        }


@dataclass(kw_only=True, frozen=True)
class GPU(OV):
    """OpenVINO GPU backend."""

    # Hardware & Runtime Execution
    device_id: int = 0
    """OpenVINO GPU device index."""
    num_streams: int = 1
    """Number of OpenVINO inference streams."""

    # Model Precision & Data Types
    fp16: bool = False
    """Request FP16 inference precision."""
    fp16_blacklist_ops: Sequence[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""

    @property
    def config(self) -> dict[str, Any]:
        return dict(super().config) | {
            "NUM_STREAMS": self.num_streams,
            "INFERENCE_PRECISION_HINT": "f16" if self.fp16 else "f32",
        }

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return super().get_args(clips) | {"device": f"{self.device}.{self.device_id}"}


@dataclass(kw_only=True, frozen=True)
class NPU(OV):
    """OpenVINO NPU backend for Intel neural processing units."""
