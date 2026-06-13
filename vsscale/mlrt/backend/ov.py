import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from jetpytools import CustomValueError, copy_signature, to_arr

from vstools import UnsupportedSampleTypeError, core, depth, vs

from .base import Backend

type Shape = tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class OV(Backend):
    """Base OpenVINO backend configuration."""

    plugin = core.lazy.ov

    device: ClassVar[str]

    custom_config: Mapping[str, Any] = field(default_factory=dict[str, Any])
    """
    Extra OpenVINO runtime configuration keys merged into the device configuration passed to `core.ov.Model`.

    https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpu__prop__cpp__api.html
    https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpp__prop__api.html
    """

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
        UnsupportedSampleTypeError.check(clips, vs.FLOAT, self.__class__)

        clips = to_arr(clips)
        bitdepth = max(c.format.bits_per_sample for c in clips)

        res = super().inference(
            # OV Plugin only accepts fp32
            [depth(c, 32) for c in clips],
            network_path,
            overlap,
            tilesize,
            flexible=flexible,
            **kwargs,
        )

        return (
            depth(res, bitdepth, sample_type=vs.FLOAT)
            if isinstance(res, vs.VideoNode)
            else [depth(r, bitdepth, sample_type=vs.FLOAT) for r in res]
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return self.custom_config

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"device": self.device, "fp16": False, "config": OVConfig(self.config)}


@dataclass(kw_only=True, frozen=True)
class OV_CPU(OV):  # noqa: N801
    """OpenVINO CPU backend."""

    device = "CPU"

    # Hardware & Runtime Execution
    num_streams: int = 1
    """Number of OpenVINO inference streams."""
    num_threads: int = 0
    """Maximum CPU inference threads. `0` lets OpenVINO choose."""
    bind_thread: bool = True
    """Enable OpenVINO CPU thread pinning."""

    # Model Precision & Data Types
    fp16: bool | None = None
    """Request FP16 inference precision. Default to True."""
    bf16: bool | None = None
    """Request BF16 inference precision. Default to False."""
    fp16_blacklist_ops: Collection[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""

    def __post_init__(self) -> None:
        if self.fp16 is self.bf16 is None:
            object.__setattr__(self, "fp16", True)

        if self.fp16 and self.bf16:
            raise CustomValueError("OV CPU does not support both fp16 and bf16")

    @property
    def config(self) -> Mapping[str, Any]:
        return dict(super().config) | {
            "NUM_STREAMS": self.num_streams,
            "INFERENCE_NUM_THREADS": self.num_threads,
            "ENABLE_CPU_PINNING": {False: "NO", True: "YES"}[self.bind_thread],
            "INFERENCE_PRECISION_HINT": "f16" if self.fp16 else "bf16" if self.bf16 else "f32",
        }


@dataclass(kw_only=True, frozen=True)
class OV_GPU(OV):  # noqa: N801
    """OpenVINO GPU backend."""

    device = "GPU"

    # Hardware & Runtime Execution
    device_id: int = 0
    """OpenVINO GPU device index."""
    num_streams: int = 1
    """Number of OpenVINO inference streams."""

    # Model Precision & Data Types
    fp16: bool = True
    """Request FP16 inference precision."""
    fp16_blacklist_ops: Collection[str] | None = None
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
class OV_NPU(OV):  # noqa: N801
    """OpenVINO NPU backend for Intel neural processing units."""

    device = "NPU"

    fp16: Literal[True] = True


class OVConfig:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def __call__(self) -> Mapping[str, Any]:
        return self.config

    def __str__(self) -> str:
        return f"OVConfig({self.config!s})"

    def __repr__(self) -> str:
        return f"OVConfig({self.config!r})"
