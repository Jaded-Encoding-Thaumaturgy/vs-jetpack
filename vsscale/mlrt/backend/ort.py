from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import IntEnum
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, getLogger
from typing import Any, ClassVar

from jetpytools import fallback

from vstools import core, vs

from .base import Backend, BackendAutoConvertFloat

logger = getLogger(__name__)


@dataclass(kw_only=True, init=False, unsafe_hash=True, frozen=True)
class ORT(BackendAutoConvertFloat):
    """Base ONNX Runtime backend configuration."""

    plugin = core.lazy.ort

    provider: ClassVar[str]

    class Verbosity(IntEnum):
        VERBOSE = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
        FATAL = 4

        @classmethod
        def from_logging(cls, level: int) -> ORT.Verbosity:
            mapping = {
                DEBUG: cls.VERBOSE,
                INFO: cls.INFO,
                WARNING: cls.WARNING,
                ERROR: cls.ERROR,
                CRITICAL: cls.FATAL,
            }
            return mapping[level]

    # Hardware & Runtime Execution
    num_streams: int
    verbosity: int

    # Model Precision & Data Types
    fp16: bool | None
    fp16_blacklist_ops: Collection[str] | None
    output_format: Backend.OutputFormat | None

    # Specify __init__ here to set arguments order for subclasses
    def __init__(
        self,
        *,
        num_streams: int = 1,
        verbosity: int | None = None,
        fp16: bool = True,
        fp16_blacklist_ops: Collection[str] | None = None,
        output_format: Backend.OutputFormat | None = None,
    ) -> None:
        """
        Initialize the backend.

        Args:
            num_streams: Number of parallel inference streams.
            verbosity: ONNX Runtime logging verbosity.
            fp16: Convert model execution to FP16 where supported.
            fp16_blacklist_ops: ONNX node or op names to keep in FP32 during FP16 conversion.
            output_format: Requested output precision. Defaults to FP16 when `fp16` is enabled, otherwise FP32.
        """
        object.__setattr__(self, "fp16", fp16)
        object.__setattr__(self, "fp16_blacklist_ops", fp16_blacklist_ops)
        object.__setattr__(self, "num_streams", num_streams)
        object.__setattr__(
            self, "verbosity", ORT.Verbosity.from_logging(logger.level) if verbosity is None else verbosity
        )
        object.__setattr__(self, "output_format", output_format)
        super().__init__()

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {
            "fp16": self.fp16,
            "output_format": int(fallback(self.output_format, self.fp16)),
            "provider": self.provider,
            "num_streams": self.num_streams,
            "verbosity": self.verbosity,
            "fp16_blacklist_ops": self.fp16_blacklist_ops,
        }


@dataclass(kw_only=True, init=False, unsafe_hash=True, frozen=True)
class ORT_CPU(ORT):  # noqa: N801
    """ONNX Runtime CPU execution provider."""

    provider = "CPU"


@dataclass(kw_only=True, init=False, unsafe_hash=True, frozen=True)
class ORT_CUDA(ORT):  # noqa: N801
    """ONNX Runtime CUDA execution provider for Nvidia GPUs."""

    provider = "CUDA"

    # Hardware & Runtime Execution
    device_id: int
    cudnn_benchmark: bool
    use_cuda_graph: bool

    # Model Precision & Data Types
    tf32: bool

    # Builder Tuning & Optimization Levels
    prefer_nhwc: bool

    def __init__(
        self,
        *,
        num_streams: int = 1,
        verbosity: int | None = None,
        device_id: int = 0,
        cudnn_benchmark: bool = True,
        use_cuda_graph: bool = False,
        fp16: bool = True,
        fp16_blacklist_ops: Collection[str] | None = None,
        output_format: Backend.OutputFormat | None = None,
        tf32: bool = False,
        prefer_nhwc: bool = False,
    ) -> None:
        """
        Initialize the backend.

        Args:
            num_streams: Number of parallel inference streams.
            verbosity: ONNX Runtime logging verbosity.
            device_id: CUDA device index.
            cudnn_benchmark: Let cuDNN search for faster convolution algorithms.
            use_cuda_graph: Enable CUDA graph capture to improve performance and reduce CPU overhead
                for compatible models.
            fp16: Convert model execution to FP16 where supported.
            fp16_blacklist_ops: ONNX node or op names to keep in FP32 during FP16 conversion.
            output_format: Requested output precision. Defaults to FP16 when `fp16` is enabled, otherwise FP32.
            tf32: Allow TensorFloat-32 math on supported Nvidia GPUs.
            prefer_nhwc: Prefer NHWC layout where ONNX Runtime supports it.
        """
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "cudnn_benchmark", cudnn_benchmark)
        object.__setattr__(self, "use_cuda_graph", use_cuda_graph)
        object.__setattr__(self, "prefer_nhwc", prefer_nhwc)
        object.__setattr__(self, "tf32", tf32)
        super().__init__(
            num_streams=num_streams,
            verbosity=verbosity,
            fp16=fp16,
            fp16_blacklist_ops=fp16_blacklist_ops,
            output_format=output_format,
        )

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return super().get_args(clips) | {
            "device_id": self.device_id,
            "cudnn_benchmark": self.cudnn_benchmark,
            "use_cuda_graph": self.use_cuda_graph,
            "prefer_nhwc": self.prefer_nhwc,
            "tf32": self.tf32,
        }


@dataclass(kw_only=True, init=False, unsafe_hash=True, frozen=True)
class ORT_DML(ORT):  # noqa: N801
    """ONNX Runtime DirectML execution provider for D3D12 devices."""

    provider = "DML"

    # Hardware & Runtime Execution
    device_id: int

    def __init__(
        self,
        *,
        device_id: int = 0,
        fp16: bool = True,
        fp16_blacklist_ops: Collection[str] | None = None,
        output_format: Backend.OutputFormat | None = None,
        num_streams: int = 1,
        verbosity: int | None = None,
    ) -> None:
        """
        Initialize the backend.

        Args:
            device_id: DirectML adapter index.
            num_streams: Number of parallel inference streams.
            verbosity: ONNX Runtime logging verbosity.
            fp16: Convert model execution to FP16 where supported.
            fp16_blacklist_ops: ONNX node or op names to keep in FP32 during FP16 conversion.
            output_format: Requested output precision. Defaults to FP16 when `fp16` is enabled, otherwise FP32.
        """
        object.__setattr__(self, "device_id", device_id)
        super().__init__(
            num_streams=num_streams,
            verbosity=verbosity,
            fp16=fp16,
            fp16_blacklist_ops=fp16_blacklist_ops,
            output_format=output_format,
        )

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return super().get_args(clips) | {"device_id": self.device_id}


@dataclass(kw_only=True, init=False, unsafe_hash=True, frozen=True)
class ORT_COREML(ORT):  # noqa: N801
    """ONNX Runtime Core ML execution provider."""

    provider = "COREML"

    class Provider(IntEnum):
        NEURAL_NETWORK = 0
        ML_PROGRAM = 1

    # Hardware & Runtime Execution
    ml_program: Provider
    """Core ML provider mode."""

    def __init__(
        self,
        *,
        ml_program: int = Provider.NEURAL_NETWORK,
        fp16: bool = True,
        fp16_blacklist_ops: Collection[str] | None = None,
        output_format: Backend.OutputFormat | None = None,
        num_streams: int = 1,
        verbosity: int | None = None,
    ) -> None:
        object.__setattr__(self, "ml_program", ORT_COREML.Provider(ml_program))
        super().__init__(
            num_streams=num_streams,
            verbosity=verbosity,
            fp16=fp16,
            fp16_blacklist_ops=fp16_blacklist_ops,
            output_format=output_format,
        )

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return super().get_args(clips) | {"ml_program": self.ml_program}
