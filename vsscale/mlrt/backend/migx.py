from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

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
