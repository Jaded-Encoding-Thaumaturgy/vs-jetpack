from collections.abc import Sequence
from dataclasses import dataclass

from .base import Backend

type Shape = tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class OV(Backend):
    @property
    def device(self) -> str:
        if type(self) is OV:
            raise NotImplementedError
        return self.__class__.__name__


@dataclass(kw_only=True, frozen=True)
class CPU(OV):
    """OpenVINO backend for x86 CPUs."""

    # Hardware & Runtime Execution
    num_streams: int = 1
    num_threads: int = 0
    bind_thread: bool = True

    # Model Precision & Data Types
    fp16: bool = False
    bf16: bool = False
    fp16_blacklist_ops: Sequence[str] | None = None


@dataclass(kw_only=True, frozen=True)
class GPU(OV):
    """OpenVINO backend for GPUs."""

    # Hardware & Runtime Execution
    device_id: int = 0
    num_streams: int = 1

    # Model Precision & Data Types
    fp16: bool = False
    fp16_blacklist_ops: Sequence[str] | None = None


@dataclass(kw_only=True, frozen=True)
class NPU(OV):
    """OpenVINO backend for Intel NPUs."""
