from __future__ import annotations

import os
import platform
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, ClassVar, Literal, overload

from jetpytools import to_arr

from vstools import UnsupportedSampleTypeError, core, vs

from ...helpers import get_gpu

type Shape = tuple[int, int]

logger = getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Backend:
    """
    Base wrapper for a MLRT VapourSynth backend plugin.
    """

    plugin: ClassVar[vs.Plugin]
    flexible_output_prop: ClassVar[str] = "MlrtFlexible"

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: Literal[False] = ...,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: Literal[True],
        **kwargs: Any,
    ) -> list[vs.VideoNode]: ...

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: bool = ...,
        **kwargs: Any,
    ) -> vs.VideoNode | list[vs.VideoNode]: ...

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
        """
        Run inference with this backend.

        Args:
            clips: Input clip or clips passed to the backend model.
            network_path: Path to the model file or backend artifact.
            overlap: Horizontal and vertical tile overlap in pixels.
            tilesize: Horizontal and vertical tile size in pixels.
            flexible: Return each flexible output plane as a separate clip.
            **kwargs: Additional backend plugin arguments forwarded unchanged.

        Returns:
            A single output clip, or a list of output clips when `flexible` is enabled.
        """
        UnsupportedSampleTypeError.check(clips, vs.FLOAT, self.__class__)

        args = self.get_args(clips)

        if flexible:
            args = args.copy()
            args["flexible_output_prop"] = self.flexible_output_prop

        logger.info("Calling %s.Model", self.plugin.namespace)
        logger.info("Clips: %r", clips)
        logger.info("Network Path: %s", network_path)
        logger.info("overlap=%s, tilesize=%s, %s", overlap, tilesize, args | kwargs)
        output = self.plugin.Model(clips, network_path, overlap, tilesize, **args | kwargs)

        if flexible:
            clip = output["clip"]
            num_planes = output["num_planes"]

            output = [clip.std.PropToClip(prop=f"{self.flexible_output_prop}{i}") for i in range(num_planes)]

        return output

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        """Return backend plugin arguments derived from this configuration."""
        return {}

    @staticmethod
    def autoselect(device_id: int = 0, **kwargs: Any) -> Backend:
        """
        Try to select the best backend for the current system.

        Args:
            device_id: The GPU device id.
            **kwargs: Additional arguments to pass to the backend.

        Returns:
            The selected backend.
        """

        gpu = get_gpu(device_id)
        vendor = None if not gpu else str(gpu.vendor).strip()

        match vendor:
            # Windows & Linux
            case "NVIDIA Corporation":
                if hasattr(core, "trt"):
                    backend = UserBackend.TRT
                elif hasattr(core, "trt_rtx"):
                    backend = UserBackend.TRT_RTX
                elif platform.system().lower() == "windows" and hasattr(core, "ort"):
                    backend = UserBackend.ORT_DML
                elif hasattr(core, "ort"):
                    backend = UserBackend.ORT_CUDA
                elif hasattr(core, "ncnn"):
                    backend = UserBackend.NCNN
                else:
                    backend = UserBackend.OV_CPU
            # Windows & Linux
            case "Advanced Micro Devices, Inc.":
                if platform.system().lower() == "windows" and hasattr(core, "ort"):
                    backend = UserBackend.ORT_DML
                elif hasattr(core, "migx"):
                    backend = UserBackend.MIGX
                elif hasattr(core, "ncnn"):
                    backend = UserBackend.NCNN_VK
                else:
                    backend = UserBackend.OV_CPU
            # Windows & Linux
            case "Intel(R) Corporation":
                if hasattr(core, "ov"):
                    backend = UserBackend.OV_GPU
                elif platform.system().lower() == "windows" and hasattr(core, "ort"):
                    backend = UserBackend.ORT_DML
                elif hasattr(core, "ncnn"):
                    backend = UserBackend.NCNN_VK
                else:
                    backend = UserBackend.OV_CPU
            # macOS ARM64 & x86_64
            case "Apple":
                if hasattr(core, "ncnn"):
                    backend = UserBackend.NCNN_VK
                elif hasattr(core, "ort"):
                    backend = UserBackend.ORT_COREML
                else:
                    backend = UserBackend.OV_CPU
            case _:
                backend = UserBackend.OV_CPU

        del gpu

        if hasattr(backend, "device_id"):
            kwargs["device_id"] = device_id

        return backend(**kwargs)


class BackendAutoConvertFloat(Backend):
    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"output_format": max(c.format.bits_per_sample for c in to_arr(clips)) == 16}


from . import migx, ncnn, ort, ov, trt  # noqa: E402


@dataclass(kw_only=True, frozen=True)
class UserBackend(ABC):
    """Backend namespace for user interaction."""

    MIGX = migx.MIGX
    NCNN = ncnn.NCNN
    NCNN_VK = ncnn.NCNN
    ORT = ort.ORT
    ORT_CPU = ort.ORT_CPU
    ORT_CUDA = ort.ORT_CUDA
    ORT_DML = ort.ORT_DML
    ORT_COREML = ort.ORT_COREML
    OV = ov.OV
    OV_CPU = ov.OV_CPU
    OV_GPU = ov.OV_GPU
    OV_NPU = ov.OV_NPU
    TRT = trt.TRT
    TRT_RTX = trt.TRT_RTX

    autoselect = Backend.autoselect


UserBackend.register(Backend)
