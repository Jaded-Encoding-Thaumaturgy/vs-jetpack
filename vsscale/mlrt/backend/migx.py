import os
import shutil
import subprocess
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

from jetpytools import CustomRuntimeError, CustomValueError, copy_signature, to_arr

from vstools import UnsupportedSampleTypeError, core, depth, vs

from ..settings import get_artifacts_folder
from .base import BackendAutoConvertFloat

type Shape = tuple[int, int]


logger = getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class MIGX(BackendAutoConvertFloat):
    """
    MIGraphX backend for AMD GPUs.

    The ONNX model is compiled to an MXR program with `migraphx-driver`
    and cached before execution through `core.migx.Model`.
    """

    plugin = core.lazy.migx

    # Hardware & Runtime Execution
    device_id: int = 0
    """AMD GPU device index."""
    num_streams: int = 1
    """Number of parallel MIGraphX inference streams."""

    # Model Precision & Data Types
    fp16: bool | None = None
    """Compile the program for FP16 where supported. Default to True"""
    bf16: bool | None = None
    """Compile the program for BF16 where supported. Default to False."""

    # Input Shapes & Optimization Profiles
    opt_shapes: Shape | None = None
    """Optimization input tile size as `(width, height)`. Defaults to the inference tile size."""

    # Builder Tuning & Optimization Levels
    fast_math: bool = True
    """Keep MIGraphX fast math optimizations enabled."""
    exhaustive_tune: bool = False
    """Enable exhaustive tuning during compilation."""

    # Miscellaneous & Custom Settings
    custom_env: Mapping[str, str] = field(default_factory=dict[str, str])
    """
    Extra environment variables for `migraphx-driver`.

    https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/reference/MIGraphX-dev-env-vars.html
    https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html
    """
    custom_args: Sequence[str] = field(default_factory=list[str])
    """
    Additional command-line arguments appended to `migraphx-driver compile`.

    migraph-driver compile --help
    """

    def __post_init__(self) -> None:
        if self.fp16 is self.bf16 is None:
            object.__setattr__(self, "fp16", True)

        if self.fp16 and self.bf16:
            raise CustomValueError("MIGX backend does not support both fp16 and bf16")

    @property
    def version(self) -> tuple[int, int]:
        from packaging.version import Version

        version_info = self.plugin.Version()

        v_run = int(version_info["hip_runtime_version"])
        run_version = Version(f"{v_run // 10000000}.{(v_run % 10000000) // 100000}.{v_run % 100000}")

        v_build = int(version_info["hip_runtime_version_build"])
        build_version = Version(f"{v_build // 10000000}.{(v_build % 10000000) // 100000}.{v_build % 100000}")

        if build_version.release[:2] != run_version.release[:2]:
            raise CustomRuntimeError(
                f"MIGraphx plugin build version {build_version} does not match runtime version {run_version}"
            )

        return run_version.release[0], run_version.release[1]

    @copy_signature(BackendAutoConvertFloat.inference)
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
        channels = sum(c.format.num_planes for c in clips)
        bitdepth = max(c.format.bits_per_sample for c in clips)

        program_path = self.build_program(Path(network_path), channels, tilesize)

        if self.fp16:
            # Clips must be in fp16 format is fp16 is enabled,
            # otherwise the MIGX plugin errors out.
            clips = [depth(c, 16, sample_type=vs.SampleType.FLOAT) for c in clips]
        else:
            clips = [depth(c, 32) for c in clips]

        res = super().inference(clips, program_path, overlap, tilesize, flexible=flexible, **kwargs)

        return (
            depth(res, bitdepth, sample_type=vs.FLOAT)
            if isinstance(res, vs.VideoNode)
            else [depth(r, bitdepth, sample_type=vs.FLOAT) for r in res]
        )

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"device_id": self.device_id, "num_streams": self.num_streams}

    def get_identity(self, network_path: Path, tilesize: Shape) -> int:
        checksum = zlib.crc32(network_path.read_bytes())

        device_props = self.plugin.DeviceProperties(self.device_id)
        device = [device_props["name"].decode().replace(" ", "-"), device_props["driver_version"]]

        components = (
            str(self),
            str(self.version),
            network_path.name,
            f"{checksum:x}",
            str(tilesize),
            *map(str, device),
        )
        logger.debug("%s: Identity %r", self.get_identity, components)
        return zlib.crc32(bytes("|".join(components), "utf-8"))

    def build_program(self, network_path: Path, channels: int, tilesize: Shape, input_name: str = "input") -> Path:
        if rocm_path := os.getenv("ROCM_PATH"):
            migraph_driver = f"{rocm_path}/bin/migraphx-driver"
        else:
            migraph_driver = shutil.which("migraphx-driver") or "migraphx-driver"

        dirname = get_artifacts_folder()
        dirname.mkdir(parents=True, exist_ok=True)
        identity = self.get_identity(network_path, tilesize)
        program_path = dirname / f"{identity}.mxr"

        command: list[Any] = [
            migraph_driver,
            "compile",
            "--onnx",
            network_path,
            "--gpu",
            "--optimize",
            "--binary",
            "--output",
            program_path,
        ]

        opt_shapes = self.opt_shapes or tilesize
        command.extend(["--input-dim", f"@{input_name}", "1", f"{channels}", f"{opt_shapes[1]}", f"{opt_shapes[0]}"])
        if self.fp16:
            command.append("--fp16")
        if self.bf16:
            command.append("--bf16")
        if not self.fast_math:
            command.append("--disable-fast-math")
        if self.exhaustive_tune:
            command.append("--exhaustive-tune")

        command.extend(self.custom_args)

        logger.debug("%s: Calling migraphx-driver with the command:", self)
        logger.debug(command)

        try:
            subprocess.run(command, env=os.environ | self.custom_env, check=True, stdout=sys.stderr)
        except subprocess.CalledProcessError as e:
            logger.debug("MIGraphx STDERR:\n%s", e.stderr)
            logger.debug("MIGraphx STDOUT:\n%s", e.stdout)
            raise CustomRuntimeError("The program compilation failed") from e

        return program_path
