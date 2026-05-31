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

from jetpytools import CustomRuntimeError, copy_signature, to_arr
from packaging.version import Version

from vstools import core, vs

from ..settings import get_artifacts_folder
from .base import Backend

type Shape = tuple[int, int]


logger = getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class MIGX(Backend):
    """MIGraphX backend for AMD GPUs."""

    plugin = core.lazy.migx

    # Hardware & Runtime Execution
    device_id: int = 0
    num_streams: int = 1

    # Model Precision & Data Types
    fp16: bool = False
    bf16: bool = False

    # Input Shapes & Optimization Profiles
    opt_shapes: Shape | None = None

    # Builder Tuning & Optimization Levels
    fast_math: bool = True
    exhaustive_tune: bool = False

    # Miscellaneous & Custom Settings
    custom_env: Mapping[str, str] = field(default_factory=dict[str, str])
    """
    https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/reference/MIGraphX-dev-env-vars.html
    https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html
    """
    custom_args: Sequence[str] = field(default_factory=list[str])
    """
    migraph-driver compile --help
    """

    @property
    def version(self) -> tuple[int, int, int]:
        v = int(self.plugin.Version()["migraphx_version_build"])
        plugin_version = Version(f"{v // 10000}.{v // 100 % 100}.{v % 100}")

        p = subprocess.run(["cat", "/opt/rocm/.info/version"], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError

        migraphx_driver_version = Version(p.stdout)

        if plugin_version.release[:3] != (version := migraphx_driver_version.release[:3]):
            raise RuntimeError

        return version  # type: ignore[return-value]

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
        network_path = Path(network_path)
        channels = sum(clip.format.num_planes for clip in to_arr(clips))
        program_path = self.build_program(network_path, channels, tilesize)

        return super().inference(clips, program_path, overlap, tilesize, flexible=flexible, **kwargs)

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {"device_id": self.device_id, "num_streams": self.num_streams}

    def get_identity(self, network_path: Path, tilesize: Shape) -> int:
        checksum = zlib.crc32(network_path.read_bytes())

        device_props = self.plugin.DeviceProperties(self.device_id)
        device = [device_props["name"].decode().replace(" ", "-"), device_props["driver_version"].decode()]

        components = (
            str(self),
            str(self.version),
            network_path.name,
            f"{checksum:x}",
            str(tilesize),
            *device,
        )
        return zlib.crc32(bytes("|".join(components), "utf-8"))

    def build_program(self, network_path: Path, channels: int, tilesize: Shape, input_name: str = "input") -> Path:
        if rocm_path := os.getenv("ROCM_PATH"):
            migraph_driver = f"{rocm_path}/bin/migraphx-driver"
        else:
            migraph_driver = shutil.which("migraph-driver") or "migraph-driver"

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

        try:
            subprocess.run(command, env=os.environ | self.custom_env, check=True, stdout=sys.stderr)
        except subprocess.CalledProcessError as e:
            logger.debug("MIGraphx STDERR:\n%s", e.stderr)
            logger.debug("MIGraphx STDOUT:\n%s", e.stdout)
            raise CustomRuntimeError("The program compilation failed") from e

        return program_path
