from __future__ import annotations

import warnings
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from logging import getLogger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

import onnx
import onnxconverter_common as onnxcc
from jetpytools import cachedproperty
from packaging.version import Version

if TYPE_CHECKING:
    import tensorrt as trt
    import tensorrt_rtx as trt_rtx

from vstools import core, vs

from ..settings import get_engines_folder, get_onnx_folder
from .base import Backend

type Shape = tuple[int, int]


logger = getLogger(__name__)

# https://docs.nvidia.com/deeplearning/tensorrt/latest/
# https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html
# https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/index.html


@dataclass(kw_only=True, frozen=True)
class TensorRT(Backend):
    """Base TensorRT backend configuration."""

    plugin: ClassVar[vs.Plugin]
    supports_onnx_serialization = False

    # Hardware & Runtime Execution
    device_id: int = 0
    num_streams: int = 1
    use_cuda_graph: bool = False

    # Model Precision & Data Types
    fp16: bool = False
    fp16_node_block_list: Sequence[str] | None = None
    bf16: bool = False
    tf32: bool = False

    # Input Shapes & Optimization Profiles
    static_shape: bool = True
    min_shapes: Shape = (0, 0)
    opt_shapes: Shape | None = None
    max_shapes: Shape | None = None

    # Builder Tactics & Kernel Selection (Libraries)
    edge_mask_convolutions: bool = True
    jit_convolutions: bool = True

    # Builder Tuning & Optimization Levels
    workspace: int | None = None
    builder_optimization_level: int = 3
    max_aux_streams: int | None = None
    max_num_tactics: int | None = None
    tiling_optimization_level: trt.TilingOptimizationLevel | trt_rtx.TilingOptimizationLevel | int = 0
    l2_limit_for_tiling: int = -1

    # Miscellaneous & Custom Settings
    custom_args: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fp16 and self.bf16:
            raise ValueError("TensorRT does not support both fp16 and bf16")

    if TYPE_CHECKING:
        import tensorrt as trt
    else:

        @cachedproperty
        def trt(self) -> ModuleType:
            if find_spec("tensorrt"):
                import tensorrt

                return tensorrt

            raise ModuleNotFoundError("The 'tensorrt' dependency is not installed.") from None

    @property
    def version(self) -> tuple[int, int, int]:
        return Version(self.trt.__version__).release[:3]  # type: ignore[return-value]

    @property
    def logger(self) -> trt.ILogger:
        from ._trt import Logger

        return Logger(logger)

    def get_identity(self, network_path: Path, channels: int, tilesize: Shape) -> int:
        checksum = zlib.crc32(network_path.read_bytes())

        device_name = self.plugin.DeviceProperties(self.device_id)["name"].decode().replace(" ", "-")

        components = (
            str(self),
            str(self.version),
            network_path.name,
            f"{checksum:x}",
            str(channels),
            str(tilesize),
            device_name,
        )
        return zlib.crc32(bytes("|".join(components), "utf-8"))

    def build_engine(self, network_path: Path, channels: int, tilesize: Shape, input_name: str = "input") -> Path:
        """
        Build or retrieve a cached TensorRT RTX engine.

        Returns the path to the serialized engine file.
        """
        if self.fp16:
            network_path = self._convert_onnx_fp16(network_path)
        elif self.bf16:
            network_path = self._convert_onnx_bf16(network_path)

        dirname = get_engines_folder()
        dirname.mkdir(parents=True, exist_ok=True)
        identity = self.get_identity(network_path, channels, tilesize)
        engine_path = dirname / f"{identity}.engine"

        if engine_path.is_file() and engine_path.stat().st_size >= 1024:
            return engine_path

        self.build(
            network_path=network_path,
            engine_path=engine_path,
            channels=channels,
            tilesize=tilesize,
            input_name=input_name,
        )

        return engine_path

    def build(
        self,
        network_path: Path,
        engine_path: Path,
        channels: int,
        tilesize: Shape,
        input_name: str,
    ) -> None:
        trt_logger = self.logger
        builder = self.trt.Builder(trt_logger)
        network = builder.create_network(1 << self.trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED.value)
        parser = self.trt.OnnxParser(network, trt_logger)

        if not parser.parse_from_file(str(network_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX model: {network_path}\n" + "\n".join(errors))

        config = builder.create_builder_config()

        # Delegate builder setup
        self.configure_builder_config(config, network)
        self.setup_optimization_profile(builder, network, config, channels, input_name, tilesize)

        # Timing Cache
        timing_cache_path = Path(f"{engine_path}.cache")
        timing_cache_data = b""
        if timing_cache_path.exists():
            timing_cache_data = timing_cache_path.read_bytes()

        timing_cache = config.create_timing_cache(timing_cache_data)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

        # Build
        logger.info(f"Building TensorRT {self.__class__.__name__} engine from {network_path}...")
        serialized = builder.build_serialized_network(network, config)

        if not serialized:
            raise RuntimeError(f"TensorRT engine build failed for {network_path}")

        engine_path.write_bytes(serialized)

        # Save Cache
        updated_cache = config.get_timing_cache()
        timing_cache_path.write_bytes(updated_cache.serialize())

        logger.info(f"Engine saved to {engine_path}")

    def configure_builder_config(self, config: trt.IBuilderConfig, network: trt.INetworkDefinition) -> None:
        if self.workspace is not None:
            config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, self.workspace)

        if not self.tf32:
            config.flags &= ~(1 << self.trt.BuilderFlag.TF32.value)

        self.configure_tactic_sources(config)
        self.configure_optimization_settings(config)

    def configure_tactic_sources(self, config: trt.IBuilderConfig) -> None:
        tactic_sources = config.get_tactic_sources()

        if self.edge_mask_convolutions:
            tactic_sources |= 1 << self.trt.TacticSource.EDGE_MASK_CONVOLUTIONS.value
        else:
            tactic_sources &= ~(1 << self.trt.TacticSource.EDGE_MASK_CONVOLUTIONS.value)

        if self.jit_convolutions:
            tactic_sources |= 1 << self.trt.TacticSource.JIT_CONVOLUTIONS.value
        else:
            tactic_sources &= ~(1 << self.trt.TacticSource.JIT_CONVOLUTIONS.value)

        config.set_tactic_sources(tactic_sources)

    def configure_optimization_settings(self, config: trt.IBuilderConfig) -> None:
        config.builder_optimization_level = self.builder_optimization_level

        if self.max_aux_streams is not None:
            config.max_aux_streams = self.max_aux_streams

        if self.max_num_tactics is not None:
            config.max_num_tactics = self.max_num_tactics

        if self.tiling_optimization_level != 0:
            config.tiling_optimization_level = self.trt.TilingOptimizationLevel(self.tiling_optimization_level)
            config.l2_limit_for_tiling = self.l2_limit_for_tiling

    def setup_optimization_profile(
        self,
        builder: trt.Builder,
        network: trt.INetworkDefinition,
        config: trt.IBuilderConfig,
        channels: int,
        input_name: str,
        tilesize: Shape,
    ) -> None:
        profile = builder.create_optimization_profile()
        opt_shapes = self.trt.Dims(self.opt_shapes or tilesize)
        max_shapes = self.trt.Dims(self.max_shapes or tilesize)

        if self.static_shape:
            shape = self.trt.Dims((1, channels, opt_shapes[1], opt_shapes[0]))

            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                if input_tensor.name == input_name:
                    input_tensor.shape = shape

            profile.set_shape(input_name, shape, shape, shape)
        else:
            profile.set_shape(
                input_name,
                self.trt.Dims((1, channels, self.min_shapes[1], self.min_shapes[0])),
                self.trt.Dims((1, channels, opt_shapes[1], opt_shapes[0])),
                self.trt.Dims((1, channels, max_shapes[1], max_shapes[0])),
            )

        config.add_optimization_profile(profile)

    def _convert_onnx_fp16(self, network_path: Path) -> Path:
        suffix = "fp16" if not self.fp16_node_block_list else f"fp16_block_{'_'.join(self.fp16_node_block_list)}"

        checksum = zlib.crc32(network_path.read_bytes())
        dirname = network_path.parent

        get_onnx_folder().mkdir(parents=True, exist_ok=True)

        fp16_path = dirname / f"{network_path.stem}_{checksum:x}_{suffix}.onnx"

        if fp16_path.is_file() and fp16_path.stat().st_size >= 1024:
            return fp16_path

        logger.info(f"Converting ONNX graph metadata to Float16 for: {network_path.name}")

        model = onnx.load(network_path)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"onnxconverter_common.*float16")
            model = onnxcc.convert_float_to_float16(
                model,
                keep_io_types=False,
                node_block_list=self.fp16_node_block_list,
            )

        onnx.save(model, fp16_path)

        return fp16_path

    def _convert_onnx_bf16(self, network_path: Path) -> Path:
        checksum = zlib.crc32(network_path.read_bytes())
        dirname = network_path.parent
        get_onnx_folder().mkdir(parents=True, exist_ok=True)

        bf16_path = dirname / f"{network_path.stem}_{checksum:x}_bf16_io.onnx"

        if bf16_path.is_file() and bf16_path.stat().st_size >= 1024:
            return bf16_path

        logger.info(f"Converting ONNX graph metadata to BFloat16 for: {network_path.name}")
        model = onnx.load(network_path)
        graph = model.graph

        for tensor in graph.input:
            if tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                tensor.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16

        for tensor in graph.output:
            if tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                tensor.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16

        for tensor in graph.value_info:
            if tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                tensor.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16

        for initializer in graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                initializer.data_type = onnx.TensorProto.BFLOAT16

        onnx.save(model, bf16_path)
        return bf16_path


@dataclass(kw_only=True, frozen=True)
class TRT(TensorRT):
    """TensorRT backend for Nvidia GPUs."""

    plugin = core.lazy.trt


@dataclass(kw_only=True, frozen=True)
class RTX(TensorRT):
    """TensorRT RTX backend for Nvidia RTX GPUs."""

    plugin = core.lazy.trt_rtx

    if TYPE_CHECKING:
        import tensorrt_rtx as trt
    else:

        @cachedproperty
        def trt(self) -> ModuleType:
            if find_spec("tensorrt_rtx"):
                import tensorrt_rtx as trt

                return trt

            raise ModuleNotFoundError("The 'tensorrt_rtx' dependency is not installed.") from None

    @property
    def logger(self) -> trt_rtx.ILogger:  # type: ignore[override]
        from ._trt_rtx import Logger

        return Logger(logger)
