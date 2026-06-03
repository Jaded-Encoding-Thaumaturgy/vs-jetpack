from __future__ import annotations

import os
import subprocess
import sys
import warnings
import zlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, getLogger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, SupportsInt

from jetpytools import cachedproperty, classproperty, copy_signature, to_arr
from packaging.version import Version

if TYPE_CHECKING:
    import tensorrt
    import tensorrt_rtx

from vstools import core, depth, vs

from ..settings import get_artifacts_folder, get_provider_folder
from .base import Backend

type Shape = tuple[int, int]


logger = getLogger(__name__)

LOGGING_VERBOSITY_MAP = {DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4}

# https://docs.nvidia.com/deeplearning/tensorrt/latest/
# https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html
# https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/index.html


@dataclass(kw_only=True, frozen=True)
class TRT(Backend):
    """TensorRT backend for Nvidia GPUs using the `core.trt` plugin."""

    plugin = core.lazy.trt

    plugin: ClassVar[vs.Plugin]

    # Hardware & Runtime Execution
    device_id: int = 0
    """CUDA device index."""
    num_streams: int = 1
    """Number of parallel plugin inference streams."""
    use_cuda_graph: bool = True
    """Enable CUDA graph execution for compatible engines to improve performance and reduce CPU overhead."""
    verbosity: SupportsInt | tensorrt.ILogger.Severity | tensorrt_rtx.ILogger.Severity | None = field(
        default=None, repr=False
    )
    """TensorRT/plugin logging severity."""

    # Model Precision & Data Types
    fp16: bool | None = None
    """Convert the ONNX model to FP16 before building. Default to True."""
    fp16_blacklist_ops: Collection[str] | None = None
    """ONNX node or op names to keep in FP32 during FP16 conversion."""
    bf16: bool | None = None
    """Convert the ONNX model to BF16 before building. Default to False."""
    tf32: bool = False
    """Allow TensorRT TF32 tactics."""
    strict_nans: bool = False
    """Disable float optimizations (0*x => 0, x-x => 0, x/x => 1) to preserve NaN/Inf propagation."""

    # Input Shapes & Optimization Profiles
    static_shape: bool = True
    """Build a fixed-shape engine when true."""
    min_shapes: Shape = (0, 0)
    """Minimum dynamic input tile size as `(width, height)`."""
    opt_shapes: Shape | None = None
    """Optimal input tile size as `(width, height)`. Defaults to the inference tile size."""
    max_shapes: Shape | None = None
    """Maximum dynamic input tile size as `(width, height)`. Defaults to the inference tile size."""

    # Builder Tactics & Kernel Selection (Libraries)
    edge_mask_convolutions: bool = True
    """Enable TensorRT edge-mask convolution tactics."""
    jit_convolutions: bool = True
    """Enable TensorRT JIT convolution tactics."""
    sparse_weights: bool = False
    """Allow the builder to exploit structured sparsity in weights."""

    # Builder Tuning & Optimization Levels
    workspace: int | None = None
    """Workspace memory pool limit in bytes."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level."""
    max_aux_streams: int | None = None
    """Maximum auxiliary streams used by TensorRT kernels."""
    max_num_tactics: int | None = None
    """Maximum number of tactics considered per layer."""
    tiling_optimization_level: SupportsInt | tensorrt.TilingOptimizationLevel | tensorrt_rtx.TilingOptimizationLevel = 0
    """TensorRT tiling optimization search level."""
    l2_limit_for_tiling: int = -1
    """L2 cache usage hint for tiling optimization."""
    avg_timing_iterations: int = 1
    """Number of averaging iterations when timing tactics. Higher values produce more stable tactic selection."""
    tactic_dram: int | None = None
    """DRAM limit in bytes for the optimizer during tactic selection. Prevents OOM on memory-constrained systems."""
    weight_streaming: bool = False
    """Stream weights from host to device to reduce GPU memory at the cost of performance."""

    # Build Process
    force_rebuild: bool = field(default=False, repr=False)
    """Force a full engine rebuild, ignoring any cached engine."""
    max_threads: int | None = field(default=None, repr=False)
    """Maximum number of builder threads. Limits CPU usage during engine build."""

    def __post_init__(self) -> None:
        if self.fp16 is self.bf16 is None:
            object.__setattr__(self, "fp16", True)

        if self.fp16 and self.bf16:
            raise ValueError("TensorRT backends do not support both fp16 and bf16")

        if self.verbosity is None:
            object.__setattr__(self, "verbosity", LOGGING_VERBOSITY_MAP.get(logger.getEffectiveLevel(), 2))

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
        v = int(self.plugin.Version()["tensorrt_version_build"])
        plugin_version = Version(f"{v // 10000}.{v // 100 % 100}.{v % 100}")
        bindings_version = Version(self.trt.__version__)

        if plugin_version.release[:3] != (version := bindings_version.release[:3]):
            raise RuntimeError(
                f"TensorRT plugin version {plugin_version} does not match TensorRT bindings version {bindings_version}"
            )

        return version  # type: ignore[return-value]

    @classproperty.cached
    @classmethod
    def logger(cls) -> tensorrt.ILogger:
        from ._trt import Logger

        return Logger(logger)

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
        clips = to_arr(clips)
        channels = sum(clip.format.num_planes for clip in clips)
        engine_path = self.build_engine(Path(network_path), channels, tilesize)

        if self.fp16 or self.bf16:
            # Clips must be in fp16 format is fp16 or bf16 mode is enabled,
            # otherwise the TRT plugins error out.
            clips = [depth(c, 16, sample_type=vs.SampleType.FLOAT) for c in clips]
        else:
            clips = [depth(c, 32) for c in clips]

        return super().inference(clips, engine_path, overlap, tilesize, flexible=flexible, **kwargs)

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "use_cuda_graph": self.use_cuda_graph,
            "num_streams": self.num_streams,
            "verbosity": self.verbosity,
        }

    def get_identity(self, network_path: Path, channels: int, tilesize: Shape) -> int:
        checksum = zlib.crc32(network_path.read_bytes())

        command = [
            "nvidia-smi",
            "-i",
            str(self.device_id),
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(command, capture_output=True, text=True, check=True)
        device = [d.strip().replace(" ", "_") for d in res.stdout.split(",")]

        components = (
            str(self),
            str(self.version),
            str(sys.version_info[:2]),
            network_path.name,
            f"{checksum:x}",
            str(channels),
            str(tilesize),
            *device,
        )
        return zlib.crc32(bytes("|".join(components), "utf-8"))

    def build_engine(self, network_path: Path, channels: int, tilesize: Shape, input_name: str = "input") -> Path:
        """
        Build or retrieve a cached TensorRT engine.

        Args:
            network_path: Path to the ONNX model.
            channels: Number of model input channels.
            tilesize: Inference tile size as `(width, height)`.
            input_name: Name of the model input tensor.

        Returns:
            Path to the serialized engine file.
        """
        if self.fp16:
            network_path = self._convert_onnx_fp16(network_path)
        elif self.bf16:
            network_path = self._convert_onnx_bf16(network_path)

        dirname = get_artifacts_folder()
        dirname.mkdir(parents=True, exist_ok=True)
        identity = self.get_identity(network_path, channels, tilesize)
        engine_path = dirname / f"{identity}.engine"

        if not self.force_rebuild and engine_path.is_file() and engine_path.stat().st_size >= 1024:
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

        if self.max_threads is not None:
            builder.max_threads = self.max_threads

        network = builder.create_network()
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

        if self.tactic_dram is not None:
            config.set_memory_pool_limit(self.trt.MemoryPoolType.TACTIC_DRAM, self.tactic_dram)

        if not self.tf32:
            config.flags &= ~(1 << self.trt.BuilderFlag.TF32.value)

        if self.sparse_weights:
            config.flags |= 1 << self.trt.BuilderFlag.SPARSE_WEIGHTS.value

        if self.strict_nans:
            config.flags |= 1 << self.trt.BuilderFlag.STRICT_NANS.value

        if self.weight_streaming:
            config.flags |= 1 << self.trt.BuilderFlag.WEIGHT_STREAMING.value

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
        config.avg_timing_iterations = self.avg_timing_iterations

        if self.max_aux_streams is not None:
            config.max_aux_streams = self.max_aux_streams

        if self.max_num_tactics is not None:
            config.max_num_tactics = self.max_num_tactics

        if int(self.tiling_optimization_level) != 0:
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

        input_names = [network.get_input(i).name for i in range(network.num_inputs)]
        if input_name not in input_names:
            logger.debug("input_name %r isn't in the input network", input_name)
            if network.num_inputs == 1:
                input_name = input_names[0]
            else:
                raise ValueError(f"Input name '{input_name}' not found in network inputs: {input_names}")

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
        import onnx
        import onnx.helper as onnxh
        import onnxconverter_common as onnxcc

        network = network_path.read_bytes()

        get_provider_folder().mkdir(parents=True, exist_ok=True)
        suffix = "fp16" if not self.fp16_blacklist_ops else f"fp16_block_{'_'.join(self.fp16_blacklist_ops)}"
        checksum = zlib.crc32(network)
        fp16_path = network_path.parent / f"{network_path.stem}_{checksum:x}_{suffix}.onnx"

        if fp16_path.is_file() and fp16_path.stat().st_size >= 1024:
            return fp16_path

        logger.info(f"Converting ONNX graph metadata to Float16 for: {network_path.name}")

        model = onnx.load_model_from_string(network)

        # Inject default value attribute to ConstantOfShape nodes if missing to ensure they get converted to Float16
        for node in model.graph.node:
            if node.op_type != "ConstantOfShape":
                continue
            if not any(attr.name == "value" for attr in node.attribute):
                node.attribute.append(
                    onnxh.make_attribute("value", onnxh.make_tensor("value", onnx.TensorProto.FLOAT, [1], [0.0]))
                )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"onnxconverter_common.*float16")
            model = onnxcc.convert_float_to_float16(
                model,
                keep_io_types=False,
                node_block_list=self.fp16_blacklist_ops,
            )

        onnx.save_model(model, fp16_path)

        return fp16_path

    def _convert_onnx_bf16(self, network_path: Path) -> Path:
        import ml_dtypes
        import onnx

        network = network_path.read_bytes()

        get_provider_folder().mkdir(parents=True, exist_ok=True)
        checksum = zlib.crc32(network)
        bf16_path = network_path.parent / f"{network_path.stem}_{checksum:x}_bf16_io.onnx"

        if bf16_path.is_file() and bf16_path.stat().st_size >= 1024:
            return bf16_path

        logger.info(f"Converting ONNX graph metadata and initializers to BFloat16 for: {network_path.name}")
        model = onnx.load_model_from_string(network)
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

        for i, initializer in enumerate(graph.initializer):
            if initializer.data_type == onnx.TensorProto.FLOAT:
                arr_bf16 = onnx.numpy_helper.to_array(initializer).astype(ml_dtypes.bfloat16)
                new_tensor = onnx.numpy_helper.from_array(arr_bf16, name=initializer.name)
                graph.initializer[i].CopyFrom(new_tensor)

        onnx.save_model(model, bf16_path)
        return bf16_path


@dataclass(kw_only=True, frozen=True)
class TRT_RTX(TRT):  # noqa: N801
    """TensorRT RTX backend for Nvidia RTX GPUs using the `core.trt_rtx` plugin."""

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

    @classproperty.cached
    @classmethod
    def logger(cls) -> tensorrt_rtx.ILogger:  # type: ignore[override]
        from ._trt_rtx import Logger

        return Logger(logger)
