"""
This module implements scalers for ONNX models.
"""

from __future__ import annotations

import re
from abc import ABC
from dataclasses import Field, asdict, fields, replace
from functools import cache
from logging import DEBUG, debug, getLogger, warning
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    SupportsFloat,
    get_args,
    runtime_checkable,
)

from jetpytools import CustomImportError, CustomValueError, SPath, SPathLike

if TYPE_CHECKING:
    from packaging.version import Version

from typing_extensions import deprecated

from vsexprtools import norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelLike, ScalerLike
from vstools import (
    ColorRange,
    DitherType,
    Matrix,
    MatrixLike,
    OutdatedPluginError,
    ProcessVariableResClip,
    check_variable_resolution,
    core,
    depth,
    get_color_family,
    get_nvidia_version,
    get_video_format,
    get_y,
    join,
    limiter,
    padder,
    vs,
)

from .generic import BaseGenericScaler

__all__ = [
    "DPIR",
    "ArtCNN",
    "BackendLike",
    "BaseOnnxScaler",
    "GenericOnnxScaler",
    "Waifu2x",
    "autoselect_backend",
]


@runtime_checkable
class _SupportsFP16(Protocol):
    fp16: bool


def _clean_keywords(kwargs: dict[str, Any], backend: Any) -> dict[str, Any]:
    valid_fields = _get_backend_fields(backend)
    return {k: v for k, v in kwargs.items() if k in valid_fields}


def _get_backend_fields(backend: Any) -> dict[str, Field[Any]]:
    return {f.name: f for f in fields(backend)}


if TYPE_CHECKING:
    from vsmlrt import backendT as Backend

    BackendLike = Backend | type[Backend] | str
    """
    Type alias for anything that can resolve to a Backend from vs-mlrt.

    This includes:

    - A string identifier.
    - A class type subclassing `Backend`.
    - An instance of a `Backend`.
    """
else:
    BackendLike = Any


def _normalize_git_version(raw: str) -> Version:
    from packaging.version import Version

    raw = raw.strip().lstrip("v")

    matched = re.match(r"(?P<tag>[0-9A-Za-z.\-_]+?)(?:-(?P<count>\d+)-g(?P<hash>[0-9a-f]+))?$", raw)

    if not matched:
        raise ValueError(f"Unrecognized git-describe string: {raw!r}")

    tag = matched.group("tag")
    count = int(matched.group("count") or 0)

    tag_parts = tag.split(".", 2)
    numeric_parts = list[str]()
    suffix_parts = list[str]()

    for part in tag_parts:
        if part.isdigit():
            numeric_parts.append(part)
        else:
            suffix_parts.append(part)

    base_version = ".".join(numeric_parts)
    normalized = base_version if count == 0 else f"{base_version}.post{count}"

    local_segments = list[str]()

    if suffix_parts:
        local_segments.append(".".join(suffix_parts))

    if local_segments:
        normalized += "+" + ".".join(local_segments)

    return Version(normalized)


@cache
def _check_vsmlrt_script_version(cls: type[BaseOnnxScaler]) -> None:
    try:
        import vsmlrt
    except ImportError:
        raise CustomImportError(cls, "vsmlrt") from None

    from packaging.version import Version

    if (current_version := Version(vsmlrt.__version__)) < Version(cls._REQUIRED_VSMLRT_SCRIPT_VERSION):
        raise CustomImportError(
            cls,
            "vsmlrt",
            f"Detected vs-mlrt version {current_version} is older than {cls._REQUIRED_VSMLRT_SCRIPT_VERSION}. "
            "Please update to a more recent version.",
        )


@cache
def _check_vsmlrt_plugin_version(backend_name: str, cls: type[BaseOnnxScaler]) -> None:
    bname = backend_name.lower().split("_", 1)
    plugin_name = "trt_rtx" if bname == ["trt", "rtx"] else bname[0]

    current_version = _normalize_git_version(getattr(core, plugin_name).Version()["version"].decode())

    from packaging.version import Version

    if current_version < Version(cls._REQUIRED_VSMLRT_PLUGIN_VERSION):
        raise OutdatedPluginError(
            cls,
            plugin_name,
            f"The plugin '{plugin_name}' version is older than {cls._REQUIRED_VSMLRT_PLUGIN_VERSION}. "
            "Please update to a more recent version.",
        )


def autoselect_backend(**kwargs: Any) -> Backend:
    """
    Try to select the best backend for the current system.

    If the system has an NVIDIA GPU: TRT > TRT_RTX > DirectML (D3D12) > NCNN (Vulkan) > CUDA (ORT) > OpenVINO GPU.
    Else: DirectML (D3D12) > MIGraphX > NCNN (Vulkan) > CPU (ORT) > CPU OpenVINO

    Args:
        **kwargs: Additional arguments to pass to the backend.

    Returns:
        The selected backend.
    """
    from os import name

    from vsmlrt import Backend

    backend: Any

    if get_nvidia_version():
        if hasattr(core, "trt"):
            backend = Backend.TRT
        elif hasattr(core, "trt_rtx"):
            backend = Backend.TRT_RTX
        elif hasattr(core, "ort") and name == "nt":
            backend = Backend.ORT_DML
        elif hasattr(core, "ncnn"):
            backend = Backend.NCNN_VK
        elif hasattr(core, "ort"):
            backend = Backend.ORT_CUDA
        else:
            backend = Backend.OV_GPU
    else:
        if hasattr(core, "ort") and name == "nt":
            backend = Backend.ORT_DML
        elif hasattr(core, "migx"):
            backend = Backend.MIGX
        elif hasattr(core, "ncnn"):
            backend = Backend.NCNN_VK
        elif hasattr(core, "ort"):
            backend = Backend.ORT_CPU
        else:
            backend = Backend.OV_CPU

    return backend(**_clean_keywords(kwargs, backend))


class BaseOnnxScaler(BaseGenericScaler, ABC):
    """
    Abstract generic scaler class for an ONNX model.
    """

    _REQUIRED_VSMLRT_SCRIPT_VERSION = "3.22.36"
    _REQUIRED_VSMLRT_PLUGIN_VERSION = "15.14"

    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            _check_vsmlrt_script_version(cls)
            return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        model: SPathLike | None = None,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        multiple: int = 1,
        max_instances: int = 2,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the scaler with the specified parameters.

        Args:
            model: Path to the ONNX model file.
            backend: The backend to be used with the vs-mlrt framework. If set to None, the most suitable backend will
                be automatically selected, prioritizing fp16 support.
            tiles: Whether to split the image into multiple tiles. This can help reduce VRAM usage, but note that the
                model's behavior may vary when they are used.
            tilesize: The size of each tile when splitting the image (if tiles are enabled).
            overlap: The size of overlap between tiles.
            multiple: Multiple of the tiles.
            max_instances: Maximum instances to spawn when scaling a variable resolution clip.
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments to pass to the backend. See the vsmlrt backend's docstring for more details.
        """
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

        if model is not None:
            self.model = str(SPath(model).resolve())

        fp16 = self.kwargs.pop("fp16", True)
        default_args = {"fp16": fp16, "output_format": int(fp16), "use_cuda_graph": True, "use_cublas": True}

        from vsmlrt import backendT

        if backend is None:
            self.backend = autoselect_backend(**default_args | self.kwargs)
        elif isinstance(backend, type):
            self.backend = backend(**_clean_keywords(default_args | self.kwargs, backend))
        elif isinstance(backend, str):
            backends_map = {b.__name__.lower(): b for b in get_args(backendT)}

            try:
                backend_t = backends_map[backend.lower().strip()]
            except KeyError:
                raise CustomValueError("Unknown backend!", self.__class__, backend)

            self.backend = backend_t(**_clean_keywords(default_args | self.kwargs, backend_t))
        else:
            self.backend = replace(backend, **_clean_keywords(self.kwargs, backend))

        _check_vsmlrt_plugin_version(self.backend.__class__.__name__, self.__class__)

        self.tiles = tiles
        self.tilesize = tilesize
        self.overlap = overlap
        self.multiple = multiple

        if self.overlap is None:
            self.overlap_w = self.overlap_h = 8
        elif isinstance(self.overlap, int):
            self.overlap_w = self.overlap_h = self.overlap
        else:
            self.overlap_w, self.overlap_h = self.overlap

        self.max_instances = max_instances

        if getLogger().level <= DEBUG:
            debug(f"{self}: Using {self.backend.__class__.__name__} backend")

            valid_fields = _get_backend_fields(self.backend)

            for k, v in asdict(self.backend).items():
                debug(f"{self}: {k}={v}, default is {valid_fields[k].default}")

            debug(f"{self}: User tiles: {self.tiles}")
            debug(f"{self}: User tilesize: {self.tilesize}")
            debug(f"{self}: User overlap: {(self.overlap_w, self.overlap_h)}")
            debug(f"{self}: User multiple: {self.multiple}")

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale the given clip using the ONNX model.

        Args:
            clip: The input clip to be scaled.
            width: The target width for scaling. If None, the width of the input clip will be used.
            height: The target height for scaling. If None, the height of the input clip will be used.
            shift: A tuple representing the shift values for the x and y axes.
            **kwargs: Additional arguments to be passed to the `preprocess_clip`, `postprocess_clip`, `inference`, and
                `_final_scale` methods. Use the prefix `preprocess_` or `postprocess_` to pass an argument to the
                respective method. Use the prefix `inference_` to pass an argument to the inference method.

        Returns:
            The scaled clip.
        """
        from vsmlrt import Backend

        width, height = self._wh_norm(clip, width, height)

        preprocess_kwargs = dict[str, Any]()
        postprocess_kwargs = dict[str, Any]()
        inference_kwargs = dict[str, Any]()

        for k in kwargs.copy():
            for prefix, ckwargs in zip(
                ("preprocess_", "postprocess_", "inference_"), (preprocess_kwargs, postprocess_kwargs, inference_kwargs)
            ):
                if k.startswith(prefix):
                    ckwargs[k.removeprefix(prefix)] = kwargs.pop(k)
                    break

        debug(f"{self}: Preprocess kwargs: {preprocess_kwargs}")
        debug(f"{self}: Postprocess kwargs: {postprocess_kwargs}")
        debug(f"{self}: Inference kwargs: {inference_kwargs}")

        wclip = self.preprocess_clip(clip, **preprocess_kwargs)

        if 0 not in {clip.width, clip.height}:
            scaled = self.inference(wclip, **inference_kwargs)
        else:
            debug(f"{self}: Variable resolution clip detected!")

            if not isinstance(self.backend, (Backend.TRT, Backend.TRT_RTX)):
                raise CustomValueError(
                    "Variable resolution clips can only be processed with TRT Backend!", self.__class__, self.backend
                )

            warning(f"{self.__class__.__name__}: Variable resolution clip detected!")

            if self.backend.static_shape:
                warning("static_shape is True, setting it to False...")
                self.backend.static_shape = False

            if not self.backend.max_shapes:
                warning("max_shapes is None, setting it to (1936, 1088). You may want to adjust it...")
                self.backend.max_shapes = (1936, 1088)

            if not self.backend.opt_shapes:
                warning("opt_shapes is None, setting it to (64, 64). You may want to adjust it...")
                self.backend.opt_shapes = (64, 64)

            scaled = ProcessVariableResClip.from_func(
                wclip, lambda c: self.inference(c, **inference_kwargs), False, wclip.format, self.max_instances
            )

        scaled = self.postprocess_clip(scaled, clip, **postprocess_kwargs)

        return self._finish_scale(scaled, clip, width, height, shift, **kwargs)

    def calc_tilesize(self, clip: vs.VideoNode, **kwargs: Any) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Reimplementation of vsmlrt.calc_tilesize helper function
        """

        from vsmlrt import calc_tilesize

        kwargs = {
            "tiles": self.tiles,
            "tilesize": self.tilesize,
            "width": clip.width,
            "height": clip.height,
            "multiple": self.multiple,
            "overlap_w": self.overlap_w,
            "overlap_h": self.overlap_h,
        } | kwargs

        return calc_tilesize(**kwargs)

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Performs preprocessing on the clip prior to inference.
        """
        debug(f"{self}.pre: Before pp; Clip format is {clip.format!r}")

        clip = depth(clip, self._pick_precision(16, 32), vs.FLOAT, **kwargs)

        debug(f"{self}.pre: After pp; Clip format is {clip.format!r}")

        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Handles postprocessing of the model's output after inference.
        """
        debug(f"{self}.post: Before pp; Clip format is {clip.format!r}")

        clip = depth(
            clip,
            input_clip,
            dither_type=DitherType.ORDERED if 0 in {clip.width, clip.height} else DitherType.AUTO,
            **kwargs,
        )

        debug(f"{self}.post: After pp; Clip format is {clip.format!r}")

        return clip

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Runs inference on the given video clip using the configured model and backend.
        """

        from vsmlrt import inference

        tilesize, overlaps = self.calc_tilesize(clip)

        debug(f"{self}: Passing clip to inference: {clip.format!r}")
        debug(f"{self}: Passing model: {self.model}")
        debug(f"{self}: Passing tiles size: {tilesize}")
        debug(f"{self}: Passing overlaps: {overlaps}")
        debug(f"{self}: Passing extra kwargs: {kwargs}")

        return inference(clip, self.model, overlaps, tilesize, self.backend, **kwargs)

    def _pick_precision[_IntT: int](self, fp16: _IntT, fp32: _IntT) -> _IntT:
        from vsmlrt import Backend

        precision = (
            fp16
            if (isinstance(self.backend, _SupportsFP16) and self.backend.fp16)
            and isinstance(
                self.backend,
                (
                    Backend.TRT,
                    Backend.TRT_RTX,
                    Backend.ORT_CPU,
                    Backend.ORT_CUDA,
                    Backend.ORT_DML,
                    Backend.ORT_COREML,
                    Backend.NCNN_VK,
                ),
            )
            else fp32
        )

        debug(f"{self}: Selecting precision: {get_video_format(precision) if precision > 32 else precision!r}")

        return precision


class BaseOnnxScalerRGB(BaseOnnxScaler):
    """
    Abstract ONNX class for RGB models.
    """

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clip = self.kernel.resample(clip, self._pick_precision(vs.RGBH, vs.RGBS), Matrix.RGB, **kwargs)
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if get_video_format(clip) != get_video_format(input_clip):
            kwargs = (
                dict[str, Any](
                    format=input_clip,
                    matrix=Matrix.from_video(input_clip, func=self.__class__),
                    range=ColorRange.from_video(input_clip, func=self.__class__),
                    dither_type=DitherType.ORDERED,
                )
                | kwargs
            )
            clip = self.kernel.resample(clip, **kwargs)

        return clip


class GenericOnnxScaler(BaseOnnxScaler, partial_abstract=True):
    """
    Generic scaler class for an ONNX model.

    Example usage:
    ```py
    from vsscale import GenericOnnxScaler

    scaled = GenericOnnxScaler("path/to/model.onnx").scale(clip, ...)

    # For Windows paths:
    scaled = GenericOnnxScaler(r"path\\to\\model.onnx").scale(clip, ...)
    ```
    """


class BaseArtCNN(BaseOnnxScaler):
    _model: ClassVar[int]
    _static_kernel_radius = 2

    def __init__(
        self,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the scaler with the specified parameters.

        Args:
            backend: The backend to be used with the vs-mlrt framework. If set to None, the most suitable backend will
                be automatically selected, prioritizing fp16 support.
            tiles: Whether to split the image into multiple tiles. This can help reduce VRAM usage, but note that the
                model's behavior may vary when they are used.
            tilesize: The size of each tile when splitting the image (if tiles are enabled).
            overlap: The size of overlap between tiles.
            max_instances: Maximum instances to spawn when scaling a variable resolution clip.
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments to pass to the backend. See the vsmlrt backend's docstring for more details.
        """
        from vsmlrt import ArtCNNModel, models_path

        super().__init__(
            (SPath(models_path) / "ArtCNN" / f"{ArtCNNModel(self._model).name}.onnx").to_str(),
            backend,
            tiles,
            tilesize,
            overlap,
            1,
            max_instances,
            kernel=kernel,
            scaler=scaler,
            shifter=shifter,
            **kwargs,
        )


class BaseArtCNNLuma(BaseArtCNN):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return super().preprocess_clip(get_y(clip), **kwargs)

    def _finish_scale(
        self,
        clip: vs.VideoNode,
        input_clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        matrix: MatrixLike | None = None,
        copy_props: bool = False,
    ) -> vs.VideoNode:
        # Changes compared to BaseGenericScaler are:
        # - extract luma if input clip is luma only is removed  since this is a no op here
        # - Chroma planes are scaled accordingly with the artcnn'd luma,
        #   avoiding getting a luma plane when passing a YUV clip.

        if (clip.width, clip.height) != (width, height):
            clip = self.scaler.scale(clip, width, height)

        if input_clip.format.color_family == vs.YUV:
            scaled_chroma = self.scaler.scale(input_clip, clip.width, clip.height)
            clip = join(clip, scaled_chroma, prop_src=scaled_chroma)

        if shift != (0, 0):
            clip = self.shifter.shift(clip, shift)

        if clip.format.id != input_clip.format.id:
            clip = self.kernel.resample(clip, input_clip, matrix)

        if copy_props:
            return vs.core.std.CopyFrameProps(clip, input_clip)

        return clip


class BaseArtCNNChroma(BaseArtCNN):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        assert clip.format.color_family == vs.YUV

        bits = self._pick_precision(16, 32)
        format = clip.format.replace(subsampling_h=0, subsampling_w=0, sample_type=vs.FLOAT, bits_per_sample=bits)

        if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
            chroma_scaler = Kernel.ensure_obj(kwargs.pop("chroma_scaler", Bilinear))

            debug(f"{self}.pre: Before pp; Clip format is {clip.format!r}")

            clip = chroma_scaler.resample(clip, format, **kwargs)

            debug(f"{self}.pre: Before pp; Clip format is {clip.format!r}")

            return norm_expr(clip, ("x 0 1 clamp", "x 0.5 + 0 1 clamp"), func=self.__class__)

        return norm_expr(clip, "x plane_min - plane_max plane_min - / 0 1 clamp", format=format, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clip = norm_expr(clip, "x 0.5 -", [1, 2], func=self.__class__)
        return super().postprocess_clip(clip, input_clip, **kwargs)

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        from vsmlrt import flexible_inference

        tilesize, overlaps = self.calc_tilesize(clip)

        debug(f"{self}: Passing clip to inference: {clip.format!r}")
        debug(f"{self}: Passing model: {self.model}")
        debug(f"{self}: Passing tiles size: {tilesize}")
        debug(f"{self}: Passing overlaps: {overlaps}")
        debug(f"{self}: Passing extra kwargs: {kwargs}")

        u, v = flexible_inference(clip, self.model, overlaps, tilesize, self.backend, **kwargs)

        debug(f"{self}: Inferenced clip: {u.format!r}")
        debug(f"{self}: Inferenced clip: {v.format!r}")

        return core.std.ShufflePlanes([clip, u, v], [0, 0, 0], vs.YUV, clip)

    def _finish_scale(
        self,
        clip: vs.VideoNode,
        input_clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        matrix: MatrixLike | None = None,
        copy_props: bool = False,
    ) -> vs.VideoNode:
        if (clip.width, clip.height) != (width, height):
            clip = self.scaler.scale(clip, width, height)

        if shift != (0, 0):
            clip = self.shifter.shift(clip, shift)

        if clip.format.id != input_clip.format.replace(subsampling_w=0, subsampling_h=0).id:
            clip = self.kernel.resample(clip, input_clip, matrix)

        if copy_props:
            return vs.core.std.CopyFrameProps(clip, input_clip)

        return clip


class ArtCNN(BaseArtCNNLuma):
    """
    Super-Resolution Convolutional Neural Networks optimised for anime.

    A quick reminder that vs-mlrt does not ship these in the base package.
    You will have to grab the extended models pack or get it from the repo itself.
    (And create an "ArtCNN" folder in your models folder yourself)

    <https://github.com/Artoriuz/ArtCNN/releases/latest>

    Defaults to R8F64.

    Example usage:
    ```py
    from vsscale import ArtCNN

    doubled = ArtCNN().scale(clip, clip.width * 2, clip.height * 2)
    ```
    """

    _model = 7

    class C4F16(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 16 filters each.

        The currently fastest variant. Not really recommended for any filtering.
        Should strictly be used for real-time applications and even then the other non R ones should be fast enough...

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 10

    class C4F16_DN(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F16 but intended to also denoise. Works well on noisy sources when you don't want any sharpening.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16_DN().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 13

    class C4F16_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F16 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16_DS().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 11

    class C4F32(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 32 filters each.

        If you need an even faster model.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 0

    @deprecated(
        "This model is no longer maintained and has been deprecated. Please use R8F64_Chroma instead.",
        category=DeprecationWarning,
    )
    class C4F32_Chroma(BaseArtCNNChroma):  # noqa: N801
        """
        The smaller of the chroma models.

        These don't double the input clip and rather just try to enhance the chroma using luma information.

        Example usage:
        ```py
        from vsscale import ArtCNN

        chroma_upscaled = ArtCNN.C4F32_Chroma().scale(clip)
        ```
        """

        _model = 4

    class C4F32_DN(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F32 but intended to also denoise. Works well on noisy sources when you don't want any sharpening.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32_DN().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 14

    class C4F32_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F32 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32_DS().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 1

    @deprecated(
        "This model is no longer maintained and has been deprecated. Please use R8F64 instead.",
        category=DeprecationWarning,
    )
    class C16F64(BaseArtCNNLuma):
        """
        Very fast and good enough for AA purposes but the onnx variant is officially deprecated.

        This has 16 internal convolution layers with 64 filters each.

        ONNX files available at https://github.com/Artoriuz/ArtCNN/tree/388b91797ff2e675fd03065953cc1147d6f972c2/ONNX

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C16F64().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 2

    @deprecated(
        "This model is no longer maintained and has been deprecated. Please use R8F64_Chroma instead.",
        category=DeprecationWarning,
    )
    class C16F64_Chroma(BaseArtCNNChroma):  # noqa: N801
        """
        The bigger of the old chroma models.

        These don't double the input clip and rather just try to enhance the chroma using luma information.

        Example usage:
        ```py
        from vsscale import ArtCNN

        chroma_upscaled = ArtCNN.C16F64_Chroma().scale(clip)
        ```
        """

        _model = 5

    @deprecated(
        "This model is no longer maintained and has been deprecated. Please use R8F64 instead.",
        category=DeprecationWarning,
    )
    class C16F64_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C16F64 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C16F64_DS().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 3

    class R8F64(BaseArtCNNLuma):
        """
        A smaller and faster version of R16F96 but very competitive.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 7

    class R8F64_Chroma(BaseArtCNNChroma):  # noqa: N801
        """
        The new and fancy big chroma model.

        These don't double the input clip and rather just try to enhance the chroma using luma information.

        Example usage:
        ```py
        from vsscale import ArtCNN

        chroma_upscaled = ArtCNN.R8F64_Chroma().scale(clip)
        ```
        """

        _model = 9

    class R8F64_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as R8F64 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_DS().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 8

    class R8F64_JPEG420(BaseArtCNN, BaseOnnxScalerRGB):  # noqa: N801
        """
        1x RGB model meant to clean JPEG artifacts and to fix chroma subsampling.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_JPEG420().scale(clip)
        ```
        """

        _model = 15

    class R8F64_JPEG444(BaseArtCNN, BaseOnnxScalerRGB):  # noqa: N801
        """
        1x RGB model meant to clean JPEG artifacts.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_JPEG444().scale(clip)
        ```
        """

        _model = 16

    class R16F96(BaseArtCNNLuma):
        """
        The biggest model. Can compete with or outperform Waifu2x Cunet.

        Also quite a bit slower but is less heavy on vram.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R16F96().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 6

    class R16F96_Chroma(BaseArtCNNChroma):  # noqa: N801
        """
        The biggest and fancy chroma model. Shows almost biblical results on the right sources.

        These don't double the input clip and rather just try to enhance the chroma using luma information.

        Example usage:
        ```py
        from vsscale import ArtCNN

        chroma_upscaled = ArtCNN.R16F96_Chroma().scale(clip)
        ```
        """

        _model = 12


class BaseWaifu2x(BaseOnnxScaler):
    scale_w2x: Literal[1, 2, 4]
    """Upscaling factor."""

    noise: Literal[-1, 0, 1, 2, 3]
    """Noise reduction level"""

    _model: ClassVar[int]
    _static_kernel_radius = 2

    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the scaler with the specified parameters.

        Args:
            scale: Upscaling factor. 1 = no uspcaling, 2 = 2x, 4 = 4x.
            noise: Noise reduction level. -1 = none, 0 = low, 1 = medium, 2 = high, 3 = highest.
            backend: The backend to be used with the vs-mlrt framework. If set to None, the most suitable backend will
                be automatically selected, prioritizing fp16 support.
            tiles: Whether to split the image into multiple tiles. This can help reduce VRAM usage, but note that the
                model's behavior may vary when they are used.
            tilesize: The size of each tile when splitting the image (if tiles are enabled).
            overlap: The size of overlap between tiles.
            max_instances: Maximum instances to spawn when scaling a variable resolution clip.
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments to pass to the backend. See the vsmlrt backend's docstring for more details.
        """
        self.scale_w2x = scale
        self.noise = noise
        super().__init__(
            None,
            backend,
            tiles,
            tilesize,
            overlap,
            1,
            max_instances,
            kernel=kernel,
            scaler=scaler,
            shifter=shifter,
            **kwargs,
        )

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        from vsmlrt import Waifu2x as mlrt_Waifu2x
        from vsmlrt import Waifu2xModel

        return mlrt_Waifu2x(
            clip,
            self.noise,
            self.scale_w2x,
            self.tiles,
            self.tilesize,
            self.overlap,
            Waifu2xModel(self._model),
            self.backend,
            **kwargs,
        )


class _Waifu2xCunet(BaseWaifu2x, BaseOnnxScalerRGB):
    _model = 6
    _static_kernel_radius = 16

    if TYPE_CHECKING:

        def scale(
            self,
            clip: vs.VideoNode,
            width: int | None = None,
            height: int | None = None,
            shift: tuple[float, float] = (0, 0),
            **kwargs: Any,
        ) -> vs.VideoNode:
            """
            Scale the given clip using the ONNX model.

            Args:
                clip: The input clip to be scaled.
                width: The target width for scaling. If None, the width of the input clip will be used.
                height: The target height for scaling. If None, the height of the input clip will be used.
                shift: A tuple representing the shift values for the x and y axes.
                **kwargs: Additional arguments to be passed to the `preprocess_clip`, `postprocess_clip`, `inference`,
                    and `_final_scale` methods. Use the prefix `preprocess_` or `postprocess_` to pass an argument to
                    the respective method. Use the prefix `inference_` to pass an argument to the inference method.

                    Additional Notes for the Cunet model:

                       - The model can cause artifacts around the image edges.
                       To mitigate this, mirrored padding is applied to the image before inference.
                       This behavior can be disabled by setting `inference_no_pad=True`.
                       - A tint issue is also present but it is not constant. It leaves flat areas alone but tints
                       detailed areas.
                       Since most people will use Cunet to rescale details, the tint fix is enabled by default.
                       This behavior can be disabled with `postprocess_no_tint_fix=True`

            Returns:
                The scaled clip.
            """
            ...

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Cunet model ruins image borders, so we need to pad it before upscale and crop it after.
        if kwargs.pop("no_pad", False):
            return super().inference(clip, **kwargs)

        with padder.ctx(16, 4) as pad:
            padded = pad.MIRROR(clip)
            scaled = super().inference(padded, **kwargs)
            cropped = pad.CROP(scaled)

        return cropped

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Cunet model also has a tint issue but it is not constant
        # It leaves flat areas alone but tints detailed areas.
        # Since most people will use Cunet to rescale details, the tint fix is enabled by default.
        if kwargs.pop("no_tint_fix", False):
            return super().postprocess_clip(clip, input_clip, **kwargs)

        tint_fix = norm_expr(
            clip,
            "x 0.5 255 / + 0 1 clamp",
            planes=0 if get_video_format(input_clip).color_family is vs.GRAY else None,
            func="Waifu2x." + self.__class__.__name__,
        )
        return super().postprocess_clip(tint_fix, input_clip, **kwargs)


class Waifu2x(_Waifu2xCunet):
    """
    Well known Image Super-Resolution for Anime-Style Art.

    Defaults to Cunet.

    Example usage:
    ```py
    from vsscale import Waifu2x

    doubled = Waifu2x().scale(clip, clip.width * 2, clip.height * 2)
    ```
    """

    class AnimeStyleArt(BaseWaifu2x):
        """
        Waifu2x model for anime-style art.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.AnimeStyleArt().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 0

    class AnimeStyleArtRGB(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        RGB version of the anime-style model.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.AnimeStyleArtRGB().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 1

    class Photo(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        Waifu2x model trained on real-world photographic images.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.Photo().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 2

    class UpConv7AnimeStyleArt(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        UpConv7 model variant optimized for anime-style images.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.UpConv7AnimeStyleArt().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 3

    class UpConv7Photo(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        UpConv7 model variant optimized for photographic images.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.UpConv7Photo().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 4

    class UpResNet10(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        UpResNet10 model offering a balance of speed and quality.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.UpResNet10().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 5

    class Cunet(_Waifu2xCunet):
        """
        CUNet (Compact U-Net) model for anime art.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.Cunet().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

    class SwinUnetArt(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        Swin-Unet-based model trained on anime-style images.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetArt().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 7

    class SwinUnetPhoto(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        Swin-Unet model trained on photographic content.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetPhoto().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 8

    class SwinUnetPhotoV2(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        Improved Swin-Unet model for photos (v2).

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetPhotoV2().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 9

    class SwinUnetArtScan(BaseWaifu2x, BaseOnnxScalerRGB):
        """
        Swin-Unet model trained on anime scans.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetArtScan().scale(clip, clip.width * 2, clip.height * 2)
        ```
        """

        _model = 10


type _DPIRGrayModel = int
type _DPIRColorModel = int


class BaseDPIR(BaseOnnxScaler):
    _model: ClassVar[tuple[_DPIRGrayModel, _DPIRColorModel]]
    _static_kernel_radius = 8

    def __init__(
        self,
        strength: SupportsFloat | vs.VideoNode = 10,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the scaler with the specified parameters.

        Args:
            strength: Threshold (8-bit scale) strength for deblocking/denoising. If a VideoNode is used, it must be in
                GRAY8, GRAYH, or GRAYS format, with pixel values representing the 8-bit thresholds.
            backend: The backend to be used with the vs-mlrt framework. If set to None, the most suitable backend will
                be automatically selected, prioritizing fp16 support.
            tiles: Whether to split the image into multiple tiles. This can help reduce VRAM usage, but note that the
                model's behavior may vary when they are used.
            tilesize: The size of each tile when splitting the image (if tiles are enabled).
            overlap: The size of overlap between tiles.
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments to pass to the backend. See the vsmlrt backend's docstring for more details.
        """
        from vsmlrt import Backend

        self.strength = strength

        super().__init__(
            None,
            backend,
            tiles,
            tilesize,
            16 if overlap is None else overlap,
            8,
            -1,
            kernel=kernel,
            scaler=scaler,
            shifter=shifter,
            **kwargs,
        )

        if isinstance(self.backend, Backend.TRT) and not self.backend.force_fp16:
            self.backend.custom_args.extend(["--precisionConstraints=obey", "--layerPrecisions=Conv_123:fp32"])

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        *,
        copy_props: bool = True,
        **kwargs: Any,
    ) -> vs.VideoNode:
        assert check_variable_resolution(clip, self.__class__)

        return super().scale(clip, width, height, shift, copy_props=copy_props, **kwargs)

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if get_color_family(clip) == vs.GRAY:
            return super().preprocess_clip(clip, **kwargs)

        clip = self.kernel.resample(clip, self._pick_precision(vs.RGBH, vs.RGBS), Matrix.RGB, **kwargs)

        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if get_video_format(clip) != get_video_format(input_clip):
            kwargs = (
                dict[str, Any](
                    format=input_clip,
                    matrix=Matrix.from_video(input_clip, func=self.__class__),
                    range=ColorRange.from_video(input_clip, func=self.__class__),
                    dither_type=DitherType.ORDERED,
                )
                | kwargs
            )
            clip = self.kernel.resample(clip, **kwargs)

        return clip

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        from vsmlrt import DPIRModel, inference, models_path

        # Normalizing the strength clip
        strength_fmt = clip.format.replace(color_family=vs.GRAY)

        if isinstance(self.strength, vs.VideoNode):
            self.strength = norm_expr(self.strength, "x 255 /", format=strength_fmt, func=self.__class__)
        else:
            self.strength = clip.std.BlankClip(format=strength_fmt.id, color=float(self.strength) / 255, keep=True)

        debug(f"{self}: Passing strength clip format: {self.strength.format!r}")

        # Get model name
        self.model = (
            SPath(models_path) / "dpir" / f"{DPIRModel(self._model[clip.format.color_family != vs.GRAY]).name}.onnx"
        ).to_str()

        # Basic inference args
        tilesize, overlaps = self.calc_tilesize(clip)

        debug(f"{self}: Passing model: {self.model}")
        debug(f"{self}: Passing tiles size: {tilesize}")
        debug(f"{self}: Passing overlaps: {overlaps}")
        debug(f"{self}: Passing extra kwargs: {kwargs}")

        # Padding
        padding = padder.mod_padding(clip, self.multiple, 0)

        if not any(padding) or kwargs.pop("no_pad", False):
            return inference([clip, self.strength], self.model, overlaps, tilesize, self.backend, **kwargs)

        clip = padder.MIRROR(clip, *padding)
        strength = padder.MIRROR(self.strength, *padding)

        return inference([clip, strength], self.model, overlaps, tilesize, self.backend, **kwargs).std.Crop(*padding)


class DPIR(BaseDPIR):
    """
    Deep Plug-and-Play Image Restoration.
    """

    _model = (2, 3)

    class DrunetDenoise(BaseDPIR):
        """
        DPIR model for denoising.
        """

        _model = (0, 1)

    class DrunetDeblock(BaseDPIR):
        """
        DPIR model for deblocking.
        """

        _model = (2, 3)
