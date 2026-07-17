"""
This module implements scalers for ONNX models.
"""

from __future__ import annotations

import dataclasses
import math
import re
from abc import ABC
from contextlib import suppress
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, SupportsFloat

from jetpytools import (
    CustomRuntimeError,
    CustomTypeError,
    CustomValueError,
    FileNotExistsError,
    FuncExcept,
    SPath,
    SPathLike,
)

from vsexprtools import ExprOp, combine_expr, norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelLike, ScalerLike
from vstools import (
    ConvMode,
    Matrix,
    Padder,
    ProcessVariableResClip,
    check_variable_resolution,
    core,
    get_color_family,
    get_y,
    join,
    limiter,
    vs,
)

from .generic import BaseGenericScaler
from .mlrt import Backend, get_model_path
from .mlrt.backend.base import Backend as RealBackend
from .mlrt.settings import get_toml_config

type BackendLike = type[Backend | RealBackend] | Backend | RealBackend

__all__ = ["DPIR", "ArtCNN", "BaseOnnxScaler", "GenericOnnxScaler", "Waifu2x"]


logger = getLogger(__name__)


class BaseOnnxScaler(BaseGenericScaler, ABC):
    """
    Abstract generic scaler class for an ONNX model.
    """

    def __init__(
        self,
        model: SPathLike | None = None,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] = 0,
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
            kernel: Base kernel to be used for certain scaling/shifting operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

        if model is not None:
            self.model = SPath(model).resolve()

        if backend is None:
            self.backend = RealBackend.autoselect(**self.kwargs)
        elif isinstance(backend, type) and issubclass(backend, RealBackend):
            self.backend = backend(**self.kwargs)
        elif self.kwargs and isinstance(backend, RealBackend):
            self.backend = dataclasses.replace(backend, **self.kwargs)
        elif isinstance(backend, RealBackend):
            self.backend = backend
        else:
            raise CustomTypeError("Invalid backend")

        if isinstance(self.backend, Backend.ORT) and self.backend.fp16:
            bl = set(self.backend.fp16_blacklist_ops or []).union(["ConstantOfShape", "Resize"])
            self.backend = dataclasses.replace(self.backend, fp16_blacklist_ops=bl)

        self.tiles = tiles
        self.tilesize = tilesize
        self.overlap = overlap
        self.multiple = multiple

        if isinstance(self.overlap, int):
            self.overlap_w = self.overlap_h = self.overlap
        else:
            self.overlap_w, self.overlap_h = self.overlap

        self.max_instances = max_instances

        logger.info("%s: Using '%s' backend", self, self.backend.__class__.__name__)
        logger.debug("%s: %s", self, self.backend)
        logger.debug("%s: User tiles: %s", self, self.tiles)
        logger.debug("%s: User tilesize: %s", self, self.tilesize)
        logger.debug("%s: User overlap: %s", self, (self.overlap_w, self.overlap_h))
        logger.debug("%s: User multiple: %s", self, self.multiple)

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
        if clip.format.sample_type != vs.SampleType.FLOAT:
            raise CustomRuntimeError("Only FLOAT formats are supported.", self.scale)

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

        logger.debug("%s: Preprocess kwargs: %s", self.scale, preprocess_kwargs)
        logger.debug("%s: Postprocess kwargs: %s", self.scale, postprocess_kwargs)
        logger.debug("%s: Inference kwargs: %s", self.scale, inference_kwargs)

        wclip = self.preprocess_clip(clip, **preprocess_kwargs)

        if 0 not in {clip.width, clip.height}:
            scaled = self.inference(wclip, **inference_kwargs)
        else:
            logger.debug("%s: Variable resolution clip detected...", self.scale)

            if not isinstance(self.backend, Backend.TRT) or self.backend.static_shape:
                raise CustomValueError("Only TRT backends support static_shape=False", self.__class__, self.backend)

            scaled = ProcessVariableResClip.from_func(
                wclip, lambda c: self.inference(c, **inference_kwargs), False, wclip.format, self.max_instances
            )

        scaled = self.postprocess_clip(scaled, clip, **postprocess_kwargs)

        return self._finish_scale(scaled, clip, width, height, shift, **kwargs)

    def calc_tilesize(self, clip: vs.VideoNode) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Reimplementation of vsmlrt.calc_tilesize helper function
        """

        def calc_size(dimension: int, tiles: int, overlap: int, multiple: int) -> int:
            return math.ceil((dimension + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple

        overlap_w = self.overlap_w
        overlap_h = self.overlap_h

        if isinstance(self.tilesize, tuple):
            tile_w, tile_h = self.tilesize
        elif isinstance(self.tilesize, int):
            tile_w, tile_h = self.tilesize, self.tilesize
        else:
            if self.tiles is None:
                overlap_w = 0
                overlap_h = 0
                tile_w = clip.width
                tile_h = clip.height
            elif isinstance(self.tiles, int):
                tile_w = calc_size(clip.width, self.tiles, self.overlap_w, self.multiple)
                tile_h = calc_size(clip.height, self.tiles, self.overlap_h, self.multiple)
            else:
                tile_w = calc_size(clip.width, self.tiles[0], self.overlap_w, self.multiple)
                tile_h = calc_size(clip.height, self.tiles[1], self.overlap_h, self.multiple)

            if tile_w % self.multiple != 0 or tile_h % self.multiple != 0:
                raise CustomValueError(
                    f"Tile size ({tile_w}, {tile_h}) must be divisible by {self.multiple}",
                    self.__class__,
                )

        return (tile_w, tile_h), (overlap_w, overlap_h)

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Performs preprocessing on the clip prior to inference.
        """
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Handles postprocessing of the model's output after inference.
        """
        return clip

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Runs inference on the given video clip using the configured model and backend.
        """

        tilesize, overlaps = self.calc_tilesize(clip)

        logger.debug("%s: Passing clip to inference: %r", self.inference, clip.format)
        logger.debug("%s: Passing model: %s", self.inference, self.model)
        logger.debug("%s: Passing tiles size: %s", self.inference, tilesize)
        logger.debug("%s: Passing overlaps: %s", self.inference, overlaps)
        logger.debug("%s: Passing extra kwargs: %s", self.inference, kwargs)

        return self.backend.inference(clip, self.model, overlaps, tilesize, flexible=False, **kwargs)


class BaseOnnxScalerRGB(BaseOnnxScaler):
    """
    Abstract ONNX class for RGB models.
    """

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clip = self.kernel.resample(
            clip,
            clip.format.replace(color_family=vs.ColorFamily.RGB, subsampling_w=0, subsampling_h=0),
            Matrix.RGB,
            **kwargs,
        )
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        logger.debug("%s.post: Before pp; Clip format is %r", self, clip.format)

        out_fmt = input_clip.format.replace(subsampling_w=0, subsampling_h=0)
        # Resamples only for color_family changes e.g. RGB -> YUV
        if clip.format != out_fmt:
            clip = self.kernel.resample(clip, out_fmt, input_clip, range=input_clip, **kwargs)

        logger.debug("%s.post: After pp; Clip format is %r", self, clip.format)

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
    _model: ClassVar[str]
    _static_kernel_radius = 2

    def __init__(
        self,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] = 8,
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
            kernel: Base kernel to be used for certain scaling/shifting operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        model = self._model if hasattr(self, "_model") else self.__class__.__name__

        super().__init__(
            _get_onnx_model("ArtCNN", f"ArtCNN_{model}", func=self.__class__),
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
        copy_props: bool = False,
    ) -> vs.VideoNode:
        if input_clip.format.color_family == vs.YUV:
            scaled_chroma = self.scaler.scale(input_clip, clip.width, clip.height)
            clip = join(clip, scaled_chroma, prop_src=scaled_chroma)

            logger.debug("%s: Chroma planes has been scaled accordingly", self)

        return super()._finish_scale(clip, input_clip, width, height, shift, copy_props)


class BaseArtCNNChroma(BaseArtCNN):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        assert clip.format.color_family == vs.YUV

        if (clip.format.subsampling_h, clip.format.subsampling_w) != (0, 0):
            logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)
            fmt = clip.format.replace(subsampling_h=0, subsampling_w=0)
            clip = Kernel.ensure_obj(kwargs.pop("chroma_scaler", Bilinear)).resample(clip, fmt, **kwargs)
            logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)

        return norm_expr(clip, ("x 0 1 clamp", "x 0.5 + 0 1 clamp"), func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clip = norm_expr(clip, "x 0.5 -", [1, 2], func=self.__class__)
        return super().postprocess_clip(clip, input_clip, **kwargs)

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        tilesize, overlaps = self.calc_tilesize(clip)

        logger.debug("%s: Passing clip to inference: %r", self.inference, clip.format)
        logger.debug("%s: Passing model: %r", self.inference, self.model)
        logger.debug("%s: Passing tiles size: %r", self.inference, tilesize)
        logger.debug("%s: Passing overlaps: %r", self.inference, overlaps)
        logger.debug("%s: Passing extra kwargs: %r", self.inference, kwargs)

        u, v = self.backend.inference(clip, self.model, overlaps, tilesize, flexible=True, **kwargs)

        logger.debug("%s: Inferenced clip: %r", self.inference, u.format)
        logger.debug("%s: Inferenced clip: %r", self.inference, v.format)

        return core.std.ShufflePlanes([clip, u, v], [0, 0, 0], vs.YUV, clip)


class ArtCNN(BaseArtCNNLuma):
    """
    Super-Resolution Convolutional Neural Networks optimised for anime.

    <https://github.com/Artoriuz/ArtCNN/releases/latest>

    Defaults to R8F64.

    Example usage:
    ```py
    from vsscale import ArtCNN

    doubled = ArtCNN().supersample(clip, 2)
    ```
    """

    _model = "R8F64"

    class C4F16(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 16 filters each.

        The currently fastest variant. Not really recommended for any filtering.
        Should strictly be used for real-time applications and even then the other non R ones should be fast enough...

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16().supersample(clip, 2)
        ```
        """

    class C4F16_DN(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F16 but intended to also denoise. Works well on noisy sources when you don't want any sharpening.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16_DN().supersample(clip, 2)
        ```
        """

    class C4F16_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F16 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F16_DS().supersample(clip, 2)
        ```
        """

    class C4F32(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 32 filters each.

        If you need an even faster model.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32().supersample(clip, 2)
        ```
        """

    class C4F32_DN(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F32 but intended to also denoise. Works well on noisy sources when you don't want any sharpening.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32_DN().supersample(clip, 2)
        ```
        """

    class C4F32_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as C4F32 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.C4F32_DS().supersample(clip, 2)
        ```
        """

    class R8F64(BaseArtCNNLuma):
        """
        A smaller and faster version of R16F96 but very competitive.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64().supersample(clip, 2)
        ```
        """

    class R8F64_DS(BaseArtCNNLuma):  # noqa: N801
        """
        The same as R8F64 but intended to also denoise and sharpen.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_DS().supersample(clip, 2)
        ```
        """

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

    class R8F64_Chroma_DN(BaseArtCNNChroma):  # noqa: N801
        """
        Noise-focused variant of R8F64_Chroma.

        Trained for noisy or heavily compressed sources, aggressively removing chroma noise and artifacts.

        Example usage:
        ```py
        from vsscale import ArtCNN

        chroma_upscaled = ArtCNN.R8F64_Chroma_DN().scale(clip)
        ```
        """

    class R8F64_JPEG420(BaseArtCNN, BaseOnnxScalerRGB):  # noqa: N801
        """
        1x RGB model meant to clean JPEG artifacts and to fix chroma subsampling.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_JPEG420().scale(clip)
        ```
        """

    class R8F64_JPEG444(BaseArtCNN, BaseOnnxScalerRGB):  # noqa: N801
        """
        1x RGB model meant to clean JPEG artifacts.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R8F64_JPEG444().scale(clip)
        ```
        """

    class R16F96(BaseArtCNNLuma):
        """
        The biggest model. Can compete with or outperform Waifu2x Cunet.

        Also quite a bit slower but is less heavy on vram.

        Example usage:
        ```py
        from vsscale import ArtCNN

        doubled = ArtCNN.R16F96().supersample(clip, 2)
        ```
        """


class BaseWaifu2x(BaseOnnxScalerRGB):
    scale_w2x: Literal[1, 2, 4]
    """Upscaling factor."""

    noise: Literal[-1, 0, 1, 2, 3]
    """Noise reduction level"""

    _model: ClassVar[str]

    _static_kernel_radius = 2

    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] = 4,
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
            kernel: Base kernel to be used for certain scaling/shifting operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        self.scale_w2x = scale
        self.noise = noise
        model_name = self._model if hasattr(self, "_model") else self.__class__.__name__
        model_name = re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()  # CamelCase -> snake_case

        if self.scale_w2x > 1:
            model_name += f"_scale{self.scale_w2x}x"

        if self.noise >= 0:
            model_name += f"_noise{self.noise}"

        super().__init__(
            _get_onnx_model("Waifu2x", model_name, func=self.__class__),
            backend,
            tiles,
            tilesize,
            overlap,
            max_instances=max_instances,
            kernel=kernel,
            scaler=scaler,
            shifter=shifter,
            **kwargs,
        )


class _Waifu2xCunet(BaseWaifu2x):
    _static_kernel_radius = 16

    if not TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs, multiple=4)

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

                    Additional note for the Cunet model:

                       - A tint issue is also present but it is not constant. It leaves flat areas alone but tints
                       detailed areas.
                       This behavior can be disabled with `postprocess_no_tint_fix=True`

            Returns:
                The scaled clip.
            """
            ...

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if kwargs.pop("no_pad", False) or (not clip.width % self.overlap_w and not clip.height % self.overlap_h):
            logger.debug("%s: Skipping padding for clip %r", self, clip)
            return super().inference(clip, **kwargs)

        logger.debug("%s: Padding clip %r", self, clip)
        pad = Padder.from_mod(clip, 4, 0)
        padded = pad.mirror(clip)
        scaled = super().inference(padded, **kwargs)
        cropped = pad.crop(scaled)

        return cropped

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Cunet model also has a tint issue but it is not constant
        # It leaves flat areas alone but tints detailed areas.
        if kwargs.pop("no_tint_fix", False):
            return super().postprocess_clip(clip, input_clip, **kwargs)

        maximum_expr = combine_expr(ExprOp.matrix("x", 1, ConvMode.SQUARE, [(0, 0)])[0])
        tint_fix = norm_expr(
            clip,
            ["x 0.5 255 / +", maximum_expr, ExprOp.MIN],
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

    doubled = Waifu2x().supersample(clip, 2)
    ```
    """

    _model = "CunetArt"

    class CunetArt(_Waifu2xCunet):
        """
        CUNet (Compact U-Net) model for anime art.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.CunetArt().supersample(clip, 2)
        ```
        """

    Cunet = CunetArt

    class SwinUnetArt(BaseWaifu2x):
        """
        Swin-Unet-based model trained on anime-style images.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetArt().supersample(clip, 2)
        ```
        """

    class SwinUnetPhoto(BaseWaifu2x):
        """
        Swin-Unet model trained on photographic content.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetPhoto().supersample(clip, 2)
        ```
        """

    class SwinUnetArtScan(BaseWaifu2x):
        """
        Swin-Unet model trained on anime scans.

        Example usage:
        ```py
        from vsscale import Waifu2x

        doubled = Waifu2x.SwinUnetArtScan().supersample(clip, 2)
        ```
        """


class BaseDPIR(BaseOnnxScalerRGB, BaseOnnxScaler):
    _kind: ClassVar[str] = ""
    _static_kernel_radius = 8

    def __init__(
        self,
        strength: SupportsFloat | vs.VideoNode = 10,
        backend: BackendLike | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] = 16,
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
            kernel: Base kernel to be used for certain scaling/shifting operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        self.strength = strength

        super().__init__(
            None,
            backend,
            tiles,
            tilesize,
            overlap,
            8,
            -1,
            kernel=kernel,
            scaler=scaler,
            shifter=shifter,
            **kwargs,
        )

        if isinstance(self.backend, (Backend.TRT, Backend.ORT, Backend.NCNN, Backend.OV_CPU, Backend.OV_GPU)):
            bl = set(self.backend.fp16_blacklist_ops or []).union(["Conv_123"])
            self.backend = dataclasses.replace(self.backend, fp16_blacklist_ops=bl)

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
        return (
            BaseOnnxScaler.preprocess_clip(self, clip, **kwargs)
            if get_color_family(clip) == vs.GRAY
            else BaseOnnxScalerRGB.preprocess_clip(self, clip, **kwargs)
        )

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Normalizing the strength clip
        strength_fmt = clip.format.replace(color_family=vs.GRAY)

        if isinstance(self.strength, vs.VideoNode):
            strength = norm_expr(self.strength, "x 255 /", format=strength_fmt, func=self.__class__)
        else:
            strength = clip.std.BlankClip(format=strength_fmt, color=float(self.strength) / 255, keep=True)

        logger.debug("%s: Passing strength clip format: %r", self.inference, strength.format)

        # Get model name
        model_name = "drunet"
        if self._kind:
            model_name += f"_{self._kind}"

        if clip.format.color_family == vs.GRAY:
            model_name += "_gray"
        else:
            model_name += "_color"

        model = _get_onnx_model("DPIR", model_name, func=self.__class__)

        # Basic inference args
        tilesize, overlaps = self.calc_tilesize(clip)

        logger.debug("%s: Passing model: %s", self.inference, model)
        logger.debug("%s: Passing tiles size: %s", self.inference, tilesize)
        logger.debug("%s: Passing overlaps: %s", self.inference, overlaps)
        logger.debug("%s: Passing extra kwargs: %s", self.inference, kwargs)

        # Padding
        pad = Padder.from_mod(clip, self.multiple, 0)

        if not any(pad) or kwargs.pop("no_pad", False):
            return self.backend.inference([clip, strength], model, overlaps, tilesize, **kwargs)

        clip = pad.mirror(clip)
        strength = pad.mirror(strength)

        return pad.crop(self.backend.inference([clip, strength], model, overlaps, tilesize, **kwargs))


class DPIR(BaseDPIR):
    """
    Deep Plug-and-Play Image Restoration.
    """

    class DrunetDenoise(BaseDPIR):
        """
        DPIR model for denoising.
        """

    class DrunetDeblock(BaseDPIR):
        """
        DPIR model for deblocking.
        """

        _kind = "deblocking"


def _get_onnx_model(
    provider: str,
    model_name: str,
    *,
    auto_download: bool | None = None,
    func: FuncExcept | None = None,
) -> Path:
    with suppress(FileNotExistsError):
        if (path := get_model_path(provider, model_name)).exists():
            return path

    logger.debug("%r does not exist", model_name)

    conf = get_toml_config()
    dconf = conf.get("onnx", {}).get("download", {})

    if dconf.get("auto", auto_download):
        from rich.logging import RichHandler

        from .mlrt.cli import app

        logger.info("Auto-downloading %r from provider %r", model_name, provider)

        user_provider = next((p for p in dconf.get("provider", []) if p.lower().startswith(provider.lower())), None)
        console = next((h.console for h in logger.handlers if isinstance(h, RichHandler)), None)

        app(["onnx", "download", user_provider or provider, "--latest"], console=console, result_action="return_value")

        with suppress(FileNotExistsError):
            if (path := get_model_path(provider, model_name)).exists():
                return path

    raise CustomRuntimeError(
        f"The specified model {model_name} does not exist. "
        "Run `vsscale onnx download --help` to see the available models.",
        func,
    )
