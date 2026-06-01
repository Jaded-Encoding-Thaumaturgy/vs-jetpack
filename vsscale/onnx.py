"""
This module implements scalers for ONNX models.
"""

from __future__ import annotations

import dataclasses
import math
import re
from abc import ABC
from logging import getLogger
from typing import TYPE_CHECKING, Any, ClassVar, Literal, SupportsFloat

from jetpytools import CustomValueError, FileNotExistsError, SPath, SPathLike

from vsexprtools import norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelLike, ScalerLike
from vsmasktools import Morpho
from vstools import (
    Matrix,
    MatrixLike,
    ProcessVariableResClip,
    Range,
    check_variable_resolution,
    core,
    depth,
    get_color_family,
    get_video_format,
    get_y,
    join,
    limiter,
    padder,
    vs,
)

from .generic import BaseGenericScaler
from .mlrt import Backend, get_model_folder

type BackendLike = type[Backend] | Backend

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
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

        if model is not None:
            self.model = SPath(model).resolve()

        if backend is None:
            self.backend = Backend.autoselect(**self.kwargs)
        elif isinstance(backend, type):
            self.backend = backend(**self.kwargs)
        elif self.kwargs:
            self.backend = dataclasses.replace(backend, **self.kwargs)
        else:
            self.backend = backend

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
        logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)

        clip = depth(clip, self._pick_precision(16, 32), vs.FLOAT, **kwargs)

        logger.debug("%s: After pp; Clip format is %r", self.preprocess_clip, clip.format)

        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """
        Handles postprocessing of the model's output after inference.
        """
        logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)
        logger.debug("%s: Before pp; Clip format is %r", self.postprocess_clip, clip.format)

        clip = depth(clip, input_clip, **kwargs)

        logger.debug("%s: After pp; Clip format is %r", self.postprocess_clip, clip.format)

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

    def _pick_precision[IntT: int](self, fp16: IntT, fp32: IntT) -> IntT:
        precision = (
            fp16
            if isinstance(self.backend, (Backend.ORT, Backend.NCNN, Backend.TRT, Backend.TRT)) and self.backend.fp16
            else fp32
        )

        logger.debug(
            "%s: Selecting precision: %r",
            self._pick_precision,
            get_video_format(precision) if precision > 32 else precision,
        )

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
                    range=Range.from_video(input_clip, func=self.__class__),
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
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
            **kwargs: Additional arguments.
        """
        model = self._model if hasattr(self, "_model") else self.__class__.__name__

        super().__init__(
            get_model_folder("ArtCNN") / f"ArtCNN_{model}.onnx",
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

            logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)

            clip = chroma_scaler.resample(clip, format, **kwargs)

            logger.debug("%s: Before pp; Clip format is %r", self.preprocess_clip, clip.format)

            return norm_expr(clip, ("x 0 1 clamp", "x 0.5 + 0 1 clamp"), func=self.__class__)

        return norm_expr(clip, "x plane_min - plane_max plane_min - / 0 1 clamp", format=format, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clip = norm_expr(clip, "x 0.5 -", [1, 2], func=self.__class__)
        return super().postprocess_clip(clip, input_clip, **kwargs)

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        tilesize, overlaps = self.calc_tilesize(clip)

        logger.debug("%s: Passing clip to inference: %s", self.inference, clip.format)
        logger.debug("%s: Passing model: %s", self.inference, self.model)
        logger.debug("%s: Passing tiles size: %s", self.inference, tilesize)
        logger.debug("%s: Passing overlaps: %s", self.inference, overlaps)
        logger.debug("%s: Passing extra kwargs: %s", self.inference, kwargs)

        u, v = self.backend.inference(clip, self.model, overlaps, tilesize, flexible=True, **kwargs)

        logger.debug("%s: Inferenced clip: %s", self.inference, u.format)
        logger.debug("%s: Inferenced clip: %s", self.inference, v.format)

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
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
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

        model = get_model_folder("Waifu2x") / f"{model_name}.onnx"

        if not model.exists():
            raise FileNotExistsError(
                "The specified model does not exist. Run `vsscale onnx show waifu2x` to see the available models.",
                self.__class__,
            )

        super().__init__(
            model,
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

                    Additional Notes for the Cunet model:

                       - The model can cause artifacts around the image edges.
                       To mitigate this, mirrored padding is applied to the image before inference.
                       This behavior can be disabled by setting `inference_no_pad=True`.
                       - A tint issue is also present but it is not constant. It leaves flat areas alone but tints
                       detailed areas.
                       This behavior can be disabled with `postprocess_no_tint_fix=True`

            Returns:
                The scaled clip.
            """
            ...

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Cunet model ruins image borders, so we need to pad it before upscale and crop it after.
        if kwargs.pop("no_pad", False):
            return super().inference(clip, **kwargs)

        with padder.ctx(4, 4) as pad:
            padded = pad.MIRROR(clip)
            scaled = super().inference(padded, **kwargs)
            cropped = pad.CROP(scaled)

        return cropped

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Cunet model also has a tint issue but it is not constant
        # It leaves flat areas alone but tints detailed areas.
        if kwargs.pop("no_tint_fix", False):
            return super().postprocess_clip(clip, input_clip, **kwargs)

        clip = depth(clip, 32)

        tint_fix = norm_expr(
            [clip, Morpho.maximum(clip)], "x 0.5 255 / + y min", func="Waifu2x." + self.__class__.__name__
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


class BaseDPIR(BaseOnnxScaler):
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
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
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
                    range=Range.from_video(input_clip, func=self.__class__),
                )
                | kwargs
            )
            clip = self.kernel.resample(clip, **kwargs)

        return clip

    def inference(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        # Normalizing the strength clip
        strength_fmt = clip.format.replace(color_family=vs.GRAY)

        if isinstance(self.strength, vs.VideoNode):
            self.strength = norm_expr(self.strength, "x 255 /", format=strength_fmt, func=self.__class__)
        else:
            self.strength = clip.std.BlankClip(format=strength_fmt, color=float(self.strength) / 255, keep=True)

        logger.debug("%s: Passing strength clip format: %s", self.inference, self.strength.format)

        # Get model name
        model = "drunet"
        if self._kind:
            model += f"_{self._kind}"

        if clip.format.color_family == vs.GRAY:
            model += "_gray"
        else:
            model += "_color"

        model = get_model_folder("DPIR") / f"{model}.onnx"

        # Basic inference args
        tilesize, overlaps = self.calc_tilesize(clip)

        logger.debug("%s: Passing model: %s", self.inference, self.model)
        logger.debug("%s: Passing tiles size: %s", self.inference, tilesize)
        logger.debug("%s: Passing overlaps: %s", self.inference, overlaps)
        logger.debug("%s: Passing extra kwargs: %s", self.inference, kwargs)

        # Padding
        padding = padder.mod_padding(clip, self.multiple, 0)

        if not any(padding) or kwargs.pop("no_pad", False):
            return self.backend.inference([clip, self.strength], model, overlaps, tilesize, **kwargs)

        clip = padder.MIRROR(clip, *padding)
        strength = padder.MIRROR(self.strength, *padding)

        return self.backend.inference([clip, strength], self.model, overlaps, tilesize, **kwargs).std.Crop(*padding)


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
