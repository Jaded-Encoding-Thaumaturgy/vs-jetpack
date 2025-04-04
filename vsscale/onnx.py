from abc import ABC
from logging import warning
from typing import Any, ClassVar, Literal

from vsexprtools import norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelT, ScalerT
from vstools import (
    ConstantFormatVideoNode, CustomValueError, DitherType, Matrix, ProcessVariableResClip, SPath, SPathLike,
    check_variable_format, core, depth, get_nvidia_version, get_video_format, get_y, inject_self, limiter, padder, vs
)

from .generic import BaseGenericScaler

__all__ = [
    "autoselect_backend",

    "BaseOnnxScaler",

    "GenericOnnxScaler",

    "ArtCNN",

    "Waifu2x"
]


def _clean_keywords(kwargs: dict[str, Any], backend: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in backend.__dataclass_fields__}


def autoselect_backend(**kwargs: Any) -> Any:
    import os

    from vsmlrt import Backend

    backend: Any

    if get_nvidia_version():
        if hasattr(core, "trt"):
            backend = Backend.TRT
        elif hasattr(core, "ort"):
            backend = Backend.ORT_CUDA
        elif hasattr(core, "ncnn"):
            backend = Backend.NCNN_VK
        else:
            backend = Backend.OV_GPU
    else:
        if hasattr(core, "ort") and os.name == "nt":
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
    """Abstract generic scaler class for an onnx model."""

    def __init__(
        self,
        model: SPathLike | None = None,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param model:           Path to the model.
        :param backend:         vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.
                                In order of trt > cuda > directml > nncn > cpu.
        :param tiles:           Splits up the frame into multiple tiles.
                                Helps if you're lacking in vram but models may behave differently.
        :param tilesize:
        :param overlap:
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        """
        if model is not None:
            self.model = str(SPath(model).resolve())

        if backend is None:
            # Default with float16 precision and output as fp16 as well
            # if the backend supports it
            self.backend = autoselect_backend(fp16=16, output_format=1)
        else:
            self.backend = backend

        self.tiles = tiles
        self.tilesize = tilesize
        self.overlap = overlap

        if self.overlap is None:
            self.overlap_w = self.overlap_h = 8
        elif isinstance(self.overlap, int):
            self.overlap_w = self.overlap_h = self.overlap
        else:
            self.overlap_w, self.overlap_h = self.overlap

        self.max_instances = max_instances

        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        from vsmlrt import Backend

        assert check_variable_format(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        wclip = self.preprocess_clip(clip)

        if 0 not in {clip.width, clip.height}:
            scaled = self.inference(wclip)
        else:
            if not isinstance(self.backend, Backend.TRT):
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

            scaled = ProcessVariableResClip[ConstantFormatVideoNode].from_func(
                wclip, self.inference, False, wclip.format, self.max_instances
            )

        scaled = self.postprocess_clip(scaled, clip)

        return self._finish_scale(scaled, clip, width, height, shift, **kwargs)

    def calc_tilesize(self, clip: vs.VideoNode) -> tuple[tuple[int, int], tuple[int, int]]:
        from vsmlrt import calc_tilesize

        return calc_tilesize(
            tiles=self.tiles,
            tilesize=self.tilesize,
            width=clip.width,
            height=clip.height,
            multiple=1,
            overlap_w=self.overlap_w,
            overlap_h=self.overlap_h,
        )

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        clip = depth(clip, 16 if self.backend.fp16 else 32, vs.FLOAT)
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return depth(
            clip, input_clip, dither_type=DitherType.ORDERED if 0 in {clip.width, clip.height} else DitherType.AUTO
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        from vsmlrt import inference

        tiles, overlaps = self.calc_tilesize(clip)

        return inference(clip, self.model, overlaps, tiles, self.backend, **kwargs)


class GenericOnnxScaler(BaseOnnxScaler):
    """Generic scaler class for an onnx model."""

    _static_kernel_radius = 2


class BaseArtCNN(BaseOnnxScaler):
    _model: ClassVar[int]
    _static_kernel_radius = 2

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        from vsmlrt import ArtCNN as mlrt_ArtCNN
        from vsmlrt import ArtCNNModel

        return mlrt_ArtCNN(clip, self.tiles, self.tilesize, self.overlap, ArtCNNModel(self._model), self.backend)


class BaseArtCNNLuma(BaseArtCNN):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return super().preprocess_clip(get_y(clip), **kwargs)


class BaseArtCNNChroma(BaseArtCNN):
    def __init__(
        self,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        chroma_scaler: KernelT = Bilinear,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param backend:         vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.
                                In order of trt > cuda > directml > nncn > cpu.
        :param tiles:           Splits up the frame into multiple tiles.
                                Helps if you're lacking in vram but models may behave differently.
        :param tilesize:
        :param overlap:
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param chroma_scaler:   Scaler to upscale the chroma with. Defaults to Bilinear.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        """
        self.chroma_scaler = Kernel.ensure_obj(chroma_scaler)

        super().__init__(
            None, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable_format(clip, self.__class__)

        if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
            clip = self.chroma_scaler.resample(
                clip, clip.format.replace(
                    subsampling_h=0, subsampling_w=0,
                    sample_type=vs.FLOAT, bits_per_sample=16 if self.backend.fp16 else 32
                )
            )
            return limiter(clip, func=self.__class__)

        return super().preprocess_clip(clip, **kwargs)


class ArtCNN(BaseArtCNNLuma):
    """
    Super-Resolution Convolutional Neural Networks optimised for anime.

    A quick reminder that vs-mlrt does not ship these in the base package.\n
    You will have to grab the extended models pack or get it from the repo itself.\n
    (And create an "ArtCNN" folder in your models folder yourself)

    https://github.com/Artoriuz/ArtCNN/releases/latest

    Defaults to R8F64.
    """

    _model = 7

    class C4F32(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 32 filters each.\n
        If you need an even faster model.
        """

        _model = 0

    class C4F32_DS(BaseArtCNNLuma):
        """The same as C4F32 but intended to also sharpen and denoise."""

        _model = 1

    class C16F64(BaseArtCNNLuma):
        """
        Very fast and good enough for AA purposes but the onnx variant is officially deprecated.\n
        This has 16 internal convolution layers with 64 filters each.

        ONNX files available at https://github.com/Artoriuz/ArtCNN/tree/388b91797ff2e675fd03065953cc1147d6f972c2/ONNX
        """

        _model = 2

    class C16F64_DS(BaseArtCNNLuma):
        """The same as C16F64 but intended to also sharpen and denoise."""

        _model = 3

    class C4F32_Chroma(BaseArtCNNChroma):
        """
        The smaller of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 4

    class C16F64_Chroma(BaseArtCNNChroma):
        """
        The bigger of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 5

    class R16F96(BaseArtCNNLuma):
        """
        The biggest model. Can compete with or outperform Waifu2x Cunet.\n
        Also quite a bit slower but is less heavy on vram.
        """

        _model = 6

    class R8F64(BaseArtCNNLuma):
        """
        A smaller and faster version of R16F96 but very competitive.
        """

        _model = 7

    class R8F64_DS(BaseArtCNNLuma):
        """The same as R8F64 but intended to also sharpen and denoise."""

        _model = 8

    class R8F64_Chroma(BaseArtCNNChroma):
        """
        The new and fancy big chroma model.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 9

    class C4F16(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 16 filters each.\n
        The currently fastest variant. Not really recommended for any filtering.\n
        Should strictly be used for real-time applications and even then the other non R ones should be fast enough...
        """

        _model = 10

    class C4F16_DS(BaseArtCNNLuma):
        """The same as C4F16 but intended to also sharpen and denoise."""

        _model = 11


class BaseWaifu2x(BaseOnnxScaler):
    scale_w2x: Literal[1, 2, 4]
    noise: Literal[-1, 0, 1, 2, 3]

    _model: ClassVar[int]
    _static_kernel_radius = 2

    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param noise:
        :param scale:
        :param backend:         vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.
                                In order of trt > cuda > directml > nncn > cpu.
        :param tiles:           Splits up the frame into multiple tiles.
                                Helps if you're lacking in vram but models may behave differently.
        :param tilesize:
        :param overlap:
        :param max_instances:
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        """
        self.scale_w2x = scale
        self.noise = noise
        super().__init__(
            None, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
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
            **kwargs
        )


class BaseWaifu2xRGB(BaseWaifu2x):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        clip = self.kernel.resample(clip, vs.RGBH if self.backend.fp16 else vs.RGBS, Matrix.RGB)
        return limiter(clip, func=self.__class__)


class BaseWaifu2xMlrtPreprocess(BaseWaifu2x):
    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        preprocess: bool = True,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        self.preprocess = preprocess
        super().__init__(
            scale, noise, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return super().inference(clip, preprocess=self.preprocess, **kwargs)


class Waifu2x(BaseWaifu2x):
    _model = 6

    class AnimeStyleArt(BaseWaifu2xMlrtPreprocess, BaseWaifu2x):
        _model = 0

    class AnimeStyleArtRGB(BaseWaifu2xMlrtPreprocess, BaseWaifu2xRGB):
        _model = 1

    class Photo(BaseWaifu2xMlrtPreprocess, BaseWaifu2xRGB):
        _model = 2

    class UpConv7AnimeStyleArt(BaseWaifu2xRGB):
        _model = 3

    class UpConv7Photo(BaseWaifu2xRGB):
        _model = 4

    class UpResNet10(BaseWaifu2xRGB):
        _model = 5

    class Cunet(BaseWaifu2xRGB):
        _model = 6

        def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
            with padder.ctx(16, 4) as pad:
                padded = pad.MIRROR(clip)
                scaled = super().inference(padded, **kwargs)
                cropped = pad.CROP(scaled)

            return cropped

        def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
            tint_fix = norm_expr(
                clip, 'x 0.5 255 / + 0 1 clamp',
                planes=0 if get_video_format(input_clip).color_family is vs.GRAY else None,
                func="Waifu2x" + self.__class__.__name__
            )
            return super().postprocess_clip(tint_fix, input_clip, **kwargs)

    class SwinUnetArt(BaseWaifu2xRGB):
        _model = 7

    class SwinUnetPhoto(BaseWaifu2xRGB):
        _model = 8

    class SwinUnetPhotoV2(BaseWaifu2xRGB):
        _model = 9

    class SwinUnetArtScan(BaseWaifu2xRGB):
        _model = 10
