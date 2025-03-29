from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

from vskernels import Bilinear, Catrom, Kernel, KernelT, ScalerT
from vstools import (
    ConstantFormatVideoNode, CustomValueError, KwargsT, NotFoundEnumValue, SPath, SPathLike, check_variable, core,
    depth, get_nvidia_version, get_y, inject_self, limiter, vs
)

from .helpers import BaseGenericScaler

__all__ = ["GenericOnnxScaler", "autoselect_backend", "ArtCNN"]


def autoselect_backend(trt_args: KwargsT = {}, **kwargs: Any) -> Any:
    import os

    from vsmlrt import Backend

    fp16 = kwargs.pop("fp16", True)

    cuda = get_nvidia_version() is not None
    if cuda:
        if hasattr(core, "trt"):
            kwargs.update(trt_args)
            return Backend.TRT(fp16=fp16, **trt_args)
        elif hasattr(core, "ort"):
            return Backend.ORT_CUDA(fp16=fp16, **kwargs)
        else:
            return Backend.OV_GPU(fp16=fp16, **kwargs)
    else:
        if hasattr(core, "ort") and os.name == "nt":
            return Backend.ORT_DML(fp16=fp16, **kwargs)
        elif hasattr(core, "ncnn"):
            return Backend.NCNN_VK(fp16=fp16, **kwargs)

        return Backend.ORT_CPU(fp16=fp16, **kwargs) if hasattr(core, "ort") else Backend.OV_CPU(fp16=fp16, **kwargs)


class BaseGenericOnnxScaler(BaseGenericScaler, ABC):
    """Abstract generic scaler class for an onnx model."""

    def __init__(
        self,
        model: SPathLike | None = None,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param model:       Path to the model.
        :param backend:     vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.
                            In order of trt > cuda > directml > nncn > cpu.
        :param tiles:       Splits up the frame into multiple tiles.
                            Helps if you're lacking in vram but models may behave differently.
        :param tilesize:    
        :param overlap:     
        :param kernel:      Base kernel to be used for certain scaling/shifting/resampling operations.
                            Defaults to Catrom.
        :param scaler:      Scaler used for scaling operations. Defaults to kernel.
        :param shifter:     Kernel used for shifting operations. Defaults to scaler.
        """
        if model is not None:
            self.model = str(SPath(model).resolve())

        self.backend = autoselect_backend() if backend is None else backend

        self.tiles = tiles
        self.tilesize = tilesize

        if overlap is None:
            self.overlap_w = self.overlap_h = 8
        elif isinstance(overlap, int):
            self.overlap_w = self.overlap_h = overlap
        else:
            self.overlap_w, self.overlap_h = overlap

        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

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

    def init_backend(self, trt_opt_shapes: tuple[int, int]) -> Any:
        from vsmlrt import init_backend

        return init_backend(backend=self.backend, trt_opt_shapes=trt_opt_shapes)

    def inference(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        from vsmlrt import inference

        (tile_w, tile_h), (overlap_w, overlap_h) = self.calc_tilesize(clip)

        if tile_w % 1 != 0 or tile_h % 1 != 0:
            raise CustomValueError("Tile size must be divisible by 1", self.__class__, (tile_w, tile_h))

        scaled = inference(
            limiter(depth(clip, 32), func=self.__class__),
            network_path=self.model,
            backend=self.init_backend(trt_opt_shapes=(tile_w, tile_h)),
            overlap=(overlap_w, overlap_h),
            tilesize=(tile_w, tile_h),
        )

        if TYPE_CHECKING:
            assert check_variable(scaled, self.__class__)

        return scaled


class GenericOnnxScaler(BaseGenericOnnxScaler):
    """Generic scaler class for an onnx model."""

    _static_kernel_radius = 2

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        scaled = self.inference(clip)

        width, height = self._wh_norm(clip, width, height)

        return self._finish_scale(scaled, clip, width, height, shift, **kwargs)


class BaseArtCNN(BaseGenericOnnxScaler):
    _model: ClassVar[int]

    _static_kernel_radius = 2

    def __init__(
        self,
        chroma_scaler: KernelT = Bilinear,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param chroma_scaler:   Scaler to upscale the chroma with. Defaults to Bilinear.
        :param backend:         vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.
                                In order of trt > cuda > directml > nncn > cpu.
        :param tiles:           Splits up the frame into multiple tiles.
                                Helps if you're lacking in vram but models may behave differently.
        :param tilesize:        
        :param overlap:         
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        """

        self.chroma_scaler = Kernel.ensure_obj(chroma_scaler)

        super().__init__(None, backend, tiles, tilesize, overlap, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        from vsmlrt import ArtCNN as mlrt_ArtCNN
        from vsmlrt import ArtCNNModel

        assert check_variable(clip, self.__class__)

        chroma_model = self._model in [4, 5, 9]

        # The chroma models aren't supposed to change the video dimensions and API wise this is more comfortable.
        if width is None or height is None:
            if chroma_model:
                width = clip.width
                height = clip.height
            else:
                raise CustomValueError(
                    "You have to pass height and width if not using a chroma model.", "ArtCNN", (width, height)
                )

        if chroma_model and clip.format.color_family != vs.YUV:
            raise CustomValueError("ArtCNN Chroma models need YUV input.", "ArtCNN")

        if not chroma_model and clip.format.color_family not in (vs.YUV, vs.GRAY):
            raise CustomValueError("Regular ArtCNN models need YUV or GRAY input.", "ArtCNN")

        if chroma_model and (clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0):
            clip = self.chroma_scaler.resample(
                clip, clip.format.replace(subsampling_h=0, subsampling_w=0)
            )

        if self._model not in ArtCNNModel.__members__.values():
            raise NotFoundEnumValue(f"Invalid model: '{self._model}'. Please update 'vsmlrt'!", "ArtCNN")

        wclip = get_y(clip) if not chroma_model else clip

        scaled = mlrt_ArtCNN(
            limiter(depth(wclip, 32), func="ArtCNN"),
            self.tiles,
            self.tilesize,
            (self.overlap_w, self.overlap_h),
            ArtCNNModel(self._model),
            backend=self.backend,
        )

        if TYPE_CHECKING:
            assert check_variable(scaled, self.__class__)

        return self._finish_scale(scaled, wclip, width, height, shift, **kwargs)


class ArtCNN(BaseArtCNN):
    """
    Super-Resolution Convolutional Neural Networks optimised for anime.

    A quick reminder that vs-mlrt does not ship these in the base package.\n
    You will have to grab the extended models pack or get it from the repo itself.\n
    (And create an "ArtCNN" folder in your models folder yourself)

    https://github.com/Artoriuz/ArtCNN/releases/latest

    Defaults to R8F64.
    """

    _model = 7

    class C4F32(BaseArtCNN):
        """
        This has 4 internal convolution layers with 32 filters each.\n
        If you need an even faster model.
        """

        _model = 0

    class C4F32_DS(BaseArtCNN):
        """The same as C4F32 but intended to also sharpen and denoise."""

        _model = 1

    class C16F64(BaseArtCNN):
        """
        Very fast and good enough for AA purposes but the onnx variant is officially deprecated.\n
        This has 16 internal convolution layers with 64 filters each.

        ONNX files available at https://github.com/Artoriuz/ArtCNN/tree/388b91797ff2e675fd03065953cc1147d6f972c2/ONNX
        """

        _model = 2

    class C16F64_DS(BaseArtCNN):
        """The same as C16F64 but intended to also sharpen and denoise."""

        _model = 3

    class C4F32_Chroma(BaseArtCNN):
        """
        The smaller of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 4

    class C16F64_Chroma(BaseArtCNN):
        """
        The bigger of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 5

    class R16F96(BaseArtCNN):
        """
        The biggest model. Can compete with or outperform Waifu2x Cunet.\n
        Also quite a bit slower but is less heavy on vram.
        """

        _model = 6

    class R8F64(BaseArtCNN):
        """
        A smaller and faster version of R16F96 but very competitive.
        """

        _model = 7

    class R8F64_DS(BaseArtCNN):
        """The same as R8F64 but intended to also sharpen and denoise."""

        _model = 8

    class R8F64_Chroma(BaseArtCNN):
        """
        The new and fancy big chroma model.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 9

    class C4F16(BaseArtCNN):
        """
        This has 4 internal convolution layers with 16 filters each.\n
        The currently fastest variant. Not really recommended for any filtering.\n
        Should strictly be used for real-time applications and even then the other non R ones should be fast enough...
        """

        _model = 10

    class C4F16_DS(BaseArtCNN):
        """The same as C4F16 but intended to also sharpen and denoise."""

        _model = 11
