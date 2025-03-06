from dataclasses import dataclass
from typing import Any, ClassVar

from vskernels import Kernel, KernelT
from vstools import (
    CustomValueError, KwargsT, NotFoundEnumValue, SPath, SPathLike, core,
    depth, expect_bits, get_nvidia_version, get_video_format, get_y, inject_self, limiter, vs
)

from .helpers import GenericScaler

__all__ = ["GenericOnnxScaler", "autoselect_backend", "ArtCNN"]


@dataclass
class GenericOnnxScaler(GenericScaler):
    """Generic scaler class for an onnx model."""

    model: SPathLike
    """Path to the model."""
    backend: Any | None = None
    """
    vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.\n
    In order of trt > cuda > directml > nncn > cpu.
    """
    tiles: int | tuple[int, int] | None = None
    """Splits up the frame into multiple tiles. Helps if you're lacking in vram but models may behave differently."""

    tilesize: int | tuple[int, int] | None = None
    overlap: int | tuple[int, int] | None = None

    _static_kernel_radius = 2

    @inject_self
    def scale(  # type: ignore
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        if self.backend is None:
            self.backend = autoselect_backend()

        wclip, _ = expect_bits(clip, 32)

        from vsmlrt import calc_tilesize, inference, init_backend  #type: ignore[import-untyped]

        if self.overlap is None:
            overlap_w = overlap_h = 8
        else:
            overlap_w, overlap_h = (self.overlap, self.overlap) if isinstance(self.overlap, int) else self.overlap

        (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
            tiles=self.tiles,
            tilesize=self.tilesize,
            width=wclip.width,
            height=wclip.height,
            multiple=1,
            overlap_w=overlap_w,
            overlap_h=overlap_h,
        )

        if tile_w % 1 != 0 or tile_h % 1 != 0:
            raise CustomValueError(f"Tile size must be divisible by 1 ({tile_w}, {tile_h})", self.__class__)

        backend = init_backend(backend=self.backend, trt_opt_shapes=(tile_w, tile_h))

        scaled = inference(
            limiter(wclip, func=self.__class__),
            network_path=str(SPath(self.model).resolve()),
            backend=backend,
            overlap=(overlap_w, overlap_h),
            tilesize=(tile_w, tile_h),
        )
        return self._finish_scale(scaled, clip, width, height, shift)


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


class _BaseArtCNN:
    _model: ClassVar[int]
    _func = "ArtCNN"


@dataclass
class BaseArtCNN(_BaseArtCNN, GenericScaler):
    backend: Any | None = None
    """
    vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.\n
    In order of trt > cuda > directml > nncn > cpu.
    """
    chroma_scaler: KernelT | None = None
    """
    Scaler to upscale the chroma with.\n
    Necessary if you're trying to use one of the chroma models but aren't passing a 444 clip.\n
    Bilinear is probably the safe option to use.
    """

    tiles: int | tuple[int, int] | None = None
    """Splits up the frame into multiple tiles. Helps if you're lacking in vram but models may behave differently."""
    tilesize: int | tuple[int, int] | None = None
    overlap: int | tuple[int, int] | None = None

    _static_kernel_radius = 2

    @inject_self
    def scale(  # type: ignore
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        from vsmlrt import ArtCNNModel, ArtCNN as mlrt_ArtCNN

        clip_format = get_video_format(clip)
        chroma_model = self._model in [4, 5, 9]

        # The chroma models aren't supposed to change the video dimensions and API wise this is more comfortable.
        if width is None or height is None:
            if chroma_model:
                width = clip.width
                height = clip.height
            else:
                raise CustomValueError("You have to pass height and width if not using a chroma model.", self._func)

        if chroma_model and clip_format.color_family != vs.YUV:
            raise CustomValueError("ArtCNN Chroma models need YUV input.", self._func)

        if not chroma_model and clip_format.color_family not in (vs.YUV, vs.GRAY):
            raise CustomValueError("Regular ArtCNN models need YUV or GRAY input.", self._func)

        if chroma_model and (clip_format.subsampling_h != 0 or clip_format.subsampling_w != 0):
            if self.chroma_scaler is None:
                raise CustomValueError(
                    "ArtCNN needs a non subsampled clip. Either pass one or set `chroma_scaler`.", self._func
                )

            clip = Kernel.ensure_obj(self.chroma_scaler).resample(
                clip, clip_format.replace(subsampling_h=0, subsampling_w=0)
            )

        if self._model not in ArtCNNModel.__members__.values():
            raise NotFoundEnumValue(f'Invalid model: \'{self._model}\'. Please update \'vsmlrt\'!', self._func)

        wclip = get_y(clip) if not chroma_model else clip

        if self.backend is None:
            self.backend = autoselect_backend()

        scaled = mlrt_ArtCNN(
            limiter(depth(wclip, 32), func=self._func),
            self.tiles,
            self.tilesize,
            self.overlap,
            ArtCNNModel(self._model),
            backend=self.backend,
        )

        return self._finish_scale(scaled, wclip, width, height, shift)


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
