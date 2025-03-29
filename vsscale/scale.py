from __future__ import annotations

from functools import partial
from math import ceil, log2
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from vsexprtools import complexpr_available, expr_func, norm_expr
from vskernels import Catrom, Hermite, KernelT, LinearScaler, Scaler, ScalerT
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ConstantFormatVideoNode, KwargsT, Matrix, MatrixT, PlanesT, ProcessVariableClip,
    ProcessVariableResClip, VSFunctionNoArgs, check_ref_clip, check_variable, check_variable_format,
    clamp, core, depth, fallback, get_nvidia_version, get_prop, inject_self, limiter, padder, vs
)

from .helpers import BaseGenericScaler

__all__ = [
    'DPID',
    'SSIM',
    'Waifu2x'
]


class DPID(BaseGenericScaler):
    """Rapid, Detail-Preserving Image Downscaler for VapourSynth"""

    def __init__(
        self,
        sigma: float = 0.1,
        ref: vs.VideoNode | ScalerT = Catrom,
        planes: PlanesT = None,
        **kwargs: Any
    ) -> None:
        """
        :param sigma:       The power factor of range kernel. It can be used to tune the amplification
                            of the weights of pixels that represent detail—from a box filter over an emphasis
                            of distinct pixels towards a selection of only the most distinct pixels.
        :param ref:         VideoNode or Scaler to obtain the downscaled reference for DPID.
        :param planes:      Sets which planes will be processed. Any unprocessed planes will be simply copied from ref.
        """
        super().__init__(**kwargs)

        self.sigma = sigma
        self.ref = ref
        self.planes = planes

        if isinstance(ref, vs.VideoNode):
            self._ref_scaler = self.scaler
        else:
            self._ref_scaler = Scaler.ensure_obj(ref, self.__class__)

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

        width, height = self._wh_norm(clip, width, height)

        ref = clip

        if isinstance(self.ref, vs.VideoNode):
            check_ref_clip(clip, self.ref)

            if TYPE_CHECKING:
                assert check_variable_format(self.ref, self.__class__)

            ref = self.ref

        if (ref.width, ref.height) != (width, height):
            ref = self._ref_scaler.scale(ref, width, height)  # type: ignore[assignment]

        kwargs = {
            'lambda': self.sigma, 'planes': self.planes,
            'src_left': shift[1], 'src_top': shift[0]
        } | self.kwargs | kwargs | {'read_chromaloc': True}

        return core.dpid.DpidRaw(clip, ref, **kwargs)

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return self._ref_scaler.kernel_radius


class SSIM(LinearScaler):
    """
    SSIM downsampler is an image downscaling technique that aims to optimize
    for the perceptual quality of the downscaled results.

    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured
    using famous Structural SIMilarity (SSIM) index.

    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.
    """

    def __init__(
        self,
        scaler: ScalerT = Hermite,
        smooth: int | float | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param scaler:      Scaler to be used for downscaling, defaults to Hermite.
        :param smooth:      Image smoothening method.
                            If you pass an int, it specifies the "radius" of the internally-used boxfilter,
                            i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
                            If you pass a float, it specifies the "sigma" of gauss_blur,
                            i.e. the standard deviation of gaussian blur.
                            If you pass a function, it acts as a general smoother.
                            Default uses a gaussian blur based on the scaler's kernel radius.
        """
        super().__init__(**kwargs)

        self.scaler = Hermite.from_param(scaler)

        if smooth is None:
            smooth = (self.scaler.kernel_radius + 1.0) / 3

        if callable(smooth):
            self.filter_func = smooth
        elif isinstance(smooth, int):
            self.filter_func = partial(box_blur, radius=smooth)
        elif isinstance(smooth, float):
            self.filter_func = partial(gauss_blur, sigma=smooth)

    def _linear_scale(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.scale)

        l1 = self.scaler.scale(clip, width, height, shift, **(kwargs | self.kwargs))

        l1_sq, c_sq = [expr_func(x, 'x dup *') for x in (l1, clip)]

        l2 = self.scaler.scale(c_sq, width, height, shift, **(kwargs | self.kwargs))

        m, sl_m_square, sh_m_square = [self.filter_func(x) for x in (l1, l1_sq, l2)]

        if complexpr_available:
            merge_expr = f'z dup * SQ! x SQ@ - SQD! SQD@ {1e-6} < 0 y SQ@ - SQD@ / sqrt ?'
        else:
            merge_expr = f'x z dup * - {1e-6} < 0 y z dup * - x z dup * - / sqrt ?'

        r = expr_func([sl_m_square, sh_m_square, m], merge_expr)

        t = expr_func([r, m], 'x y *')

        return expr_func([self.filter_func(m), self.filter_func(r), l1, self.filter_func(t)], 'x y z * + a -')

    @inject_self.property
    def kernel_radius(self) -> int:  # type: ignore
        return self.scaler.kernel_radius


class Waifu2xPadHelper(ProcessVariableResClip):
    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int]) -> vs.VideoNode:
        padding = padder.mod_padding(cast_to)

        return padder.MIRROR(super().normalize(clip, cast_to), *padding).std.SetFrameProp('_PadValues', padding)


class Waifu2xCropHelper(ProcessVariableClip[tuple[int, int, int, int, int, int]]):
    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int, int, int, int, int]:
        return (frame.width, frame.height, *get_prop(frame, '_PadValues', list))

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int, int, int, int, int]) -> vs.VideoNode:
        width, height, *padding = cast_to

        return (
            clip.std.RemoveFrameProps()
            .resize.Point(width, height)
            .std.CopyFrameProps(clip)
            .std.Crop(*(p * 2 for p in padding))
        )


class Waifu2xScaleHelper(ProcessVariableResClip):
    def __init__(
        self, clip: vs.VideoNode, backend: type, backend_kwargs: KwargsT, kwargs: KwargsT, cache_size: int
    ) -> None:
        super().__init__(clip, cache_size=cache_size)

        self.kwargs = kwargs
        self.backend = backend
        self.backend_kwargs = backend_kwargs

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int]) -> vs.VideoNode:
        from vsmlrt import Waifu2x as MlrtWaifu2x  # type: ignore

        if (max_shapes := self.backend_kwargs.get('max_shapes', None)):
            if cast_to[0] > max_shapes[0] or cast_to[1] > max_shapes[1]:
                self.backend_kwargs.update(max_shapes=cast_to)

        return MlrtWaifu2x(
            super().normalize(clip, cast_to), backend=self.backend(**self.backend_kwargs), **self.kwargs
        )


class Waifu2xResizeHelper(ProcessVariableResClip):
    def __init__(
        self, clip: vs.VideoNode, width: int, height: int, planes: PlanesT, is_gray: bool,
        scaler: Scaler, do_padding: bool, w2x_kwargs: KwargsT, w2x_cache_size: int,
        backend: type, backend_kwargs: KwargsT
    ) -> None:
        super().__init__(clip, (width, height))

        self.width = width
        self.height = height
        self.planes = planes
        self.is_gray = is_gray
        self.scaler = scaler
        self.do_padding = do_padding
        self.w2x_kwargs = w2x_kwargs
        self.w2x_cache_size = w2x_cache_size
        self.backend = backend
        self.backend_kwargs = backend_kwargs.copy()

    def normalize(self, wclip: vs.VideoNode, cast_to: tuple[int, int]) -> vs.VideoNode:
        mult = max(int(log2(ceil(size))) for size in (self.width / cast_to[0], self.height / cast_to[1]))

        wclip = limiter(wclip, func=self.__class__)

        for _ in range(mult):
            if self.do_padding:
                wclip = Waifu2xPadHelper.from_clip(wclip)

            wclip = Waifu2xScaleHelper(
                wclip, self.backend, self.backend_kwargs, self.w2x_kwargs, self.w2x_cache_size
            ).eval_clip()

            if self.do_padding:
                cropped = Waifu2xCropHelper.from_clip(wclip)

                wclip = norm_expr(cropped, 'x 0.5 255 / + 0 1 clamp', planes=self.planes, func=self.__class__)

        return wclip

    def process(self, wclip: vs.VideoNode) -> vs.VideoNode:
        if self.is_gray:
            wclip = wclip.std.ShufflePlanes(0, vs.GRAY)

        return self.scaler.scale(wclip, self.width, self.height)


class _BaseWaifu2x:
    _model: ClassVar[int]
    _needs_gray = False
    _static_args = dict(noise=-1, scale=2)


@dataclass
class BaseWaifu2x(_BaseWaifu2x, GenericScaler):
    """Use Waifu2x neural network to scale clips."""

    cuda: bool | Literal['trt'] | None = None
    """Whether to run this on cpu, gpu, or use trt technology. None will pick the fastest automatically."""

    num_streams: int | None = None
    """Number of gpu streams for the model."""

    fp16: bool = True
    """Whether to use float16 precision if available."""

    device_id: int = 0
    """Id of the cuda device to use."""

    matrix: MatrixT | None = None
    """Input clip's matrix. Set only if necessary."""

    tiles: int | tuple[int, int] | None = None
    """Process in separate tiles instead of the whole frame. Use if [V]RAM limited."""

    tilesize: int | tuple[int, int] | None = None
    """Manually specify the size of a single tile."""

    overlap: int | tuple[int, int] | None = None
    """Overlap for reducing blocking artifacts between tile borders."""

    backend_kwargs: KwargsT | None = None
    """Kwargs passed to create the backend instance."""

    dynamic_shape: bool | None = None
    """
    Use a single model for 0-max_shapes resolutions.
    None to automatically detect it. Will be True when previewing and TRT is available.
    """

    max_shapes: tuple[int, int] | None = (1936, 1088)
    """
    Max shape for a dynamic model when using TRT and variable resolution clip.
    This can be overridden if the frame size is bigger.
    """

    max_instances: int = 2
    """Maximum instances to spawn when scaling a variable resolution clip."""

    def __post_init__(self) -> None:
        cuda = self.cuda

        if self.dynamic_shape is None:
            try:
                from vspreview.api import is_preview

                self.dynamic_shape = is_preview()
            except Exception:
                self.dynamic_shape = False

        if cuda is True:
            self.fp16 = False
        elif self.fp16:
            self.fp16 = complexpr_available.fp16

        bkwargs = (self.backend_kwargs or KwargsT()) | KwargsT(fp16=self.fp16, device_id=self.device_id)

        # All this will eventually be in vs-nn
        if cuda is None:
            try:
                data: KwargsT = core.trt.DeviceProperties(self.device_id)  # type: ignore
                memory = data.get('total_global_memory', 0)
                def_num_streams = clamp(data.get('async_engine_count', 1), 1, 2)

                cuda = 'trt'

                def_bkwargs = KwargsT(
                    workspace=memory / (1 << 22) if memory else None,
                    use_cuda_graph=True, use_cublas=True, use_cudnn=True,
                    use_edge_mask_convolutions=True, use_jit_convolutions=True,
                    static_shape=True, heuristic=True, output_format=int(self.fp16),
                    num_streams=def_num_streams
                )

                if self._model >= Waifu2x.SwinUnetArt._model:
                    def_bkwargs |= KwargsT(tf32=not self.fp16)

                bkwargs = def_bkwargs | bkwargs

                streams_info = 'OK' if bkwargs['num_streams'] == def_num_streams else 'MISMATCH'

                core.log_message(
                    vs.MESSAGE_TYPE_DEBUG,
                    f'Selected [{data.get("name", b"<unknown>").decode("utf8")}] '
                    f'with {f"{(memory / (1 << 30))}GiB" if memory else "<unknown>"} of VRAM, '
                    f'num_streams={def_num_streams} ({streams_info})'
                )
            except Exception:
                self.fp16 = False
                bkwargs['fp16'] = False
                cuda = get_nvidia_version() is not None

        if self.num_streams is not None:
            bkwargs.update(num_streams=self.num_streams)
        elif bkwargs.get('num_streams', None) is None:
            bkwargs.update(num_streams=fallback(self.num_streams, 1))

        self._cuda = cuda
        self._bkwargs = bkwargs

        super().__post_init__()

    @property
    def _backend(self) -> object:
        from vsmlrt import Backend

        if self._cuda is True:
            if hasattr(core, 'ort'):
                return Backend.ORT_CUDA

            return Backend.OV_GPU
        elif self._cuda is False:
            if hasattr(core, 'ncnn'):
                return Backend.NCNN_VK

            if hasattr(core, 'ort'):
                return Backend.ORT_CPU

            return Backend.OV_CPU

        return Backend.TRT

    @inject_self.init_kwargs.clean
    def scale(  # type:ignore
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        from vsmlrt import Backend

        width, height = self._wh_norm(clip, width, height)

        wclip = clip

        assert check_variable_format(clip, self.scale)

        matrix = self.matrix
        is_gray = clip.format.color_family is vs.GRAY
        planes = 0 if is_gray else None

        _static_args = kwargs.pop('_static_args', self._static_args)
        force = _static_args.pop('force', False)
        do_scale = _static_args.get('scale') > 1

        bkwargs = self._bkwargs.copy()

        dynamic_shapes = self.dynamic_shape or (0 in (clip.width, clip.height)) or not bkwargs.get('static_shape', True)

        kwargs.update(tiles=self.tiles, tilesize=self.tilesize, overlap=self.overlap)

        if dynamic_shapes and self._backend is Backend.TRT:
            bkwargs.update(static_shape=False, opt_shapes=(64, 64), max_shapes=self.max_shapes)

        if (is_upscale := width > clip.width or height > clip.width or force):
            model = self._model

            if clip.format.color_family is vs.YUV:
                if not matrix:
                    matrix = Matrix.from_param_or_video(matrix or self.matrix, clip, False, self.__class__)

                wclip = self._kernel.resample(wclip, vs.RGBH if self.fp16 else vs.RGBS, Matrix.RGB, matrix)
            else:
                wclip = depth(wclip, 16 if self.fp16 else 32, vs.FLOAT)

                if is_gray and model != 0:
                    wclip = wclip.std.ShufflePlanes(0, vs.RGB)

            assert wclip.format

            if wclip.format.color_family is vs.RGB:
                if model == 0:
                    model = 1

            wclip = Waifu2xResizeHelper(
                wclip, width, height, planes, is_gray, self._scaler,
                do_scale and self._model == Waifu2x.Cunet._model,
                KwargsT(
                    **_static_args, model=model,
                    preprocess=False, **kwargs
                ), self.max_instances, self._backend, bkwargs  # type: ignore[arg-type]
            ).eval_clip()

        return self._finish_scale(wclip, clip, width, height, shift, matrix, is_upscale)

    _static_kernel_radius = 2


class Waifu2x(BaseWaifu2x):
    _model = 6

    class AnimeStyleArt(BaseWaifu2x):
        _model = 0

    class Photo(BaseWaifu2x):
        _model = 2

    class UpConv7AnimeStyleArt(BaseWaifu2x):
        _model = 3

    class UpConv7Photo(BaseWaifu2x):
        _model = 4

    class UpResNet10(BaseWaifu2x):
        _model = 5

    class Cunet(BaseWaifu2x):
        _model = 6

    class SwinUnetArt(BaseWaifu2x):
        _model = 7

    class SwinUnetPhoto(BaseWaifu2x):
        _model = 8

    class SwinUnetPhotoV2(BaseWaifu2x):
        _model = 9

    class SwinUnetArtScan(BaseWaifu2x):
        _model = 10
