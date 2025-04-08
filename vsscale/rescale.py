from __future__ import annotations

from functools import cached_property, wraps
from typing import Any, Callable, TypeVar

from vsexprtools import norm_expr
from vskernels import Bilinear, BorderHandling, Hermite, Kernel, KernelT, Point, Scaler, ScalerT
from vskernels.types import LeftShift, TopShift
from vsmasktools import KirschTCanny, based_diff_mask
from vsmasktools.utils import _get_region_expr
from vstools import (
    ColorRange, ConstantFormatVideoNode, DitherType, FieldBased, FieldBasedT, FrameRangeN, FrameRangesN, VideoNodeT,
    check_variable, core, depth, get_peak_value, get_y, join, limiter, replace_ranges, split, vs
)

from .helpers import BottomCrop, CropRel, LeftCrop, RightCrop, ScalingArgs, TopCrop
from .onnx import ArtCNN

__all__ = [
    'Rescale',
    'RescaleBase',
    'RescaleT',
]

RescaleT = TypeVar('RescaleT', bound="RescaleBase")


class RescaleBase:
    descale_args: ScalingArgs
    field_based: FieldBased | None

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        kernel: KernelT,
        upscaler: ScalerT = ArtCNN,
        downscaler: ScalerT = Hermite(linear=True),
        field_based: FieldBasedT | bool | None = None,
        border_handling: int | BorderHandling = BorderHandling.MIRROR
    ) -> None:
        assert check_variable(clip, self.__class__)

        self.clipy, *chroma = split(clip)
        self.chroma = chroma

        self.kernel = Kernel.ensure_obj(kernel)
        self.upscaler = Scaler.ensure_obj(upscaler)

        self.downscaler = Scaler.ensure_obj(downscaler)

        self.field_based = FieldBased.from_param(field_based)

        self.border_handling = BorderHandling(int(border_handling))

    def __delattr__(self, __name: str) -> None:
        match __name:
            case 'descale':
                self._trydelattr('rescale')
                self._trydelattr('doubled')
            case 'doubled':
                self._trydelattr('upscale')
            case _:
                pass
        delattr(self, __name)

    def _trydelattr(self, attr: str) -> None:
        try:
            delattr(self, attr)
        except AttributeError:
            pass

    @staticmethod
    def _apply_field_based(
        function: Callable[[RescaleT, VideoNodeT], VideoNodeT]
    ) -> Callable[[RescaleT, VideoNodeT], VideoNodeT]:
        @wraps(function)
        def wrap(self: RescaleT, clip: VideoNodeT) -> VideoNodeT:
            if self.field_based:
                clip = self.field_based.apply(clip)
                clip = function(self, clip)
                return FieldBased.PROGRESSIVE.apply(clip)
            else:
                return function(self, clip)
        return wrap

    @staticmethod
    def _add_props(
        function: Callable[[RescaleT, VideoNodeT], VideoNodeT]
    ) -> Callable[[RescaleT, VideoNodeT], VideoNodeT]:
        @wraps(function)
        def wrap(self: RescaleT, clip: VideoNodeT) -> VideoNodeT:
            w, h = (
                f"{int(d)}" if d.is_integer() else f"{d:.2f}"
                for d in [self.descale_args.src_width, self.descale_args.src_height]
            )
            return core.std.SetFrameProp(
                function(self, clip),
                "Rescale" + function.__name__.split('_')[-1].capitalize() + 'From',
                data=f'{self.kernel.__class__.__name__} - {w} x {h}'
            )
        return wrap

    @_add_props
    @_apply_field_based
    def _generate_descale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self.kernel.descale(
            clip,
            self.descale_args.width, self.descale_args.height,
            **self.descale_args.kwargs(),
            border_handling=self.border_handling
        )

    @_add_props
    @_apply_field_based
    def _generate_rescale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self.kernel.scale(  # type: ignore[return-value]
            clip,
            self.clipy.width, self.clipy.height,
            **self.descale_args.kwargs(),
            border_handling=self.border_handling
        )

    @_add_props
    def _generate_doubled(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self.upscaler.supersample(clip, 2)  # type: ignore[return-value]

    @_add_props
    def _generate_upscale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self.downscaler.scale(  # type: ignore[return-value]
            clip,
            self.clipy.width, self.clipy.height,
            **self.descale_args.kwargs(clip)
        )

    @cached_property
    def descale(self) -> ConstantFormatVideoNode:
        return self._generate_descale(self.clipy)

    @cached_property
    def rescale(self) -> ConstantFormatVideoNode:
        return self._generate_rescale(self.descale)

    @cached_property
    def doubled(self) -> ConstantFormatVideoNode:
        return self._generate_doubled(self.descale)

    @cached_property
    def upscale(self) -> ConstantFormatVideoNode:
        """Returns the upscaled clip"""
        upscaled = join(
            self._generate_upscale(self.doubled),
            *self.chroma
        )
        return core.std.CopyFrameProps(upscaled, self.clipy, '_ChromaLocation')


class Rescale(RescaleBase):
    """
    Rescale wrapper supporting everything you need for (fractional) descaling,
    re-upscaling and masking-out details.

    Examples usage:
    Basic 720p rescale:
    ```py
    from vsscale import Rescale
    from vskernels import Bilinear

    rs = Rescale(clip, 720, Bilinear)
    final = rs.upscale
    ```

    Adding aa and dehalo on doubled clip:
    ```py
    from vsaa import based_aa
    from vsdehalo import fine_dehalo

    aa = based_aa(rs.doubled, supersampler=False)
    dehalo = fine_dehalo(aa, ...)

    rs.doubled = dehalo
    ```

    Loading line_mask and credit_mask:
    ```py
    from vsmasktools import diff_creditless_oped
    from vsexprtools import ExprOp

    rs.default_line_mask()

    oped_credit_mask = diff_creditless_oped(...)
    credit_mask = rs.default_credit_mask(thr=0.209, ranges=(200, 300), postfilter=4)
    rs.credit_mask = ExprOp.ADD.combine(oped_credit_mask, credit_mask)
    ```

    Fractional rescale:
    ```py
    from vsscale import Rescale
    from vskernels import Bilinear

    # Forcing the height to a float will ensure a fractional descale
    rs = Rescale(clip, 800.0, Bilinear)
    >>> rs.descale_args
    ScalingArgs(
        width=1424, height=800, src_width=1422.2222222222222, src_height=800.0,
        src_top=0.0, src_left=0.8888888888889142, mode='hw'
    )

    # while doing this will not
    rs = Rescale(clip, 800, Bilinear)
    >>> rs.descale_args
    ScalingArgs(width=1422, height=800, src_width=1422, src_height=800, src_top=0, src_left=0, mode='hw')
    ```

    Cropping is also supported:
    ```py
    from vsscale import Rescale
    from vskernels import Bilinear

    # Descaling while cropping the letterboxes at the top and bottom
    rs = Rescale(clip, 874, Bilinear, crop=(0, 0, 202, 202))
    >>> rs.descale_args
    ScalingArgs(
        width=1554, height=548, src_width=1554.0, src_height=547.0592592592592,
        src_top=0.4703703703703752, src_left=0.0, mode='hw'
    )

    # Same thing but ensuring the width is fractional descaled
    rs = Rescale(clip, 874.0, Bilinear, crop=(0, 0, 202, 202))
    >>> rs.descale_args
    ScalingArgs(
        width=1554, height=548, src_width=1553.7777777777778, src_height=547.0592592592592,
        src_top=0.4703703703703752, src_left=0.11111111111108585, mode='hw'
    )
    ```
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        height: int | float,
        kernel: KernelT,
        upscaler: ScalerT = ArtCNN,
        downscaler: ScalerT = Hermite(linear=True),
        width: int | float | None = None,
        base_height: int | None = None,
        base_width: int | None = None,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = CropRel(),
        shift: tuple[TopShift, LeftShift] = (0, 0),
        field_based: FieldBasedT | bool | None = None,
        border_handling: int | BorderHandling = BorderHandling.MIRROR
    ) -> None:
        """Initialize the rescaling process.

        :param clip:                Clip to be rescaled
        :param height:              Height to be descaled to.
                                    Forcing the value to float will ensure a fractional descale
        :param kernel:              Kernel used for descaling
        :param upscaler:            Scaler that supports doubling, defaults to ArtCNN
        :param downscaler:          Scaler used for downscaling the upscaled clip back to input res,
                                    defaults to Hermite(linear=True)
        :param width:               Width to be descaled to. If None, automatically calculated from the height
        :param base_height:         Integer height at which the clip will be contained.
                                    If None, automatically calculated from the height
        :param base_width:          Integer width at which the clip will be contained.
                                    If None, automatically calculated from the width
        :param crop:                Cropping values to apply before descale.
                                    The ratio descale height / source height will be preserved even after descale.
                                    The cropped area is restored when calling the `upscale` property.
        :param shift:               Shifts to apply during descale and upscale, defaults to (0, 0)
        :param field_based:         Parameter specifying the source is a cross-converted/interlaced upscaled content
        :param border_handling:     Adjust the way the clip is padded internally during the scaling process.
                                    Accepted values are:
                                        0: Assume the image was resized with mirror padding.
                                        1: Assume the image was resized with zero padding.
                                        2: Assume the image was resized with extend padding,
                                           where the outermost row was extended infinitely far.
                                    Defaults to 0
        """
        self._line_mask: ConstantFormatVideoNode | None = None
        self._credit_mask: ConstantFormatVideoNode | None = None
        self._ignore_mask: ConstantFormatVideoNode | None = None
        self._crop = crop
        self._pre = clip

        self.descale_args = ScalingArgs.from_args(
            clip, height, width, base_height, base_width, shift[0], shift[1], crop, mode='hw'
        )

        super().__init__(clip, kernel, upscaler, downscaler, field_based, border_handling)

        if self._crop > (0, 0, 0, 0):
            self.clipy = self.clipy.std.Crop(*self._crop)

    def _generate_descale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        if not self._ignore_mask:
            return super()._generate_descale(clip)

        @self._add_props
        @self._apply_field_based
        def _generate_descale_ignore_mask(self: Rescale, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            assert self._ignore_mask

            self.descale_args.mode = 'h'

            descale_h = self.kernel.descale(
                clip,
                None, self.descale_args.height,
                **self.descale_args.kwargs(),
                border_handling=self.border_handling,
                ignore_mask=self._ignore_mask
            )

            self.descale_args.mode = 'w'

            descale_w = self.kernel.descale(
                descale_h,
                self.descale_args.width, None,
                **self.descale_args.kwargs(),
                border_handling=self.border_handling,
                ignore_mask=Point.scale(self._ignore_mask, height=descale_h.height)
            )

            self.descale_args.mode = 'hw'

            return descale_w

        return _generate_descale_ignore_mask(self, clip)

    def _generate_upscale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        upscale = super()._generate_upscale(clip)

        merged_mask = norm_expr([self.line_mask, self.credit_mask], "x y - 0 yrange_max clamp", func=self.__class__)

        upscale = core.std.CopyFrameProps(core.std.MaskedMerge(self.clipy, upscale, merged_mask), upscale)

        if self._crop > (0, 0, 0, 0):
            pre_y = get_y(self._pre)
            black = pre_y.std.BlankClip()
            mask = norm_expr(
                black, _get_region_expr(
                    black, *self._crop, replace=f'{get_peak_value(black, False, ColorRange.FULL)} x'
                ), func=self.__class__
            )

            upscale = core.std.MaskedMerge(upscale.std.AddBorders(*self._crop), pre_y, mask)

        return upscale

    @property
    def line_mask(self) -> ConstantFormatVideoNode:
        lm = self._line_mask or core.std.BlankClip(
            self.clipy,
            color=get_peak_value(self.clipy, False, ColorRange.FULL)
        )

        if self.border_handling:
            px = (self.kernel.kernel_radius, ) * 4
            lm = norm_expr(
                lm, _get_region_expr(lm, *px, replace=f'{get_peak_value(lm, False, ColorRange.FULL)} x'),
                func=self.__class__
            )

        self._line_mask = lm

        return self._line_mask

    @line_mask.setter
    def line_mask(self, mask: vs.VideoNode | None) -> None:
        if mask is not None:
            self._line_mask = limiter(
                depth(mask, self.clipy, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL)
            )
        else:
            self._line_mask = None

    @line_mask.deleter
    def line_mask(self) -> None:
        self._line_mask = None

    @property
    def credit_mask(self) -> ConstantFormatVideoNode:
        if self._credit_mask:
            return self._credit_mask

        self.credit_mask = core.std.BlankClip(self.clipy)

        return self.credit_mask

    @credit_mask.setter
    def credit_mask(self, mask: vs.VideoNode | None) -> None:
        if mask is not None:
            self._credit_mask = limiter(
                depth(mask, self.clipy, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL)
            )
        else:
            self._credit_mask = None

    @credit_mask.deleter
    def credit_mask(self) -> None:
        self._credit_mask = None

    @property
    def ignore_mask(self) -> ConstantFormatVideoNode:
        if self._ignore_mask:
            return self._ignore_mask
        self.ignore_mask = core.std.BlankClip(self.clipy, format=vs.GRAY8)
        return self.ignore_mask

    @ignore_mask.setter
    def ignore_mask(self, mask: vs.VideoNode | None) -> None:
        if mask is not None:
            self._ignore_mask = depth(
                mask, 8, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL
            )
        else:
            self._ignore_mask = None

    @ignore_mask.deleter
    def ignore_mask(self) -> None:
        self._ignore_mask = None

    def default_line_mask(
        self, clip: vs.VideoNode | None = None, scaler: ScalerT = Bilinear, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Load a default Kirsch line mask in the class instance. Additionnaly, it is returned.

        :param clip:    Reference clip, defaults to doubled clip if None.
        :param scaler:  Scaled used for matching the source clip format, defaults to Bilinear
        :return:        Generated mask.
        """
        scaler = Scaler.ensure_obj(scaler)
        scale_kwargs = scaler.kwargs if clip else self.descale_args.kwargs(self.doubled) | scaler.kwargs

        clip = clip if clip else self.doubled

        line_mask = KirschTCanny.edgemask(clip, **kwargs).std.Maximum().std.Minimum()
        line_mask = scaler.scale(  # type: ignore[assignment]
            line_mask, self.clipy.width, self.clipy.height, format=self.clipy.format, **scale_kwargs
        )

        self.line_mask = line_mask

        return self.line_mask

    def default_credit_mask(
        self, rescale: vs.VideoNode | None = None, src: vs.VideoNode | None = None,
        thr: float = 0.216, expand: int = 4,
        ranges: FrameRangeN | FrameRangesN | None = None, exclusive: bool = False,
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Load a credit mask by making a difference mask between src and rescaled clips

        :param rescale:     Rescaled clip, defaults to rescaled instance clip
        :param src:         Source clip, defaults to source instance clip
        :param thr:         Threshold of the amplification expr, defaults to 0.216
        :param expand:      Additional expand radius applied to the mask, defaults to 4
        :param ranges:      If specified, ranges to apply the credit clip to
        :param exclusive:   Use exclusive ranges (Default: False)
        :return:            Generated mask
        """
        if not src:
            src = self.clipy
        if not rescale:
            rescale = self.rescale

        src, rescale = get_y(src), get_y(rescale)

        credit_mask = based_diff_mask(
            src, rescale, thr=thr, expand=expand, func=self.default_credit_mask, **kwargs
        )

        if ranges is not None:
            credit_mask = replace_ranges(credit_mask.std.BlankClip(keep=True), credit_mask, ranges, exclusive)

        self.credit_mask = credit_mask

        return self.credit_mask
