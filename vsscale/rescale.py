from __future__ import annotations

import contextlib
from functools import cached_property, wraps
from typing import Any, Callable, Sequence, TypeVar

from vsexprtools import ExprToken, norm_expr
from vskernels import Bilinear, BorderHandling, Hermite, Kernel, KernelLike, Scaler, ScalerLike
from vskernels.types import LeftShift, TopShift
from vsmasktools import KirschTCanny, based_diff_mask, region_rel_mask
from vsrgtools import BlurMatrix
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    ConvMode,
    DitherType,
    FieldBased,
    FieldBasedT,
    FrameRangeN,
    FrameRangesN,
    VideoNodeT,
    check_variable,
    core,
    depth,
    get_peak_value,
    get_y,
    join,
    normalize_ranges,
    replace_ranges,
    scale_mask,
    split,
    vs,
    vs_object,
)

from .helpers import BottomCrop, CropRel, LeftCrop, RightCrop, ScalingArgs, TopCrop
from .onnx import ArtCNN

__all__ = [
    "Rescale",
    "RescaleBase",
]

_RescaleT = TypeVar("_RescaleT", bound="RescaleBase")


class RescaleBase(vs_object):
    """Base class for Rescale wrapper"""

    descale_args: ScalingArgs
    """
    Descale arguments. See [ScalingArgs][vsscale.ScalingArgs]
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        kernel: KernelLike,
        upscaler: ScalerLike = ArtCNN,
        downscaler: ScalerLike = Hermite(linear=True),
        field_based: FieldBasedT | bool | None = None,
        border_handling: int | BorderHandling = BorderHandling.MIRROR,
        **kwargs: Any,
    ) -> None:
        assert check_variable(clip, self.__class__)

        self._clipy, *chroma = split(clip)
        self._chroma = chroma

        self._kernel = Kernel.ensure_obj(kernel)
        self._upscaler = Scaler.ensure_obj(upscaler)

        self._downscaler = Scaler.ensure_obj(downscaler)

        self._field_based = FieldBased.from_param(field_based)

        self._border_handling = BorderHandling(int(border_handling))

        self.__add_props = kwargs.get("_add_props")

    def __delattr__(self, name: str) -> None:
        def _delattr(attr: str) -> None:
            with contextlib.suppress(AttributeError):
                delattr(self, attr)

        match name:
            case "descale":
                _delattr("rescale")
                _delattr("doubled")
            case "doubled":
                _delattr("upscale")
            case _:
                pass

        with contextlib.suppress(AttributeError):
            super().__delattr__(name)

    @staticmethod
    def _apply_field_based(
        function: Callable[[_RescaleT, VideoNodeT], VideoNodeT],
    ) -> Callable[[_RescaleT, VideoNodeT], VideoNodeT]:
        @wraps(function)
        def wrap(self: _RescaleT, clip: VideoNodeT) -> VideoNodeT:
            if self._field_based:
                clip = self._field_based.apply(clip)
                clip = function(self, clip)
                return FieldBased.PROGRESSIVE.apply(clip)
            else:
                return function(self, clip)

        return wrap

    @staticmethod
    def _add_props(
        function: Callable[[_RescaleT, VideoNodeT], VideoNodeT],
    ) -> Callable[[_RescaleT, VideoNodeT], VideoNodeT]:
        @wraps(function)
        def wrap(self: _RescaleT, clip: VideoNodeT) -> VideoNodeT:
            if not self.__add_props:
                return function(self, clip)

            w, h = (
                f"{int(d)}" if d.is_integer() else f"{d:.2f}"
                for d in [self.descale_args.src_width, self.descale_args.src_height]
            )
            return core.std.SetFrameProp(
                function(self, clip),
                "Rescale" + function.__name__.split("_")[-1].capitalize() + "From",
                data=f"{self._kernel.__class__.__name__} - {w} x {h}",
            )

        return wrap

    @_add_props
    @_apply_field_based
    def _generate_descale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self._kernel.descale(
            clip,
            self.descale_args.width,
            self.descale_args.height,
            **self.descale_args.kwargs(),
            border_handling=self._border_handling,
        )

    @_add_props
    @_apply_field_based
    def _generate_rescale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self._kernel.scale(  # type: ignore[return-value]
            clip,
            self._clipy.width,
            self._clipy.height,
            **self.descale_args.kwargs(),
            border_handling=self._border_handling,
        )

    @_add_props
    def _generate_doubled(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self._upscaler.supersample(clip, 2)

    @_add_props
    def _generate_upscale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return self._downscaler.scale(  # type: ignore[return-value]
            clip, self._clipy.width, self._clipy.height, **self.descale_args.kwargs(clip)
        )

    @cached_property
    def descale(self) -> ConstantFormatVideoNode:
        """
        Gets the descaled clip.
        """
        return self._generate_descale(self._clipy)

    @cached_property
    def rescale(self) -> ConstantFormatVideoNode:
        """
        Gets the rescaled clip.
        """
        return self._generate_rescale(self.descale)

    @cached_property
    def doubled(self) -> ConstantFormatVideoNode:
        """
        Gets the doubled clip.
        """
        return self._generate_doubled(self.descale)

    @cached_property
    def upscale(self) -> ConstantFormatVideoNode:
        """
        Returns the upscaled clip
        """
        return core.std.CopyFrameProps(
            join(self._generate_upscale(self.doubled), *self._chroma), self._clipy, "_ChromaLocation"
        )

    def __vs_del__(self, core_id: int) -> None:
        del self._clipy
        del self._chroma
        del self.descale
        del self.rescale
        del self.doubled
        del self.upscale


class Rescale(RescaleBase):
    """
    Rescale wrapper supporting everything you need for (fractional) descaling,
    re-upscaling and masking-out details.

    Examples usage:

    - Basic 720p rescale:

        ```py
        from vsscale import Rescale
        from vskernels import Bilinear

        rs = Rescale(clip, 720, Bilinear)
        final = rs.upscale
        ```

    - Adding aa and dehalo on doubled clip:

        ``` py
        from vsaa import based_aa
        from vsdehalo import fine_dehalo

        aa = based_aa(rs.doubled, supersampler=False)
        dehalo = fine_dehalo(aa, ...)

        rs.doubled = dehalo
        ```

    - Loading line_mask and credit_mask:

        ``` py
        from vsmasktools import diff_creditless_oped
        from vsexprtools import ExprOp

        rs.default_line_mask()

        oped_credit_mask = diff_creditless_oped(...)
        credit_mask = rs.default_credit_mask(thr=0.209, ranges=(200, 300), postfilter=4)
        rs.credit_mask = ExprOp.ADD.combine(oped_credit_mask, credit_mask)
        ```

    - Fractional rescale:

        ``` py
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

    - Cropping is also supported:

        ``` py
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
        kernel: KernelLike,
        upscaler: ScalerLike = ArtCNN,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | float | None = None,
        base_height: int | None = None,
        base_width: int | None = None,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = CropRel(),
        shift: tuple[TopShift, LeftShift] = (0, 0),
        field_based: FieldBasedT | bool | None = None,
        border_handling: int | BorderHandling = BorderHandling.MIRROR,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the rescaling process.

        Args:
            clip: Clip to be rescaled.
            height: Height to be descaled to. If passed as a float, a fractional descale is performed.
            kernel: Kernel used for descaling.
            upscaler: Scaler that supports doubling. Defaults to ``ArtCNN``.
            downscaler: Scaler used to downscale the upscaled clip back to input resolution. Defaults to
                ``Hermite(linear=True)``.
            width: Width to be descaled to. If ``None``, it is automatically calculated from the height.
            base_height: Integer height to contain the clip within. If ``None``, it is automatically calculated from the
                height.
            base_width: Integer width to contain the clip within. If ``None``, it is automatically calculated from the
                width.
            crop: Cropping values to apply before descale. The ratio ``descale_height / source_height`` is preserved
                even after descale. The cropped area is restored when calling the ``upscale`` property.
            shift: Pixel shifts to apply during descale and upscale. Defaults to ``(0, 0)``.
            field_based: Whether the input is cross-converted or interlaced content.
            border_handling: Adjusts how the clip is padded internally during the scaling process.
                Accepted values are:

                   - ``0`` (MIRROR): Assume the image was resized with mirror padding.
                   - ``1`` (ZERO):   Assume the image was resized with zero padding.
                   - ``2`` (EXTEND): Assume the image was resized with extend padding,
                     where the outermost row was extended infinitely far.

                Defaults to ``0``.
        """
        self._line_mask: ConstantFormatVideoNode | None = None
        self._credit_mask: ConstantFormatVideoNode | None = None
        self._ignore_mask: ConstantFormatVideoNode | None = None
        self._crop = crop
        self._pre = clip

        self.descale_args = ScalingArgs.from_args(
            clip, height, width, base_height, base_width, shift[0], shift[1], crop, mode="hw"
        )

        super().__init__(clip, kernel, upscaler, downscaler, field_based, border_handling, **kwargs)

        if self._crop > (0, 0, 0, 0):
            self._clipy = self._clipy.std.Crop(*self._crop)

    def _generate_descale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        if not self._ignore_mask:
            return super()._generate_descale(clip)

        @self._add_props
        @self._apply_field_based
        def _generate_descale_ignore_mask(self: Rescale, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            assert self._ignore_mask

            self.descale_args.mode = "h"

            descale_h = self._kernel.descale(
                clip,
                None,
                self.descale_args.height,
                **self.descale_args.kwargs(),
                border_handling=self._border_handling,
                ignore_mask=self._ignore_mask,
            )

            self.descale_args.mode = "w"

            descale_w = self._kernel.descale(
                descale_h,
                self.descale_args.width,
                None,
                **self.descale_args.kwargs(),
                border_handling=self._border_handling,
                ignore_mask=core.resize.Point(self._ignore_mask, height=descale_h.height),
            )

            self.descale_args.mode = "hw"

            return descale_w

        return _generate_descale_ignore_mask(self, clip)

    def _generate_upscale(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        upscale = super()._generate_upscale(clip)

        merged_mask = norm_expr([self.line_mask, self.credit_mask], "x y - 0 yrange_max clamp", func=self.__class__)

        upscale = core.std.CopyFrameProps(core.std.MaskedMerge(self._clipy, upscale, merged_mask), upscale)

        if self._crop > (0, 0, 0, 0):
            pre_y = get_y(self._pre)

            mask = region_rel_mask(
                pre_y.std.BlankClip(length=1, keep=True),
                *self._crop,
                replace_in=ExprToken.RangeMax,
                replace_out=0,
                func=self.__class__,
            )

            upscale = core.std.MaskedMerge(upscale.std.AddBorders(*self._crop), pre_y, mask)

        return upscale

    @property
    def line_mask(self) -> ConstantFormatVideoNode:
        """Gets the lineart mask to be applied on the upscaled clip."""
        lm = self._line_mask or core.std.BlankClip(
            self._clipy, color=get_peak_value(self._clipy, False, ColorRange.FULL), keep=True
        )

        if self._border_handling:
            lm = region_rel_mask(
                lm,
                self._kernel.kernel_radius,
                self._kernel.kernel_radius,
                self._kernel.kernel_radius,
                self._kernel.kernel_radius,
                replace_out=ExprToken.RangeMax,
                func=self.__class__,
            )

        self._line_mask = lm

        return self._line_mask

    @line_mask.setter
    def line_mask(self, mask: vs.VideoNode | None) -> None:
        if mask is not None:
            self._line_mask = depth(
                mask, self._clipy, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL
            )
        else:
            self._line_mask = None

    @line_mask.deleter
    def line_mask(self) -> None:
        self._line_mask = None

    @property
    def credit_mask(self) -> ConstantFormatVideoNode:
        """Gets the credit mask to be applied on the upscaled clip."""
        if self._credit_mask:
            return self._credit_mask

        self.credit_mask = core.std.BlankClip(self._clipy, keep=True)

        return self.credit_mask

    @credit_mask.setter
    def credit_mask(self, mask: vs.VideoNode | None) -> None:
        if mask is not None:
            self._credit_mask = depth(
                mask, self._clipy, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL
            )
        else:
            self._credit_mask = None

    @credit_mask.deleter
    def credit_mask(self) -> None:
        self._credit_mask = None

    @property
    def ignore_mask(self) -> ConstantFormatVideoNode:
        """Gets the ignore mask to be applied on the descaled clip."""
        if self._ignore_mask:
            return self._ignore_mask
        self.ignore_mask = core.std.BlankClip(self._clipy, format=vs.GRAY8, keep=True)
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
        self, clip: vs.VideoNode | None = None, scaler: ScalerLike = Bilinear, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Load a default Kirsch line mask in the class instance. Additionally, it is returned.

        Args:
            clip: Reference clip, defaults to doubled clip if None.
            scaler: Scaled used for matching the source clip format, defaults to Bilinear.

        Returns:
            Generated mask.
        """
        scaler = Scaler.ensure_obj(scaler)
        scale_kwargs = scaler.kwargs if clip else self.descale_args.kwargs(self.doubled) | scaler.kwargs

        clip = clip if clip else self.doubled

        line_mask = KirschTCanny.edgemask(clip, **kwargs).std.Maximum().std.Minimum()
        line_mask = scaler.scale(
            line_mask, self._clipy.width, self._clipy.height, format=self._clipy.format, **scale_kwargs
        )

        self.line_mask = line_mask

        return self.line_mask

    def default_credit_mask(
        self,
        rescale: vs.VideoNode | None = None,
        src: vs.VideoNode | None = None,
        thr: float = 0.216,
        expand: int = 4,
        ranges: FrameRangeN | FrameRangesN | None = None,
        stabilize: bool | tuple[Sequence[float], float] = False,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Load a credit mask by making a difference mask between src and rescaled clips

        Args:
            rescale: Rescaled clip, defaults to rescaled instance clip.
            src: Source clip, defaults to source instance clip.
            thr: Threshold of the amplification expr, defaults to 0.216.
            expand: Additional expand radius applied to the mask, defaults to 4.
            ranges: If specified, ranges to apply the credit clip to.
            stabilize: Try to stabilize the mask by applying a temporal convolution and then binarized by a threshold.
                Only works when there are ranges specified.

        Returns:
            Generated mask.
        """
        if not src:
            src = self._clipy
        if not rescale:
            rescale = self.rescale

        src, rescale = get_y(src), get_y(rescale)

        credit_mask = based_diff_mask(src, rescale, thr=thr, expand=expand, func=self.default_credit_mask, **kwargs)

        if ranges is not None:
            if stabilize:
                ranges = normalize_ranges(credit_mask, ranges)

                scprev = [x for s, e in ranges for x in (s, e + int(not replace_ranges.exclusive))]
                scnext = [x - 1 for x in scprev]

                def set_credit_ranges(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                    f = f.copy()
                    f.props.update(_SceneChangePrev=False, _SceneChangeNext=False)

                    if n in scprev:
                        f.props["_SceneChangePrev"] = True
                        return f

                    if n in scnext:
                        f.props["_SceneChangeNext"] = True

                    return f

                credit_mask = core.std.ModifyFrame(credit_mask, credit_mask, set_credit_ranges)

                if isinstance(stabilize, tuple):
                    weights, thr = stabilize
                else:
                    weights, thr = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 0.24

                credit_mask = BlurMatrix.custom(weights, ConvMode.T)(credit_mask, scenechange=True).std.Binarize(
                    scale_mask(thr, 32, credit_mask)
                )

            credit_mask = replace_ranges(credit_mask.std.BlankClip(keep=True), credit_mask, ranges)

        self.credit_mask = credit_mask

        return self.credit_mask

    def __vs_del__(self, core_id: int) -> None:
        del self._line_mask
        del self._credit_mask
        del self._ignore_mask
        del self._pre

        super().__vs_del__(core_id)
