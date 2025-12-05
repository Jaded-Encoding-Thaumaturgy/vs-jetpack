from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

from jetpytools import cachedproperty

from vsexprtools import ExprToken, norm_expr
from vskernels import (
    Bilinear,
    BorderHandling,
    ComplexKernel,
    ComplexKernelLike,
    Hermite,
    SampleGridModel,
    Scaler,
    ScalerLike,
)
from vskernels.types import LeftShift, TopShift
from vsmasktools import Kirsch, based_diff_mask, region_rel_mask, stabilize_mask
from vstools import (
    ColorRange,
    DitherType,
    FieldBased,
    FieldBasedLike,
    FrameRangeN,
    FrameRangesN,
    VSObjectABC,
    core,
    depth,
    get_peak_value,
    get_y,
    join,
    replace_ranges,
    split,
    vs,
)

from .helpers import BottomCrop, CropRel, LeftCrop, RightCrop, ScalingArgs, TopCrop
from .onnx import ArtCNN

__all__ = [
    "Rescale",
    "RescaleBase",
]


class RescaleBase(VSObjectABC):
    """Base class for Rescale wrapper"""

    descale_args: ScalingArgs
    """
    Descale arguments. See [ScalingArgs][vsscale.ScalingArgs]
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        /,
        kernel: ComplexKernelLike,
        upscaler: ScalerLike = ArtCNN,
        downscaler: ScalerLike = Hermite(linear=True),
        field_based: FieldBasedLike | bool | None = None,
        border_handling: int = BorderHandling.MIRROR,
        **kwargs: Any,
    ) -> None:
        self._clipy, *chroma = split(clip)
        self._chroma = chroma

        self._kernel = ComplexKernel.ensure_obj(kernel)
        self._upscaler = Scaler.ensure_obj(upscaler)

        self._downscaler = Scaler.ensure_obj(downscaler)

        self._field_based = FieldBased.from_param_with_fallback(field_based)

        self._border_handling = BorderHandling.from_param(border_handling)

        self.__add_props = kwargs.get("_add_props")

    @staticmethod
    def _apply_field_based[RescaleT: RescaleBase](
        function: Callable[[RescaleT, vs.VideoNode], vs.VideoNode],
    ) -> Callable[[RescaleT, vs.VideoNode], vs.VideoNode]:
        @wraps(function)
        def wrap(self: RescaleT, clip: vs.VideoNode) -> vs.VideoNode:
            if self._field_based:
                clip = self._field_based.apply(clip)
                clip = function(self, clip)
                return FieldBased.PROGRESSIVE.apply(clip)
            else:
                return function(self, clip)

        return wrap

    @staticmethod
    def _add_props[RescaleT: RescaleBase](
        function: Callable[[RescaleT, vs.VideoNode], vs.VideoNode],
    ) -> Callable[[RescaleT, vs.VideoNode], vs.VideoNode]:
        @wraps(function)
        def wrap(self: RescaleT, clip: vs.VideoNode) -> vs.VideoNode:
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
    def _generate_descale(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._kernel.descale(
            clip,
            self.descale_args.width,
            self.descale_args.height,
            **self.descale_args.kwargs(),
            border_handling=self._border_handling,
        )

    @_add_props
    @_apply_field_based
    def _generate_rescale(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._kernel.scale(
            clip,
            self._clipy.width,
            self._clipy.height,
            **self.descale_args.kwargs(),
            border_handling=self._border_handling,
        )

    @_add_props
    def _generate_doubled(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._upscaler.supersample(clip, 2)

    @_add_props
    def _generate_upscale(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self._downscaler.scale(clip, self._clipy.width, self._clipy.height, **self.descale_args.kwargs(clip))

    descale = cachedproperty[vs.VideoNode, vs.VideoNode](
        lambda self: self._generate_descale(self._clipy),
        lambda self, value: cachedproperty.update_cache(self, "descale", value),
        lambda self: cachedproperty.clear_cache(self, ["descale", "rescale", "doubled", "upscale"]),
    )
    """
    Gets the descaled clip.
    """

    rescale = cachedproperty[vs.VideoNode, vs.VideoNode](
        lambda self: self._generate_rescale(self.descale),
        lambda self, value: cachedproperty.update_cache(self, "rescale", value),
        lambda self: cachedproperty.clear_cache(self, "rescale"),
    )
    """
    Gets the rescaled clip.
    """

    doubled = cachedproperty[vs.VideoNode, vs.VideoNode](
        lambda self: self._generate_doubled(self.descale),
        lambda self, value: cachedproperty.update_cache(self, "doubled", value),
        lambda self: cachedproperty.clear_cache(self, ["doubled", "upscale"]),
    )
    """
    Gets the doubled clip.
    """

    upscale = cachedproperty[vs.VideoNode, vs.VideoNode](
        lambda self: core.std.CopyFrameProps(
            join([self._generate_upscale(self.doubled), *self._chroma]), self._clipy, "_ChromaLocation"
        ),
        lambda self, value: cachedproperty.update_cache(self, "upscale", value),
        lambda self: cachedproperty.clear_cache(self, "upscale"),
    )
    """
    Returns the upscaled clip
    """


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
        height: int | float,  # noqa: PYI041
        kernel: ComplexKernelLike,
        upscaler: ScalerLike = ArtCNN,
        downscaler: ScalerLike = Hermite(linear=True),
        width: int | float | None = None,  # noqa: PYI041
        base_height: int | None = None,
        base_width: int | None = None,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        field_based: FieldBasedLike | bool | None = None,
        border_handling: int = BorderHandling.MIRROR,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
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
            sample_grid_model: Model used to align sampling grid.
        """
        self._line_mask: vs.VideoNode | None = None
        self._credit_mask: vs.VideoNode | None = None
        self._ignore_mask: vs.VideoNode | None = None
        self._crop = crop or CropRel()
        self._pre = clip

        self.descale_args = ScalingArgs.from_args(
            clip, height, width, base_height, base_width, shift[0], shift[1], crop, sample_grid_model, mode="hw"
        )

        super().__init__(clip, kernel, upscaler, downscaler, field_based, border_handling, **kwargs)

        if self._crop > (0, 0, 0, 0):
            self._clipy = self._clipy.std.Crop(*self._crop)

    def _generate_descale(self, clip: vs.VideoNode) -> vs.VideoNode:
        if not self._ignore_mask:
            return super()._generate_descale(clip)

        @self._add_props
        @self._apply_field_based
        def _generate_descale_ignore_mask(self: Rescale, clip: vs.VideoNode) -> vs.VideoNode:
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

    def _generate_upscale(self, clip: vs.VideoNode) -> vs.VideoNode:
        upscale = super()._generate_upscale(clip)

        merged_mask = norm_expr([self.line_mask, self.credit_mask], "x y - 0 mask_max clamp", func=self.__class__)

        upscale = core.std.CopyFrameProps(core.std.MaskedMerge(self._clipy, upscale, merged_mask), upscale)

        if self._crop > (0, 0, 0, 0):
            pre_y = get_y(self._pre)

            mask = region_rel_mask(
                pre_y.std.BlankClip(length=1, keep=True),
                *self._crop,
                replace_in=0,
                replace_out=ExprToken.RangeMax,
                func=self.__class__,
            )

            upscale = core.std.MaskedMerge(upscale.std.AddBorders(*self._crop), pre_y, mask)

        return upscale

    @property
    def line_mask(self) -> vs.VideoNode:
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
    def credit_mask(self) -> vs.VideoNode:
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
    def ignore_mask(self) -> vs.VideoNode:
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
    ) -> vs.VideoNode:
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

        line_mask = Kirsch.edgemask(clip, **kwargs).std.Maximum().std.Minimum()
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
        stabilize: bool = True,
        scenechanges: Iterable[int] | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
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
            scenechanges: Explicit list of scenechange frames for stabilizing the mask.
            **kwargs: Additional keyword arguments for [based_diff_mask][vsmasktools.based_diff_mask]

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
                credit_mask = stabilize_mask(credit_mask, 3, ranges, scenechanges, func=self.default_credit_mask)

            credit_mask = replace_ranges(credit_mask.std.BlankClip(keep=True), credit_mask, ranges)

        self.credit_mask = credit_mask

        return self.credit_mask
