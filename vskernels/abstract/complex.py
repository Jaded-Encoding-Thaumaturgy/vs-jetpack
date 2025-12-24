"""
This module defines the abstract classes for scaling, descaling and resampling operations.
"""

from __future__ import annotations

from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

from jetpytools import (
    CustomIndexError,
    CustomNotImplementedError,
    CustomValueError,
    FuncExcept,
    fallback,
    normalize_seq,
)

from vstools import (
    ChromaLocation,
    Dar,
    FieldBased,
    FieldBasedLike,
    Resolution,
    Sar,
    depth,
    expect_bits,
    get_video_format,
    split,
    vs,
)

from ..types import (
    BorderHandling,
    BotFieldLeftShift,
    BotFieldTopShift,
    Center,
    FieldShift,
    LeftShift,
    SampleGridModel,
    Slope,
    TopFieldLeftShift,
    TopFieldTopShift,
    TopShift,
)
from .base import Descaler, Kernel, Resampler, Scaler

__all__ = [
    "ComplexDescaler",
    "ComplexDescalerLike",
    "ComplexKernel",
    "ComplexKernelLike",
    "ComplexScaler",
    "ComplexScalerLike",
    "KeepArScaler",
    "LinearDescaler",
    "LinearScaler",
]


def _check_dynamic_keeparscaler_params(
    border_handling: int,
    sample_grid_model: int,
    sar: Any,
    dar: Any,
    dar_in: Any,
    keep_ar: Any,
    func: FuncExcept,
) -> bool:
    exceptions = list[CustomNotImplementedError]()

    if border_handling != BorderHandling.MIRROR:
        exceptions.append(
            CustomNotImplementedError(
                'When passing a dynamic size clip, "border_handling" must be MIRROR', func, border_handling
            )
        )
    if sample_grid_model != SampleGridModel.MATCH_EDGES:
        exceptions.append(
            CustomNotImplementedError(
                'When passing a dynamic size clip, "sample_grid_model" must be MATCH_EDGES', func, sample_grid_model
            )
        )
    if any(p is not None for p in [sar, dar, dar_in, keep_ar]):
        exceptions.append(
            CustomNotImplementedError(
                'When passing a dynamic size clip, "sar", "dar", "dar_in" and "keep_ar" must all be None',
                func,
                (sar, dar, dar_in, keep_ar),
            )
        )

    if exceptions:
        raise ExceptionGroup("Multiple exceptions occurred!", exceptions)

    return True


@overload
def _descale_shift_norm(
    shift: tuple[TopShift, LeftShift] | FieldShift, assume_progressive: Literal[True], func: FuncExcept | None = None
) -> tuple[TopShift, LeftShift]: ...


@overload
def _descale_shift_norm(
    shift: tuple[TopShift, LeftShift] | FieldShift, assume_progressive: Literal[False], func: FuncExcept | None = None
) -> tuple[tuple[TopFieldTopShift, BotFieldTopShift], tuple[TopFieldLeftShift, BotFieldLeftShift]]: ...


def _descale_shift_norm(
    shift: tuple[TopShift, LeftShift] | FieldShift, assume_progressive: bool = True, func: FuncExcept | None = None
) -> Any:
    if assume_progressive:
        if any(isinstance(sh, tuple) for sh in shift):
            raise CustomValueError("You can't descale per-field when the input is progressive!", func, shift)
    else:
        shift_y, shift_x = tuple[tuple[float, float], ...](sh if isinstance(sh, tuple) else (sh, sh) for sh in shift)
        shift = shift_y, shift_x

    return shift


def _linearize(
    obj: Scaler | Descaler,
    clip: vs.VideoNode,
    linear: bool | None,
    sigmoid: bool | tuple[Slope, Center],
    op_partial: partial[vs.VideoNode],
    func: FuncExcept,
    **kwargs: Any,
) -> vs.VideoNode:
    if linear is False and sigmoid is not False:
        raise CustomValueError("If sigmoid is not False, linear can't be False as well!", func, (linear, sigmoid))

    # if a _linear_scale or _linear_descale method is specified in the class,
    # use this method instead of the super().scale or super().descale method.
    # args and keywords are also forwarded.
    if hasattr(obj, f"_linear_{op_partial.func.__name__}"):
        op_partial = partial(
            getattr(obj, f"_linear_{op_partial.func.__name__}"), *op_partial.args, **op_partial.keywords
        )

    if sigmoid or linear:
        from ..util import LinearLight

        fmt = obj.kwargs.pop("format", kwargs.pop("format", None))

        llargs = dict[str, Any](clip=clip, sigmoid=sigmoid, out_fmt=fmt)

        if isinstance(obj, Resampler):
            llargs.update(resampler=obj)

        with LinearLight(**llargs) as ll:
            ll.linear = op_partial(ll.linear, **kwargs)

        return ll.out

    return op_partial(clip, **kwargs)


class LinearScaler(Scaler):
    """
    Abstract scaler class that applies linearization before scaling.

    Only affects scaling results when `linear` or `sigmoid` parameters are specified.

    Optionally, subclasses can implement `_linear_scale` to override the default behavior
    with a custom linear scaling algorithm.
    """

    if TYPE_CHECKING:

        def _linear_scale(
            self,
            clip: vs.VideoNode,
            width: int | None,
            height: int | None,
            shift: tuple[TopShift, LeftShift],
            **kwargs: Any,
        ) -> vs.VideoNode:
            """
            An optional function to be implemented by subclasses.

            If implemented, this will override the default scale behavior,
            allowing custom linear scaling logic to be applied instead of the base scaler's method.
            """
            ...

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # LinearScaler adds `linear` and `sigmoid` parameters
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale a clip to the given resolution with optional linearization.

        This method behaves like the base [Scaler.scale()][vskernels.Scaler.scale] but adds support for linear
        or sigmoid-based preprocessing and postprocessing.

        When enabled, the clip is linearized before the scaling operation and de-linearized afterward.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.
            linear: Whether to linearize the input before scaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).
            **kwargs: Additional arguments forwarded to the scale function.

        Returns:
            Scaled video clip.
        """
        return _linearize(
            self,
            clip,
            linear,
            sigmoid,
            partial(super().scale, width=width, height=height, shift=shift),
            self.scale,
            **kwargs,
        )


class LinearDescaler(Descaler):
    """
    Abstract descaler class that applies linearization before descaling.

    Only affects descaling results when `linear` or `sigmoid` parameters are specified.

    Optionally, subclasses can implement `_linear_descale` to override the default behavior
    with a custom linear descaling algorithm.
    """

    if TYPE_CHECKING:

        def _linear_descale(
            self,
            clip: vs.VideoNode,
            width: int | None,
            height: int | None,
            shift: tuple[TopShift, LeftShift],
            **kwargs: Any,
        ) -> vs.VideoNode:
            """
            An optional function to be implemented by subclasses.

            If implemented, this will override the default descale behavior,
            allowing custom linear descaling logic to be applied instead of the base descaler's method.
            """
            ...

        def _linear_rescale(
            self,
            clip: vs.VideoNode,
            width: int | None,
            height: int | None,
            shift: tuple[TopShift, LeftShift],
            **kwargs: Any,
        ) -> vs.VideoNode:
            """
            An optional function to be implemented by subclasses.

            If implemented, this will override the default rescale behavior,
            allowing custom linear rescaling logic to be applied instead of the base descaler's method.
            """
            ...

    def descale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # LinearDescaler adds `linear` and `sigmoid` parameters
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Descale a clip to the specified resolution, optionally using linear light processing.

        This method behaves like the base [Descaler.descale()][vskernels.Descaler.descale] but adds support for linear
        or sigmoid-based preprocessing and postprocessing.

        When enabled, the clip is linearized before the descaling operation and de-linearized afterward.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target descaled width (defaults to clip width if None).
            height: Target descaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during descaling.
            linear: Whether to linearize the input before descaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).

        Returns:
            The descaled video node, optionally processed in linear light.
        """
        return _linearize(
            self,
            clip,
            linear,
            sigmoid,
            partial(super().descale, width=width, height=height, shift=shift),
            self.descale,
            **kwargs,
        )

    def rescale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # LinearDescaler adds `linear` and `sigmoid` parameters
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Rescale a clip to the given resolution from a previously descaled clip,
        optionally using linear light processing.

        This method behaves like the base [Descaler.rescale()][vskernels.Descaler.rescale] but adds support for linear
        or sigmoid-based preprocessing and postprocessing.

        When enabled, the clip is linearized before the rescaling operation and de-linearized afterward.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target scaled width (defaults to clip width if None).
            height: Target scaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during rescaling.
            linear: Whether to linearize the input before rescaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).

        Returns:
            The rescaled video node, optionally processed in linear light.
        """
        return _linearize(
            self,
            clip,
            linear,
            sigmoid,
            partial(super().rescale, width=width, height=height, shift=shift),
            self.rescale,
            **kwargs,
        )


class KeepArScaler(Scaler):
    """
    Abstract scaler class that preserves or adjusts aspect ratio during scaling.

    Extends the base [Scaler][vskernels.Scaler] with logic for handling sample aspect ratio (SAR)
    and display aspect ratio (DAR) transformations.

    Provides automatic normalization and correction of scaling parameters to maintain proper image proportions
    when `keep_ar` is enabled.
    """

    def _ar_params_norm(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        sar: Sar | float | bool | None,
        dar: Dar | float | bool | None,
        dar_in: Dar | float | bool | None,
        sar_scale: Fraction | float | Literal[False],
        keep_ar: bool | None,
    ) -> tuple[float, float, float]:
        if keep_ar is not None and None not in (sar, dar, dar_in):
            raise CustomValueError(
                'If "keep_ar" is not None, then at least one of "sar", "dar", or "dar_in" must be None.'
            )

        def _norm_xar[XarT: (Dar, Sar)](
            xar: Fraction | float | bool | None, if_true: XarT, if_false: XarT, xar_t: type[XarT]
        ) -> XarT:
            if xar is None:
                xar = bool(keep_ar)

            if xar is True:
                return if_true

            if xar is False:
                return if_false

            return xar_t(xar)

        src_sar = _norm_xar(sar, Sar.from_clip(clip), Sar(1, 1), Sar)
        out_dar = _norm_xar(dar, Dar.from_res(width, height), Dar(0), Dar)
        src_dar = _norm_xar(dar_in, Dar.from_clip(clip, False), out_dar, Dar)

        if sar_scale is not False:
            src_sar *= sar_scale

        return float(src_sar), float(src_dar), float(out_dar)

    def _handle_crop_resize_kwargs(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[TopShift, LeftShift],
        sar: Sar | bool | float | None,
        dar: Dar | bool | float | None,
        dar_in: Dar | bool | float | None,
        keep_ar: bool | None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], tuple[TopShift, LeftShift], Sar | Literal[False]]:
        kwargs.setdefault("src_top", kwargs.pop("sy", shift[0]))
        kwargs.setdefault("src_left", kwargs.pop("sx", shift[1]))
        kwargs.setdefault("src_width", kwargs.pop("sw", clip.width))
        kwargs.setdefault("src_height", kwargs.pop("sh", clip.height))

        sar_scale = kwargs.pop("_sar_scale", False)

        src_res = Resolution(kwargs["src_width"], kwargs["src_height"])

        src_sar, src_dar, out_dar = self._ar_params_norm(clip, width, height, sar, dar, dar_in, sar_scale, keep_ar)
        out_sar: Sar | Literal[False] = False

        if src_sar != 1.0:
            out_dar = width / src_sar / height if src_sar > 1.0 else width / (height * src_sar)

            out_sar = Sar(1, 1)

        if src_dar != out_dar:
            if src_dar > out_dar:
                src_shift, src_window = "src_left", "src_width"

                fix_crop = src_res.width - (src_res.height * out_dar)
            else:
                src_shift, src_window = "src_top", "src_height"

                fix_crop = src_res.height - (src_res.width / out_dar)

            fix_shift = fix_crop / 2

            kwargs[src_shift] += fix_shift
            kwargs[src_window] -= fix_crop

        out_shift = (kwargs.pop("src_top"), kwargs.pop("src_left"))

        return kwargs, out_shift, out_sar

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        *,
        # KeepArScaler adds `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar`
        border_handling: int = BorderHandling.MIRROR,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None,
        dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None,
        keep_ar: bool | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale a clip to the given resolution with aspect ratio support.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.
            border_handling: Method for handling image borders during sampling.
            sample_grid_model: Model used to align sampling grid.
            sar: Sample aspect ratio to assume or convert to.
            dar: Desired display aspect ratio.
            dar_in: Input display aspect ratio, if different from clip's.
            keep_ar: Whether to adjust dimensions to preserve aspect ratio.

        Returns:
            Scaled clip, optionally aspect-corrected.
        """
        width, height = self._wh_norm(clip, width, height)

        if 0 in (clip.width, clip.height):
            _check_dynamic_keeparscaler_params(
                border_handling, sample_grid_model, sar, dar, dar_in, keep_ar, self.scale
            )
            return super().scale(clip, width, height, shift, **kwargs)

        if border_handling == sample_grid_model == 0 and sar is dar is dar_in is keep_ar is None:
            return super().scale(clip, width, height, shift, **kwargs)

        kwargs, shift, out_sar = self._handle_crop_resize_kwargs(
            clip, width, height, shift, sar, dar, dar_in, keep_ar, **kwargs
        )

        kwargs, shift = SampleGridModel.from_param(sample_grid_model, self.scale).for_dst(
            clip, width, height, shift, **kwargs
        )
        padded, shift = BorderHandling.from_param(border_handling, self.scale).prepare_clip(
            clip, width, height, shift, self.kernel_radius, **kwargs
        )

        scaled = super().scale(padded, width, height, shift, **kwargs)

        if out_sar:
            return out_sar.apply(scaled)

        return scaled


class ComplexScaler(KeepArScaler, LinearScaler):
    """
    Abstract composite scaler class with support for aspect ratio preservation, linear light processing,
    and per-plane subpixel shifting.

    Combines [KeepArScaler][vskernels.KeepArScaler] for handling sample/display aspect ratios
    and [LinearScaler][vskernels.LinearScaler] for linear and sigmoid processing.

    Additionally, it introduces support for specifying per-plane subpixel shifts.
    """

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        # ComplexScaler adds shift per planes
        shift: tuple[TopShift | list[TopShift], LeftShift | list[LeftShift]] = (0, 0),
        *,
        # `linear` and `sigmoid` from LinearScaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
        border_handling: int = BorderHandling.MIRROR,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None,
        dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None,
        keep_ar: bool | None = None,
        # ComplexScaler adds blur
        blur: float | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale a clip to the given resolution, with aspect ratio and linear light support.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling. If a tuple is provided, it is used uniformly. If a
                list is given, the shift is applied per plane.
            linear: Whether to linearize the input before descaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).
            border_handling: Method for handling image borders during sampling.
            sample_grid_model: Model used to align sampling grid.
            sar: Sample aspect ratio to assume or convert to.
            dar: Desired display aspect ratio.
            dar_in: Input display aspect ratio, if different from clip's.
            keep_ar: Whether to adjust dimensions to preserve aspect ratio.
            blur: Amount of blur to apply during scaling.

        Returns:
            Scaled clip, optionally aspect-corrected and linearized.
        """
        kwargs.update(
            linear=linear,
            sigmoid=sigmoid,
            border_handling=border_handling,
            sample_grid_model=sample_grid_model,
            sar=sar,
            dar=dar,
            dar_in=dar_in,
            keep_ar=keep_ar,
            blur=blur,
        )

        shift_top, shift_left = shift

        if isinstance(shift_top, (int, float)) and isinstance(shift_left, (int, float)):
            return super().scale(clip, width, height, (shift_top, shift_left), **kwargs)

        n_planes = clip.format.num_planes

        shift_top = normalize_seq(shift_top, n_planes)
        shift_left = normalize_seq(shift_left, n_planes)

        if n_planes == 1:
            return super().scale(clip, width, height, (shift_top[0], shift_left[0]), **kwargs)

        width, height = self._wh_norm(clip, width, height)

        format_in = clip.format
        format_out = get_video_format(fallback(kwargs.pop("format", None), self.kwargs.get("format"), clip.format))

        chromaloc = ChromaLocation.from_video(clip, func=self.scale)
        chromaloc_in = ChromaLocation(
            fallback(kwargs.pop("chromaloc_in", None), self.kwargs.get("chromaloc_in"), chromaloc)
        )
        chromaloc_out = ChromaLocation(fallback(kwargs.pop("chromaloc", None), self.kwargs.get("chromaloc"), chromaloc))

        off_left, off_top = chromaloc_in.get_offsets(format_in)
        off_left_out, off_top_out = chromaloc_out.get_offsets(format_out)

        factor_w = 1 / 2**format_in.subsampling_w
        factor_h = 1 / 2**format_in.subsampling_h

        # Offsets for format out
        offc_left = (abs(off_left) + off_left_out) * factor_w
        offc_top = (abs(off_top) + off_top_out) * factor_h

        # Offsets for scale out
        if format_out.subsampling_w:
            offc_left = ((abs(off_left) + off_left * (clip.width / width)) * factor_w) + offc_left
        if format_out.subsampling_h:
            offc_top = ((abs(off_top) + off_top * (clip.height / height)) * factor_h) + offc_top

        for i in range(1, n_planes):
            shift_left[i] += offc_left
            shift_top[i] += offc_top

        scaled_planes = list[vs.VideoNode]()

        for i, (plane, top, left) in enumerate(zip(split(clip), shift_top, shift_left)):
            if i:
                w = round(width * 1 / 2**format_out.subsampling_w)
                h = round(height * 1 / 2**format_out.subsampling_h)
            else:
                w, h = width, height

            scaled_planes.append(
                super().scale(
                    plane,
                    w,
                    h,
                    (top, left),
                    format=format_out.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0),
                    **kwargs,
                )
            )

        merged = vs.core.std.ShufflePlanes(scaled_planes, [0, 0, 0], format_out.color_family, scaled_planes[0])

        return chromaloc_out.apply(merged)


class ComplexDescaler(LinearDescaler):
    """
    Abstract descaler class with support for border handling and sampling grid alignment.

    Extends [LinearDescaler][vskernels.LinearDescaler] by introducing mechanisms to control how image borders
    are handled and how the sampling grid is aligned during descaling.
    """

    def descale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] | FieldShift = (0, 0),
        *,
        # `linear` and `sigmoid` parameters from LinearDescaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # ComplexDescaler adds border_handling, sample_grid_model, field_based,  ignore_mask and blur
        border_handling: int = BorderHandling.MIRROR,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        field_based: FieldBasedLike | None = None,
        ignore_mask: vs.VideoNode | None = None,
        blur: float | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Descale a clip to the given resolution, with image borders handling and sampling grid alignment,
        optionally using linear light processing.

        Supports both progressive and interlaced sources. When interlaced, it will separate fields,
        perform per-field descaling, and weave them back.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target descaled width (defaults to clip width if None).
            height: Target descaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) or per-field shifts.
            linear: Whether to linearize the input before descaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).
            border_handling: Method for handling image borders during sampling.
            sample_grid_model: Model used to align sampling grid.
            field_based: Field-based processing mode (interlaced or progressive).
            ignore_mask: Optional mask specifying areas to ignore during descaling.
            blur: Amount of blur to apply during scaling.
            **kwargs: Additional arguments passed to `descale_function`.

        Returns:
            The descaled video node, optionally processed in linear light.
        """
        width, height = self._wh_norm(clip, width, height)

        field_based = FieldBased.from_param_or_video(field_based, clip)

        input_clip = clip
        clip = depth(clip, 32, vs.FLOAT)

        de_base_args = (width, height // (1 + field_based.is_inter))
        kwargs.update(
            linear=linear,
            sigmoid=sigmoid,
            border_handling=BorderHandling.from_param(border_handling, self.descale),
            ignore_mask=ignore_mask,
            blur=blur,
        )

        sample_grid_model = SampleGridModel(sample_grid_model)

        if field_based.is_inter:
            from vsdeinterlace import reweave

            shift_y, shift_x = _descale_shift_norm(shift, False, self.descale)

            kwargs_tf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[0], shift_x[0]), **kwargs)
            kwargs_bf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[1], shift_x[1]), **kwargs)

            de_kwargs_tf = self.get_descale_args(clip, (shift_y[0], shift_x[0]), *de_base_args, **kwargs_tf)
            de_kwargs_bf = self.get_descale_args(clip, (shift_y[1], shift_x[1]), *de_base_args, **kwargs_bf)

            if height % 2:
                raise CustomIndexError("You can't descale to odd resolution when crossconverted!", self.descale)

            field_shift = 0.125 * height / clip.height

            fields = clip.std.SeparateFields(field_based.is_tff)

            descaled_tf = super().descale(
                fields[0::2],
                **de_kwargs_tf | {"src_top": de_kwargs_tf.get("src_top", 0.0) + field_shift},
            )
            descaled_bf = super().descale(
                fields[1::2],
                **de_kwargs_bf | {"src_top": de_kwargs_bf.get("src_top", 0.0) - field_shift},
            )
            descaled = reweave(descaled_tf, descaled_bf, field_based)
        else:
            shift = _descale_shift_norm(shift, True, self.descale)

            kwargs, shift = sample_grid_model.for_src(clip, width, height, shift, **kwargs)

            descaled = super().descale(clip, **self.get_descale_args(clip, shift, *de_base_args, **kwargs))

        return depth(descaled, input_clip)

    def rescale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] | FieldShift = (0, 0),
        *,
        # `linear` and `sigmoid` parameters from LinearDescaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # ComplexDescaler adds border_handling, sample_grid_model, field_based, ignore_mask and blur
        border_handling: int = BorderHandling.MIRROR,
        sample_grid_model: int = SampleGridModel.MATCH_EDGES,
        field_based: FieldBasedLike | None = None,
        ignore_mask: vs.VideoNode | None = None,
        blur: float | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Rescale a clip to the given resolution from a previously descaled clip,
        with image borders handling and sampling grid alignment, optionally using linear light processing.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip.
            width: Target scaled width (defaults to clip width if None).
            height: Target scaled height (defaults to clip height if None).
            shift: Subpixel shift (top, left) or per-field shifts.
            linear: Whether to linearize the input before rescaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).
            border_handling: Method for handling image borders during sampling.
            sample_grid_model: Model used to align sampling grid.
            field_based: Field-based processing mode (interlaced or progressive).
            ignore_mask: Optional mask specifying areas to ignore during rescaling.
            blur: Amount of blur to apply during rescaling.
            **kwargs: Additional arguments passed to `rescale_function`.

        Returns:
            The scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)

        field_based = FieldBased.from_param_or_video(field_based, clip)

        clip, bits = expect_bits(clip, 32)

        de_base_args = (width, height // (1 + field_based.is_inter))
        kwargs.update(
            border_handling=BorderHandling.from_param(border_handling, self.rescale), ignore_mask=ignore_mask, blur=blur
        )

        sample_grid_model = SampleGridModel(sample_grid_model)

        if field_based.is_inter:
            raise NotImplementedError
        else:
            shift = _descale_shift_norm(shift, True, self.rescale)

            kwargs, shift = sample_grid_model.for_src(clip, width, height, shift, **kwargs)

            rescaled = super().rescale(
                clip, **self.get_rescale_args(clip, shift, *de_base_args, **kwargs), linear=linear, sigmoid=sigmoid
            )

        return depth(rescaled, bits)


class ComplexKernel(Kernel, ComplexDescaler, ComplexScaler):
    """
    Comprehensive abstract kernel class combining scaling, descaling, and resampling with linear light and aspect ratio support.

    This class merges the full capabilities of [Kernel][vskernels.Kernel], [ComplexDescaler][vskernels.ComplexDescaler],
    and [ComplexScaler][vskernels.ComplexScaler].
    """  # noqa: E501


type ComplexScalerLike = str | type[ComplexScaler] | ComplexScaler
"""
Type alias for anything that can resolve to a ComplexScaler.

This includes:

- A string identifier.
- A class type subclassing [ComplexScaler][vskernels.ComplexScaler].
- An instance of a [ComplexScaler][vskernels.ComplexScaler].
"""

type ComplexDescalerLike = str | type[ComplexDescaler] | ComplexDescaler
"""
Type alias for anything that can resolve to a ComplexDescaler.

This includes:

- A string identifier.
- A class type subclassing [ComplexDescaler][vskernels.ComplexDescaler].
- An instance of a [ComplexDescaler][vskernels.ComplexDescaler].
"""

type ComplexKernelLike = str | type[ComplexKernel] | ComplexKernel
"""
Type alias for anything that can resolve to a ComplexKernel.

This includes:

- A string identifier.
- A class type subclassing [ComplexKernel][vskernels.ComplexKernel].
- An instance of a [ComplexKernel][vskernels.ComplexKernel].
"""
