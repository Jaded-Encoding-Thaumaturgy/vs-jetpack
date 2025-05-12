from __future__ import annotations

from functools import partial
from math import ceil
from typing import TYPE_CHECKING, Any, Literal, SupportsFloat, TypeVar, Union, overload

from jetpytools import CustomIndexError, CustomValueError, FuncExceptT

from vstools import (
    ConstantFormatVideoNode, Dar, FieldBased, FieldBasedT, KwargsT, Resolution, Sar, VideoNodeT,
    check_correct_subsampling, depth, expect_bits, fallback, vs
)

from ..types import (
    BorderHandling, BotFieldLeftShift, BotFieldTopShift, Center, LeftShift, SampleGridModel, ShiftT, Slope,
    TopFieldLeftShift, TopFieldTopShift, TopShift
)
from .base import BaseScaler, Descaler, Kernel, Resampler, Scaler

__all__ = [
    "LinearScaler",
    "LinearDescaler",
    "KeepArScaler",
    "ComplexScaler",
    "ComplexScalerT",
    "ComplexKernel",
    "ComplexKernelT",
    "CustomComplexKernel",
    "CustomComplexTapsKernel",
]

XarT = TypeVar("XarT", Sar, Dar)


def _from_param(cls: type[XarT], value: XarT | bool | float | None, fallback: XarT) -> XarT | None:
    if value is False:
        return fallback

    if value is True:
        return None

    if isinstance(value, cls):
        return value

    if isinstance(value, SupportsFloat):
        return cls.from_float(float(value))

    return None


class _BaseLinear(BaseScaler):
    def _linearize(
        self,
        clip: vs.VideoNode,
        linear: bool | None,
        sigmoid: bool | tuple[Slope, Center],
        op_partial: partial[VideoNodeT],
        func: FuncExceptT,
        **kwargs: Any,
    ) -> VideoNodeT:

        if linear is False and sigmoid is not False:
            raise CustomValueError("If sigmoid is not False, linear can't be False as well!", func, (linear, sigmoid))

        # if a _linear_scale or _linear_descale method is specified in the class,
        # use this method instead of the super().scale or super().descale method.
        # args and keywords are also forwarded.
        if hasattr(self, f"_linear_{op_partial.func.__name__}"):
            op_partial = partial(
                getattr(self, f"_linear_{op_partial.func.__name__}"), *op_partial.args, **op_partial.keywords
            )

        if sigmoid or linear:
            from ..util import LinearLight

            fmt = self.kwargs.pop("format", kwargs.pop("format", None))

            with LinearLight(clip, linear, sigmoid, self if isinstance(self, Resampler) else None, fmt) as ll:
                ll.linear = op_partial(ll.linear, **kwargs)

            return ll.out  # type: ignore[return-value]

        return op_partial(clip, **kwargs)


class LinearScaler(_BaseLinear, Scaler):
    """
    Abctract scaler class that applies linearization before scaling.

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
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        """
        Scale a clip to the given resolution with optional linearization.

        This method behaves like the base `Scaler.descale()` but adds support for
        linear or sigmoid-based preprocessing and postprocessing. When enabled, the clip
        is linearized before the scaling operation and de-linearized afterward.

        :param clip:        The source clip.
        :param width:       Target width (defaults to clip width if None).
        :param height:      Target height (defaults to clip height if None).
        :param shift:       Subpixel shift (top, left) applied during scaling.
        :param linear:      Whether to linearize the input before scaling. If None, inferred from sigmoid.
        :param sigmoid:     Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                            `True` applies the defaults values (6.5, 0.75).
                            Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                            and sigmoid center has to be in range 0.0-1.0 (inclusive).
        :param kwargs:      Additional arguments forwarded to the scale function.
        :return:            Scaled video clip.
        """
        return self._linearize(
            clip, linear, sigmoid, partial(super().scale, width=width, height=height, shift=shift), self.scale, **kwargs
        )


class LinearDescaler(_BaseLinear, Descaler):
    """
    Abctract descaler class that applies linearization before descaling.

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
        ) -> ConstantFormatVideoNode:
            """
            An optional function to be implemented by subclasses.

            If implemented, this will override the default descale behavior,
            allowing custom linear descaling logic to be applied instead of the base descaler's method.
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
    ) -> ConstantFormatVideoNode:
        """
        Descale a clip to the specified resolution, optionally using linear light processing.

        This method behaves like the base `Descaler.descale()` but adds support for
        linear or sigmoid-based preprocessing and postprocessing. When enabled, the clip
        is linearized before the descaling operation and de-linearized afterward.

        :param clip:                The source clip.
        :param width:               Target descaled width (defaults to clip width if None).
        :param height:              Target descaled height (defaults to clip height if None).
        :param shift:               Subpixel shift (top, left) applied during descaling.
        :param linear:              Whether to linearize the input before descaling. If None, inferred from sigmoid.
        :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                    `True` applies the defaults values (6.5, 0.75).
                                    Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                    and sigmoid center has to be in range 0.0-1.0 (inclusive).
        :return:                    The descaled video node, optionally processed in linear light.
        """
        return self._linearize(
            clip,
            linear,
            sigmoid,
            partial(super().descale, width=width, height=height, shift=shift),
            self.descale,
            **kwargs,
        )


class KeepArScaler(Scaler):
    def _get_kwargs_keep_ar(
        self,
        sar: Sar | float | bool | None = None,
        dar: Dar | float | bool | None = None,
        dar_in: Dar | float | bool | None = None,
        keep_ar: bool | None = None,
        **kwargs: Any,
    ) -> KwargsT:
        kwargs = KwargsT(keep_ar=keep_ar, sar=sar, dar=dar, dar_in=dar_in) | kwargs

        if keep_ar is not None:
            if None not in set(kwargs.get(x) for x in ("keep_ar", "sar", "dar", "dar_in")):
                print(
                    UserWarning(
                        f'{self.__class__.__name__}.scale: "keep_ar" set '
                        'with non-None values set in "sar", "dar" and "dar_in" won\'t do anything!'
                    )
                )
        else:
            kwargs["keep_ar"] = False

        default_val = kwargs.pop("keep_ar")

        for key in ("sar", "dar", "dar_in"):
            if kwargs[key] is None:
                kwargs[key] = default_val

        return kwargs

    def _handle_crop_resize_kwargs(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[TopShift, LeftShift],
        sar: Sar | bool | float | None,
        dar: Dar | bool | float | None,
        dar_in: Dar | bool | float | None,
        **kwargs: Any,
    ) -> tuple[KwargsT, tuple[TopShift, LeftShift], Sar | None]:
        kwargs.setdefault("src_top", kwargs.pop("sy", shift[0]))
        kwargs.setdefault("src_left", kwargs.pop("sx", shift[1]))
        kwargs.setdefault("src_width", kwargs.pop("sw", clip.width))
        kwargs.setdefault("src_height", kwargs.pop("sh", clip.height))

        src_res = Resolution(kwargs["src_width"], kwargs["src_height"])

        src_sar = float(_from_param(Sar, sar, Sar(1, 1)) or Sar.from_clip(clip))
        out_sar = None

        out_dar = float(_from_param(Dar, dar, Dar(0)) or Dar.from_res(width, height))
        src_dar = float(fallback(_from_param(Dar, dar_in, Dar(out_dar)), Dar.from_clip(clip, False)))

        if src_sar not in {0.0, 1.0}:
            if src_sar > 1.0:
                out_dar = (width / src_sar) / height
            else:
                out_dar = width / (height * src_sar)

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
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None,
        dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None,
        keep_ar: bool | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        width, height = self._wh_norm(clip, width, height)

        check_correct_subsampling(clip, width, height)

        const_size = 0 not in (clip.width, clip.height)

        if const_size:
            kwargs = self._get_kwargs_keep_ar(sar, dar, dar_in, keep_ar, **kwargs)

            kwargs, shift, out_sar = self._handle_crop_resize_kwargs(clip, width, height, shift, **kwargs)

            kwargs, shift = sample_grid_model.for_dst(clip, width, height, shift, **kwargs)

            if (bh := BorderHandling.from_param(border_handling, self.scale)) is not None:
                border_handling = bh
            else:
                raise TypeError

            padded = border_handling.prepare_clip(clip, self.kernel_radius)

            shift, clip = (
                tuple(s + ((p - c) // 2) for s, c, p in zip(shift, *((x.height, x.width) for x in (clip, padded)))),
                padded,
            )

        clip = super().scale(clip, width, height, shift, **kwargs)

        if const_size and out_sar:
            clip = out_sar.apply(clip)

        return clip


class ComplexScaler(KeepArScaler, LinearScaler):
    """
    Abstract composite scaler class that supports both aspect ratio preservation and linear light processing.

    This class combines the capabilities of `KeepArScaler` (for handling display/sample aspect ratio)
    and `LinearScaler` (for linear and sigmoid processing).
    """

    if TYPE_CHECKING:
        def scale(
            self,
            clip: vs.VideoNode,
            width: int | None = None,
            height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *,
            # `linear` and `sigmoid` from LinearScaler
            linear: bool | None = False,
            sigmoid: bool | tuple[Slope, Center] = False,
            # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            sar: Sar | float | bool | None = None,
            dar: Dar | float | bool | None = None,
            dar_in: Dar | bool | float | None = None,
            keep_ar: bool | None = None,
            # ComplexScaler adds blur
            blur: float | None = None,
            **kwargs: Any,
        ) -> vs.VideoNode | ConstantFormatVideoNode:
            """
            Scale a clip to the given resolution, with aspect ratio and linear light support.

            :param clip:                The source clip.
            :param width:               Target width (defaults to clip width if None).
            :param height:              Target height (defaults to clip height if None).
            :param shift:               Subpixel shift (top, left) applied during scaling.
            :param linear:              Whether to linearize the input before descaling. If None, inferred from sigmoid.
            :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                        `True` applies the defaults values (6.5, 0.75).
                                        Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                        and sigmoid center has to be in range 0.0-1.0 (inclusive).
            :param border_handling:     Method for handling image borders during sampling.
            :param sample_grid_model:   Model used to align sampling grid.
            :param sar:                 Sample aspect ratio to assume or convert to.
            :param dar:                 Desired display aspect ratio.
            :param dar_in:              Input display aspect ratio, if different from clip's.
            :param keep_ar:             Whether to adjust dimensions to preserve aspect ratio.
            :param blur:                Amount of blur to apply during scaling.
            :return:                    Scaled clip, optionally aspect-corrected and linearized.
            """
            ...


class ComplexDescaler(LinearDescaler):
    """
    Abstract descaler class with support for border handling and sampling grid alignment.

    Extends `LinearDescaler` by introducing mechanisms to control how image borders
    are handled and how the sampling grid is aligned during descaling.
    """

    def descale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: ShiftT = (0, 0),
        *,
        # `linear` and `sigmoid` parameters from LinearDescaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # ComplexDescaler adds border_handling, sample_grid_model, field_based,  ignore_mask and blur
        border_handling: int | BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: int | SampleGridModel = SampleGridModel.MATCH_EDGES,
        field_based: FieldBasedT | None = None,
        ignore_mask: vs.VideoNode | None = None,
        blur: float | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Descale a clip to the given resolution, with image borders handling and sampling grid alignment,
        optionally using linear light processing.

        Supports both progressive and interlaced sources. When interlaced, it will separate fields,
        perform per-field descaling, and weave them back.

        :param clip:                The source clip.
        :param width:               Target descaled width (defaults to clip width if None).
        :param height:              Target descaled height (defaults to clip height if None).
        :param shift:               Subpixel shift (top, left) or per-field shifts.
        :param linear:              Whether to linearize the input before descaling. If None, inferred from sigmoid.
        :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                    `True` applies the defaults values (6.5, 0.75).
                                    Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                    and sigmoid center has to be in range 0.0-1.0 (inclusive).
        :param border_handling:     Method for handling image borders during sampling.
        :param sample_grid_model:   Model used to align sampling grid.
        :param field_based:         Field-based processing mode (interlaced or progressive).
        :param ignore_mask:         Optional mask specifying areas to ignore during descaling.
        :param blur:                Amount of blur to apply during scaling.
        :param kwargs:              Additional arguments passed to `descale_function`.
        :return:                    The descaled video node, optionally processed in linear light.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

        field_based = FieldBased.from_param_or_video(field_based, clip)

        clip, bits = expect_bits(clip, 32)

        de_base_args = (width, height // (1 + field_based.is_inter))
        kwargs.update(
            border_handling=BorderHandling.from_param(border_handling, self.descale), ignore_mask=ignore_mask, blur=blur
        )

        sample_grid_model = SampleGridModel(sample_grid_model)

        if field_based.is_inter:
            shift_y, shift_x = self._shift_norm(shift, False, self.descale)

            kwargs_tf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[0], shift_x[0]), **kwargs)
            kwargs_bf, shift = sample_grid_model.for_src(clip, width, height, (shift_y[1], shift_x[1]), **kwargs)

            de_kwargs_tf = self.get_descale_args(clip, (shift_y[0], shift_x[0]), *de_base_args, **kwargs_tf)
            de_kwargs_bf = self.get_descale_args(clip, (shift_y[1], shift_x[1]), *de_base_args, **kwargs_bf)

            if height % 2:
                raise CustomIndexError("You can't descale to odd resolution when crossconverted!", self.descale)

            field_shift = 0.125 * height / clip.height

            fields = clip.std.SeparateFields(field_based.is_tff)

            interleaved = vs.core.std.Interleave(
                [
                    super().descale(
                        fields[offset::2],
                        **de_kwargs | dict(src_top=de_kwargs.get("src_top", 0.0) + (field_shift * mult)),
                        linear=linear,
                        sigmoid=sigmoid,
                    )
                    for offset, mult, de_kwargs in [(0, 1, de_kwargs_tf), (1, -1, de_kwargs_bf)]
                ]
            )

            descaled = interleaved.std.DoubleWeave(field_based.is_tff)[::2]
        else:
            shift = self._shift_norm(shift, True, self.descale)

            kwargs, shift = sample_grid_model.for_src(clip, width, height, shift, **kwargs)

            descaled = super().descale(
                clip, **self.get_descale_args(clip, shift, *de_base_args, **kwargs), linear=linear, sigmoid=sigmoid
            )

        return depth(descaled, bits)

    def rescale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: ShiftT = (0, 0),
        *,
        # `linear` and `sigmoid` parameters from LinearDescaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # ComplexDescaler adds border_handling, sample_grid_model, field_based,  ignore_mask and blur
        border_handling: int | BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: int | SampleGridModel = SampleGridModel.MATCH_EDGES,
        field_based: FieldBasedT | None = None,
        ignore_mask: vs.VideoNode | None = None,
        blur: float | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Rescale a clip to the given resolution from a previously descaled clip.

        :param clip:                The source clip.
        :param width:               Target scaled width (defaults to clip width if None).
        :param height:              Target scaled height (defaults to clip height if None).
        :param shift:               Subpixel shift (top, left) or per-field shifts.
        :param linear:              Whether to linearize the input before rescaling. If None, inferred from sigmoid.
        :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                    `True` applies the defaults values (6.5, 0.75).
                                    Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                    and sigmoid center has to be in range 0.0-1.0 (inclusive).
        :param border_handling:     Method for handling image borders during sampling.
        :param sample_grid_model:   Model used to align sampling grid.
        :param field_based:         Field-based processing mode (interlaced or progressive).
        :param ignore_mask:         Optional mask specifying areas to ignore during rescaling.
        :param blur:                Amount of blur to apply during rescaling.
        :param kwargs:              Additional arguments passed to `rescale_function`.
        :return:                    Scaled clip.
        """
        width, height = self._wh_norm(clip, width, height)
        check_correct_subsampling(clip, width, height)

        field_based = FieldBased.from_param_or_video(field_based, clip)

        clip, bits = expect_bits(clip, 32)

        de_base_args = (width, height // (1 + field_based.is_inter))
        kwargs.update(
            border_handling=BorderHandling.from_param(border_handling, self.descale), ignore_mask=ignore_mask, blur=blur
        )

        sample_grid_model = SampleGridModel(sample_grid_model)

        if field_based.is_inter:
            raise NotImplementedError
        else:
            shift = self._shift_norm(shift, True, self.descale)

            kwargs, shift = sample_grid_model.for_src(clip, width, height, shift, **kwargs)

            descaled = super().descale(
                clip, **self.get_descale_args(clip, shift, *de_base_args, **kwargs), linear=linear, sigmoid=sigmoid
            )

        return depth(descaled, bits)

    @overload
    def _shift_norm(
        self, shift: ShiftT, assume_progressive: Literal[True] = ..., func: FuncExceptT | None = None
    ) -> tuple[TopShift, LeftShift]: ...

    @overload
    def _shift_norm(
        self, shift: ShiftT, assume_progressive: Literal[False] = ..., func: FuncExceptT | None = None
    ) -> tuple[tuple[TopFieldTopShift, BotFieldTopShift], tuple[TopFieldLeftShift, BotFieldLeftShift]]: ...

    def _shift_norm(self, shift: ShiftT, assume_progressive: bool = True, func: FuncExceptT | None = None) -> Any:
        """
        Normalize shift values depending on field-based status.

        :param shift:               Shift values (single or per-field).
        :param assume_progressive:  Whether to assume the input is progressive.
        :param func:                Function returned for custom error handling.
        :raises CustomValueError:   If per-field shift is used in progressive mode.
        :return:                    Normalized shift values.
        """
        if assume_progressive:
            if any(isinstance(sh, tuple) for sh in shift):
                raise CustomValueError("You can't descale per-field when the input is progressive!", func, shift)
        else:
            shift_y, shift_x = tuple[tuple[float, float], ...](
                sh if isinstance(sh, tuple) else (sh, sh) for sh in shift
            )
            shift = shift_y, shift_x

        return shift


class ComplexKernel(Kernel, ComplexDescaler, ComplexScaler):
    """
    Comprehensive abstract kernel class combining scaling, descaling,
    and resampling with linear light and aspect ratio support.

    This class merges the full capabilities of `Kernel`, `ComplexDescaler`, and `ComplexScaler`.
    """


class CustomComplexKernel(CustomKernel, ComplexKernel):
    """
    Abstract kernel class that combines custom kernel behavior with advanced scaling and descaling capabilities.

    This class extends both `CustomKernel` and `ComplexKernel`, enabling the definition
    of custom mathematical kernels with the advanced rescaling logic provided by
    linear and aspect-ratio-aware components.
    """


class CustomComplexTapsKernel(CustomComplexKernel):
    """
    Extension of `CustomComplexKernel` that introduces configurable kernel taps.
    """

    def __init__(self, taps: float, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific number of taps.

        :param taps:    Determines the radius of the kernel.
        :param kwargs:  Additional keyword arguments passed to the superclass.
        """
        self.taps = taps
        super().__init__(**kwargs)

    @Scaler.cached_property
    def kernel_radius(self) -> int:
        """
        Compute the effective kernel radius based on the number of taps.

        :return: Radius as the ceiling of `taps`.
        """
        return ceil(self.taps)

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**dict(taps=self.taps) | attrs)


ComplexScalerT = Union[str, type[ComplexScaler], ComplexScaler]
"""
Type alias for anything that can resolve to a ComplexScaler.

This includes:
- A string identifier.
- A class type subclassing `ComplexScaler`.
- An instance of a `ComplexScaler`.
"""

ComplexDescalerT = Union[str, type[ComplexDescaler], ComplexDescaler]
"""
Type alias for anything that can resolve to a ComplexDescaler.

This includes:
- A string identifier.
- A class type subclassing `ComplexDescaler`.
- An instance of a `ComplexDescaler`.
"""

ComplexKernelT = Union[str, type[ComplexKernel], ComplexKernel]
"""
Type alias for anything that can resolve to a ComplexKernel.

This includes:
- A string identifier.
- A class type subclassing `ComplexKernel`.
- An instance of a `ComplexKernel`.
"""
