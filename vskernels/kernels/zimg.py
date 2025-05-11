from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vstools import ConstantFormatVideoNode, Dar, FieldBased, Sar, vs

from ..types import BorderHandling, Center, LeftShift, SampleGridModel, ShiftT, Slope, TopShift
from .abstract import Descaler
from .complex import ComplexKernel

__all__ = [
    'ZimgDescaler',
    'ZimgComplexKernel'
]


class ZimgDescaler(Descaler):
    """
    An abstract descaler that uses the Zimg library for descaling operation.
    """

    if TYPE_CHECKING:
        def descale(
            self, clip: vs.VideoNode, width: int | None, height: int | None,
            shift: ShiftT = (0, 0),
            *,
            # `border_handling`, `sample_grid_model` and `field_based` from Descaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            field_based: FieldBased | None = None,
            # ZimgDescaler adds `blur` and `ignore_mask` parameters
            blur: float = 1.0, ignore_mask: vs.VideoNode | None = None,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Descale a clip to the given resolution.

            Supports both progressive and interlaced sources. When interlaced, it will separate fields,
            perform per-field descaling, and weave them back.

            :param clip:                The source clip.
            :param width:               Target descaled width (defaults to clip width if None).
            :param height:              Target descaled height (defaults to clip height if None).
            :param shift:               Subpixel shift (top, left) or per-field shifts.
            :param border_handling:     Method for handling image borders during sampling.
            :param sample_grid_model:   Model used to align sampling grid.
            :param field_based:         Field-based processing mode (interlaced or progressive).
            :param blur:                Amount of blur to apply during descaling.
            :param ignore_mask:         Optional mask specifying areas to ignore during descaling.
            :param kwargs:              Additional parameters passed to the underlying descaler.
            :return:                    The descaled clip.
            """
            ...


class ZimgComplexKernel(ComplexKernel, ZimgDescaler):
    """
    A abstract complex kernel class that combines scaling and descaling operations using Zimg.
    """

    if TYPE_CHECKING:
        # Override signature to add `blur`
        def scale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *,
            # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
            dar_in: Dar | bool | float | None = None, keep_ar: bool | None = None,
            # `linear` and `sigmoid` from LinearScaler
            linear: bool | None = None, sigmoid: bool | tuple[Slope, Center] = False,
            # ZimgComplexKernel adds blur parameter
            blur: float = 1.0,
            **kwargs: Any
        ) -> vs.VideoNode | ConstantFormatVideoNode:
            """
            Scale a clip to the given resolution, with aspect ratio and linear light support.

            :param clip:                The source clip.
            :param width:               Target width (defaults to clip width if None).
            :param height:              Target height (defaults to clip height if None).
            :param shift:               Subpixel shift (top, left) applied during scaling.
            :param border_handling:     Method for handling image borders during sampling.
            :param sample_grid_model:   Model used to align sampling grid.
            :param sar:                 Sample aspect ratio to assume or convert to.
            :param dar:                 Desired display aspect ratio.
            :param dar_in:              Input display aspect ratio, if different from clip’s.
            :param keep_ar:             Whether to adjust dimensions to preserve aspect ratio.
            :param linear:              Whether to linearize the input before descaling. If None, inferred from sigmoid.
            :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                        `True` applies the defaults values (6.5, 0.75).
                                        Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                        and sigmoid center has to be in range 0.0-1.0 (inclusive).
            :param blur:                Amount of blur to apply during scaling.
            :return:                    The scaled clip.
            """
            ...

        def descale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: ShiftT = (0, 0),
            *,
            # `border_handling`, `sample_grid_model` and `field_based` from Descaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            field_based: FieldBased | None = None,
            # `linear` and `sigmoid` from LinearDescaler
            linear: bool | None = None, sigmoid: bool | tuple[Slope, Center] = False,
            # `blur` and `ignore_mask` parameters from ZimgDescaler
            blur: float = 1.0, ignore_mask: vs.VideoNode | None = None,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            """
            Descale a clip to the given resolution.

            Supports both progressive and interlaced sources. When interlaced, it will separate fields,
            perform per-field descaling, and weave them back.

            :param clip:                The source clip.
            :param width:               Target descaled width (defaults to clip width if None).
            :param height:              Target descaled height (defaults to clip height if None).
            :param shift:               Subpixel shift (top, left) or per-field shifts.
            :param border_handling:     Method for handling image borders during sampling.
            :param sample_grid_model:   Model used to align sampling grid.
            :param field_based:         Field-based processing mode (interlaced or progressive).
            :param linear:              Whether to linearize the input before descaling. If None, inferred from sigmoid.
            :param sigmoid:             Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center).
                                        `True` applies the defaults values (6.5, 0.75).
                                        Keep in mind sigmoid slope has to be in range 1.0-20.0 (inclusive)
                                        and sigmoid center has to be in range 0.0-1.0 (inclusive).
            :param blur:                Amount of blur to apply during descaling.
            :param ignore_mask:         Optional mask specifying areas to ignore during descaling.
            :param kwargs:              Additional parameters passed to the underlying descaler.
            :return:                    The descaled clip.
            """
            ...

