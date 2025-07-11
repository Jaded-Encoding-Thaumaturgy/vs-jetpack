from __future__ import annotations

from typing import Callable, Generic, Sequence

from jetpytools import CustomIntEnum, CustomStrEnum, P, R

from vsexprtools import norm_expr
from vstools import ConstantFormatVideoNode, KwargsNotNone, PlanesT, check_variable, normalize_param_planes, vs

from .aka_expr import removegrain_aka_exprs, repair_aka_exprs

__all__ = ["clense", "remove_grain", "removegrain", "repair", "vertical_cleaner"]


class Repair(Generic[P, R]):
    """
    Class decorator that wraps the [repair][vsrgtools.rgtools.repair] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, repair_func: Callable[P, R]) -> None:
        self._func = repair_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Mode(CustomIntEnum):
        """
        Enum that specifies the mode for repairing or limiting a source clip using a reference clip.

        These modes define different spatial strategies for constraining each pixel in the source clip
        based on the reference clip's local neighborhood, typically using 3x3 squares or line-sensitive patterns.

        Commonly used in denoising, artifact removal, or detail-preserving restoration.
        """

        NONE = 0
        """
        No repair. The input plane is passed through unchanged.
        """

        MINMAX_SQUARE1 = 1
        """
        Clamp using the 1st min/max from a 3x3 square in the reference clip.
        """

        MINMAX_SQUARE2 = 2
        """
        Clamp using the 2nd min/max from a 3x3 square in the reference clip.
        """

        MINMAX_SQUARE3 = 3
        """
        Clamp using the 3rd min/max from a 3x3 square in the reference clip.
        """

        MINMAX_SQUARE4 = 4
        """
        Clamp using the 4th min/max from a 3x3 square in the reference clip.
        """

        LINE_CLIP_MIN = 5
        """
        Line-sensitive clamping with minimal alteration.
        """

        LINE_CLIP_LIGHT = 6
        """
        Line-sensitive clamping with a light effect.
        """

        LINE_CLIP_MEDIUM = 7
        """
        Line-sensitive clamping with a moderate effect.
        """

        LINE_CLIP_STRONG = 8
        """
        Line-sensitive clamping with a strong effect.
        """

        LINE_CLIP_CLOSE = 9
        """
        Line-sensitive clamp using the closest neighbors.
        """

        MINMAX_SQUARE_REF_CLOSE = 10
        """
        Replace pixel with the closest value in the 3x3 reference square.
        """

        MINMAX_SQUARE_REF1 = 11
        """
        Same as mode 1, but clips with min/max of 1st rank and the center pixel of the reference clip.
        """

        MINMAX_SQUARE_REF2 = 12
        """
        Same as mode 2, but clips with min/max of 2nd rank and the center pixel of the reference clip.
        """

        MINMAX_SQUARE_REF3 = 13
        """
        Same as mode 3, but clips with min/max of 3rd rank and the center pixel of the reference clip.
        """

        MINMAX_SQUARE_REF4 = 14
        """
        Same as mode 4, but clips with min/max of 4th rank and the center pixel of the reference clip.
        """

        CLIP_REF_RG5 = 15
        """
        Use RemoveGrain mode 5's result to constrain the pixel.
        """

        CLIP_REF_RG6 = 16
        """
        Use RemoveGrain mode 6's result to constrain the pixel.
        """

        CLIP_REF_RG17 = 17
        """
        Use RemoveGrain mode 17's result to constrain the pixel.
        """

        CLIP_REF_RG18 = 18
        """
        Use RemoveGrain mode 18's result to constrain the pixel.
        """

        CLIP_REF_RG19 = 19
        """
        Use RemoveGrain mode 19's result to constrain the pixel.
        """

        CLIP_REF_RG20 = 20
        """
        Use RemoveGrain mode 20's result to constrain the pixel.
        """

        CLIP_REF_RG21 = 21
        """
        Use RemoveGrain mode 21's result to constrain the pixel.
        """

        CLIP_REF_RG22 = 22
        """
        Use RemoveGrain mode 22's result to constrain the pixel.
        """

        CLIP_REF_RG23 = 23
        """
        Use RemoveGrain mode 23's result to constrain the pixel.
        """

        CLIP_REF_RG24 = 24
        """
        Use RemoveGrain mode 24's result to constrain the pixel.
        """

        # Mode 25 is not available

        CLIP_REF_RG26 = 26
        """
        Use RemoveGrain mode 26's result to constrain the pixel.
        """

        CLIP_REF_RG27 = 27
        """
        Use RemoveGrain mode 27's result to constrain the pixel.
        """

        CLIP_REF_RG28 = 28
        """
        Use RemoveGrain mode 28's result to constrain the pixel.
        """

        def __call__(
            self, clip: vs.VideoNode, repairclip: vs.VideoNode, planes: PlanesT = None
        ) -> ConstantFormatVideoNode:
            """
            Apply the selected repair mode to a `clip` using a `repairclip`.

            Args:
                clip: Input clip to process (typically filtered).
                repairclip: Reference clip for bounds (often the original or a less-processed version).
                planes: Planes to process. Defaults to all.

            Returns:
                Clip with repaired pixels, bounded by the reference.
            """
            return repair(clip, repairclip, self, planes)


@Repair
def repair(
    clip: vs.VideoNode,
    repairclip: vs.VideoNode,
    mode: int | Repair.Mode | Sequence[int | Repair.Mode],
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    Constrains the input `clip` using a `repairclip` by clamping pixel values
    based on a chosen [mode][vsrgtools.rgtools.Repair.Mode].

    This is typically used to limit over-aggressive filtering (e.g., from `RemoveGrain`) while keeping
    the corrections within reasonable bounds derived from the reference (`repairclip`). Often used in detail
    restoration workflows.

    - Modes 1-24 directly map to [zsmooth.Repair](https://github.com/adworacz/zsmooth?tab=readme-ov-file#repair)
      plugin modes.
    - Modes 26+ fall back to expression-based implementations.

    Example:
        ```py
        repaired = repair(filtered, clip, repair.Mode.MINMAX_SQUARE_REF3)
        ```

        Alternatively, directly using the enum:
        ```py
        repaired = repair.Mode.MINMAX_SQUARE_REF3(clip)
        ```

    Args:
        clip: Input clip to process (typically filtered).
        repairclip: Reference clip for bounds (often the original or a less-processed version).
        mode: Repair mode(s) used to constrain pixels. Can be a single mode or a list per plane. See
            [Repair.Mode][vsrgtools.rgtools.Repair.Mode] for details.
        planes: Planes to process. Default to all.

    Returns:
        Clip with repaired pixels, bounded by the reference.
    """
    assert check_variable(clip, repair)
    assert check_variable(repairclip, repair)

    mode = normalize_param_planes(clip, mode, planes, Repair.Mode.NONE)

    if not sum(mode):
        return clip

    if all(m in range(24 + 1) for m in mode):
        return clip.zsmooth.Repair(repairclip, mode)

    return norm_expr([clip, repairclip], tuple([repair_aka_exprs[m]() for m in mode]), func=repair)


class RemoveGrain(Generic[P, R]):
    """
    Class decorator that wraps the [remove_grain][vsrgtools.rgtools.remove_grain] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, repair_func: Callable[P, R]) -> None:
        self._func = repair_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Mode(CustomIntEnum):
        """
        Enum representing available spatial filtering strategies in RemoveGrain.

        These modes serve a wide range of use cases, such as clamping outliers,
        removing noise, or simple dehaloing.

        More information [here](https://blog.kageru.moe/legacy/removegrain.html).
        """

        NONE = 0
        """
        The input plane is simply passed through.
        """

        MINMAX_AROUND1 = 1
        """
        Clamp pixel to the min/max of its 3x3 neighborhood (excluding center).
        """

        MINMAX_AROUND2 = 2
        """
        Clamp to the second lowest/highest in the neighborhood.
        """

        MINMAX_AROUND3 = 3
        """
        Clamp to the third lowest/highest in the neighborhood.
        """

        MINMAX_MEDIAN = 4
        """
        Deprecated. Similar to mode 1, but clamps to fourth-lowest/highest.
        """

        EDGE_CLIP_STRONG = 5
        """
        Line-sensitive clipping that minimizes change to the center pixel.
        """

        EDGE_CLIP_MODERATE = 6
        """
        Line-sensitive clipping with moderate sensitivity (change prioritized 2:1).
        """

        EDGE_CLIP_MEDIUM = 7
        """
        Balanced version of mode 6 (change vs. range ratio 1:1).
        """

        EDGE_CLIP_LIGHT = 8
        """
        Light edge clipping (prioritizes the range between opposites).
        """

        LINE_CLIP_CLOSE = 9
        """
        Clip using the line with closest neighbors. Useful for fixing 1-pixel gaps.
        """

        MIN_SHARP = 10
        """
        Replaces pixel with its closest neighbor. A poor but sharp denoiser.
        """

        BINOMIAL_BLUR = 11
        """
        Deprecated. Use `BlurMatrix.BINOMIAL`. Applies weighted 3x3 blur.
        """

        BOB_TOP_CLOSE = 13
        """
        Interpolate top field using the closest neighboring pixels.
        """

        BOB_BOTTOM_CLOSE = 14
        """
        Interpolate bottom field using the closest neighboring pixels.
        """

        BOB_TOP_INTER = 15
        """
        Top field interpolation using a more complex formula than mode 13.
        """

        BOB_BOTTOM_INTER = 16
        """
        Bottom field interpolation using a more complex formula than mode 14.
        """

        MINMAX_MEDIAN_OPP = 17
        """
        Clip using min/max of opposing neighbor pairs.
        """

        LINE_CLIP_OPP = 18
        """
        Line-sensitive clipping using opposite neighbors with minimal deviation.
        """

        MEAN_NO_CENTER = 19
        """
        Deprecated. Use `BlurMatrix.MEAN_NO_CENTER`. Mean of neighborhood (excluding center).
        """

        MEAN = 20
        """
        Deprecated. Use `BlurMatrix.MEAN`. Arithmetic mean of 3x3 neighborhood.
        """

        BOX_BLUR_NO_CENTER = MEAN_NO_CENTER
        """
        Alias for MEAN_NO_CENTER.
        """

        BOX_BLUR = MEAN
        """
        Alias for MEAN.
        """

        OPP_CLIP_AVG = 21
        """
        Clip to min/max of averages of 4 opposing pixel pairs.
        """

        OPP_CLIP_AVG_FAST = 22
        """
        Faster variant of mode 21 with simpler rounding.
        """

        EDGE_DEHALO = 23
        """
        Very light dehaloing. Rarely useful.
        """

        EDGE_DEHALO2 = 24
        """
        More conservative version of mode 23.
        """

        MIN_SHARP2 = 25
        """
        Minimal sharpening also known as "non destructive sharpen".
        """

        SMART_RGC = 26
        """
        Like mode 17, but preserves corners. Does not preserve thin lines.
        """

        SMART_RGCL = 27
        """
        Uses 12 pixel pairs instead of 8. Similar to 26, but slightly stronger.
        """

        SMART_RGCL2 = 28
        """
        Variant of 27 with different pairs. Usually visually similar.
        """

        def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> ConstantFormatVideoNode:
            """
            Apply the selected remove grain mode to a `clip`.

            Args:
                clip: Clip to process.
                planes: Planes to process. Defaults to all.

            Returns:
                Processed (denoised) clip.
            """
            return remove_grain(clip, self, planes)


@RemoveGrain
def remove_grain(
    clip: vs.VideoNode, mode: int | RemoveGrain.Mode | Sequence[int | RemoveGrain.Mode], planes: PlanesT = None
) -> ConstantFormatVideoNode:
    """
    Apply spatial denoising using the RemoveGrain algorithm.

    Supports a variety of pixel clamping, edge-aware filtering, and blur strategies.
    See [RemoveGrain.Mode][vsrgtools.rgtools.RemoveGrain.Mode] for all available modes.

    - Modes 1-24 are natively implemented in [zsmooth.RemoveGrain](https://github.com/adworacz/zsmooth?tab=readme-ov-file#removegrain).
    - Modes 25+ fall back to expression-based implementations.

    Example:
        ```py
        denoised = remove_grain(clip, remove_grain.Mode.EDGE_CLIP_STRONG)
        ```

        Alternatively, directly using the enum:
        ```py
        denoised = remove_grain.Mode.EDGE_CLIP_STRONG(clip)
        ```

    Args:
        clip: Clip to process.
        mode: Mode(s) to use. Can be a single mode or per-plane list. See
            [RemoveGrain.Mode][vsrgtools.rgtools.RemoveGrain.Mode] for details.
        planes: Planes to process. Default to all.

    Returns:
        Processed (denoised) clip.
    """
    assert check_variable(clip, remove_grain)

    mode = normalize_param_planes(clip, mode, planes, RemoveGrain.Mode.NONE)

    if not sum(mode):
        return clip

    if all(m in range(24 + 1) for m in mode):
        return clip.zsmooth.RemoveGrain(mode)

    return norm_expr(clip, tuple([removegrain_aka_exprs[m]() for m in mode]), func=remove_grain)


class Clense(Generic[P, R]):
    """
    Class decorator that wraps the [clense][vsrgtools.rgtools.clense] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, clense_func: Callable[P, R]) -> None:
        self._func = clense_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Mode(CustomStrEnum):
        """
        Enum that specifies the temporal clense mode to use.

        Clense modes refer to different ways of applying temporal median filtering over multiple frames.

        Each mode maps to a function provided
        by the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#clense--forwardclense--backwardclense)
        plugin.
        """

        NONE = ""
        """
        No clense filtering. Returns the original clip unchanged.
        """

        BACKWARD = "BackwardClense"
        """
        Use the current and previous two frames for temporal median filtering.
        """

        FORWARD = "ForwardClense"
        """
        Use the current and next two frames for temporal median filtering.
        """

        BOTH = "Clense"
        """
        Standard clense: median of previous, current, and next frame.
        """

        def __call__(
            self,
            clip: vs.VideoNode,
            previous_clip: vs.VideoNode | None = None,
            next_clip: vs.VideoNode | None = None,
            planes: PlanesT = None,
        ) -> ConstantFormatVideoNode:
            """
            Apply the selected clense mode to a clip using
            the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#clense--forwardclense--backwardclense)
            plugin.

            Args:
                clip: Source clip to process.
                previous_clip: Optional alternate clip to source previous frames. Defaults to `clip`.
                next_clip: Optional alternate clip to source next frames. Defaults to `clip`.
                planes: Planes to process. Defaults to all.

            Returns:
                Clensed clip with temporal median filtering.
            """
            return clense(clip, previous_clip, next_clip, self, planes)


@Clense
def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None,
    next_clip: vs.VideoNode | None = None,
    mode: Clense.Mode | str = Clense.Mode.NONE,
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    Apply a clense (temporal median) filter based on the specified mode.

    Example:
        ```py
        clensed = clense(clip, ..., mode=clense.Mode.BOTH)
        ```

        Alternatively, directly using the enum:
        ```py
        clensed = clense.Mode.BOTH(clip, ...)
        ```

    Args:
        clip: Source clip to process.
        previous_clip: Optional alternate clip to source previous frames. Defaults to `clip`.
        next_clip: Optional alternate clip to source next frames. Defaults to `clip`.
        mode: Mode of filtering. One of [Mode][vsrgtools.rgtools.Clense.Mode] or its string values.
        planes: Planes to process. Defaults to all.

    Returns:
        Clensed clip with temporal median filtering.
    """
    assert check_variable(clip, clense)

    kwargs = KwargsNotNone(previous=previous_clip, next=next_clip)

    if mode == Clense.Mode.NONE:
        return clip

    return getattr(clip.zsmooth, mode)(planes=planes, **kwargs)


class VerticalCleaner(Generic[P, R]):
    """
    Class decorator that wraps the [vertical_cleaner][vsrgtools.rgtools.vertical_cleaner] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, vertical_cleaner_func: Callable[P, R]) -> None:
        self._func = vertical_cleaner_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Mode(CustomIntEnum):
        """
        Enum that specifies the vertical cleaner mode to use
        in the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#verticalcleaner) plugin.
        """

        NONE = 0
        """
        The input plane is simply passed through.
        """

        MEDIAN = 1
        """
        Applies a strict vertical median filter.
        """

        PRESERVING = 2
        """
        Applies a detail-preserving vertical median filter (less aggressive).
        """

        def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> ConstantFormatVideoNode:
            """
            Applies the vertical cleaning mode to the given clip.

            Args:
                clip: Source clip to process.
                planes: Planes to process. Defaults to all.

            Returns:
                Filtered clip.
            """
            return vertical_cleaner(clip, self, planes)


@VerticalCleaner
def vertical_cleaner(
    clip: vs.VideoNode,
    mode: int | VerticalCleaner.Mode | Sequence[int | VerticalCleaner.Mode],
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    Applies a fast vertical median or relaxed median filter to the clip
    using the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#verticalcleaner) plugin.

    Example:
        ```py
        cleaned = vertical_cleaner(clip, vertical_cleaner.Mode.PRESERVING)
        ```

        Alternatively, directly using the enum:
        ```py
        cleaned = vertical_cleaner.Mode.PRESERVING(clip)
        ```

    Args:
        clip: Source clip to process.
        mode: Mode of vertical cleaning to apply.
            Can be:

               - A single enum/int (applied to all planes),
               - A sequence of enums/ints (one per plane).

        planes: Planes to process. Defaults to all.

    Returns:
        Filtered clip.
    """
    assert check_variable(clip, vertical_cleaner)

    mode = normalize_param_planes(clip, mode, planes, VerticalCleaner.Mode.NONE)

    if not sum(mode):
        return clip

    return clip.zsmooth.VerticalCleaner(mode)


removegrain = remove_grain  # TODO: remove
