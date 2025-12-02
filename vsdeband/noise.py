from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from enum import auto
from typing import Any, Literal, Protocol, overload

from jetpytools import MISSING, CustomEnum, CustomValueError, EnumABCMeta, FuncExcept, MissingT, mod_x, to_arr

from vsexprtools import norm_expr
from vsjetpack import TypeVar
from vskernels import BicubicAuto, Lanczos, LeftShift, Scaler, ScalerLike, ScalerSpecializer, TopShift
from vsrgtools import BlurMatrix
from vstools import (
    ColorRange,
    ConvMode,
    Planes,
    UnsupportedColorFamilyError,
    check_variable_resolution,
    core,
    get_lowest_values,
    get_neutral_values,
    get_peak_values,
    get_u,
    get_v,
    normalize_param_planes,
    normalize_planes,
    scale_delta,
    split,
    vs,
)

from .debanders import placebo_deband

__all__ = [
    "GrainFactoryBicubic",
    "Grainer",
    "LanczosTwoPasses",
    "ScalerTwoPasses",
]


type EdgeLimits = tuple[float | Sequence[float] | bool, float | Sequence[float] | bool]
"""
Tuple representing lower and upper edge limits for each plane.

Format: (low, high)

Each element can be:

- A float: the same limit is applied to all planes.
- A sequence of floats: individual limits for each plane.
- True: use the default legal range per plane.
- False: no limits are applied.
"""


class _GrainerFunc(Protocol):
    """
    Protocol for a graining function applied to a VideoNode.
    """

    def __call__(
        self, clip: vs.VideoNode, strength: float | Sequence[float], planes: Planes, **kwargs: Any
    ) -> vs.VideoNode: ...


_ScalerWithLanczosDefaultT = TypeVar("_ScalerWithLanczosDefaultT", bound=Scaler, default=Lanczos)


# TODO: class ScalerTwoPasses[_ScalerT: Scaler = Lanczos] python 3.13
class ScalerTwoPasses(ScalerSpecializer[_ScalerWithLanczosDefaultT]):
    """
    Scaler class that applies scaling in two passes.
    """

    _default_scaler = Lanczos

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        assert check_variable_resolution(clip, self.__class__)

        if any(shift):
            raise CustomValueError("Shifting is unsupported.", self.__class__, shift)

        width, height = self._wh_norm(clip, width, height)

        if width / clip.width > 1.5 or height / clip.height > 1.5:
            # If the scale is too big, we need to scale it in two passes, else the window
            # will be too big and the grain will be dampened down too much
            clip = super().scale(
                clip,
                mod_x((width + clip.width) / 2, 2**clip.format.subsampling_w),
                mod_x((height + clip.height) / 2, 2**clip.format.subsampling_h),
                **kwargs,
            )

        return super().scale(clip, width, height, (0, 0), **kwargs)


LanczosTwoPasses = ScalerTwoPasses[Lanczos]
"""
Lanczos resizer that applies scaling in two passes.
"""


class GrainFactoryBicubic(BicubicAuto):
    """
    Bicubic scaler originally implemented in GrainFactory with a sharp parameter.
    """

    def __init__(self, sharp: float = 50, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional arguments.

        Args:
            sharp: Sharpness of the scaler. Defaults to 50 which corresponds to Catrom scaling.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(sharp / -50 + 1, None, **kwargs)


class AbstractGrainer(ABC):
    """
    Abstract grainer base class.
    """

    @abstractmethod
    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode | GrainerPartial: ...


class Grainer(AbstractGrainer, CustomEnum, metaclass=EnumABCMeta):
    """
    Enum representing different grain/noise generation algorithms.
    """

    GAUSS = 0
    """
    Gaussian noise. Built-in [vs-noise](https://github.com/wwww-wwww/vs-noise) plugin.
    """

    PERLIN = 1
    """
    Perlin noise. Built-in [vs-noise](https://github.com/wwww-wwww/vs-noise) plugin.
    """

    SIMPLEX = 2
    """
    Simplex noise. Built-in [vs-noise](https://github.com/wwww-wwww/vs-noise) plugin.
    """

    FBM_SIMPLEX = 3
    """
    Fractional Brownian Motion based on Simplex noise.
    Built-in [vs-noise](https://github.com/wwww-wwww/vs-noise) plugin.
    """

    POISSON = 4
    """
    Poisson-distributed noise. Built-in [vs-noise](https://github.com/wwww-wwww/vs-noise) plugin.
    """

    PLACEBO = auto()
    """
    Grain effect provided by the `libplacebo` rendering library.
    """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.GAUSS, Grainer.POISSON],
        clip: vs.VideoNode,
        /,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.GAUSS, Grainer.POISSON],
        /,
        *,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> GrainerPartial: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.PERLIN, Grainer.SIMPLEX, Grainer.FBM_SIMPLEX],
        clip: vs.VideoNode,
        /,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        *,
        size: int | tuple[float | None, float | None] | None = (2.0, 2.0),
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.PERLIN, Grainer.SIMPLEX, Grainer.FBM_SIMPLEX],
        /,
        *,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        size: int | tuple[float | None, float | None] | None = (2.0, 2.0),
        **kwargs: Any,
    ) -> GrainerPartial: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.PLACEBO],
        clip: vs.VideoNode,
        /,
        strength: float | Sequence[float] = ...,
        *,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Grainer.PLACEBO],
        /,
        *,
        strength: float | Sequence[float] = ...,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> GrainerPartial: ...

    @overload
    def __call__(
        self,
        clip: vs.VideoNode,
        /,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self,
        /,
        *,
        strength: float | tuple[float, float] = ...,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> GrainerPartial: ...

    def __call__(
        self,
        clip: vs.VideoNode | MissingT = MISSING,
        /,
        strength: float | Sequence[float] = 0,
        static: bool = False,
        scale: float | tuple[float, float] = 1.0,
        scaler: ScalerLike = LanczosTwoPasses,
        temporal: float | tuple[float, int] = (0.0, 0),
        post_process: Callable[[vs.VideoNode], vs.VideoNode]
        | Iterable[Callable[[vs.VideoNode], vs.VideoNode]]
        | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode | GrainerPartial:
        """
        Apply grain to a clip using the selected graining method.

        If no clip is passed, a partially applied grainer with the provided arguments is returned instead.

        Example usage:
            ```py
            # For PERLIN, SIMPLEX, and FBM_SIMPLEX, it is recommended to use `size` instead of `scale`,
            # as `size` allows for direct internal customization of each grain type.
            grained = Grainer.PERLIN(clip, (1.65, 0.65), temporal=(0.25, 2), luma_scaling=4, size=3.0, seed=333)
            ```

        Args:
            clip:
                The input clip to apply grain to. If omitted, returns a partially applied grainer.
            strength:
                Grain strength.
                A single float applies uniform strength to all planes.
                A sequence allows per-plane control.

            static:
                If True, the grain pattern is static (unchanging across frames).
            scale:
                Scaling divisor for the grain layer.
                Can be a float (uniform scaling) or a tuple (width, height scaling).
            scaler:
                Scaler used to resize the grain layer when `scale` is not 1.0.
            temporal:
                Temporal grain smoothing parameters. Either a float (weight) or a tuple of (weight, radius).
            post_process:
                One or more functions applied after grain generation (and temporal smoothing, if used).
            protect_edges:
                Protects edge regions of each plane from graining.

                   - True: Use legal range based on clip format.
                   - False: Disable edge protection.
                   - Tuple: Specify custom edge limits per plane (see [EdgeLimits][vsdeband.noise.EdgeLimits]).

            protect_neutral_chroma:
                Whether to disable graining on neutral chroma.
            luma_scaling:
                Sensitivity of the luma-adaptive graining mask.
                Higher values reduce grain in brighter areas; negative values invert behavior.
            planes:
                Which planes to process. Default to all.
            **kwargs:
                Additional arguments to pass to the graining function or additional advanced options:

                   - ``temporal_avg_func``: Temporal average function to use instead of the default standard mean.
                   - ``protect_edges_blend``: Blend range (float) to soften edge protection thresholds.
                   - ``protect_neutral_chroma_blend``: Blend range (float) for neutral chroma protection.
                   - ``neutral_out``: (Boolean) Output the neutral layer instead of the merged clip.

        Returns:
            Grained video clip, or a [GrainerPartial][vsdeband.noise.GrainerPartial] if `clip` is not provided.
        """
        kwargs.update(
            strength=strength,
            scale=scale,
            scaler=scaler,
            temporal=temporal,
            protect_edges=protect_edges,
            post_process=post_process,
            protect_neutral_chroma=protect_neutral_chroma,
            luma_scaling=luma_scaling,
            planes=planes,
        )

        if clip is MISSING:
            return GrainerPartial(self, **kwargs)

        if self == Grainer.PLACEBO:
            assert static is False, "PlaceboGrain does not support static noise!"

            return _apply_grainer(
                clip,
                lambda clip, strength, planes, **kwds: placebo_deband(
                    clip, 8, 0.0, strength, planes, iterations=1, **kwds
                ),
                **kwargs,
                func=self.name,
            )

        if not isinstance(size := kwargs.pop("size", (None, None)), tuple):
            size = (size, size)

        kwargs.update(xsize=size[0], ysize=size[1])

        def _noise_function(
            clip: vs.VideoNode, strength: float | Sequence[float], planes: Planes, **kwds: Any
        ) -> vs.VideoNode:
            strength = normalize_param_planes(clip, strength, planes, 0)

            if len(set(strength[1:])) != 1:
                raise CustomValueError("Inconsistent grain values on chroma planes.", self.name, strength[1:])

            return core.noise.Add(clip, strength[0], strength[1], type=self.value, constant=static, **kwds)

        return _apply_grainer(clip, _noise_function, **kwargs, func=self.name)

    @staticmethod
    def norm_brightness() -> Callable[[vs.VideoNode], vs.VideoNode]:
        """
        Normalize the brightness of the grained clip to match the original clip's average luminance.

        Designed for use in the `post_process` parameter of [Grainer()][vsdeband.Grainer.__call__].

        Returns:
            A function that takes a grained clip and returns a brightness-normalized version.
        """

        def _funtion(grained: vs.VideoNode) -> vs.VideoNode:
            for i in range(grained.format.num_planes):
                grained = core.std.PlaneStats(grained, plane=i, prop=f"PS{i}")

            if grained.format.sample_type is vs.FLOAT:
                expr = "x x.PS{plane_idx}Average -"
            else:
                expr = "x neutral range_size / x.PS{plane_idx}Average - range_size * +"

            return norm_expr(grained, expr, func=Grainer.norm_brightness)

        return _funtion


def _apply_grainer(
    clip: vs.VideoNode,
    grainer_function: _GrainerFunc,
    strength: float | Sequence[float],
    scale: float | tuple[float, float],
    scaler: ScalerLike,
    temporal: float | tuple[float, int],
    protect_edges: bool | EdgeLimits,
    post_process: Callable[[vs.VideoNode], vs.VideoNode] | Iterable[Callable[[vs.VideoNode], vs.VideoNode]] | None,
    protect_neutral_chroma: bool | None,
    luma_scaling: float | None,
    planes: Planes,
    func: FuncExcept,
    **kwargs: Any,
) -> vs.VideoNode:
    # Normalize params
    planes = normalize_planes(clip, planes)
    scale = scale if isinstance(scale, tuple) else (scale, scale)
    scaler = Scaler.ensure_obj(scaler, func)
    temporal_avg, temporal_rad = temporal if isinstance(temporal, tuple) else (temporal, 1)
    temporal_avg_func = kwargs.pop("temporal_avg_func", BlurMatrix.MEAN(temporal_rad, mode=ConvMode.TEMPORAL))
    protect_neutral_chroma = (
        (clip.format.color_family is vs.YUV)
        if protect_neutral_chroma is None and any(p in planes for p in [1, 2])
        else protect_neutral_chroma
    )
    protect_edges = protect_edges if isinstance(protect_edges, tuple) else (protect_edges, protect_edges)
    protect_edges_blend = kwargs.pop("protect_edges_blend", 0.0)
    protect_neutral_chroma_blend = kwargs.pop("protect_neutral_chroma_blend", scale_delta(2, 8, clip))
    neutral_out = kwargs.pop("neutral_out", False)

    # Making a neutral blank clip
    base_clip = clip.std.BlankClip(
        mod_x(clip.width / scale[0], 2**clip.format.subsampling_w),
        mod_x(clip.height / scale[1], 2**clip.format.subsampling_h),
        length=clip.num_frames + temporal_rad * 2,
        color=get_neutral_values(clip),
        keep=True,
    )

    if not planes:
        return clip if not neutral_out else base_clip[temporal_rad:-temporal_rad]

    # Applying grain
    grained = grainer_function(base_clip, strength, planes, **kwargs)

    # Scaling up if needed
    if (base_clip.width, base_clip.height) != (clip.width, clip.height):
        grained = scaler.scale(grained, clip.width, clip.height)

    # Temporal average if radius > 0
    if temporal_rad > 0:
        average = temporal_avg_func(grained, planes)
        grained = core.std.Merge(grained, average, normalize_param_planes(grained, temporal_avg, planes, 0))[
            temporal_rad:-temporal_rad
        ]

    # Protect edges eg. excluding grain outside of the legal limited ranges
    if protect_edges != (False, False):
        lo, hi = protect_edges

        if lo is True:
            lo = get_lowest_values(clip, ColorRange.from_video(clip))
        elif lo is False:
            lo = get_lowest_values(clip, ColorRange.FULL)

        if hi is True:
            hi = get_peak_values(clip, ColorRange.from_video(clip))
        elif hi is False:
            hi = get_peak_values(clip, ColorRange.FULL)

        grained = _protect_pixel_range(clip, grained, to_arr(lo), to_arr(hi), protect_edges_blend, planes, func)

    # Postprocess
    if post_process:
        if callable(post_process):
            grained = post_process(grained)
        else:
            for pp in post_process:
                grained = pp(grained)

    if protect_neutral_chroma or luma_scaling is not None:
        base_clip = clip.std.BlankClip(length=clip.num_frames, color=get_neutral_values(clip), keep=True)

        if protect_neutral_chroma:
            grained = _protect_neutral_chroma(clip, grained, base_clip, protect_neutral_chroma_blend, planes, func)

        if luma_scaling is not None:
            from vsmasktools import adg_mask

            grained = core.std.MaskedMerge(base_clip, grained, adg_mask(clip, luma_scaling), planes)

    return core.std.MergeDiff(clip, grained, planes) if not neutral_out else grained


def _protect_pixel_range(
    clip: vs.VideoNode,
    grained: vs.VideoNode,
    low: list[float],
    high: list[float],
    blend: float = 0.0,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    if not blend:
        expr = "y neutral - abs A! x A@ - {lo} < x A@ + {hi} > or neutral y ?"
    else:
        expr = (
            "y neutral - N! N@ abs A! "
            "x A@ - range_min - {lo}      - {blend} / "
            "x A@ + range_min + {hi} swap - {blend} / "
            "min 0 1 clamp "
            "N@ * neutral + "
        )

    return norm_expr([clip, grained], expr, planes, func=func, lo=low, hi=high, blend=blend)


def _protect_neutral_chroma(
    clip: vs.VideoNode,
    grained: vs.VideoNode,
    base_clip: vs.VideoNode,
    blend: float,
    planes: list[int],
    func: FuncExcept | None,
) -> vs.VideoNode:
    match clip.format.color_family:
        case vs.YUV:
            if not blend:
                expr = "x neutral = y neutral = and mask_max 0 ?"
            else:
                expr = "x neutral - abs {blend} / 1 min 1 swap - y neutral - abs {blend} / 1 min 1 swap - * mask_max *"

            mask = norm_expr([get_u(clip), get_v(clip)], expr, func=func, blend=blend)

            return core.std.MaskedMerge(
                grained,
                base_clip,
                core.std.ShufflePlanes([clip, mask, mask], [0, 0, 0], vs.YUV, clip),
                {1, 2}.intersection(planes),
            )
        case vs.RGB:
            return core.std.MaskedMerge(
                grained, base_clip, norm_expr(split(clip), "x y = x z = and mask_max *", planes, func=func)
            )

    raise UnsupportedColorFamilyError(
        func, clip, (vs.YUV, vs.RGB), "Can't use `protect_neutral_chroma=True` when input clip is {wrong}"
    )


class GrainerPartial(AbstractGrainer):
    """
    A partially-applied grainer wrapper.
    """

    def __init__(self, grainer: Grainer, **kwargs: Any) -> None:
        """
        Stores a grainer function, allowing it to be reused with different clips.

        Args:
            grainer: [Grainer][vsdeband.noise.Grainer] enumeration.
            **kwargs: Arguments for the specified grainer.
        """
        self._grainer = grainer
        self._kwargs = kwargs

    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode:
        """
        Apply the grainer to the given clip with optional argument overrides.

        Args:
            clip: Clip to be processed.
            **kwargs: Additional keyword arguments to override or extend the stored ones.

        Returns:
            Processed clip.
        """
        return self._grainer(clip, **self._kwargs | kwargs)


type GrainerLike = Grainer | GrainerPartial
"""
Grainer-like type, which can be a single grainer or a partial grainer.
"""
