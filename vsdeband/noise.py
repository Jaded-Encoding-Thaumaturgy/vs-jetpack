from __future__ import annotations

from enum import auto
from typing import Any, Callable, ClassVar, Iterable, Literal, Protocol, Sequence, TypeAlias, TypeVar, overload

from jetpytools import MISSING, CustomEnum, FuncExceptT, MissingT, fallback, inject_self
from typing_extensions import deprecated

from vsexprtools import norm_expr
from vskernels import BaseScalerSpecializer, BicubicAuto, Lanczos, LeftShift, Scaler, ScalerLike, TopShift
from vsmasktools import adg_mask
from vsrgtools import BlurMatrix
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    ConvMode,
    PlanesT,
    UnsupportedColorFamilyError,
    check_variable,
    core,
    get_lowest_values,
    get_neutral_values,
    get_peak_values,
    get_u,
    get_v,
    mod_x,
    normalize_param_planes,
    normalize_seq,
    scale_delta,
    split,
    to_arr,
    vs,
)

from .debanders import placebo_deband

__all__ = [
    "AddNoise",
    "GrainFactoryBicubic",
    "Grainer",
    "LanczosTwoPasses",
    "ScalerTwoPasses",
]


EdgeLimits = tuple[float | Sequence[float] | bool, float | Sequence[float] | bool]
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
        self, clip: vs.VideoNode, strength: Sequence[float], planes: PlanesT, **kwargs: Any
    ) -> vs.VideoNode: ...


class _PostProcessFunc(Protocol):
    """
    Protocol for a post-processing function applied after graining.
    """

    def __call__(self, grained: vs.VideoNode) -> vs.VideoNode: ...


_ScalerT = TypeVar("_ScalerT", bound=Scaler)


class ScalerTwoPasses(BaseScalerSpecializer[_ScalerT], Scaler, partial_abstract=True):
    """
    Abstract scaler class that applies scaling in two passes.
    """

    _default_scaler = Lanczos

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode | ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        if width / clip.width > 1.5 or height / clip.height > 1.5:
            # If the scale is too big, we need to scale it in two passes, else the window
            # will be too big and the grain will be dampened down too much
            mod = max(clip.format.subsampling_w, clip.format.subsampling_h) << 1
            clip = super().scale(
                clip, mod_x((width + clip.width) / 2, mod), mod_x((height + clip.height) / 2, mod), **kwargs
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


class AbstractGrainer:
    """
    Abstract grainer base class.
    """

    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode | GrainerPartial:
        raise NotImplementedError


class Grainer(AbstractGrainer, CustomEnum):
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
        post_process: _PostProcessFunc | Iterable[_PostProcessFunc] | None = None,
        protect_edges: bool | EdgeLimits = True,
        protect_neutral_chroma: bool | None = None,
        luma_scaling: float | None = None,
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
                   - Tuple: Specify custom edge limits per plane (see `EdgeLimits`).

            protect_neutral_chroma:
                Whether to disable graining on neutral chroma.
            luma_scaling:
                Sensitivity of the luma-adaptive graining mask.
                Higher values reduce grain in brighter areas; negative values invert behavior.
            **kwargs:
                Additional arguments to pass to the graining function or additional advanced options:

                   - ``temporal_avg_func``: Temporal average function to use instead of the default standard mean.
                   - ``protect_edges_blend``: Blend range (float) to soften edge protection thresholds.
                   - ``protect_neutral_chroma_blend``: Blend range (float) for neutral chroma protection.
                   - ``neutral_out``: (Boolean) Output the neutral layer instead of the merged clip.

        Returns:
            Grained video clip, or a `GrainerPartial` if `clip` is not provided.
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
        )

        if clip is MISSING:
            return GrainerPartial(self, **kwargs)

        assert check_variable(clip, self.name)

        if self == Grainer.PLACEBO:
            assert static is False, "PlaceboGrain does not support static noise!"

            grained = _apply_grainer(
                clip,
                lambda clip, strength, planes, **kwds: placebo_deband(
                    clip, 8, 0.0, strength, planes, iterations=1, **kwds
                ),
                **kwargs,
                func=self.name,
            )
        else:
            if not isinstance(size := kwargs.pop("size", (None, None)), tuple):
                size = (size, size)

            kwargs.update(xsize=size[0], ysize=size[1])

            grained = _apply_grainer(
                clip,
                lambda clip, strength, planes, **kwds: core.noise.Add(
                    clip,
                    *strength[:2],  # type: ignore[misc]
                    type=self.value,
                    constant=static,
                    **kwds,
                ),
                **kwargs,
                func=self.name,
            )

        return grained


def _apply_grainer(
    clip: ConstantFormatVideoNode,
    grainer_function: _GrainerFunc,
    strength: float | Sequence[float],
    scale: float | tuple[float, float],
    scaler: ScalerLike,
    temporal: float | tuple[float, int],
    protect_edges: bool | EdgeLimits,
    post_process: Callable[..., vs.VideoNode] | Iterable[Callable[..., vs.VideoNode]] | None,
    protect_neutral_chroma: bool | None,
    luma_scaling: float | None,
    func: FuncExceptT,
    **kwargs: Any,
) -> vs.VideoNode:
    # Normalize params
    strength = normalize_seq(strength, clip.format.num_planes)
    scale = scale if isinstance(scale, tuple) else (scale, scale)
    scaler = Scaler.ensure_obj(scaler, func)
    temporal_avg, temporal_rad = temporal if isinstance(temporal, tuple) else (temporal, 1)
    temporal_avg_func = kwargs.pop("temporal_avg_func", BlurMatrix.MEAN(temporal_rad, mode=ConvMode.TEMPORAL))
    protect_neutral_chroma = (
        (clip.format.color_family is vs.YUV) if protect_neutral_chroma is None else protect_neutral_chroma
    )
    protect_edges = protect_edges if isinstance(protect_edges, tuple) else (protect_edges, protect_edges)
    protect_edges_blend = kwargs.pop("protect_edges_blend", 0.0)
    protect_neutral_chroma_blend = kwargs.pop("protect_neutral_chroma_blend", scale_delta(2, 8, clip))
    neutral_out = kwargs.pop("neutral_out", False)

    planes = [i for i, s in zip(range(clip.format.num_planes), strength) if s]
    (scalex, scaley), mod = scale, max(clip.format.subsampling_w, clip.format.subsampling_h) << 1

    if not planes:
        return clip

    # Making a neutral blank clip
    base_clip = clip.std.BlankClip(
        mod_x(clip.width / scalex, mod),
        mod_x(clip.height / scaley, mod),
        length=clip.num_frames + temporal_rad * 2,
        color=get_neutral_values(clip),
        keep=True,
    )
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

        grained = _protect_pixel_range(clip, grained, to_arr(lo), to_arr(hi), protect_edges_blend, func)

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
            grained = _protect_neutral_chroma(clip, grained, base_clip, protect_neutral_chroma_blend, func)

        if luma_scaling is not None:
            grained = core.std.MaskedMerge(base_clip, grained, adg_mask(clip, luma_scaling), planes)

    return core.std.MergeDiff(clip, grained, planes) if not neutral_out else grained


def _protect_pixel_range(
    clip: ConstantFormatVideoNode,
    grained: vs.VideoNode,
    low: list[float],
    high: list[float],
    blend: float = 0.0,
    func: FuncExceptT | None = None,
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

    return norm_expr([clip, grained], expr, func=func, lo=low, hi=high, blend=blend)


def _protect_neutral_chroma(
    clip: ConstantFormatVideoNode,
    grained: vs.VideoNode,
    base_clip: vs.VideoNode,
    blend: float = 0.0,
    func: FuncExceptT | None = None,
) -> vs.VideoNode:
    match clip.format.color_family:
        case vs.YUV:
            if not blend:
                expr = "x neutral = y neutral = and range_max 0 ?"
            else:
                expr = "x neutral - abs {blend} / 1 min 1 swap - y neutral - abs {blend} / 1 min 1 swap - * range_max *"

            mask = norm_expr([get_u(clip), get_v(clip)], expr, func=func, blend=blend)

            return core.std.MaskedMerge(
                grained, base_clip, core.std.ShufflePlanes([clip, mask, mask], [0, 0, 0], vs.YUV, clip), [1, 2]
            )
        case vs.RGB:
            return core.std.MaskedMerge(
                grained, base_clip, norm_expr(split(clip), "x y = x z = and range_max *", func=func)
            )
        case _:
            raise UnsupportedColorFamilyError("Can't use `protect_neutral_chroma=True` when input clip is GRAY", func)


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
        self.grainer = grainer
        self.kwargs = kwargs

    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode:
        """
        Apply the grainer to the given clip with optional argument overrides.

        Args:
            clip: Clip to be processed.
            **kwargs: Additional keyword arguments to override or extend the stored ones.

        Returns:
            Processed clip.
        """
        return self.grainer(clip, **self.kwargs | kwargs)


GrainerLike: TypeAlias = Grainer | GrainerPartial
"""
Grainer-like type, which can be a single grainer or a partial grainer.
"""


class AddNoiseBase:
    _noise_type: ClassVar[int]

    def __init__(
        self,
        strength: float | tuple[float, float] = 0.25,
        size: float | tuple[float, float] = (1.0, 1.0),
        sharp: float | ScalerLike = Lanczos,
        dynamic: bool = True,
        temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: Any | None = None,
        protect_chroma: bool = True,
        luma_scaling: float | None = None,
        fade_limits: bool | Any = True,
        *,
        matrix: Any | None = None,
        kernel: Any = None,
        neutral_out: bool = False,
        **kwargs: Any,
    ) -> None:
        self.strength = strength
        self.size = size
        self.sharp = sharp
        self.dynamic = dynamic
        self.temporal_average = temporal_average
        self.postprocess = postprocess
        self.protect_chroma = protect_chroma
        self.luma_scaling = luma_scaling
        self.fade_limits = fade_limits
        self.neutral_out = neutral_out
        self.kwargs = kwargs

    @inject_self
    def grain(
        self,
        clip: vs.VideoNode,
        strength: float | tuple[float, float] | None = None,
        dynamic: bool | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        scaler = (
            GrainFactoryBicubic(self.sharp + 50)
            if isinstance(self.sharp, (float, int))
            else ScalerTwoPasses[Scaler.from_param(self.sharp)]  # type: ignore[misc]
        )
        return Grainer(self._noise_type)(  # type: ignore[misc]
            clip,
            fallback(strength, self.strength),
            not fallback(dynamic, self.dynamic),
            self.size,
            scaler,  # type: ignore[arg-type]
            self.temporal_average,
            self.postprocess,
            self.fade_limits,
            self.protect_chroma,
            self.luma_scaling,
            neutral_out=self.neutral_out,
            **self.kwargs | kwargs,
        )


class AddNoise(AddNoiseBase):
    @deprecated(
        '"AddNoise.GAUSS" is deprecated and will be removed in a future version. Use Grainer.GAUSS instead.',
        category=DeprecationWarning,
    )
    class GAUSS(AddNoiseBase):
        _noise_type = 0

    @deprecated(
        '"AddNoise.PERLIN" is deprecated and will be removed in a future version. Use Grainer.PERLIN instead.',
        category=DeprecationWarning,
    )
    class PERLIN(AddNoiseBase):
        _noise_type = 1

    @deprecated(
        '"AddNoise.SIMPLEX" is deprecated and will be removed in a future version. Use Grainer.SIMPLEX instead.',
        category=DeprecationWarning,
    )
    class SIMPLEX(AddNoiseBase):
        _noise_type = 2

    @deprecated(
        '"AddNoise.FBM_SIMPLEX" is deprecated and will be removed in a future version. '
        "Use Grainer.FBM_SIMPLEX instead.",
        category=DeprecationWarning,
    )
    class FBM_SIMPLEX(AddNoiseBase):  # noqa: N801
        _noise_type = 3

    @deprecated(
        '"AddNoise.POISSON" is deprecated and will be removed in a future version. Use Grainer.POISSON instead.',
        category=DeprecationWarning,
    )
    class POISSON(AddNoiseBase):
        _noise_type = 4
