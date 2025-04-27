from __future__ import annotations

from functools import cache, cached_property
from inspect import signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast

from jetpytools import (
    CustomEnum, CustomRuntimeError, CustomStrEnum, CustomValueError, KwargsT, P, R, fallback, interleave_arr,
    normalize_seq
)

from vsexprtools import norm_expr
from vskernels import Point
from vstools import (
    ConstantFormatVideoNode, FunctionUtil, PlanesT, UnsupportedVideoFormatError, check_progressive, check_ref_clip,
    check_variable, core, depth, get_y, join, vs
)

from .prefilters import Prefilter

__all__ = [
    'wnnm',
    'bm3d'
]


def _recursive_denoise(
    clip: vs.VideoNode, func: vs.Function, ref_key: str | None,
    refine: int, merge_factor: float, planes: PlanesT, kwargs: KwargsT
) -> vs.VideoNode:
    denoised: vs.VideoNode = None  # type: ignore

    assert refine >= 0

    for i in range(refine + 1):
        if i == 0:
            prev = clip
        elif i == 1:
            prev = denoised
        else:
            prev = norm_expr([clip, prev, denoised], f'x y - {merge_factor} * z +', planes, func=_recursive_denoise)

        dkwargs = (kwargs | {ref_key: denoised}) if ref_key and denoised else kwargs

        denoised = cast(vs.VideoNode, func(prev, **dkwargs))

    return denoised


def wnnm(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, tr: int = 0, ref: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, ps_num: int = 2, ps_range: int = 4,
    residual: bool = False, adaptive_aggregation: bool = True,
    merge_factor: float = 0.1, self_refine: bool = False, planes: PlanesT = None
) -> vs.VideoNode:
    """
    Weighted Nuclear Norm Minimization Denoise algorithm.

    Block matching, which is popularized by BM3D, finds similar blocks and then stacks together in a 3-D group.
    The similarity between these blocks allows details to be preserved during denoising.

    In contrast to BM3D, which denoises the 3-D group based on frequency domain filtering,
    WNNM utilizes weighted nuclear norm minimization, a kind of low rank matrix approximation.
    Because of this, WNNM exhibits less blocking and ringing artifact compared to BM3D,
    but the computational complexity is much higher. This stage is called collaborative filtering in BM3D.

    For more information, see the `WNNM README <https://github.com/WolframRhodium/VapourSynth-WNNM>`_.

    :param clip:                    Clip to process.
    :param sigma:                   Strength of denoising, valid range is [0, +inf].
                                    If a float is passed, this strength will be applied to every plane.
                                    Values higher than 4.0 are not recommended. Recommended values are [0.35, 1.0].
                                    Default: 3.0.
    :param refine:                  The amount of iterations for iterative regularization.
                                    Default: 0.
    :param tr:                      Temporal radius. To enable spatial-only denoising, set this to 0.
                                    Higher values will rapidly increase filtering time and RAM usage.
                                    Default: 0.
    :param ref:                     Reference clip. Must be the same dimensions and format as input clip.
                                    Alternatively, a :py:class:`Prefilter` can be passed.
                                    Default: None.
    :param block_size:              The size of a block. Blocks are basic processing units.
                                    Larger blocks will take more time to process, but combined with `block_step`,
                                    may result in fewer blocks being processed overall.
                                    Valid ranges are [1, 64]. Default: 8.
    :param block_step:              Sliding step to process every next reference block.
                                    The total amount of blocks to process can be calculated with the following equation:
                                    `(width / block_step) * (height / block_step)`.
                                    Smaller values results in more reference blocks being processed.
                                    Default: 8.
    :param group_size:              Maximum number of similar blocks allowed per group (the 3rd dimension).
                                    Valid range is [1, 256]. By allowing more similar blocks to be grouped together,
                                    fewer blocks will be given to a transformed group,
                                    increasing the denoising strength.
                                    Setting this to 1 means no block matching will be performed.
                                    Default: 8.
    :param bm_range:                Length of the side of the searching neighborhood. Valid range is [0, +inf].
                                    The size of the search window is `(bm_range * 2 + 1) x (bm_range * 2 + 1)`.
                                    Larger values take more time to process, but increases the likelihood
                                    of finding similar patches.
                                    Default: 7.
    :param ps_num:                  The number of matched locations used for the predictive search.
                                    Valid ranges are [1, `group_size`].
                                    Larger values increases the possibility to match similar blocks,
                                    at the cost of taking more processing power.
                                    Default: 2.
    :param ps_range:                Length of the side of the search neighborhood for `pd_num`.
                                    Valid range is [1, +inf]. Default: 4.
    :param residual:                Whether to center blocks before performing collaborative filtering.
                                    Default: False.
    :param adaptive_aggregation:    Whether to aggregate similar blocks weighted by the inverse of the number
                                    of non-zero singular values after WNNM. Default: True.
    :param merge_factor:            Merge amount of the last recalculation into the new one
                                    when performing iterative regularization.
    :param self_refine:             If True, in the iterative recalculation step it will pass the
                                    last recalculation as ref clip instead of the original ``ref``.
    :param planes:                  Planes to process. If None, all planes. Default: None.

    :return:                        Denoised clip.
    """

    assert check_progressive(clip, wnnm)

    func = FunctionUtil(clip, wnnm, planes, bitdepth=32)

    sigma = func.norm_seq(sigma, 0)

    if isinstance(ref, Prefilter):
        ref = ref(func.work_clip, planes)
    elif ref is not None:
        ref = depth(ref, 32)
        ref = get_y(ref) if func.luma_only else ref
        check_ref_clip(func.work_clip, ref, func.func)

    return func.return_clip(
        _recursive_denoise(
            func.work_clip, core.wnnm.WNNM, self_refine and 'rclip' or None,
            refine, merge_factor, planes, dict(
                sigma=sigma, block_size=block_size, block_step=block_step, group_size=group_size,
                bm_range=bm_range, radius=tr, ps_num=ps_num, ps_range=ps_range, rclip=ref,
                adaptive_aggregation=adaptive_aggregation, residual=residual
            )
        )
    )


# TODO: remove this when vs-stubs will be a thing™
if TYPE_CHECKING:
    from vapoursynth import Function, Plugin

    class _VSFunction(Function):
        def __call__(self, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode: ...

    class _VSPlugin(Plugin):
        BM3D: _VSFunction
        BM3Dv2: _VSFunction
        VAggregate: _VSFunction
else:
    _Plugin = Any


def _clean_keywords(
    kwargs: dict[str, Any], function: _VSFunction
) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


class UnsupportedProfileError(CustomValueError):
    """Raised when an unsupported profile is passed."""


class BM3D(Generic[P, R]):
    """
    Class decorator that wraps the [bm3d][vsdenoise.blockmatch.bm3d] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, bm3d_func: Callable[P, R]) -> None:
        self._func = bm3d_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Backend(CustomEnum):
        """Enum representing the available backends for running the BM3D plugin."""

        AUTO = "auto"
        """
        Automatically selects the best available backend.
        Selection priority: "CUDA_RTC" → "CUDA" → "HIP" → "SYCL" → "CPU" → "OLD".
        When the filter chain is executed within vspreview, the priority of "cuda_rtc" and "cuda" is reversed.
        """

        CUDA_RTC = "bm3dcuda_rtc"
        """GPU implementation using NVIDIA CUDA with NVRTC (runtime compilation)."""

        CUDA = "bm3dcuda"
        """GPU implementation using NVIDIA CUDA."""

        HIP = "bm3dhip"
        """GPU implementation using AMD HIP."""

        SYCL = "bm3dsycl"
        """GPU implementation using Intel SYCL."""

        CPU = "bm3dcpu"
        """Optimized CPU implementation using AVX and AVX2 intrinsics."""

        OLD = "bm3d"
        """Reference VapourSynth-BM3D implementation."""

        @cache
        def resolve(self) -> BM3D.Backend:
            """
            Resolves the appropriate BM3D backend to use based on availability and context.

            If the current instance is not BM3D.Backend.AUTO, it returns itself.
            Otherwise, it attempts to select the best available backend.

            :raises CustomRuntimeError:     If no supported BM3D implementation is available on the system.
            :return:                        The resolved BM3D.Backend to use for processing.
            """
            if self is not BM3D.Backend.AUTO:
                return self

            try:
                from vspreview.api import is_preview
            except ImportError:

                def is_preview() -> bool:
                    return False

            if is_preview() and hasattr(core, "bm3dcuda"):
                return BM3D.Backend.CUDA

            for member in list(BM3D.Backend.__members__.values())[1:]:
                if hasattr(core, member.value):
                    return BM3D.Backend(member.value)

            raise CustomRuntimeError(
                "No compatible plugin found. Please install one from: "
                "https://github.com/WolframRhodium/VapourSynth-BM3DCUDA "
                "or https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D/"
            )

        @property
        def plugin(self) -> _VSPlugin:
            """
            Returns the appropriate BM3D plugin based on the current backend.

            :return:        The corresponding BM3D plugin for the resolved backend.
            """
            return getattr(core.lazy, self.resolve().value)

    class Profile(CustomStrEnum):
        """
        Enum representing the available BM3D profiles, each with default parameter settings.

        For more detailed information on these profiles,
        refer to the original [documentation](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D#profile-default)
        """

        FAST = "fast"
        """A profile optimized for maximum processing speed."""

        LOW_COMPLEXITY = "lc"
        """A profile designed for content with low-complexity noise."""

        NORMAL = "np"
        """A neutral profile."""

        HIGH = "high"
        """A profile focused on achieving high-precision denoising."""

        VERY_NOISY = "vn"
        """A profile tailored for handling very noisy content."""

        @cached_property
        def config(self) -> MappingProxyType[str, MappingProxyType[str, MappingProxyType[str, Any]]]:
            """
            Retrieves the configuration for each BM3D profile.
            """

            def freeze_dict(d: dict[str, Any]) -> Any:
                """Recursively convert all dictionaries into MappingProxyType."""
                return MappingProxyType({
                    k: freeze_dict(v) if isinstance(v, dict) else v
                    for k, v in d.items()
                })

            config = {
                BM3D.Profile.FAST: {
                    "basic": {
                        "spatial": {
                            "block_step": 8,
                            "bm_range": 9,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 8,
                        },
                        "temporal": {
                            "bm_range": 7,
                            "radius": 1,
                            "ps_range": 4,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                    "final": {
                        "spatial": {
                            "block_step": 7,
                            "bm_range": 9,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 8,
                        },
                        "temporal": {
                            "bm_range": 7,
                            "radius": 1,
                            "ps_range": 5,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                },
                BM3D.Profile.LOW_COMPLEXITY: {
                    "basic": {
                        "spatial": {
                            "block_step": 6,
                            "bm_range": 9,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 16,
                        },
                        "temporal": {
                            "bm_range": 9,
                            "radius": 2,
                            "ps_range": 4,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                    "final": {
                        "spatial": {
                            "block_step": 5,
                            "bm_range": 9,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 16,
                        },
                        "temporal": {
                            "bm_range": 9,
                            "radius": 2,
                            "ps_range": 5,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                },
                BM3D.Profile.NORMAL: {
                    "basic": {
                        "spatial": {
                            "block_step": 4,
                            "bm_range": 16,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 16,
                        },
                        "temporal": {
                            "bm_range": 12,
                            "radius": 3,
                            "ps_range": 5,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                    "final": {
                        "spatial": {
                            "block_step": 3,
                            "bm_range": 16,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 32,
                        },
                        "temporal": {
                            "bm_range": 12,
                            "radius": 3,
                            "ps_range": 6,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                },
                BM3D.Profile.HIGH: {
                    "basic": {
                        "spatial": {
                            "block_step": 3,
                            "bm_range": 16,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 16,
                        },
                        "temporal": {
                            "bm_range": 16,
                            "radius": 4,
                            "ps_range": 7,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                    "final": {
                        "spatial": {
                            "block_step": 2,
                            "bm_range": 16,
                            # Only available in OLD
                            "block_size": 8,
                            "group_size": 32,
                        },
                        "temporal": {
                            "bm_range": 16,
                            "radius": 4,
                            "ps_range": 8,
                            # Only available in OLD
                            "group_size": 8,
                        },
                    },
                },
                # Not available in wolfram implementation
                BM3D.Profile.VERY_NOISY: {
                    "basic": {
                        "spatial": {
                            "block_step": 4,
                            "bm_range": 16,
                            "block_size": 8,
                            "group_size": 32,
                        },
                        "temporal": {
                            "bm_range": 12,
                            "radius": 4,
                            "ps_range": 5,
                            # Only available in OLD
                            "group_size": 16,
                        },
                    },
                    "final": {
                        "spatial": {
                            "block_step": 6,
                            "bm_range": 16,
                            "block_size": 11,
                            "group_size": 32,
                        },
                        "temporal": {
                            "bm_range": 12,
                            "radius": 4,
                            "ps_range": 6,
                            # Only available in OLD
                            "group_size": 16,
                        },
                    },
                },
            }
            return freeze_dict(config[self])

        def _get_args(self, radius: int | None, estimate_step: str) -> dict[str, Any]:
            config = self.config[estimate_step]

            args = config["spatial"].copy()

            if radius is None or radius > 0:
                args.update(config["temporal"])

                if radius:
                    args.update(radius=radius)

            return args

        def basic_args(self, radius: int | None) -> dict[str, Any]:
            """
            Retrieves the arguments for the basic estimate step based on the specified radius.

            :param radius:  The temporal radius for denoising. If None, a default value is used.
            :return:        A dictionary of arguments for the basic denoising step.
            """
            return self._get_args(radius, "basic")

        def final_args(self, radius: int | None) -> dict[str, Any]:
            """
            Retrieves the arguments for the final estimate step based on the specified radius.

            :param radius:  The temporal radius for denoising. If None, a default value is used.
            :return:        A dictionary of arguments for the final denoising step.
            """
            return self._get_args(radius, "final")

    matrix_rgb2opp: list[float] = [
        1 / 3, 1 / 3, 1 / 3,
        1 / 2, 0, -1 / 2,
        1 / 4, -1 / 2, 1 / 4,
    ]
    """
    Matrix to convert RGB color space to OPP (Opponent) color space.
    """

    matrix_opp2rgb: list[float] = [
        1, 1, 2 / 3,
        1, 0, -4 / 3,
        1, -1, 2 / 3
    ]
    """
    Matrix to convert OPP (Opponent) color space back to RGB color space.
    """


@BM3D
def bm3d(
    clip: vs.VideoNode,
    sigma: float | Sequence[float] = 0.5,
    tr: int | Sequence[int | None] | None = None,
    refine: int = 1,
    profile: BM3D.Profile = BM3D.Profile.FAST,
    pre: vs.VideoNode | None = None,
    ref: vs.VideoNode | None = None,
    backend: BM3D.Backend = BM3D.Backend.AUTO,
    basic_args: dict[str, Any] | None = None,
    final_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    """
    Block-Matching and 3D filtering (BM3D) is a state-of-the-art algorithm for image denoising.

    More information at:
        - https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D/
        - https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    Example:
        ```py
        denoised = bm3d(clip, 1.25, 1, backend=bm3d.Backend.CUDA_RTC, ...)
        ```

    :param clip:                            The clip to process.
                                            If using BM3D.Backend.OLD, the clip format must be YUV444 or RGB,
                                            as filtering is always performed in the OPPonent color space.
                                            If using another device type and the clip format is:
                                                - RGB       -> Processed in OPP format (BM3D algorithm, aka `chroma=False`).
                                                - YUV444    -> Processed in YUV444 format (CBM3D algorithm, aka `chroma=True`).
                                                - GRAY      -> Processed as-is.
                                                - YUVXXX    -> Each plane is processed separately.

    :param sigma:                           Strength of denoising. Valid range is [0, +inf).
                                            A sequence of up to 3 elements can be used to set different sigma values
                                            for the Y, U, and V channels.
                                            If fewer than 3 elements are given, the last value is repeated.
                                            Defaults to 0.5.

    :param tr:                              The temporal radius for denoising. Valid range is [1, 16].
                                            Defaults to the radius defined by the profile.

    :param refine:                          Number of refinement steps.
                                                * 0 means basic estimate only.
                                                * 1 means basic estimate with one final estimate.
                                                * n means basic estimate refined with a final estimate n times.

    :param profile:                         The preset profile. Defaults to BM3D.Profile.FAST.

    :param pre:                             A pre-filtered clip for the basic estimate.
                                            It should be more suitable for block-matching than the input clip,
                                            and must be of the same format and dimensions.
                                            Either `pre` or `ref` can be specified, not both.
                                            Defaults to None.

    :param ref:                             A clip to be used as the basic estimate.
                                            It replaces BM3D’s internal basic estimate and serves as the reference
                                            for the final estimate.
                                            Must be of the same format and dimensions as the input clip.
                                            Either `ref` or `pre` can be specified, not both.
                                            Defaults to None.

    :param backend:                         The backend to use for processing. Defaults to BM3D.Backend.AUTO.

    :param basic_args:                      Additional arguments to pass to the basic estimate step.
                                            Defaults to None.

    :param final_args:                      Additional arguments to pass to the final estimate step.
                                            Defaults to None.

    :param **kwargs:                        Internal keyword arguments for testing purposes.

    :raises CustomValueError:               If both `pre` and `ref` are specified at the same time.
    :raises UnsupportedProfileError:        If the VERY_NOISY profile is not supported by the selected device type.
    :raises UnsupportedVideoFormatError:    If the video format is not supported when using BM3D.Backend.OLD.

    :return:                                Denoised clip.
    """

    func = kwargs.pop("func", None) or bm3d

    if ref and pre:
        raise CustomValueError("You cannot specify both 'pre' and 'ref' at the same time.", func)

    if pre is not None:
        pre = check_ref_clip(clip, pre, func)

    if ref is not None:
        ref = check_ref_clip(clip, ref, func)

    radius_basic, radius_final = normalize_seq(tr, 2)
    nsigma = normalize_seq(sigma, 3)

    backend = backend.resolve()
    nbasic_args = fallback(basic_args, KwargsT())
    nfinal_args = fallback(final_args, KwargsT())

    matrix_rgb2opp = kwargs.pop("matrix_rgb2opp", BM3D.matrix_rgb2opp)
    matrix_opp2rgb = kwargs.pop("matrix_rgb2opp", BM3D.matrix_opp2rgb)

    if backend != BM3D.Backend.OLD and profile == BM3D.Profile.VERY_NOISY:
        raise UnsupportedProfileError(
            "The VERY_NOISY profile is only supported with BM3D.Backend.OLD.", func
        )

    def _bm3d_wolfram(
        preclip: ConstantFormatVideoNode,
        pre: ConstantFormatVideoNode | None,
        ref: ConstantFormatVideoNode | None,
        chroma: bool = False,
    ) -> ConstantFormatVideoNode:
        """Internal function for WolframRhodium implementation."""

        if not ref:
            b_args = (
                _clean_keywords(profile.basic_args(radius_basic), backend.plugin.BM3Dv2)
                | nbasic_args
                | kwargs
            )
            b_args.update(chroma=chroma)

            basic = backend.plugin.BM3Dv2(preclip, pre, nsigma, **b_args)
        else:
            basic = ref

        if not refine:
            final = basic
        else:
            f_args = (
                _clean_keywords(profile.final_args(radius_final), backend.plugin.BM3Dv2)
                | nfinal_args
                | kwargs
            )
            f_args.update(chroma=chroma)

            final = basic

            for _ in range(refine):
                final = backend.plugin.BM3Dv2(preclip, final, nsigma, **f_args)

        return final

    def _bm3d_mawen(
        preclip: ConstantFormatVideoNode,
        pre: ConstantFormatVideoNode | None,
        ref: ConstantFormatVideoNode | None,
    ) -> ConstantFormatVideoNode:
        """Internal function for mawen1250 implementation."""

        preclip = core.bm3d.RGB2OPP(preclip, preclip.format.sample_type)

        if pre:
            pre = core.bm3d.RGB2OPP(pre, pre.format.sample_type)

        if ref:
            ref = core.bm3d.RGB2OPP(ref, ref.format.sample_type)

        if not ref:
            b_args = profile.basic_args(radius_basic) | nbasic_args | kwargs
            r = b_args["radius"]

            if r > 0:
                basic = core.bm3d.VBasic(
                    preclip, pre, profile, nsigma, matrix=100, **b_args
                ).bm3d.VAggregate(r, preclip.format.sample_type)
            else:
                basic = core.bm3d.Basic(
                    preclip, pre, profile, nsigma, matrix=100, **b_args
                )
        else:
            basic = ref

        if not refine:
            final = basic
        else:
            f_args = profile.final_args(radius_final) | nfinal_args | kwargs
            r = f_args["radius"]

            final = basic

            for _ in range(refine):
                if r > 0:
                    final = core.bm3d.VFinal(
                        preclip, final, profile, nsigma, matrix=100, **f_args
                    ).bm3d.VAggregate(r, preclip.format.sample_type)
                else:
                    final = core.bm3d.Final(
                        preclip, final, profile, nsigma, matrix=100, **f_args
                    )

        if 0 in nsigma:
            final = join({p: preclip if s == 0 else final for p, s in zip(range(3), nsigma)}, vs.YUV)

        return core.bm3d.OPP2RGB(final, preclip.format.sample_type)

    assert check_variable(clip, func)

    if clip.format.color_family == vs.RGB:
        if backend == BM3D.Backend.OLD:
            return _bm3d_mawen(clip, pre, ref)

        coefs = list(interleave_arr(matrix_rgb2opp, [0, 0, 0], 3))

        clip_opp = core.fmtc.matrix(clip, coef=coefs, col_fam=vs.YUV, bits=32)
        pre_opp = core.fmtc.matrix(pre, coef=coefs, col_fam=vs.YUV, bits=32) if pre else pre
        ref_opp = core.fmtc.matrix(ref, coef=coefs, col_fam=vs.YUV, bits=32) if ref else ref

        denoised = _bm3d_wolfram(clip_opp, pre_opp, ref_opp)

        denoised = core.fmtc.matrix(
            denoised,
            coef=list(interleave_arr(matrix_opp2rgb, [0, 0, 0], 3)),
            col_fam=vs.RGB,
        )

        return depth(denoised, clip)

    preclip = depth(clip, 32)

    if clip.format.color_family == vs.YUV and {clip.format.subsampling_w, clip.format.subsampling_h} == {0}:
        if backend == BM3D.Backend.OLD:
            preclip = Point.resample(clip, clip.format.replace(color_family=vs.RGB))

            denoised = bm3d(
                preclip,
                sigma,
                (radius_basic, radius_final),
                refine,
                profile,
                pre,
                ref,
                backend,
                basic_args,
                final_args,
                **kwargs,
            )

            return Point.resample(denoised, clip, clip)

        denoised = _bm3d_wolfram(preclip, pre, ref, chroma=True)

        return depth(denoised, clip)

    if backend == BM3D.Backend.OLD:
        raise UnsupportedVideoFormatError(
            "When using `BM3D.Backend.OLD`, the input clip must be in YUV444 or RGB format.",
            func, clip.format.color_family
        )

    denoised = _bm3d_wolfram(preclip, pre, ref)

    return depth(denoised, clip)
