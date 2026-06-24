from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from logging import getLogger
from types import MappingProxyType
from typing import Any

from jetpytools import (
    CustomEnum,
    CustomRuntimeError,
    CustomStrEnum,
    CustomValueError,
    cachedproperty,
    fallback,
    interleave_arr,
    normalize_seq,
)

from vsexprtools import norm_expr
from vskernels import Point
from vstools import (
    Planes,
    Range,
    core,
    depth,
    join,
    normalize_param_planes,
    vs,
)

__all__ = ["bm3d", "wnnm"]

logger = getLogger(__name__)


def wnnm(
    clip: vs.VideoNode,
    sigma: float | Sequence[float] = 3.0,
    tr: int | None = None,
    ref: vs.VideoNode | None = None,
    block_size: int | None = None,
    block_step: int | None = None,
    group_size: int | None = None,
    bm_range: int | None = None,
    ps_num: int | None = None,
    ps_range: int | None = None,
    residual: bool | None = None,
    adaptive_aggregation: bool | None = None,
    refine: int = 0,
    merge_factor: float = 0.1,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    WNNM is a denoising algorithm based on block-matching and weighted nuclear norm minimization.

    Block matching, which is popularized by BM3D, finds similar blocks and then stacks together in a 3-D group.
    The similarity between these blocks allows details to be preserved during denoising.

    In contrast to BM3D, which denoises the 3-D group based on frequency domain filtering,
    WNNM utilizes weighted nuclear norm minimization, a kind of low rank matrix approximation.
    Because of this, WNNM exhibits less blocking and ringing artifact compared to BM3D,
    but the computational complexity is much higher. This stage is called collaborative filtering in BM3D.

    For more information, see the [WNNM README](https://github.com/WolframRhodium/VapourSynth-WNNM).

    Args:
        clip: The input clip. Must be of 32 bit float format. Each plane is denoised separately.
        sigma: Denoising strength of each plane. Larger values remove more noise but may smooth fine details.
            Accepts either a single float (applied to all planes) or a per-plane sequence.
            The valid range is [0, +inf), though practical values usually fall between **0.35 and 1.0**.
            Values above 4.0 are rarely useful.
        tr: The temporal radius for denoising, valid range [1, 16].
            For each processed frame, (radius * 2 + 1) frames will be requested,
            Increasing radius only increases tiny computational cost in block-matching and aggregation,
            and will not affect collaborative filtering, but the memory consumption can grow quadratically.
            Thus, feel free to use large radius as long as your RAM is large enough.
        ref: Reference clip for block matching. Must be of the same dimensions and format as `clip`.
        block_size: The size of a block is block_size x block_size (the 1st and the 2nd dimension), valid range [1,64].
            A block is the basic processing unit of WNNM, representing a local patch.
            Generally, larger block will be slower, especially in the DCT/IDCT part. While at the same time,
            larger block_size allows you to set larger block_step, resulting in less block to be processed.
            8 is a well-balanced value, both for quality and speed.
        block_step: Sliding step to process every next reference block, valid range [1,block_size].
            Total number of reference blocks to be processed can be calculated approximately by
            (width / block_step) * (height / block_step).
            Smaller step results in processing more reference blocks, and is slower.
        group_size: Maximum number of similar blocks in each group (the 3rd dimension), valid range [1,256].
            Larger value allows more blocks in a single group. Thus, the sparsity in a transformed group raises,
            the filtering will be stronger, and also slower in the DCT/IDCT part.
            When set to 1, no block-matching will be performed and each group only consists of the
            reference block.
        bm_range: Length of the side of the search neighborhood for block-matching, valid range [1, +inf).
            The size of search window is (bm_range * 2 + 1) x (bm_range * 2 + 1).
            Larger is slower, with more chances to find similar patches.
        ps_num: The number of matched locations used for predictive search, valid range [1, group_size].
            Larger value increases the possibility to match more similar blocks,
            with tiny increasing in computational cost.
        ps_range: Length of the side of the search neighborhood for predictive-search block-matching,
            valid range [1, +inf)
        residual: Whether to center blocks before collaborative filtering. Default: False.
        adaptive_aggregation: Whether to aggregate blocks adaptively. Default: True.
        refine: Number of additional refinement iterations to perform.

            A value of 0 corresponds to a single WNNM pass (equivalent to `num_iterations=1`
            in the original implementation).
            Each increment adds another iterative regularization step using the previously denoised result.

            Valid range is [0, +inf).
        merge_factor: Blend factor for merging the last and current iteration during iterative regularization.
        planes: Which planes to process. Default to all.

    Returns:
        Denoised clip.
    """

    sigma = normalize_param_planes(clip, sigma, planes, 0)
    kwargs = dict[str, Any](
        sigma=sigma,
        block_size=block_size,
        block_step=block_step,
        group_size=group_size,
        bm_range=bm_range,
        radius=tr,
        ps_num=ps_num,
        ps_range=ps_range,
        residual=residual,
        adaptive_aggregation=adaptive_aggregation,
        rclip=ref,
    )

    previous = clip
    denoised = core.wnnm.WNNM(clip, **kwargs)

    for i in range(refine):
        if i == 0:
            previous = denoised
        else:
            previous = norm_expr([clip, previous, denoised], f"x y - {merge_factor} * z +", planes, func=wnnm)

        denoised = core.wnnm.WNNM(previous, **kwargs)

    return denoised


def _clean_keywords(kwargs: dict[str, Any], function: vs.Function) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in function.__signature__.parameters}


class UnsupportedProfileError(CustomValueError):
    """
    Raised when an unsupported profile is passed.
    """


class BM3D[**P, R]:
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
        """
        Enum representing the available backends for running the BM3D plugin.
        """

        AUTO = "auto"
        """
        Automatically selects the best available backend.

        Selection priority: "CUDA_RTC" → "CUDA" → "HIP" → "SYCL" → "METAL" → "CPU" → "OLD".
        """

        CUDA_RTC = "bm3dcuda_rtc"
        """
        GPU implementation using NVIDIA CUDA with NVRTC (runtime compilation).
        """

        CUDA = "bm3dcuda"
        """
        GPU implementation using NVIDIA CUDA.
        """

        HIP = "bm3dhip"
        """
        GPU implementation using AMD HIP.
        """

        SYCL = "bm3dsycl"
        """
        GPU implementation using Intel SYCL.
        """

        METAL = "bm3dmetal"
        """
        GPU implementation using Apple Metal.
        """

        CPU = "bm3dcpu"
        """
        Optimized CPU implementation using AVX and AVX2 intrinsics.
        """

        OLD = "bm3d"
        """
        Reference VapourSynth-BM3D implementation.
        """

        @cache
        def resolve(self) -> BM3D.Backend:
            """
            Resolves the appropriate BM3D backend to use.

            If the current instance is not [AUTO][vsdenoise.blockmatch.BM3D.Backend.AUTO], it returns itself.
            Otherwise, it attempts to select the best available backend.

            Raises:
                CustomRuntimeError: If no supported BM3D implementation is available on the system.

            Returns:
                The resolved BM3D.Backend to use for processing.
            """
            if self is not BM3D.Backend.AUTO:
                return self

            for member in list(BM3D.Backend.__members__.values())[1:]:
                if hasattr(core, member.value):
                    backend = BM3D.Backend(member.value)
                    logger.debug("%s: Auto selecting 'BM3D.Backend.%s'", BM3D.Backend.resolve, backend.name)
                    return backend

            raise CustomRuntimeError("No available BM3D plugin found. Please install one.")

        @property
        def plugin(self) -> vs.Plugin:
            """
            Returns the appropriate BM3D plugin based on the current backend.

            Returns:
                The corresponding BM3D plugin for the resolved backend.
            """
            return getattr(core.lazy, self.resolve().value)

    class Profile(CustomStrEnum):
        """
        Enum representing the available BM3D profiles, each with default parameter settings.

        For more detailed information on these profiles,
        refer to the original [documentation](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D#profile-default).
        """

        FAST = "fast"
        """
        A profile optimized for maximum processing speed.
        """

        LOW_COMPLEXITY = "lc"
        """
        A profile designed for content with low-complexity noise.
        """

        NORMAL = "np"
        """
        A neutral profile.
        """

        HIGH = "high"
        """
        A profile focused on achieving high-precision denoising.
        """

        VERY_NOISY = "vn"
        """
        A profile tailored for handling very noisy content.
        """

        @cachedproperty
        def config(self) -> MappingProxyType[str, MappingProxyType[str, MappingProxyType[str, Any]]]:
            """
            Retrieves the configuration for each BM3D profile.
            """

            def freeze_dict(d: dict[str, Any]) -> Any:
                """
                Recursively convert all dictionaries into MappingProxyType.
                """
                return MappingProxyType({k: freeze_dict(v) if isinstance(v, dict) else v for k, v in d.items()})

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

            Args:
                radius: The temporal radius for denoising. If None, a default value is used.

            Returns:
                A dictionary of arguments for the basic denoising step.
            """
            return self._get_args(radius, "basic")

        def final_args(self, radius: int | None) -> dict[str, Any]:
            """
            Retrieves the arguments for the final estimate step based on the specified radius.

            Args:
                radius: The temporal radius for denoising. If None, a default value is used.

            Returns:
                A dictionary of arguments for the final denoising step.
            """
            return self._get_args(radius, "final")

    matrix_rgb2opp: tuple[float, ...] = (
        1 / 3,
        1 / 3,
        1 / 3,
        1 / 2,
        0,
        -1 / 2,
        1 / 4,
        -1 / 2,
        1 / 4,
    )
    """
    Matrix to convert RGB color space to OPP (Opponent) color space.
    """

    matrix_opp2rgb: tuple[float, ...] = (1, 1, 2 / 3, 1, 0, -4 / 3, 1, -1, 2 / 3)
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
    planes: Planes = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Block-Matching and 3D filtering (BM3D) is a state-of-the-art algorithm for image denoising.

    More information at:
        - https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D/
        - https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    Example:
        ```py
        denoised = bm3d(clip, 1.25, 1, profile=bm3d.Profile.NORMAL, backend=bm3d.Backend.CUDA_RTC, ...)
        ```

    Args:
        clip: The clip to process. If using BM3D.Backend.OLD, the clip format must be YUV444 or RGB, as filtering is
            always performed in the OPPonent color space. If using another device type and the clip format is:

               - RGB       -> Processed in OPP format (BM3D algorithm, aka `chroma=False`).
               - YUV444    -> Processed in YUV444 format (CBM3D algorithm, aka `chroma=True`).
               - GRAY      -> Processed as-is.
               - YUVXXX    -> Each plane is processed separately.
        sigma: Strength of denoising. Valid range is [0, +inf). A sequence of up to 3 elements can be used to set
            different sigma values for the Y, U, and V channels. If fewer than 3 elements are given, the last value is
            repeated. Defaults to 0.5.
        tr: The temporal radius for denoising. Valid range is [1, 16]. Defaults to the radius defined by the profile.
        refine: Number of refinement steps.

               * 0 means basic estimate only.
               * 1 means basic estimate with one final estimate.
               * n means basic estimate refined with a final estimate n times.
        profile: The preset profile. Defaults to BM3D.Profile.FAST.
        pre: A pre-filtered clip for the basic estimate. It should be more suitable for block-matching than the input
            clip, and must be of the same format and dimensions. Either `pre` or `ref` can be specified, not both.
            Defaults to None.
        ref: A clip to be used as the basic estimate. It replaces BM3D's internal basic estimate and serves as the
            reference for the final estimate. Must be of the same format and dimensions as the input clip. Either `ref`
            or `pre` can be specified, not both. Defaults to None.
        backend: The backend to use for processing. Defaults to BM3D.Backend.AUTO.
        basic_args: Additional arguments to pass to the basic estimate step. Defaults to None.
        final_args: Additional arguments to pass to the final estimate step. Defaults to None.
        planes: Planes to process. Default to all.
        **kwargs: Internal keyword arguments for testing purposes.

    Raises:
        CustomValueError: If both `pre` and `ref` are specified at the same time.
        UnsupportedProfileError: If the VERY_NOISY profile is not supported by the selected device type.
        UnsupportedVideoFormatError: If the video format is not supported when using BM3D.Backend.OLD.

    Returns:
        Denoised clip.
    """
    func = kwargs.pop("func", None) or bm3d

    radius_basic, radius_final = normalize_seq(tr, 2)
    nsigma = normalize_param_planes(clip, sigma, planes, 0)

    backend = backend.resolve()
    nbasic_args = fallback(basic_args, {})
    nfinal_args = fallback(final_args, {})

    matrix_rgb2opp = kwargs.pop("matrix_rgb2opp", BM3D.matrix_rgb2opp)
    matrix_opp2rgb = kwargs.pop("matrix_opp2rgb", BM3D.matrix_opp2rgb)

    plugins_args = dict[str, Any](
        nsigma=nsigma,
        refine=refine,
        profile=profile,
        radius_basic=radius_basic,
        radius_final=radius_final,
        nbasic_args=nbasic_args,
        nfinal_args=nfinal_args,
    )

    if ref and pre:
        raise CustomValueError("You cannot specify both 'pre' and 'ref' at the same time.", func)

    if backend != BM3D.Backend.OLD:
        if profile == BM3D.Profile.VERY_NOISY:
            raise UnsupportedProfileError("The VERY_NOISY profile is only supported with BM3D.Backend.OLD.", func)
        if (
            clip.format.bits_per_sample != 32
            or (ref and ref.format.bits_per_sample != 32)
            or (pre and pre.format.bits_per_sample != 32)
        ):
            raise CustomRuntimeError(f"The backend {backend} only supports 32 bit float input.")

    match backend, clip.format.color_family:
        # We can pass directly to OLD backend when it's RGB or GRAY input.
        case BM3D.Backend.OLD, vs.RGB | vs.GRAY:
            return _bm3d_mawen(clip, pre, ref, **plugins_args, **kwargs)

        # When input is YUV444, we convert to RGB and call the function again.
        # Convert back to YUV after processing.
        case BM3D.Backend.OLD, vs.YUV if clip.format.subsampling_w == clip.format.subsampling_h == 0:
            point = Point()

            denoised = bm3d(
                point.resample(clip, clip.format.replace(color_family=vs.RGB)),
                sigma,
                (radius_basic, radius_final),
                refine,
                profile,
                point.resample(pre, pre.format.replace(color_family=vs.RGB)) if pre else pre,
                point.resample(ref, ref.format.replace(color_family=vs.RGB)) if ref else ref,
                backend,
                basic_args,
                final_args,
                **kwargs,
            )

            final = point.resample(denoised, clip, clip)

            if 0 in nsigma:
                final = join({p: clip if s == 0 else final for p, s in zip(range(3), nsigma)}, vs.YUV)
            return final

        # When input is RGB, BM3D CUDA and others need manual conversion to OPP.
        # Convert back to RGB after processing.
        case _, vs.RGB:
            coefs = list(interleave_arr(matrix_rgb2opp, [0, 0, 0], 3))

            clip_opp = core.fmtc.matrix(clip, coef=coefs, col_fam=vs.YUV, bits=32)
            pre_opp = core.fmtc.matrix(pre, coef=coefs, col_fam=vs.YUV, bits=32) if pre else pre
            ref_opp = core.fmtc.matrix(ref, coef=coefs, col_fam=vs.YUV, bits=32) if ref else ref

            denoised = _bm3d_wolfram(clip_opp, pre_opp, ref_opp, **plugins_args, backend=backend, **kwargs)
            return core.fmtc.matrix(denoised, coef=list(interleave_arr(matrix_opp2rgb, [0, 0, 0], 3)), col_fam=vs.RGB)

        # YUVXXX and GRAY inputs.
        # `chroma=True` when YUV444, otherwise it's False.
        case _, _:
            return _bm3d_wolfram(
                clip,
                pre,
                ref,
                **plugins_args,
                backend=backend,
                chroma=clip.format.subsampling_w == clip.format.subsampling_h == 0
                and clip.format.color_family == vs.YUV,
                **kwargs,
            )


def _bm3d_mawen(
    preclip: vs.VideoNode,
    pre: vs.VideoNode | None,
    ref: vs.VideoNode | None,
    nsigma: list[float],
    refine: int,
    profile: BM3D.Profile,
    radius_basic: int | None,
    radius_final: int | None,
    nbasic_args: dict[str, Any],
    nfinal_args: dict[str, Any],
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Internal function for mawen1250 implementation.
    """

    preclip = (
        core.bm3d.RGB2OPP(preclip, preclip.format.sample_type)
        if preclip.format.color_family != vs.GRAY
        else depth(preclip, range_out=Range.FULL)
    )

    if pre:
        pre = (
            core.bm3d.RGB2OPP(pre, pre.format.sample_type)
            if pre.format.color_family != vs.GRAY
            else depth(pre, range_out=Range.FULL)
        )

    if ref:
        ref = (
            core.bm3d.RGB2OPP(ref, ref.format.sample_type)
            if ref.format.color_family != vs.GRAY
            else depth(ref, range_out=Range.FULL)
        )

    if not ref:
        b_args = profile.basic_args(radius_basic) | nbasic_args | kwargs
        r = b_args["radius"]

        if r > 0:
            basic = core.bm3d.VBasic(preclip, pre, profile, nsigma, matrix=100, **b_args).bm3d.VAggregate(
                r, preclip.format.sample_type
            )
        else:
            basic = core.bm3d.Basic(preclip, pre, profile, nsigma, matrix=100, **b_args)
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
                final = core.bm3d.VFinal(preclip, final, profile, nsigma, matrix=100, **f_args).bm3d.VAggregate(
                    r, preclip.format.sample_type
                )
            else:
                final = core.bm3d.Final(preclip, final, profile, nsigma, matrix=100, **f_args)

    if 0 in nsigma:
        final = join({p: preclip if s == 0 else final for p, s in zip(range(3), nsigma)}, vs.YUV)

    return (
        core.bm3d.OPP2RGB(final, preclip.format.sample_type)
        if preclip.format.color_family != vs.GRAY
        else depth(final, range_out=preclip)
    )


def _bm3d_wolfram(
    preclip: vs.VideoNode,
    pre: vs.VideoNode | None,
    ref: vs.VideoNode | None,
    nsigma: list[float],
    refine: int,
    profile: BM3D.Profile,
    radius_basic: int | None,
    radius_final: int | None,
    backend: BM3D.Backend,
    nbasic_args: dict[str, Any],
    nfinal_args: dict[str, Any],
    chroma: bool = False,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Internal function for WolframRhodium implementation.
    """

    if not ref:
        b_args = _clean_keywords(profile.basic_args(radius_basic), backend.plugin.BM3Dv2) | nbasic_args | kwargs
        b_args.update(chroma=chroma)

        basic = backend.plugin.BM3Dv2(preclip, pre, nsigma, **b_args)
    else:
        basic = ref

    if not refine:
        final = basic
    else:
        f_args = _clean_keywords(profile.final_args(radius_final), backend.plugin.BM3Dv2) | nfinal_args | kwargs
        f_args.update(chroma=chroma)

        final = basic

        for _ in range(refine):
            final = backend.plugin.BM3Dv2(preclip, final, nsigma, **f_args)

    return final
