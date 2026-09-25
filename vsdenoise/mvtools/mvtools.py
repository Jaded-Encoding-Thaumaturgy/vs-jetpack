from __future__ import annotations

import weakref
from collections.abc import Sequence
from fractions import Fraction
from itertools import chain
from typing import Any, Literal, NamedTuple, cast, overload

from jetpytools import KwargsNotNone, fallback, normalize_seq, to_arr

from vstools import (
    Field,
    FieldLike,
    Planes,
    UnsupportedColorFamilyError,
    VSFunctionNoArgs,
    VSObject,
    core,
    normalize_planes,
    scale_delta,
    vs,
)

from .enums import MaskMode, PenaltyMode, RFilterMode, SearchMode, SharpMode
from .motion import MotionVectors
from .presets import (
    AnalyzeArgs,
    CompensateArgs,
    DegrainArgs,
    FlowArgs,
    FlowBlurArgs,
    FlowFpsArgs,
    FlowInterpolateArgs,
    MaskArgs,
    RecalculateArgs,
    ScDetectionArgs,
    SuperArgs,
)
from .utils import calc_super_pad, normalize_thscd

__all__ = ["MVTools"]


class _SuperConfigKey(NamedTuple):
    blksize: tuple[int, int]
    overlap: tuple[int, int]
    pad: tuple[int, int]
    pel: int
    sharp: SharpMode
    rfilter: RFilterMode
    onelevel: bool
    pelclip: vs.VideoNode | None


class _SuperConfigCache(VSObject, dict[_SuperConfigKey, vs.VideoNode]):
    def __vs_del__(self, core_id: int) -> None:
        self.clear()

    def get_clip(
        self,
        clip: vs.VideoNode,
        blksize: tuple[int, int],
        overlap: tuple[int, int],
        pad: tuple[int, int],
        pel: int,
        sharp: SharpMode,
        rfilter: RFilterMode,
        onelevel: bool,
        pelclip: vs.VideoNode | None,
    ) -> vs.VideoNode:
        key = _SuperConfigKey(blksize, overlap, pad, pel, sharp, rfilter, onelevel, pelclip)

        if key in self:
            return self[key]

        # Get the required padding amount from clip dimension, blksize, and overlap.
        # pad & pel must match.
        # If pel > 1, pelclip must match. If pelclip is used, sharp does not need to match.
        # If onelevel=False, blksize & rfilter must match.
        req_padx = calc_super_pad(clip.width, blksize[0], overlap[0])
        req_pady = calc_super_pad(clip.height, blksize[1], overlap[1])

        for cached_key, cached_clip in self.items():
            if all(
                (
                    calc_super_pad(clip.width, cached_key.blksize[0], cached_key.overlap[0]) == req_padx,
                    calc_super_pad(clip.height, cached_key.blksize[1], cached_key.overlap[1]) == req_pady,
                    cached_key.pad == pad,
                    cached_key.pel == pel,
                    pel == 1 or (cached_key.pelclip == pelclip and (pelclip or cached_key.sharp == sharp)),
                    onelevel
                    or all((not cached_key.onelevel, cached_key.blksize == blksize, cached_key.rfilter == rfilter)),
                )
            ):
                self[key] = cached_clip
                return cached_clip

        self[key] = core.mvu.Super(
            clip,
            blksize=blksize,
            overlap=overlap,
            pad=pad,
            pel=pel,
            sharp=sharp,
            rfilter=rfilter,
            onelevel=onelevel,
            pelclip=pelclip,
        )

        return self[key]


class _ClipSuperCache(VSObject):
    def __init__(self) -> None:
        self._cache = weakref.WeakKeyDictionary[vs.VideoNode, _SuperConfigCache]()

    def __vs_del__(self, core_id: int) -> None:
        self.clear()

    def get(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        cache = self._cache.get(clip)

        if cache is None:
            self._cache[clip] = cache = _SuperConfigCache()

        return cache.get_clip(clip, **kwargs)

    def clear(self) -> None:
        self._cache.clear()


_super_clip_cache = _ClipSuperCache()


class MVTools(VSObject):
    """
    MVTools wrapper for motion analysis, degraining, compensation, interpolation, etc.
    """

    super_args: SuperArgs
    """Arguments passed to every [MVTools.super][vsdenoise.MVTools.super] call."""

    analyze_args: AnalyzeArgs
    """Arguments passed to every [MVTools.analyze][vsdenoise.MVTools.analyze] call."""

    recalculate_args: RecalculateArgs
    """Arguments passed to every [MVTools.recalculate][vsdenoise.MVTools.recalculate] call."""

    compensate_args: CompensateArgs
    """Arguments passed to every [MVTools.compensate][vsdenoise.MVTools.compensate] call."""

    flow_args: FlowArgs
    """Arguments passed to every [MVTools.flow][vsdenoise.MVTools.flow] call."""

    degrain_args: DegrainArgs
    """Arguments passed to every [MVTools.degrain][vsdenoise.MVTools.degrain] call."""

    flow_interpolate_args: FlowInterpolateArgs
    """Arguments passed to every [MVTools.flow_interpolate][vsdenoise.MVTools.flow_interpolate] call."""

    flow_fps_args: FlowFpsArgs
    """Arguments passed to every [MVTools.flow_fps][vsdenoise.MVTools.flow_fps] call."""

    flow_blur_args: FlowBlurArgs
    """Arguments passed to every [MVTools.flow_blur][vsdenoise.MVTools.flow_blur] call."""

    mask_args: MaskArgs
    """Arguments passed to every [MVTools.mask][vsdenoise.MVTools.mask] call."""

    sc_detection_args: ScDetectionArgs
    """Arguments passed to every [MVTools.sc_detection][vsdenoise.MVTools.sc_detection] call."""

    vectors: MotionVectors
    """Motion vectors analyzed and used for all operations."""

    clip: vs.VideoNode
    """Clip to process."""

    def __init__(
        self,
        clip: vs.VideoNode,
        search_clip: vs.VideoNode | VSFunctionNoArgs | None = None,
        vectors: MotionVectors | None = None,
        chroma: bool | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        field: FieldLike | None = None,
        *,
        super_args: SuperArgs | None = None,
        analyze_args: AnalyzeArgs | None = None,
        recalculate_args: RecalculateArgs | None = None,
        compensate_args: CompensateArgs | None = None,
        flow_args: FlowArgs | None = None,
        degrain_args: DegrainArgs | None = None,
        flow_interpolate_args: FlowInterpolateArgs | None = None,
        flow_fps_args: FlowFpsArgs | None = None,
        flow_blur_args: FlowBlurArgs | None = None,
        mask_args: MaskArgs | None = None,
        sc_detection_args: ScDetectionArgs | None = None,
    ) -> None:
        """
        MVTools is a collection of functions for motion estimation and compensation in video.

        Motion compensation may be used for strong temporal denoising, advanced framerate conversions,
        image restoration, and other similar tasks.

        The plugin uses a block-matching method of motion estimation (similar methods as used in MPEG2, MPEG4, etc.).
        During the analysis stage the plugin divides frames into smaller blocks and tries to find the most similar
        matching block for every block in current frame in the second frame (which is either the previous
        or next frame).
        The relative shift of these blocks is the motion vector.

        The main method of measuring block similarity is by calculating the sum of absolute differences (SAD)
        of all pixels of these two blocks, which indicates how correct the motion estimation was.

        More information:
            - [VapourSynth plugin](https://github.com/myrsloik/mvutensils)
            - [AviSynth docs](https://htmlpreview.github.io/?https://github.com/pinterf/mvtools/blob/mvtools-pfmod/Documentation/mvtools2.html)

        Args:
            clip: The clip to process.
            search_clip: Optional clip or callable to be used for motion vector gathering only.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            chroma: Whether to consider chroma in motion vector calculations.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

            field: Set field order for interlaced processing, input is expected to be separated fields.
            super_args: Arguments passed to every [MVTools.super][vsdenoise.MVTools.super] call.
            analyze_args: Arguments passed to every [MVTools.analyze][vsdenoise.MVTools.analyze] call.
            recalculate_args: Arguments passed to every [MVTools.recalculate][vsdenoise.MVTools.recalculate] call.
            compensate_args: Arguments passed to every [MVTools.compensate][vsdenoise.MVTools.compensate] call.
            flow_args: Arguments passed to every [MVTools.flow][vsdenoise.MVTools.flow] call.
            degrain_args: Arguments passed to every [MVTools.degrain][vsdenoise.MVTools.degrain] call.
            flow_interpolate_args: Arguments passed to every
                [MVTools.flow_interpolate][vsdenoise.MVTools.flow_interpolate] call.
            flow_fps_args: Arguments passed to every [MVTools.flow_fps][vsdenoise.MVTools.flow_fps] call.
            flow_blur_args: Arguments passed to every [MVTools.flow_blur][vsdenoise.MVTools.flow_blur] call.
            mask_args: Arguments passed to every [MVTools.mask][vsdenoise.MVTools.mask] call.
            sc_detection_args: Arguments passed to every [MVTools.sc_detection][vsdenoise.MVTools.sc_detection] call.
        """

        UnsupportedColorFamilyError.check(clip, (vs.YUV, vs.GRAY), self.__class__)

        self.clip = clip
        self.chroma = chroma
        self.thscd = thscd
        self.fields = field is not None
        self.tff = Field.from_param_with_fallback(field)

        self.vectors = fallback(vectors, MotionVectors())

        if callable(search_clip):
            self.search_clip = search_clip(clip)
        else:
            self.search_clip = fallback(search_clip, clip)

        self.super_args = fallback(super_args, SuperArgs())
        self.analyze_args = fallback(analyze_args, AnalyzeArgs())
        self.recalculate_args = fallback(recalculate_args, RecalculateArgs())
        self.compensate_args = fallback(compensate_args, CompensateArgs())
        self.degrain_args = fallback(degrain_args, DegrainArgs())
        self.flow_args = fallback(flow_args, FlowArgs())
        self.flow_interpolate_args = fallback(flow_interpolate_args, FlowInterpolateArgs())
        self.flow_fps_args = fallback(flow_fps_args, FlowFpsArgs())
        self.flow_blur_args = fallback(flow_blur_args, FlowBlurArgs())
        self.mask_args = fallback(mask_args, MaskArgs())
        self.sc_detection_args = fallback(sc_detection_args, ScDetectionArgs())

    def super(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        pad: int | tuple[int | None, int | None] | None = None,
        pel: int | None = None,
        sharp: SharpMode | None = None,
        rfilter: RFilterMode | None = None,
        onelevel: bool = False,
        pelclip: vs.VideoNode | VSFunctionNoArgs | None = None,
    ) -> vs.VideoNode:
        """
        Get source clip and prepare special "super" clip with multilevel (hierarchical scaled) frames data.
        The super clip is used by both [analyze][vsdenoise.MVTools.analyze] and motion compensation (client) functions.

        You can use different Super clip for generation vectors with [analyze][vsdenoise.MVTools.analyze]
        and a different super clip format for the actual action.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            pad: How much padding to add to the source frame. Small padding is added to help with motion estimation near
                frame borders.
            pel: Subpixel precision for motion estimation (1=pixel, 2=half-pixel, 4=quarter-pixel). Default: 2.
            sharp: Subpixel interpolation method if pel is 2 or 4.
                For more information, see [SharpMode][vsdenoise.SharpMode].
            rfilter: Hierarchical levels smoothing and reducing (halving) filter. For more information, see
                [RFilterMode][vsdenoise.RFilterMode].
            onelevel: Generate only one hierarchical level.
                Only [analyze][vsdenoise.MVTools.analyze] uses more than one level,
                so if the super clip is only passed to other functions set it to True
                to save memory and a small speedup.
            pelclip: Optional upsampled source clip to use instead of internal subpixel interpolation (if pel > 1). The
                clip must contain the original source pixels at positions that are multiples of pel (e.g., positions 0,
                2, 4, etc. for pel=2), with interpolated pixels in between. The clip should not be padded.

        Returns:
            The original clip with MVUtensils frame properties attached to it.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)

        pad = fallback(pad, self.super_args.get("pad"), default=None)
        pel = fallback(pel, self.super_args.get("pel"), default=2)
        sharp = fallback(sharp, self.super_args.get("sharp"), default=SharpMode.WIENER)
        rfilter = fallback(rfilter, self.super_args.get("rfilter"), default=RFilterMode.BILINEAR)
        pelclip = fallback(pelclip, self.super_args.get("pelclip"), default=None)

        assert vectors.blksize
        assert vectors.overlap

        npad = normalize_seq(pad, 2)
        npad = (fallback(npad[0], vectors.blksize[0]), fallback(npad[1], vectors.blksize[1]))

        vectors.pad = fallback(vectors.pad, npad)
        vectors.pel = fallback(vectors.pel, pel)

        npelclip = pelclip(clip) if callable(pelclip) else pelclip

        return _super_clip_cache.get(
            clip,
            blksize=vectors.blksize,
            overlap=vectors.overlap,
            pad=vectors.pad,
            pel=vectors.pel,
            sharp=sharp,
            rfilter=rfilter,
            onelevel=onelevel,
            pelclip=npelclip,
        )

    def analyze(
        self,
        super: vs.VideoNode | None = None,
        tr: int = 1,
        delta: int | Sequence[int] | None = None,
        blksize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        levels: int | None = None,
        search: SearchMode | None = None,
        searchparam: int | None = None,
        pelsearch: int | None = None,
        mvlambda: int | None = None,
        lsad: int | None = None,
        plevel: PenaltyMode | None = None,
        globalmv: bool | None = None,
        pnew: int | None = None,
        pzero: int | None = None,
        pglobal: int | None = None,
        badsad: int | None = None,
        badrange: int | None = None,
        meander: bool | None = None,
        trymany: bool | None = None,
        satd: bool | None = None,
    ) -> None:
        """
        Analyze motion vectors in a clip using block matching.

        Takes a prepared super clip (containing hierarchical frame data) and estimates motion by comparing blocks
        between frames.
        Set motion vector data that can be used by other functions for motion compensation.

        The motion vector search is performed hierarchically, starting from a coarse image scale and progressively
        refining to finer scales.
        For each block, the function first checks predictors like the zero vector and neighboring block vectors.

        This method calculates the Sum of Absolute Differences (SAD) for these predictors,
        then iteratively tests new candidate vectors by adjusting the current best vector.
        The vector with the lowest SAD value is chosen as the final motion vector,
        with a penalty applied to maintain motion coherence between blocks.

        Args:
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
                Default: 1.
            delta: Specific delta(s) of motion vectors to use.
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
            overlap: Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal,
                vertical) overlap values. Each value must be even and less than its corresponding block size dimension.
            levels: Number of levels used in hierarchical motion vector analysis. A positive value specifies how many
                levels to use. A negative or zero value specifies how many coarse levels to skip. Lower values generally
                give better results since vectors of any length can be found. Sometimes adding more levels can help
                prevent false vectors in CGI or similar content.
            search: Search algorithm to use at the finest level. See [SearchMode][vsdenoise.SearchMode] for options.
            searchparam: Search radius/step for the chosen.
            mvlambda: Controls the coherence of the motion vector field. Higher values enforce more coherent/smooth
                motion between blocks. Too high values may cause the algorithm to miss the optimal vectors.
            lsad: SAD limit for mvlambda. When the SAD value of a vector predictor (formed from neighboring blocks)
                exceeds this limit, the local mvlambda value is decreased. This helps prevent the use of bad predictors,
                but reduces motion coherence between blocks.
            plevel: Controls how the penalty factor (mvlambda) scales with hierarchical levels.
                For more information, see [PenaltyMode][vsdenoise.PenaltyMode].
            globalmv: Whether to estimate global motion at each level and use it as an additional predictor. This can
                help with camera motion.
            pnew: Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                Higher values make the search more conservative.
            pzero: Penalty multiplier (relative to 256) applied to the SAD cost for the zero motion vector. Higher
                values discourage using zero motion.
            pglobal: Penalty multiplier (relative to 256) applied to the SAD cost when using the global motion
                predictor.
            badsad: SAD threshold above which a wider secondary search will be performed to find better motion vectors.
                Higher values mean fewer blocks will trigger the secondary search.
            badrange: Search radius for the secondary search when a block's SAD exceeds badsad.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            trymany: Whether to test multiple predictor vectors during the search process at coarser levels. Enabling
                this can find better vectors but increases processing time.
            satd: Whether to use Sum of Absolute Transformed Differences (SATD) instead of SAD for luma comparison.
        """

        blksize = fallback(blksize, self.analyze_args.get("blksize"), default=8)
        overlap = fallback(overlap, self.analyze_args.get("overlap"), default=0)

        nblksize = cast(tuple[int, int], tuple(normalize_seq(blksize, 2)))
        noverlap = cast(tuple[int, int], tuple(normalize_seq(overlap, 2)))

        self.vectors.clear()
        self.vectors.blksize = nblksize
        self.vectors.overlap = noverlap

        super_clip = self.super(fallback(super, self.search_clip), self.vectors, onelevel=False)

        analyze_args: dict[str, Any] = self.analyze_args | KwargsNotNone(
            blksize=nblksize,
            overlap=noverlap,
            levels=levels,
            search=search,
            searchparam=searchparam,
            pelsearch=pelsearch,
            mvlambda=mvlambda,
            chroma=self.chroma,
            lsad=lsad,
            plevel=plevel,
            globalmv=globalmv,
            pnew=pnew,
            pzero=pzero,
            pglobal=pglobal,
            badsad=badsad,
            badrange=badrange,
            meander=meander,
            trymany=trymany,
            satd=satd,
            fields=self.fields,
            tff=self.tff,
        )

        if not delta:
            vects = core.mvu.AnalyseMany(super_clip, radius=tr, delta=1 + self.fields, **analyze_args)
            for d in range(1, tr + 1):
                self.vectors[d], self.vectors[-d] = vects[2 * d - 2 : 2 * d]
        else:
            for d in to_arr(delta):
                if not d:
                    continue

                self.vectors[d] = core.mvu.Analyse(super_clip, delta=d * (self.fields + 1), **analyze_args)

    def recalculate(
        self,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        thsad: int | None = None,
        smooth: bool | None = None,
        blksize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        search: SearchMode | None = None,
        searchparam: int | None = None,
        mvlambda: int | None = None,
        pnew: int | None = None,
        meander: bool | None = None,
        satd: bool | None = None,
    ) -> None:
        """
        Refines and recalculates motion vectors that were previously estimated,
        optionally using a different super clip or parameters.

        This two-stage approach can provide more stable and robust motion estimation.

        The refinement only occurs at the finest hierarchical level.
        It uses the interpolated vectors from the original blocks as predictors for the new vectors,
        and recalculates their SAD values.

        Only vectors with poor quality (SAD above threshold) will be re-estimated through a new search.
        The SAD threshold is normalized to an 8x8 block size. Vectors with good quality are preserved,
        though their SAD values are still recalculated and updated.

        Args:
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            thsad: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value is
                scaled to 8x8 block size.
            smooth: Whether to interpolate the new (finer) vector field from neighbours or take the nearest old vector.
            blksize: Size of blocks for motion estimation. Can be an int or tuple of (width, height). Larger blocks are
                less sensitive to noise and faster to process, but will produce less accurate vectors.
            overlap: Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal,
                vertical) overlap values. Each value must be even and less than its corresponding block size dimension.
            search: Search algorithm to use at the finest level. See [SearchMode][vsdenoise.SearchMode] for options.
            searchparam: Search radius/step for the chosen.
            mvlambda: Controls the coherence of the motion vector field. Higher values enforce more coherent/smooth
                motion between blocks. Too high values may cause the algorithm to miss the optimal vectors.
            pnew: Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                Higher values make the search more conservative.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            satd: Whether to use Sum of Absolute Transformed Differences (SATD) instead of SAD for luma comparison.
        """

        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, self.search_clip), vectors, onelevel=True)

        recalculate_args: dict[str, Any] = self.recalculate_args | KwargsNotNone(
            thsad=thsad,
            smooth=smooth,
            blksize=blksize,
            overlap=overlap,
            search=search,
            searchparam=searchparam,
            mvlambda=mvlambda,
            chroma=self.chroma,
            pnew=pnew,
            meander=meander,
            satd=satd,
            fields=self.fields,
            tff=self.tff,
        )

        recalculated = core.mvu.Recalculate(super_clip, tuple(vectors.values()), **recalculate_args)
        if isinstance(recalculated, vs.VideoNode):
            recalculated = [recalculated]

        vectors.update(zip(vectors, recalculated))

    @overload
    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: Literal[True] = True,
        temporal_func: None = None,
    ) -> tuple[vs.VideoNode, tuple[int, int]]: ...

    @overload
    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: Literal[True] = True,
        *,
        temporal_func: VSFunctionNoArgs,
    ) -> vs.VideoNode: ...

    @overload
    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        *,
        interleave: Literal[False],
        temporal_func: None = None,
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]: ...

    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: bool = True,
        temporal_func: VSFunctionNoArgs | None = None,
    ) -> vs.VideoNode | tuple[list[vs.VideoNode], list[vs.VideoNode]] | tuple[vs.VideoNode, tuple[int, int]]:
        """
        Perform motion compensation by moving blocks from reference frames to the current frame
        according to motion vectors.

        This creates a prediction of the current frame by taking blocks from neighboring frames
        and moving them along their estimated motion paths.

        Args:
            clip: The clip to process.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
            delta: Specific delta(s) of motion vectors to use.
            thsad: SAD threshold for safe compensation. If block SAD is above thsad, the source block is used instead of
                the compensated block.
            time: Time position between frames as a percentage (0.0-100.0). Controls the interpolation position between
                frames.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

            interleave: Whether to interleave the compensated frames with the input.
            temporal_func: Temporal function to apply to the motion compensated frames.

        Returns:
            Motion compensated frames if func is provided, otherwise returns a tuple containing:

                   - The interleaved compensated frames.
                   - A tuple of (total_frames, center_offset) for manual frame selection.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr, delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.compensate_args.get("thscd"), default=self.thscd))

        compensate_args: dict[str, Any] = self.compensate_args | KwargsNotNone(
            thsad=thsad, time=time, thscd1=thscd1, thscd2=thscd2, fields=self.fields, tff=self.tff
        )

        comp_fwrd, comp_back = [
            [core.mvu.Compensate(clip, super_clip, vectors=vect, **compensate_args) for vect in vectors_list]
            for vectors_list in (reversed(vect_f), vect_b)
        ]

        if not interleave:
            return (comp_fwrd, comp_back)

        comp_clips = [*comp_fwrd, clip, *comp_back]
        cycle = len(comp_clips)
        offset = len(comp_fwrd)

        interleaved = core.std.Interleave(comp_clips)

        if temporal_func:
            return core.std.SelectEvery(temporal_func(interleaved), cycle, offset)

        return interleaved, (cycle, offset)

    @overload
    def flow(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: Literal[True] = True,
        temporal_func: None = None,
    ) -> tuple[vs.VideoNode, tuple[int, int]]: ...

    @overload
    def flow(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: Literal[True] = True,
        *,
        temporal_func: VSFunctionNoArgs,
    ) -> vs.VideoNode: ...

    @overload
    def flow(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        *,
        interleave: Literal[False],
        temporal_func: None = None,
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]: ...

    def flow(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: bool = True,
        temporal_func: VSFunctionNoArgs | None = None,
    ) -> vs.VideoNode | tuple[list[vs.VideoNode], list[vs.VideoNode]] | tuple[vs.VideoNode, tuple[int, int]]:
        """
        Performs motion compensation using pixel-level motion vectors interpolated from block vectors.

        Unlike block-based compensation, this calculates a unique motion vector for each pixel
        by bilinearly interpolating between the motion vectors of the current block and its neighbors
        based on the pixel's position.
        The pixels in the reference frame are then moved along these interpolated vectors
        to their estimated positions in the current frame.

        Args:
            clip: The clip to process.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
            delta: Specific delta(s) of motion vectors to use.
            time: Time position between frames as a percentage (0.0-100.0). Controls the interpolation position between
                frames.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

            interleave: Whether to interleave the compensated frames with the input.
            temporal_func: Optional function to process the motion compensated frames. Takes the interleaved frames as
                input and returns processed frames.

        Returns:
            Motion compensated frames if func is provided, otherwise returns a tuple containing:

                   - The interleaved compensated frames.
                   - A tuple of (total_frames, center_offset) for manual frame selection.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr, delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.flow_args.get("thscd"), default=self.thscd))

        flow_args: dict[str, Any] = self.flow_args | KwargsNotNone(
            time=time, thscd1=thscd1, thscd2=thscd2, fields=self.fields, tff=self.tff
        )

        flow_fwrd, flow_back = [
            [core.mvu.Flow(clip, super_clip, vectors=vect, **flow_args) for vect in vectors_list]
            for vectors_list in (reversed(vect_f), vect_b)
        ]

        if not interleave:
            return (flow_fwrd, flow_back)

        flow_clips = [*flow_fwrd, clip, *flow_back]
        cycle = len(flow_clips)
        offset = len(flow_fwrd)

        interleaved = core.std.Interleave(flow_clips)

        if temporal_func:
            return core.std.SelectEvery(temporal_func(interleaved), cycle, offset)

        return interleaved, (cycle, offset)

    def degrain(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
        thsad: int | tuple[int, int] | None = None,
        thsad2: int | tuple[int, int] | None = None,
        limit: float | tuple[float, float] | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        weights: Sequence[int] | None = None,
        planes: Planes = None,
    ) -> vs.VideoNode:
        """
        Perform temporal denoising using motion compensation.

        Motion compensated blocks from previous and next frames are averaged with the current frame.
        The weighting factors for each block depend on their SAD from the current frame.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
            delta: Specific delta(s) of motion vectors to use.
            thsad: Defines the soft threshold of block sum absolute differences. Blocks with SAD above this threshold
                have zero weight for averaging (denoising). Blocks with low SAD have highest weight. The remaining
                weight is taken from pixels of source clip.
                Applies to the nearest references (temporal distance 1);
                more distant references interpolate towards `thsad2`.
            thsad2: Defines the soft threshold for the furthest references (temporal distance radius).
                Each reference at distance `d` in 1...radius uses a raised-cosine interpolation
                between `thsad` (at d=1) and `thsad2` (at d=radius). Defaults to `thsad`.
            limit: Maximum allowed change in pixel values (8 bits scale).
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

            weights: Optional per-frame bias applied on top of the SAD-derived weights. Given in temporal order:
                `[bw_radius, ..., bw_1, centre, fw_1, ..., fw_radius]` (exactly `2 * radius + 1` non-negative values).
            planes: Which planes to process. Default: None (all planes).

        Returns:
            Motion compensated and temporally filtered clip with reduced noise.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr, delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.degrain_args.get("thscd"), default=self.thscd))

        limit = fallback(limit, self.degrain_args.get("limit"), default=None)
        nlimit = [scale_delta(thr, 8, clip) for thr in normalize_seq(limit, 2)] if limit is not None else limit

        degrain_args: dict[str, Any] = self.degrain_args | KwargsNotNone(
            thsad=thsad,
            thsad2=thsad2,
            planes=normalize_planes(clip, planes),
            limit=nlimit,
            thscd1=thscd1,
            thscd2=thscd2,
            weights=weights,
        )

        return core.mvu.Degrain(clip, super_clip, list(chain.from_iterable(zip(vect_b, vect_f))), **degrain_args)

    def flow_interpolate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int | tuple[int, int] | None = None,
        time: float | None = None,
        ml: float | None = None,
        blend: bool | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        interleave: bool = True,
    ) -> vs.VideoNode:
        """
        Motion interpolation function that creates an intermediate frame between two frames.

        Uses both backward and forward motion vectors to estimate motion and create a frame at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        Args:
            clip: The clip to process.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Specific delta(s) of motion vectors to use.
            time: Time position between frames as a percentage (0.0-100.0). Controls the interpolation position between
                frames. Does nothing if multi is specified.
            ml: Mask scale parameter that controls occlusion mask strength. Higher values produce weaker occlusion
                masks. Used in MakeVectorOcclusionMaskTime for modes 3-5. Used in MakeSADMaskTime for modes 6-8.
            blend: Whether to blend frames at scene changes. If True, frames will be blended. If False, frames will be
                copied.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

            interleave: Whether to interleave the interpolated frames with the source clip.

        Returns:
            Motion interpolated clip.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr=1, delta=delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.flow_interpolate_args.get("thscd"), default=self.thscd))

        flow_interpolate_args: dict[str, Any] = self.flow_interpolate_args | KwargsNotNone(
            time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        interpolated = core.mvu.FlowInter(clip, super_clip, (*vect_b, *vect_f), **flow_interpolate_args)
        return core.std.Interleave([clip, interpolated]) if interleave else interpolated

    def flow_fps(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int | tuple[int, int] | None = None,
        fps: Fraction | None = None,
        extramask: bool | None = None,
        ml: float | None = None,
        blend: bool | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        Args:
            clip: The clip to process.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Specific delta(s) of motion vectors to use.
            fps: Target output framerate as a Fraction.
            extramask: Whether to generate an extra mask for occlusion handling.
            ml: Mask scale parameter that controls occlusion mask strength. Higher values produce weaker occlusion
                masks. Used in MakeVectorOcclusionMaskTime for modes 3-5. Used in MakeSADMaskTime for modes 6-8.
            blend: Whether to blend frames at scene changes. If True, frames will be blended. If False, frames will be
                copied.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Clip with its framerate resampled.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr=1, delta=delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.flow_fps_args.get("thscd"), default=self.thscd))

        flow_fps_args: dict[str, Any] = self.flow_fps_args | KwargsNotNone(
            extramask=extramask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        if fps is not None:
            flow_fps_args.update(num=fps.numerator, den=fps.denominator)

        return core.mvu.FlowFPS(clip, super_clip, (*vect_b, *vect_f), **flow_fps_args)

    def flow_blur(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int | tuple[int, int] | None = None,
        blur: float | None = None,
        prec: int | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Creates a motion blur effect by simulating finite shutter time, similar to film cameras.

        Uses backward and forward motion vectors to create and overlay multiple copies of motion compensated pixels
        at intermediate time positions within a blurring interval around the current frame.

        Args:
            clip: The clip to process.
            super: The clip to be prepared by [super][vsdenoise.MVTools.super]. If None, super will be obtained from the
                main clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Specific delta(s) of motion vectors to use.
            blur: Blur time interval between frames as a percentage (0.0-100.0). Controls the simulated shutter
                time/motion blur strength.
            prec: Blur precision in pixel units. Controls the accuracy of the motion blur.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Motion blurred clip.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors, onelevel=True)

        vect_b, vect_f = vectors.get_vectors(tr=1, delta=delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.flow_blur_args.get("thscd"), default=self.thscd))

        flow_blur_args: dict[str, Any] = self.flow_blur_args | KwargsNotNone(
            blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2
        )

        return core.mvu.FlowBlur(clip, super_clip, (*vect_b, *vect_f), **flow_blur_args)

    def mask(
        self,
        vectors: MotionVectors | None = None,
        delta: int = 1,
        ml: float | None = None,
        gamma: float | None = None,
        kind: MaskMode = MaskMode.VECTOR_LENGTH,
        time: float | None = None,
        scval: float | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Creates a mask clip from motion vectors data.

        Args:
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Motion vector delta to use.
            ml: Motion length scale factor. When the vector's length (or other mask value) is greater than or equal to
                ml, the output is saturated to 255.
            gamma: Exponent for the relation between input and output values. 1.0 gives a linear relation, 2.0 gives a
                quadratic relation.
            kind: Type of mask to generate. See [MaskMode][vsdenoise.MaskMode] for options.
            time: Time position between frames as a percentage (0.0-100.0).
            scval: Value assigned to the mask on scene changes.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Motion mask clip.
        """

        vectors = fallback(vectors, self.vectors)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.mask_args.get("thscd"), default=self.thscd))

        mask_args: dict[str, Any] = self.mask_args | KwargsNotNone(
            ml=ml, gamma=gamma, time=time, scval=scval, thscd1=thscd1, thscd2=thscd2
        )

        match kind:
            case MaskMode.VECTOR_LENGTH:
                mask_func = core.mvu.VectorLengthMask
            case MaskMode.SAD:
                mask_func = core.mvu.SADMask
            case MaskMode.OCCLUSION:
                mask_func = core.mvu.OcclusionMask

        return mask_func(vectors[delta], **mask_args)

    def sc_detection(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int | tuple[int, int] | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Creates scene change frameprops from motion vectors data.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Specific delta(s) of motion vectors to use.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Clip with scene change properties set.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)

        vect_b, vect_f = vectors.get_vectors(tr=1, delta=delta)

        thscd1, thscd2 = normalize_thscd(fallback(thscd, self.sc_detection_args.get("thscd"), default=self.thscd))

        detect = clip
        for vect in (*vect_b, *vect_f):
            detect = core.mvu.SCDetection(detect, vect, thscd1=thscd1, thscd2=thscd2)

        return detect
