from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import chain
from typing import Any, Literal, NamedTuple, cast, overload

from jetpytools import KwargsNotNone, fallback, normalize_seq

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

from .enums import MaskMode, MVDirection, PenaltyMode, RFilterMode, SearchMode, SharpMode
from .motion import MotionVectors
from .utils import normalize_thscd, refine_blksize

__all__ = ["MVTools"]


class _SuperConfigKey(NamedTuple):
    onelevel: bool
    args: tuple[tuple[str, Any], ...]


class _SuperConfigCache(VSObject, dict[_SuperConfigKey, vs.VideoNode]):
    def get_cached_super(self, clip: vs.VideoNode, onelevel: bool, **args: Any) -> vs.VideoNode:
        args_key = tuple(sorted(args.items()))
        key = _SuperConfigKey(onelevel, args_key)

        # Check if there is a cached onelevel=False (hierarchical) clip with same args
        if (key_hierarchical := _SuperConfigKey(False, args_key)) in self:
            return self[key_hierarchical]

        if key not in self:
            self[key] = core.mvu.Super(clip, onelevel=onelevel, **args)

        return self[key]


class _ClipSuperCache(VSObject, dict[vs.VideoNode, _SuperConfigCache]):
    def get_cached_super(self, clip: vs.VideoNode, onelevel: bool, **args: Any) -> vs.VideoNode:
        cache = self.get(clip)

        if cache is None:
            self[clip] = cache = _SuperConfigCache()

        return cache.get_cached_super(clip, onelevel, **args)


_super_clip_cache = _ClipSuperCache()


class MVTools(VSObject):
    """
    MVTools wrapper for motion analysis, degraining, compensation, interpolation, etc.
    """

    super_args: dict[str, Any]
    """Arguments passed to every [MVTools.super][vsdenoise.MVTools.super] call."""

    analyze_args: dict[str, Any]
    """Arguments passed to every [MVTools.analyze][vsdenoise.MVTools.analyze] call."""

    recalculate_args: dict[str, Any]
    """Arguments passed to every [MVTools.recalculate][vsdenoise.MVTools.recalculate] call."""

    compensate_args: dict[str, Any]
    """Arguments passed to every [MVTools.compensate][vsdenoise.MVTools.compensate] call."""

    flow_args: dict[str, Any]
    """Arguments passed to every [MVTools.flow][vsdenoise.MVTools.flow] call."""

    degrain_args: dict[str, Any]
    """Arguments passed to every [MVTools.degrain][vsdenoise.MVTools.degrain] call."""

    flow_interpolate_args: dict[str, Any]
    """Arguments passed to every [MVTools.flow_interpolate][vsdenoise.MVTools.flow_interpolate] call."""

    flow_fps_args: dict[str, Any]
    """Arguments passed to every [MVTools.flow_fps][vsdenoise.MVTools.flow_fps] call."""

    flow_blur_args: dict[str, Any]
    """Arguments passed to every [MVTools.flow_blur][vsdenoise.MVTools.flow_blur] call."""

    mask_args: dict[str, Any]
    """Arguments passed to every [MVTools.mask][vsdenoise.MVTools.mask] call."""

    sc_detection_args: dict[str, Any]
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
        pad: int | tuple[int | None, int | None] | None = None,
        pel: int | None = None,
        chroma: bool | None = None,
        field: FieldLike | None = None,
        *,
        super_args: Mapping[str, Any] | None = None,
        analyze_args: Mapping[str, Any] | None = None,
        recalculate_args: Mapping[str, Any] | None = None,
        compensate_args: Mapping[str, Any] | None = None,
        flow_args: Mapping[str, Any] | None = None,
        degrain_args: Mapping[str, Any] | None = None,
        flow_interpolate_args: Mapping[str, Any] | None = None,
        flow_fps_args: Mapping[str, Any] | None = None,
        flow_blur_args: Mapping[str, Any] | None = None,
        mask_args: Mapping[str, Any] | None = None,
        sc_detection_args: Mapping[str, Any] | None = None,
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
            - [VapourSynth plugin](https://github.com/dubhater/vapoursynth-mvtools)
            - [AviSynth docs](https://htmlpreview.github.io/?https://github.com/pinterf/mvtools/blob/mvtools-pfmod/Documentation/mvtools2.html)

        Args:
            clip: The clip to process.
            search_clip: Optional clip or callable to be used for motion vector gathering only.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            pad: How much padding to add to the source frame. Small padding is added to help with motion estimation near
                frame borders.
            pel: Subpixel precision for motion estimation (1=pixel, 2=half-pixel, 4=quarter-pixel). Default: 1.
            chroma: Whether to consider chroma in motion vector calculations.
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
        self.pel = pel
        self.pad = normalize_seq(pad, 2)
        self.chroma = chroma
        self.fields = field is not None
        self.tff = Field.from_param_with_fallback(field)

        self.vectors = fallback(vectors, MotionVectors())

        if callable(search_clip):
            self.search_clip = search_clip(self.clip)
        else:
            self.search_clip = fallback(search_clip, self.clip)

        self.super_args = dict(super_args) if super_args else {}
        self.analyze_args = dict(analyze_args) if analyze_args else {}
        self.recalculate_args = dict(recalculate_args) if recalculate_args else {}
        self.compensate_args = dict(compensate_args) if compensate_args else {}
        self.degrain_args = dict(degrain_args) if degrain_args else {}
        self.flow_args = dict(flow_args) if flow_args else {}
        self.flow_interpolate_args = dict(flow_interpolate_args) if flow_interpolate_args else {}
        self.flow_fps_args = dict(flow_fps_args) if flow_fps_args else {}
        self.flow_blur_args = dict(flow_blur_args) if flow_blur_args else {}
        self.mask_args = dict(mask_args) if mask_args else {}
        self.sc_detection_args = dict(sc_detection_args) if sc_detection_args else {}

        self.blksize: tuple[int, int] = tuple(normalize_seq(self.analyze_args.get("blksize", 16), 2))
        self.overlap_div: tuple[int, int] = tuple(normalize_seq(self.analyze_args.get("overlap_div", 2), 2))

    def super(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        onelevel: bool | None = None,
        sharp: SharpMode | None = None,
        rfilter: RFilterMode | None = None,
        pelclip: vs.VideoNode | VSFunctionNoArgs | None = None,
        blksize: int | tuple[int, int] | None = None,
        overlap_div: int | tuple[int, int] | None = None,
    ) -> vs.VideoNode:
        """
        Get source clip and prepare special "super" clip with multilevel (hierarchical scaled) frames data.
        The super clip is used by both [analyze][vsdenoise.MVTools.analyze] and motion compensation (client) functions.

        You can use different Super clip for generation vectors with [analyze][vsdenoise.MVTools.analyze]
        and a different super clip format for the actual action.
        Source clip is appended to clip's frameprops, [get_super][vsdenoise.MVTools.get_super] can be used
        to extract the super clip if you wish to view it yourself.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            onelevel: Generate only one hierarchical level.
                Only [analyze][vsdenoise.MVTools.analyze] uses more than one level,
                so if the super clip is only passed to other functions set it to True
                to save memory and a small speedup.
            sharp: Subpixel interpolation method if pel is 2 or 4.
                For more information, see [SharpMode][vsdenoise.SharpMode].
            rfilter: Hierarchical levels smoothing and reducing (halving) filter. For more information, see
                [RFilterMode][vsdenoise.RFilterMode].
            pelclip: Optional upsampled source clip to use instead of internal subpixel interpolation (if pel > 1). The
                clip must contain the original source pixels at positions that are multiples of pel (e.g., positions 0,
                2, 4, etc. for pel=2), with interpolated pixels in between. The clip should not be padded.
            blksize: Size of blocks for padding. If None, resolves from vectors or default configuration.
            overlap_div: Divisor for block overlap size. If None, resolves from vectors or default configuration.

        Returns:
            The original clip with MVUtensils frame properties attached to it.
        """
        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)

        s_blksize = fallback(blksize, vectors.blksize, self.blksize)
        s_overlap_div = fallback(overlap_div, vectors.overlap_div, self.overlap_div)
        s_overlap = refine_blksize(s_blksize, s_overlap_div)  # type: ignore[arg-type]

        # if vectors.scaled:
        #     hpad, vpad = vectors.analysis_data["Analysis_Padding"]
        # else:
        #     hpad, vpad = self.pad
        hpad, vpad = self.pad

        if pelclip is not None:
            pelclip = pelclip(clip) if callable(pelclip) else pelclip
        else:
            pelclip_arg = self.super_args.get("pelclip")
            pelclip = pelclip_arg(clip) if isinstance(pelclip_arg, VSFunctionNoArgs) else pelclip_arg

        super_args = KwargsNotNone(
            blksize=s_blksize,
            overlap=s_overlap,
            pad=(fallback(hpad, 16), fallback(vpad, 16)),
            pel=fallback(self.pel, 2),
            sharp=fallback(sharp, self.super_args.get("sharp"), 2),
            rfilter=fallback(rfilter, self.super_args.get("rfilter"), 1),
            pelclip=pelclip,
        )

        return _super_clip_cache.get_cached_super(clip, bool(onelevel), **super_args)

    def analyze(
        self,
        super: vs.VideoNode | None = None,
        tr: int = 1,
        delta: int | Sequence[int] | None = None,
        blksize: int | tuple[int, int] | None = None,
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
        overlap_div: int | tuple[int, int] | None = None,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
                Default: 1.
            delta: Specific delta(s) of motion vectors to use.
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
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
            overlap_div: Divisor for the block overlap.
                Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap divisors.
            badsad: SAD threshold above which a wider secondary search will be performed to find better motion vectors.
                Higher values mean fewer blocks will trigger the secondary search.
            badrange: Search radius for the secondary search when a block's SAD exceeds badsad.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            trymany: Whether to test multiple predictor vectors during the search process at coarser levels. Enabling
                this can find better vectors but increases processing time.
            satd: Whether to use Sum of Absolute Transformed Differences (SATD) instead of SAD for luma comparison.
        """
        nblksize = cast(tuple[int, int], tuple(normalize_seq(fallback(blksize, self.blksize), 2)))
        noverlap_div = cast(tuple[int, int], tuple(normalize_seq(fallback(overlap_div, self.overlap_div), 2)))
        noverlap = refine_blksize(nblksize, noverlap_div)

        super_clip = self.super(
            fallback(super, self.search_clip),
            blksize=nblksize,
            overlap_div=noverlap_div,
            onelevel=False,
        )

        analyze_args = KwargsNotNone(
            blksize=nblksize,
            overlap=noverlap,
            levels=levels,
            search=fallback(search, self.analyze_args.get("search"), default=None),
            searchparam=fallback(searchparam, self.analyze_args.get("searchparam"), default=None),
            pelsearch=fallback(pelsearch, self.analyze_args.get("pelsearch"), default=None),
            mvlambda=fallback(mvlambda, self.analyze_args.get("mvlambda"), default=None),
            chroma=fallback(self.chroma, True),
            lsad=fallback(lsad, self.analyze_args.get("lsad"), default=None),
            plevel=fallback(plevel, self.analyze_args.get("plevel"), default=None),
            globalmv=fallback(globalmv, self.analyze_args.get("globalmv"), default=None),
            pnew=fallback(pnew, self.analyze_args.get("pnew"), default=None),
            pzero=fallback(pzero, self.analyze_args.get("pzero"), default=None),
            pglobal=fallback(pglobal, self.analyze_args.get("pglobal"), default=None),
            badsad=fallback(badsad, self.analyze_args.get("badsad"), default=None),
            badrange=fallback(badrange, self.analyze_args.get("badrange"), default=None),
            meander=fallback(meander, self.analyze_args.get("meander"), default=None),
            trymany=fallback(trymany, self.analyze_args.get("trymany"), default=None),
            satd=fallback(satd, self.analyze_args.get("satd"), default=None),
            fields=self.fields,
            tff=self.tff,
        )

        self.vectors.clear()
        self.vectors.blksize = nblksize
        self.vectors.overlap_div = noverlap_div

        if delta is None:
            vects = core.mvu.AnalyseMany(super_clip, radius=tr, delta=2 if self.fields else 1, **analyze_args)
            for i, d in enumerate(range(1, tr + 1)):
                self.vectors.set_vector(vects[i * 2], MVDirection.BACKWARD, d)
                self.vectors.set_vector(vects[i * 2 + 1], MVDirection.FORWARD, d)
        else:
            deltas = [delta] if isinstance(delta, int) else delta

            for d in deltas:
                # Scaled delta for interlaced fields
                plugin_delta = d * 2 if self.fields else d

                for direction in MVDirection:
                    actual_delta = plugin_delta if direction is MVDirection.BACKWARD else -plugin_delta

                    self.vectors.set_vector(
                        core.mvu.Analyse(super_clip, delta=actual_delta, **analyze_args), direction, d
                    )

    def recalculate(
        self,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        thsad: int | None = None,
        blksize: int | tuple[int, int] | None = None,
        search: SearchMode | None = None,
        searchparam: int | None = None,
        mvlambda: int | None = None,
        pnew: int | None = None,
        overlap_div: int | tuple[int, int] | None = None,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            thsad: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value is
                scaled to 8x8 block size.
            blksize: Size of blocks for motion estimation. Can be an int or tuple of (width, height). Larger blocks are
                less sensitive to noise and faster to process, but will produce less accurate vectors.
            search: Search algorithm to use at the finest level. See [SearchMode][vsdenoise.SearchMode] for options.
            searchparam: Search radius/step for the chosen.
            mvlambda: Controls the coherence of the motion vector field. Higher values enforce more coherent/smooth
                motion between blocks. Too high values may cause the algorithm to miss the optimal vectors.
            pnew: Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                Higher values make the search more conservative.
            overlap_div: Divisor for the block overlap.
                Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap divisors.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            satd: Whether to use Sum of Absolute Transformed Differences (SATD) instead of SAD for luma comparison.
        """
        vectors = fallback(vectors, self.vectors)
        blksize = fallback(blksize, self.recalculate_args.get("blksize"), self.blksize)
        overlap_div = fallback(overlap_div, self.recalculate_args.get("overlap_div"), self.overlap_div)
        nblksize = cast(tuple[int, int], tuple(normalize_seq(blksize, 2)))
        noverlap_div = cast(tuple[int, int], tuple(normalize_seq(overlap_div, 2)))
        noverlap = refine_blksize(nblksize, noverlap_div)

        super_clip = self.super(
            fallback(super, self.search_clip),
            vectors=vectors,
            blksize=nblksize,
            overlap_div=noverlap_div,
            onelevel=True,
        )

        recalculate_args = KwargsNotNone(
            thsad=fallback(thsad, self.recalculate_args.get("thsad"), default=None),
            blksize=nblksize,
            overlap=noverlap,
            search=fallback(search, self.recalculate_args.get("search"), default=None),
            searchparam=fallback(searchparam, self.recalculate_args.get("searchparam"), default=None),
            mvlambda=fallback(mvlambda, self.recalculate_args.get("mvlambda"), default=None),
            chroma=fallback(self.chroma, True),
            pnew=fallback(pnew, self.recalculate_args.get("pnew"), default=None),
            meander=fallback(meander, self.recalculate_args.get("meander"), default=None),
            satd=fallback(satd, self.recalculate_args.get("satd"), default=None),
            fields=self.fields,
            tff=self.tff,
        )
        vects = list[vs.VideoNode]()
        keys = list[tuple[MVDirection, int]]()

        for d in vectors.deltas:
            for direction in MVDirection:
                if d in vectors[direction]:
                    vects.append(vectors.get_vector(direction, d))
                    keys.append((direction, d))

        recalculated = core.mvu.Recalculate(super_clip, vects, **recalculate_args)

        if isinstance(recalculated, vs.VideoNode):
            recalculated = [recalculated]

        for (direction, d), vect in zip(keys, recalculated):
            vectors.set_vector(vect, direction, d)

        vectors.blksize = nblksize
        vectors.overlap_div = noverlap_div

    @overload
    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            direction: Motion vector direction to use.
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
        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)
        vect_b, vect_f = vectors.get_vectors(direction, tr, delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        compensate_args = self.compensate_args | KwargsNotNone(
            thsad=thsad,
            time=time,
            thscd1=thscd1,
            thscd2=thscd2,
            fields=self.fields,
            tff=self.tff,
        )

        comp_back, comp_fwrd = [
            [core.mvu.Compensate(clip, super_clip, vectors=vect, **compensate_args) for vect in vectors_list]
            for vectors_list in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (comp_back, comp_fwrd)

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
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
        direction: MVDirection = MVDirection.BOTH,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            direction: Motion vector direction to use.
            delta: Specific delta(s) of motion vectors to use.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
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
        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)
        vect_b, vect_f = vectors.get_vectors(direction, tr, delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_args = self.flow_args | KwargsNotNone(
            time=time,
            thscd1=thscd1,
            thscd2=thscd2,
            fields=self.fields,
            tff=self.tff,
        )

        flow_back, flow_fwrd = [
            [core.mvu.Flow(clip, super_clip, vectors=vect, **flow_args) for vect in vectors_list]
            for vectors_list in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (flow_back, flow_fwrd)

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
        limit: float | tuple[float, float] | None = None,
        weights: Sequence[int] | None = None,
        thscd: int | tuple[int | None, float | None] | None = None,
        planes: Planes = None,
    ) -> vs.VideoNode:
        """
        Perform temporal denoising using motion compensation.

        Motion compensated blocks from previous and next frames are averaged with the current frame.
        The weighting factors for each block depend on their SAD from the current frame.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
            delta: Specific delta(s) of motion vectors to use.
            thsad: Defines the soft threshold of block sum absolute differences. Blocks with SAD above this threshold
                have zero weight for averaging (denoising). Blocks with low SAD have highest weight. The remaining
                weight is taken from pixels of source clip.
            limit: Maximum allowed change in pixel values (8 bits scale).
            weights: Optional per-frame bias applied on top of the SAD-derived weights. Given in temporal order:
                `[bw_radius, ..., bw_1, centre, fw_1, ..., fw_radius]` (exactly `2 * radius + 1` non-negative values).
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.
            planes: Which planes to process. Default: None (all planes).

        Returns:
            Motion compensated and temporally filtered clip with reduced noise.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)
        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)

        if delta is not None:
            deltas = [delta] if isinstance(delta, int) else list(delta)
        else:
            tr_val = fallback(tr, vectors.tr)
            deltas = range(1, tr_val + 1)

        vect_b, vect_f = vectors.get_vectors(tr=None, delta=deltas)

        thscd1, thscd2 = normalize_thscd(thscd)

        limit = fallback(limit, self.degrain_args.get("limit"), default=None)
        limit_list = normalize_seq(limit, 2) if limit is not None else limit
        limit_list = (
            [scale_delta(v, 8, clip) if math.isfinite(v) else v for v in limit_list]
            if limit_list is not None
            else limit_list
        )

        planes_list = normalize_planes(clip, planes)

        degrain_args = self.degrain_args | KwargsNotNone(
            thsad=thsad,
            planes=planes_list,
            limit=limit_list,
            thscd1=thscd1,
            thscd2=thscd2,
            weights=weights,
        )

        vects_combined = list(chain.from_iterable(zip(vect_b, vect_f)))

        return core.mvu.Degrain(clip, super_clip, vects_combined, **degrain_args)

    def flow_interpolate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
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

        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_interpolate_args = self.flow_interpolate_args | KwargsNotNone(
            time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        interpolated = core.mvu.FlowInter(clip, super_clip, [vect_b[0], vect_f[0]], **flow_interpolate_args)
        return core.std.Interleave([clip, interpolated]) if interleave else interpolated

    def flow_fps(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
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
        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_fps_args: dict[str, Any] = KwargsNotNone(
            extramask=extramask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        if fps is not None:
            flow_fps_args.update(num=fps.numerator, den=fps.denominator)

        return core.mvu.FlowFPS(clip, super_clip, [vect_b[0], vect_f[0]], **self.flow_fps_args | flow_fps_args)

    def flow_blur(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
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
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
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
        super_clip = self.super(fallback(super, clip), vectors=vectors, onelevel=True)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_blur_args = self.flow_blur_args | KwargsNotNone(blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2)

        return core.mvu.FlowBlur(clip, super_clip, [vect_b[0], vect_f[0]], **flow_blur_args)

    def mask(
        self,
        vectors: MotionVectors | None = None,
        direction: Literal[MVDirection.FORWARD, MVDirection.BACKWARD] = MVDirection.FORWARD,
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
            direction: Motion vector direction to use.
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
        vect = vectors.get_vector(direction, delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        match kind:
            case MaskMode.VECTOR_LENGTH:
                mask_func = core.mvu.VectorLengthMask
            case MaskMode.SAD:
                mask_func = core.mvu.SADMask
            case MaskMode.OCCLUSION:
                mask_func = core.mvu.OcclusionMask

        return mask_func(
            vect,
            **self.mask_args | KwargsNotNone(ml=ml, gamma=gamma, time=time, scval=scval, thscd1=thscd1, thscd2=thscd2),
        )

    def sc_detection(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int = 1,
        thscd: int | tuple[int | None, float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Creates scene change frameprops from motion vectors data.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            delta: Motion vector delta to use.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Clip with scene change properties set.
        """

        clip = fallback(clip, self.clip)
        vectors = fallback(vectors, self.vectors)

        thscd1, thscd2 = normalize_thscd(thscd)

        sc_detection_args = self.sc_detection_args | KwargsNotNone(thscd1=thscd1, thscd2=thscd2)

        detect = clip
        for direction in MVDirection:
            detect = core.mvu.SCDetection(detect, vectors.get_vector(direction, delta), **sc_detection_args)

        return detect
