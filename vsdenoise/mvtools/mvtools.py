from __future__ import annotations

from fractions import Fraction
from itertools import chain
from typing import Any, Literal, Union, overload

from jetpytools import KwargsNotNone, KwargsT, fallback, normalize_seq

from vstools import (
    ColorRange,
    Field,
    Planes,
    UnsupportedColorFamilyError,
    VSFunctionNoArgs,
    VSObject,
    core,
    depth,
    scale_delta,
    vs,
)

from .enums import (
    FlowMode,
    MaskMode,
    MotionMode,
    MVDirection,
    PenaltyMode,
    RFilterMode,
    SADMode,
    SearchMode,
    SharpMode,
)
from .motion import MotionVectors
from .utils import normalize_thscd, planes_to_mvtools

__all__ = ["MVTools"]


class MVTools(VSObject):
    """
    MVTools wrapper for motion analysis, degraining, compensation, interpolation, etc.
    """

    super_args: KwargsT
    """Arguments passed to every [MVTools.super][vsdenoise.MVTools.super] call."""

    analyze_args: KwargsT
    """Arguments passed to every [MVTools.analyze][vsdenoise.MVTools.analyze] call."""

    recalculate_args: KwargsT
    """Arguments passed to every [MVTools.recalculate][vsdenoise.MVTools.recalculate] call."""

    compensate_args: KwargsT
    """Arguments passed to every [MVTools.compensate][vsdenoise.MVTools.compensate] call."""

    flow_args: KwargsT
    """Arguments passed to every [MVTools.flow][vsdenoise.MVTools.flow] call."""

    degrain_args: KwargsT
    """Arguments passed to every [MVTools.degrain][vsdenoise.MVTools.degrain] call."""

    flow_interpolate_args: KwargsT
    """Arguments passed to every [MVTools.flow_interpolate][vsdenoise.MVTools.flow_interpolate] call."""

    flow_fps_args: KwargsT
    """Arguments passed to every [MVTools.flow_fps][vsdenoise.MVTools.flow_fps] call."""

    block_fps_args: KwargsT
    """Arguments passed to every [MVTools.block_fps][vsdenoise.MVTools.block_fps] call."""

    flow_blur_args: KwargsT
    """Arguments passed to every [MVTools.flow_blur][vsdenoise.MVTools.flow_blur] call."""

    mask_args: KwargsT
    """Arguments passed to every [MVTools.mask][vsdenoise.MVTools.mask] call."""

    sc_detection_args: KwargsT
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
        field: Field | None = None,
        *,
        super_args: KwargsT | None = None,
        analyze_args: KwargsT | None = None,
        recalculate_args: KwargsT | None = None,
        compensate_args: KwargsT | None = None,
        flow_args: KwargsT | None = None,
        degrain_args: KwargsT | None = None,
        flow_interpolate_args: KwargsT | None = None,
        flow_fps_args: KwargsT | None = None,
        block_fps_args: KwargsT | None = None,
        flow_blur_args: KwargsT | None = None,
        mask_args: KwargsT | None = None,
        sc_detection_args: KwargsT | None = None,
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
            block_fps_args: Arguments passed to every [MVTools.block_fps][vsdenoise.MVTools.block_fps] call.
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

        self.super_args = fallback(super_args, KwargsT())
        self.analyze_args = fallback(analyze_args, KwargsT())
        self.recalculate_args = fallback(recalculate_args, KwargsT())
        self.compensate_args = fallback(compensate_args, KwargsT())
        self.degrain_args = fallback(degrain_args, KwargsT())
        self.flow_args = fallback(flow_args, KwargsT())
        self.flow_interpolate_args = fallback(flow_interpolate_args, KwargsT())
        self.flow_fps_args = fallback(flow_fps_args, KwargsT())
        self.block_fps_args = fallback(block_fps_args, KwargsT())
        self.flow_blur_args = fallback(flow_blur_args, KwargsT())
        self.mask_args = fallback(mask_args, KwargsT())
        self.sc_detection_args = fallback(sc_detection_args, KwargsT())

    def super(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        levels: int | None = None,
        sharp: SharpMode | None = None,
        rfilter: RFilterMode | None = None,
        pelclip: vs.VideoNode | VSFunctionNoArgs | None = None,
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
            levels: The number of hierarchical levels in super clip frames.
                More levels are needed for [analyze][vsdenoise.MVTools.analyze] to get better vectors,
                but fewer levels are needed for the actual motion compensation.
                0 = auto, all possible levels are produced.
            sharp: Subpixel interpolation method if pel is 2 or 4.
                For more information, see [SharpMode][vsdenoise.SharpMode].
            rfilter: Hierarchical levels smoothing and reducing (halving) filter. For more information, see
                [RFilterMode][vsdenoise.RFilterMode].
            pelclip: Optional upsampled source clip to use instead of internal subpixel interpolation (if pel > 1). The
                clip must contain the original source pixels at positions that are multiples of pel (e.g., positions 0,
                2, 4, etc. for pel=2), with interpolated pixels in between. The clip should not be padded.

        Returns:
            The original clip with the super clip attached as a frame property.
        """

        clip = fallback(clip, self.clip)

        vectors = fallback(vectors, self.vectors)

        if vectors.scaled:
            hpad, vpad = vectors.analysis_data["Analysis_Padding"]
        else:
            hpad, vpad = self.pad

        if callable(pelclip):
            pelclip = pelclip(clip)

        super_args = self.super_args | KwargsNotNone(
            hpad=hpad,
            vpad=vpad,
            pel=self.pel,
            levels=levels,
            chroma=self.chroma,
            sharp=sharp,
            rfilter=rfilter,
            pelclip=pelclip,
        )

        if levels := super_args.pop("levels", None) is None and clip is not self.search_clip:
            levels = 1

        super_clip = core.proxied.mv.Super(clip, levels=levels, **super_args)

        super_clip = clip.std.ClipToProp(super_clip, prop="MSuper")

        if clip is self.clip:
            self.clip = super_clip

        if clip is self.search_clip:
            self.search_clip = super_clip

        return super_clip

    def analyze(
        self,
        super: vs.VideoNode | None = None,
        tr: int = 1,
        blksize: int | tuple[int | None, int | None] | None = None,
        levels: int | None = None,
        search: SearchMode | None = None,
        searchparam: int | None = None,
        pelsearch: int | None = None,
        lambda_: int | None = None,
        truemotion: MotionMode | None = None,
        lsad: int | None = None,
        plevel: PenaltyMode | None = None,
        global_: bool | None = None,
        pnew: int | None = None,
        pzero: int | None = None,
        pglobal: int | None = None,
        overlap: int | tuple[int | None, int | None] | None = None,
        divide: bool | None = None,
        badsad: int | None = None,
        badrange: int | None = None,
        meander: bool | None = None,
        trymany: bool | None = None,
        dct: SADMode | None = None,
        scale_lambda: bool = True,
    ) -> None:
        """
        Analyze motion vectors in a clip using block matching.

        Takes a prepared super clip (containing hierarchical frame data) and estimates motion by comparing blocks
        between frames.
        Outputs motion vector data that can be used by other functions for motion compensation.

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
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
            levels: Number of levels used in hierarchical motion vector analysis. A positive value specifies how many
                levels to use. A negative or zero value specifies how many coarse levels to skip. Lower values generally
                give better results since vectors of any length can be found. Sometimes adding more levels can help
                prevent false vectors in CGI or similar content.
            search: Search algorithm to use at the finest level. See [SearchMode][vsdenoise.SearchMode] for options.
            searchparam: Additional parameter for the search algorithm. For NSTEP, this is the step size. For
                EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
            lambda_: Controls the coherence of the motion vector field. Higher values enforce more coherent/smooth
                motion between blocks. Too high values may cause the algorithm to miss the optimal vectors.
            truemotion: Preset that controls the default values of motion estimation parameters to optimize for true
                motion. For more information, see [MotionMode][vsdenoise.MotionMode].
            lsad: SAD limit for lambda. When the SAD value of a vector predictor (formed from neighboring blocks)
                exceeds this limit, the local lambda value is decreased. This helps prevent the use of bad predictors,
                but reduces motion coherence between blocks.
            plevel: Controls how the penalty factor (lambda) scales with hierarchical levels. For more information, see
                [PenaltyMode][vsdenoise.PenaltyMode].
            global_: Whether to estimate global motion at each level and use it as an additional predictor. This can
                help with camera motion.
            pnew: Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                Higher values make the search more conservative.
            pzero: Penalty multiplier (relative to 256) applied to the SAD cost for the zero motion vector. Higher
                values discourage using zero motion.
            pglobal: Penalty multiplier (relative to 256) applied to the SAD cost when using the global motion
                predictor.
            overlap: Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal,
                vertical) overlap values. Each value must be even and less than its corresponding block size dimension.
            divide: Whether to divide each block into 4 subblocks during post-processing. This may improve accuracy at
                the cost of performance.
            badsad: SAD threshold above which a wider secondary search will be performed to find better motion vectors.
                Higher values mean fewer blocks will trigger the secondary search.
            badrange: Search radius for the secondary search when a block's SAD exceeds badsad.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            trymany: Whether to test multiple predictor vectors during the search process at coarser levels. Enabling
                this can find better vectors but increases processing time.
            dct: SAD calculation mode using block DCT (frequency spectrum) for comparing blocks. For more information,
                see [SADMode][vsdenoise.SADMode].
            scale_lambda: Whether to scale lambda_ value according to truemotion's default value formula.

        Returns:
            A [MotionVectors][vsdenoise.MotionVectors] object containing the analyzed motion vectors for each frame.
            These vectors describe the estimated motion between frames and can be used for motion compensation.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        if scale_lambda and lambda_ and blksize:
            lambda_ = lambda_ * blksize * fallback(blksizev, blksize) // 64

        analyze_args = self.analyze_args | KwargsNotNone(
            blksize=blksize,
            blksizev=blksizev,
            levels=levels,
            search=search,
            searchparam=searchparam,
            pelsearch=pelsearch,
            lambda_=lambda_,
            chroma=self.chroma,
            truemotion=truemotion,
            lsad=lsad,
            plevel=plevel,
            global_=global_,
            pnew=pnew,
            pzero=pzero,
            pglobal=pglobal,
            overlap=overlap,
            overlapv=overlapv,
            divide=divide,
            badsad=badsad,
            badrange=badrange,
            meander=meander,
            trymany=trymany,
            dct=dct,
            fields=self.fields,
            tff=self.tff,
        )

        self.vectors.clear()

        for delta in range(1, tr + 1):
            for direction in MVDirection:
                self.vectors.set_vector(
                    core.proxied.mv.Analyse(
                        super_clip, isb=direction is MVDirection.BACKWARD, delta=delta, **analyze_args
                    ),
                    direction,
                    delta,
                )

    def recalculate(
        self,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        thsad: int | None = None,
        blksize: int | tuple[int | None, int | None] | None = None,
        search: SearchMode | None = None,
        searchparam: int | None = None,
        lambda_: int | None = None,
        truemotion: MotionMode | None = None,
        pnew: int | None = None,
        overlap: int | tuple[int | None, int | None] | None = None,
        divide: bool | None = None,
        meander: bool | None = None,
        dct: SADMode | None = None,
        scale_lambda: bool = True,
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
            searchparam: Additional parameter for the search algorithm. For NSTEP, this is the step size. For
                EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
            lambda_: Controls the coherence of the motion vector field. Higher values enforce more coherent/smooth
                motion between blocks. Too high values may cause the algorithm to miss the optimal vectors.
            truemotion: Preset that controls the default values of motion estimation parameters to optimize for true
                motion. For more information, see [MotionMode][vsdenoise.MotionMode].
            pnew: Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                Higher values make the search more conservative.
            overlap: Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal,
                vertical) overlap values. Each value must be even and less than its corresponding block size dimension.
            divide: Whether to divide each block into 4 subblocks during post-processing. This may improve accuracy at
                the cost of performance.
            meander: Whether to use a meandering scan pattern when processing blocks. If True, alternates between left-
                to-right and right-to-left scanning between rows to improve motion coherence.
            dct: SAD calculation mode using block DCT (frequency spectrum) for comparing blocks. For more information,
                see [SADMode][vsdenoise.SADMode].
            scale_lambda: Whether to scale lambda_ value according to truemotion's default value formula.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        vectors = fallback(vectors, self.vectors)

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        if scale_lambda and lambda_ and blksize:
            lambda_ = lambda_ * blksize * fallback(blksizev, blksize) // 64

        recalculate_args = self.recalculate_args | KwargsNotNone(
            thsad=thsad,
            blksize=blksize,
            blksizev=blksizev,
            search=search,
            searchparam=searchparam,
            lambda_=lambda_,
            chroma=self.chroma,
            truemotion=truemotion,
            pnew=pnew,
            overlap=overlap,
            overlapv=overlapv,
            divide=divide,
            meander=meander,
            dct=dct,
            fields=self.fields,
            tff=self.tff,
        )

        del vectors.analysis_data

        for delta in range(1, vectors.tr + 1):
            for direction in MVDirection:
                vectors.set_vector(
                    core.proxied.mv.Recalculate(super_clip, vectors.get_vector(direction, delta), **recalculate_args),
                    direction,
                    delta,
                )

    @overload
    def compensate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None,
        scbehavior: bool | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        scbehavior: bool | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        scbehavior: bool | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        scbehavior: bool | None = None,
        thsad: int | None = None,
        time: float | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = True,
        temporal_func: VSFunctionNoArgs | None = None,
    ) -> Union[
        vs.VideoNode,
        tuple[list[vs.VideoNode], list[vs.VideoNode]],
        tuple[vs.VideoNode, tuple[int, int]],
    ]:
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
            scbehavior: Whether to keep the current frame on scene changes. If True, the frame is left unchanged. If
                False, the reference frame is copied.
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
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(direction, tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        compensate_args = self.compensate_args | KwargsNotNone(
            scbehavior=scbehavior,
            thsad=thsad,
            time=time,
            thscd1=thscd1,
            thscd2=thscd2,
            fields=self.fields,
            tff=self.tff,
        )

        comp_back, comp_fwrd = [
            [core.proxied.mv.Compensate(clip, super_clip, vectors=vect, **compensate_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (comp_back, comp_fwrd)

        comp_clips = [*comp_fwrd, clip, *comp_back]
        cycle = len(comp_clips)
        offset = (cycle - 1) // 2

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
        time: float | None = None,
        mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        time: float | None = None,
        mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        time: float | None = None,
        mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        time: float | None = None,
        mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = True,
        temporal_func: VSFunctionNoArgs | None = None,
    ) -> Union[
        vs.VideoNode,
        tuple[list[vs.VideoNode], list[vs.VideoNode]],
        tuple[vs.VideoNode, tuple[int, int]],
    ]:
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
            tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
            time: Time position between frames as a percentage (0.0-100.0). Controls the interpolation position between
                frames.
            mode: Method for positioning pixels during motion compensation.
                See [FlowMode][vsdenoise.FlowMode] for options.
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
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(direction, tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_args = self.flow_args | KwargsNotNone(
            time=time,
            mode=mode,
            thscd1=thscd1,
            thscd2=thscd2,
            fields=self.fields,
            tff=self.tff,
        )

        flow_back, flow_fwrd = [
            [core.proxied.mv.Flow(clip, super_clip, vectors=vect, **flow_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (flow_back, flow_fwrd)

        flow_clips = [*flow_fwrd, clip, *flow_back]
        cycle = len(flow_clips)
        offset = (cycle - 1) // 2

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
        thsad: int | tuple[int | None, int | None] | None = None,
        limit: float | tuple[float | None, float | None] | None = None,
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
            thsad: Defines the soft threshold of block sum absolute differences. Blocks with SAD above this threshold
                have zero weight for averaging (denoising). Blocks with low SAD have highest weight. The remaining
                weight is taken from pixels of source clip.
            limit: Maximum allowed change in pixel values (8 bits scale).
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.
            planes: Which planes to process. Default: None (all planes).

        Returns:
            Motion compensated and temporally filtered clip with reduced noise.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        tr = fallback(tr, vectors.tr)
        vect_b, vect_f = vectors.get_vectors(tr=tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        thsad, thsadc = normalize_seq(thsad, 2)
        nlimit, nlimitc = normalize_seq(limit, 2)

        if nlimit is not None:
            nlimit = scale_delta(nlimit, 8, clip)

        if nlimitc is not None:
            nlimitc = scale_delta(nlimitc, 8, clip)

        degrain_args = self.degrain_args | KwargsNotNone(
            thsad=thsad,
            thsadc=thsadc,
            plane=planes_to_mvtools(clip, planes),
            limit=nlimit,
            limitc=nlimitc,
            thscd1=thscd1,
            thscd2=thscd2,
        )

        return getattr(core.proxied.mv, f"Degrain{tr}")(
            clip, super_clip, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
        )

    def flow_interpolate(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        time: float | None = None,
        ml: float | None = None,
        blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
            List of the motion interpolated frames if interleave=False else a motion interpolated clip.
        """
        clip = fallback(clip, self.clip)

        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_interpolate_args = self.flow_interpolate_args | KwargsNotNone(
            time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        interpolated = core.proxied.mv.FlowInter(clip, super_clip, vect_b[0], vect_f[0], **flow_interpolate_args)

        if interleave:
            interpolated = core.std.Interleave([clip, interpolated])

        return interpolated

    def flow_fps(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        fps: Fraction | None = None,
        mask: int | None = None,
        ml: float | None = None,
        blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
            mask: Processing mask mode for handling occlusions and motion failures.
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
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_fps_args: dict[str, Any] = KwargsNotNone(mask=mask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            flow_fps_args.update(num=fps.numerator, den=fps.denominator)

        flow_fps_args = self.flow_fps_args | flow_fps_args

        return core.proxied.mv.FlowFPS(clip, super_clip, vect_b[0], vect_f[0], **flow_fps_args)

    def block_fps(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        fps: Fraction | None = None,
        mode: int | None = None,
        ml: float | None = None,
        blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames
        using block-based motion compensation.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        Args:
            clip: The clip to process.
            super: The multilevel super clip prepared by [super][vsdenoise.MVTools.super].
                If None, super will be obtained from clip.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            fps: Target output framerate as a Fraction.
            mode: Processing mask mode for handling occlusions and motion failures.
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
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        block_fps_args: dict[str, Any] = KwargsNotNone(mode=mode, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            block_fps_args.update(num=fps.numerator, den=fps.denominator)

        block_fps_args = self.block_fps_args | block_fps_args

        return core.proxied.mv.BlockFPS(clip, super_clip, vect_b[0], vect_f[0], **block_fps_args)

    def flow_blur(
        self,
        clip: vs.VideoNode | None = None,
        super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        blur: float | None = None,
        prec: int | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        vect_b, vect_f = vectors.get_vectors(tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_blur_args = self.flow_blur_args | KwargsNotNone(blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2)

        return core.proxied.mv.FlowBlur(clip, super_clip, vect_b[0], vect_f[0], **flow_blur_args)

    def mask(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        direction: Literal[MVDirection.FORWARD, MVDirection.BACKWARD] = MVDirection.FORWARD,
        delta: int = 1,
        ml: float | None = None,
        gamma: float | None = None,
        kind: MaskMode | None = None,
        time: float | None = None,
        ysc: int | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
    ) -> vs.VideoNode:
        """
        Creates a mask clip from motion vectors data.

        Args:
            clip: The clip to process. If None, the [clip][vsdenoise.MVTools.clip] attribute is used.
            vectors: Motion vectors to use. If None, uses the vectors from this instance.
            direction: Motion vector direction to use.
            delta: Motion vector delta to use.
            ml: Motion length scale factor. When the vector's length (or other mask value) is greater than or equal to
                ml, the output is saturated to 255.
            gamma: Exponent for the relation between input and output values. 1.0 gives a linear relation, 2.0 gives a
                quadratic relation.
            kind: Type of mask to generate. See [MaskMode][vsdenoise.MaskMode] for options.
            time: Time position between frames as a percentage (0.0-100.0).
            ysc: Value assigned to the mask on scene changes.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

        Returns:
            Motion mask clip.
        """

        clip = fallback(clip, self.clip)

        vectors = fallback(vectors, self.vectors)
        vect = vectors.get_vector(direction, delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        mask_args = self.mask_args | KwargsNotNone(
            ml=ml, gamma=gamma, kind=kind, time=time, ysc=ysc, thscd1=thscd1, thscd2=thscd2
        )

        mask_clip = core.proxied.mv.Mask(depth(clip, 8), vect, **mask_args)

        return depth(mask_clip, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)

    def sc_detection(
        self,
        clip: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None,
        delta: int = 1,
        thscd: int | tuple[int | None, int | float | None] | None = None,
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
            detect = core.proxied.mv.SCDetection(detect, vectors.get_vector(direction, delta), **sc_detection_args)

        return detect

    def get_super(self, clip: vs.VideoNode | None = None) -> vs.VideoNode:
        """
        Get the super clips from the specified clip.

        If [super][vsdenoise.MVTools.super] wasn't previously called,
        it will do so here with default values or kwargs specified in the constructor.

        Args:
            clip: The clip to get the super clip from.

        Returns:
            VideoNode containing the super clip.
        """

        clip = fallback(clip, self.clip)

        try:
            super_clip = clip.std.PropToClip(prop="MSuper")
        except vs.Error:
            clip = self.super(clip)
            super_clip = clip.std.PropToClip(prop="MSuper")

        return super_clip
