from __future__ import annotations

from fractions import Fraction
from itertools import chain
from typing import Any, Literal, overload

from vstools import (
    ColorRange, CustomRuntimeError, FieldBased, GenericVSFunction, InvalidColorFamilyError,
    KwargsNotNone, KwargsT, PlanesT, VSFunction, check_variable, core, depth,
    disallow_variable_format, disallow_variable_resolution, fallback, get_prop, normalize_planes,
    normalize_seq, scale_delta, vs
)

from .enums import (
    FlowMode, MaskMode, MotionMode, MVDirection, MVToolsPlugin, PenaltyMode, RFilterMode, SADMode,
    SearchMode, SharpMode, SmoothMode
)
from .motion import MotionVectors
from .utils import normalize_thscd, planes_to_mvtools

__all__ = [
    'MVTools'
]


class MVTools:
    """MVTools wrapper for motion analysis, degraining, compensation, interpolation, etc."""

    super_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Super` call."""

    analyze_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Analyze` call."""

    recalculate_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Recalculate` call."""

    compensate_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Compensate` call."""

    flow_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Flow` call."""

    degrain_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Degrain` call."""

    flow_interpolate_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.FlowInter` call."""

    flow_fps_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.FlowFPS` call."""

    block_fps_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.BlockFPS` call."""

    flow_blur_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.FlowBlur` call."""

    mask_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.Mask` call."""

    sc_detection_args: KwargsT
    """Arguments passed to every :py:attr:`MVToolsPlugin.SCDetection` call."""

    vectors: MotionVectors
    """Motion vectors analyzed and used for all operations."""

    clip: vs.VideoNode
    """Clip to process."""

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode, search_clip: vs.VideoNode | GenericVSFunction | None = None,
        vectors: MotionVectors | MVTools | None = None,
        tr: int = 1, pad: int | tuple[int | None, int | None] | None = None,
        pel: int | None = None, planes: PlanesT = None,
        *,
        # kwargs for mvtools calls
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
        sc_detection_args: KwargsT | None = None
    ) -> None:
        """
        MVTools is a collection of functions for motion estimation and compensation in video.

        Motion compensation may be used for strong temporal denoising, advanced framerate conversions,
        image restoration, and other similar tasks.

        The plugin uses a block-matching method of motion estimation (similar methods as used in MPEG2, MPEG4, etc.).
        During the analysis stage the plugin divides frames into smaller blocks and tries to find the most similar matching block
        for every block in current frame in the second frame (which is either the previous or next frame).
        The relative shift of these blocks is the motion vector.

        The main method of measuring block similarity is by calculating the sum of absolute differences (SAD)
        of all pixels of these two blocks, which indicates how correct the motion estimation was.

        :param clip:                     The clip to process.
        :param search_clip:              Optional clip or callable to be used for motion vector gathering only.
        :param vectors:                  Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                                         If None, uses the vectors from this instance.
        :param tr:                       The temporal radius. This determines how many frames are analyzed before/after the current frame.
                                         Default: 1.
        :param pad:                      How much padding to add to the source frame.
                                         Small padding is added to help with motion estimation near frame borders.
        :param pel:                      Subpixel precision for motion estimation (1=pixel, 2=half-pixel, 4=quarter-pixel).
                                         Default: 1.
        :param planes:                   Which planes to process. Default: None (all planes).
        :param super_args:               Arguments passed to every :py:attr:`MVToolsPlugin.Super` calls.
        :param analyze_args:             Arguments passed to every :py:attr:`MVToolsPlugin.Analyze` calls.
        :param recalculate_args:         Arguments passed to every :py:attr:`MVToolsPlugin.Recalculate` calls.
        :param compensate_args:          Arguments passed to every :py:attr:`MVToolsPlugin.Compensate` calls.
        :param flow_args:                Arguments passed to every :py:attr:`MVToolsPlugin.Flow` calls.
        :param degrain_args:             Arguments passed to every :py:attr:`MVToolsPlugin.Degrain` calls.
        :param flow_interpolate_args:    Arguments passed to every :py:attr:`MVToolsPlugin.FlowInter` calls.
        :param flow_fps_args:            Arguments passed to every :py:attr:`MVToolsPlugin.FlowFPS` calls.
        :param block_fps_args:           Arguments passed to every :py:attr:`MVToolsPlugin.BlockFPS` calls.
        :param flow_blur_args:           Arguments passed to every :py:attr:`MVToolsPlugin.FlowBlur` calls.
        :param mask_args:                Arguments passed to every :py:attr:`MVToolsPlugin.Mask` calls.
        :param sc_detection_args:        Arguments passed to every :py:attr:`MVToolsPlugin.SCDetection` calls.
        """

        assert check_variable(clip, self.__class__)

        InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), self.__class__)

        if isinstance(vectors, MVTools):
            self.vectors = vectors.vectors
        elif isinstance(vectors, MotionVectors):
            self.vectors = vectors
        else:
            self.vectors = MotionVectors()

        self.mvtools = MVToolsPlugin.from_video(clip)
        self.fieldbased = FieldBased.from_video(clip, False, self.__class__)
        self.clip = clip.std.SeparateFields(self.fieldbased.is_tff) if self.fieldbased.is_inter else clip

        self.planes = normalize_planes(self.clip, planes)
        self.mv_plane = planes_to_mvtools(self.planes)
        self.chroma = self.mv_plane != 0

        self.tr = tr
        self.pel = pel
        self.pad = normalize_seq(pad, 2)

        if callable(search_clip):
            try:
                self.search_clip = search_clip(self.clip, planes=self.planes)
            except TypeError:
                self.search_clip = search_clip(self.clip)
        else:
            self.search_clip = fallback(search_clip, self.clip)

        self.disable_compensate = False

        if self.mvtools is MVToolsPlugin.FLOAT:
            self.disable_manipmv = True
            self.disable_degrain = True if tr == 1 else False
        else:
            self.disable_manipmv = False
            self.disable_degrain = False

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
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | MVTools | None = None, 
        levels: int | None = None, sharp: SharpMode | None = None,
        rfilter: RFilterMode | None = None, pelclip: vs.VideoNode | VSFunction | None = None
    ) -> vs.VideoNode:
        """
        Get source clip and prepare special "super" clip with multilevel (hierarchical scaled) frames data.
        The super clip is used by both :py:attr:`analyze` and motion compensation (client) functions.

        You can use different Super clip for generation vectors with :py:attr:`analyze` and a different super clip format for the actual action.
        Source clip is appended to clip's frameprops, :py:attr:`get_super` can be used to extract the super clip if you wish to view it yourself.

        :param clip:       The clip to process. If None, the :py:attr:`clip` attribute is used.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param levels:     The number of hierarchical levels in super clip frames.
                           More levels are needed for :py:attr:`analyze` to get better vectors,
                           but fewer levels are needed for the actual motion compensation.
                           0 = auto, all possible levels are produced.
        :param sharp:      Subpixel interpolation method if pel is 2 or 4.
                           For more information, see :py:class:`SharpMode`.
        :param rfilter:    Hierarchical levels smoothing and reducing (halving) filter.
                           For more information, see :py:class:`RFilterMode`.
        :param pelclip:    Optional upsampled source clip to use instead of internal subpixel interpolation (if pel > 1).
                           The clip must contain the original source pixels at positions that are multiples of pel
                           (e.g., positions 0, 2, 4, etc. for pel=2), with interpolated pixels in between.
                           The clip should not be padded.

        :return:           The original clip with the super clip attached as a frame property.
        """

        clip = fallback(clip, self.clip)

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        if vectors.scaled:
            self.expand_analysis_data(vectors)

            hpad, vpad = vectors.analysis_data['Analysis_Padding']
        else:
            hpad, vpad = self.pad

        if callable(pelclip):
            pelclip = pelclip(clip)

        super_args = self.super_args | KwargsNotNone(
            hpad=hpad, vpad=vpad, pel=self.pel, levels=levels, chroma=self.chroma,
            sharp=sharp, rfilter=rfilter, pelclip=pelclip
        )

        super_clip = self.mvtools.Super(clip, **super_args)

        super_clip = clip.std.ClipToProp(super_clip, prop='MSuper')

        if clip is self.clip:
            self.clip = super_clip
        if clip is self.search_clip:
            self.search_clip = super_clip

        return super_clip

    def analyze(
        self, super: vs.VideoNode | None = None, blksize: int | tuple[int | None, int | None] | None = None,
        levels: int | None = None, search: SearchMode | None = None, searchparam: int | None = None,
        pelsearch: int | None = None, lambda_: int | None = None, truemotion: MotionMode | None = None,
        lsad: int | None = None, plevel: PenaltyMode | None = None, global_: bool | None = None,
        pnew: int | None = None, pzero: int | None = None, pglobal: int | None = None,
        overlap: int | tuple[int | None, int | None] | None = None, divide: bool | None = None,
        badsad: int | None = None, badrange: int | None = None, meander: bool | None = None,
        trymany: bool | None = None, dct: SADMode | None = None
    ) -> None:
        """
        Analyze motion vectors in a clip using block matching.

        Takes a prepared super clip (containing hierarchical frame data) and estimates motion by comparing blocks between frames.
        Outputs motion vector data that can be used by other functions for motion compensation.

        The motion vector search is performed hierarchically, starting from a coarse image scale and progressively refining to finer scales.
        For each block, the function first checks predictors like the zero vector and neighboring block vectors.

        This method calculates the Sum of Absolute Differences (SAD) for these predictors,
        then iteratively tests new candidate vectors by adjusting the current best vector.
        The vector with the lowest SAD value is chosen as the final motion vector,
        with a penalty applied to maintain motion coherence between blocks.

        :param super:          The multilevel super clip prepared by :py:attr:`super`.
                               If None, super will be obtained from clip.
        :param blksize:        Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
        :param levels:         Number of levels used in hierarchical motion vector analysis.
                               A positive value specifies how many levels to use.
                               A negative or zero value specifies how many coarse levels to skip.
                               Lower values generally give better results since vectors of any length can be found.
                               Sometimes adding more levels can help prevent false vectors in CGI or similar content.
        :param search:         Search algorithm to use at the finest level. See :py:class:`SearchMode` for options.
        :param searchparam:    Additional parameter for the search algorithm. For NSTEP, this is the step size.
                               For EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
        :param lambda_:        Controls the coherence of the motion vector field.
                               Higher values enforce more coherent/smooth motion between blocks.
                               Too high values may cause the algorithm to miss the optimal vectors.
        :param truemotion:     Preset that controls the default values of motion estimation parameters to optimize for true motion.
                               For more information, see :py:class:`MotionMode`.
        :param lsad:           SAD limit for lambda.
                               When the SAD value of a vector predictor (formed from neighboring blocks) exceeds this limit,
                               the local lambda value is decreased. This helps prevent the use of bad predictors,
                               but reduces motion coherence between blocks.
        :param plevel:         Controls how the penalty factor (lambda) scales with hierarchical levels.
                               For more information, see :py:class:`PenaltyMode`.
        :param global_:        Whether to estimate global motion at each level and use it as an additional predictor.
                               This can help with camera motion.
        :param pnew:           Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                               Higher values make the search more conservative.
        :param pzero:          Penalty multiplier (relative to 256) applied to the SAD cost for the zero motion vector.
                               Higher values discourage using zero motion.
        :param pglobal:        Penalty multiplier (relative to 256) applied to the SAD cost when using the global motion predictor.
        :param overlap:        Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap values.
                               Each value must be even and less than its corresponding block size dimension.
        :param divide:         Whether to divide each block into 4 subblocks during post-processing.
                               This may improve accuracy at the cost of performance.
        :param badsad:         SAD threshold above which a wider secondary search will be performed to find better motion vectors.
                               Higher values mean fewer blocks will trigger the secondary search.
        :param badrange:       Search radius for the secondary search when a block's SAD exceeds badsad.
        :param meander:        Whether to use a meandering scan pattern when processing blocks.
                               If True, alternates between left-to-right and right-to-left scanning between rows to improve motion coherence.
        :param trymany:        Whether to test multiple predictor vectors during the search process at coarser levels.
                               Enabling this can find better vectors but increases processing time.
        :param dct:            SAD calculation mode using block DCT (frequency spectrum) for comparing blocks.
                               For more information, see :py:class:`SADMode`.

        :return:               A :py:class:`MotionVectors` object containing the analyzed motion vectors for each frame.
                               These vectors describe the estimated motion between frames and can be used for motion compensation.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        analyze_args = self.analyze_args | KwargsNotNone(
            blksize=blksize, blksizev=blksizev, levels=levels,
            search=search, searchparam=searchparam, pelsearch=pelsearch,
            lambda_=lambda_, chroma=self.chroma, truemotion=truemotion,
            lsad=lsad, plevel=plevel, global_=global_,
            pnew=pnew, pzero=pzero, pglobal=pglobal,
            overlap=overlap, overlapv=overlapv, divide=divide,
            badsad=badsad, badrange=badrange, meander=meander, trymany=trymany,
            fields=self.fieldbased.is_inter, tff=self.fieldbased.is_tff, dct=dct
        )

        if self.mvtools is MVToolsPlugin.INTEGER and not any(
            (analyze_args.get('overlap'), analyze_args.get('overlapv'))
        ):
            self.disable_compensate = True

        if self.mvtools is MVToolsPlugin.FLOAT:
            self.vectors.vmulti = self.mvtools.Analyze(super_clip, radius=self.tr, **analyze_args)
        else:
            for i in range(1, self.tr + 1):
                for direction in MVDirection:
                    vector = self.mvtools.Analyze(
                        super_clip, isb=direction is MVDirection.BACK, delta=i, **analyze_args
                    )
                    self.vectors.set_mv(direction, i, vector)
                    
            self.vectors.analysis_data.clear()

    def recalculate(
        self, super: vs.VideoNode | None = None, vectors: MotionVectors | MVTools | None = None,
        thsad: int | None = None, smooth: SmoothMode | None = None,
        blksize: int | tuple[int | None, int | None] | None = None, search: SearchMode | None = None,
        searchparam: int | None = None, lambda_: int | None = None, truemotion: MotionMode | None = None,
        pnew: int | None = None, overlap: int | tuple[int | None, int | None] | None = None,
        divide: bool | None = None, meander: bool | None = None, dct: SADMode | None = None
    ) -> None:
        """
        Refines and recalculates motion vectors that were previously estimated, optionally using a different super clip or parameters.
        This two-stage approach can provide more stable and robust motion estimation.

        The refinement only occurs at the finest hierarchical level. It uses the interpolated vectors from the original blocks
        as predictors for the new vectors, and recalculates their SAD values.

        Only vectors with poor quality (SAD above threshold) will be re-estimated through a new search.
        The SAD threshold is normalized to an 8x8 block size. Vectors with good quality are preserved,
        though their SAD values are still recalculated and updated.

        :param super:          The multilevel super clip prepared by :py:attr:`super`.
                               If None, super will be obtained from clip.
        :param vectors:        Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                               If None, uses the vectors from this instance.
        :param thsad:          Only bad quality new vectors with a SAD above thid will be re-estimated by search.
                               thsad value is scaled to 8x8 block size.
        :param blksize:        Size of blocks for motion estimation. Can be an int or tuple of (width, height).
                               Larger blocks are less sensitive to noise and faster to process, but will produce less accurate vectors.
        :param smooth:         This is method for dividing coarse blocks into smaller ones.
                               Only used with the FLOAT MVTools plugin.
        :param search:         Search algorithm to use at the finest level. See :py:class:`SearchMode` for options.
        :param searchparam:    Additional parameter for the search algorithm. For NSTEP, this is the step size.
                               For EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
        :param lambda_:        Controls the coherence of the motion vector field.
                               Higher values enforce more coherent/smooth motion between blocks.
                               Too high values may cause the algorithm to miss the optimal vectors.
        :param truemotion:     Preset that controls the default values of motion estimation parameters to optimize for true motion.
                               For more information, see :py:class:`MotionMode`.
        :param pnew:           Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                               Higher values make the search more conservative.
        :param overlap:        Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap values.
                               Each value must be even and less than its corresponding block size dimension.
        :param divide:         Whether to divide each block into 4 subblocks during post-processing.
                               This may improve accuracy at the cost of performance.
        :param meander:        Whether to use a meandering scan pattern when processing blocks.
                               If True, alternates between left-to-right and right-to-left scanning between rows to improve motion coherence.
        :param dct:            SAD calculation mode using block DCT (frequency spectrum) for comparing blocks.
                               For more information, see :py:class:`SADMode`.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        if not vectors.has_vectors:
            raise CustomRuntimeError('You must run `analyze` before `recalculate`!', self.recalculate)

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        recalculate_args = self.recalculate_args | KwargsNotNone(
            thsad=thsad, smooth=smooth, blksize=blksize, blksizev=blksizev, search=search, searchparam=searchparam,
            lambda_=lambda_, chroma=self.chroma, truemotion=truemotion, pnew=pnew, overlap=overlap, overlapv=overlapv,
            divide=divide, meander=meander, fields=self.fieldbased.is_inter, tff=self.fieldbased.is_tff, dct=dct
        )

        if self.mvtools is MVToolsPlugin.INTEGER and not any(
            (recalculate_args.get('overlap'), recalculate_args.get('overlapv'))
        ):
            self.disable_compensate = True

        if self.mvtools is MVToolsPlugin.FLOAT:
            vectors.vmulti = self.mvtools.Recalculate(super_clip, vectors=vectors.vmulti, **recalculate_args)
        else:
            for i in range(1, self.tr + 1):
                for direction in MVDirection:
                    vector = self.mvtools.Recalculate(super_clip, vectors.get_mv(direction, i), **recalculate_args)
                    vectors.set_mv(direction, i, vector)

            vectors.analysis_data.clear()

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[True] = True, temporal_func: None = None
    ) -> vs.VideoNode:
        ...

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[True] = True, temporal_func: VSFunction = ...
    ) -> tuple[vs.VideoNode, tuple[int, int]]:
        ...

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[False] = False
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        ...

    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | None] | None = None,
        interleave: bool = True, temporal_func: VSFunction | None = None
    ) -> vs.VideoNode | tuple[list[vs.VideoNode], list[vs.VideoNode]] | tuple[vs.VideoNode, tuple[int, int]]:
        """
        Perform motion compensation by moving blocks from reference frames to the current frame according to motion vectors.
        This creates a prediction of the current frame by taking blocks from neighboring frames and moving them along their estimated motion paths.

        :param clip:             The clip to process.
        :param super:            The multilevel super clip prepared by :py:attr:`super`.
                                 If None, super will be obtained from clip.
        :param vectors:          Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                                 If None, uses the vectors from this instance.
        :param direction:        Motion vector direction to use.
        :param tr:               The temporal radius. This determines how many frames are analyzed before/after the current frame.
        :param scbehavior:       Whether to keep the current frame on scene changes.
                                 If True, the frame is left unchanged. If False, the reference frame is copied.
        :param thsad:            SAD threshold for safe compensation.
                                 If block SAD is above thsad, the source block is used instead of the compensated block.
        :param thsad2:           Define the SAD soft threshold for frames with the largest temporal distance.
                                 The actual SAD threshold for each reference frame is interpolated between thsad (nearest frames)
                                 and thsad2 (furthest frames).
                                 Only used with the FLOAT MVTools plugin.
        :param time:             Time position between frames as a percentage (0.0-100.0).
                                 Controls the interpolation position between frames.
        :param thscd:            Scene change detection thresholds.
                                 First value is the block change threshold between frames.
                                 Second value is the number of changed blocks needed for a scene change.
        :param interleave:       Whether to interleave the compensated frames with the input.
        :param temporal_func:    Temporal function to apply to the motion compensated frames.

        :return:                 Motion compensated frames if func is provided, otherwise returns a tuple containing:
                                 - The interleaved compensated frames.
                                 - A tuple of (total_frames, center_offset) for manual frame selection.
        """

        if self.disable_compensate:
            raise CustomRuntimeError('Motion analysis was performed without block overlap!', self.compensate)

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        tr = fallback(tr, self.tr)

        vect_b, vect_f = self.get_vectors(self.vectors, direction=direction, tr=tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        compensate_args = self.compensate_args | KwargsNotNone(
            scbehavior=scbehavior, thsad=thsad, thsad2=thsad2, time=time, fields=self.fieldbased.is_inter,
            thscd1=thscd1, thscd2=thscd2, tff=self.fieldbased.is_tff
        )

        comp_back, comp_fwrd = [
            [self.mvtools.Compensate(clip, super_clip, vectors=vect, **compensate_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (comp_back, comp_fwrd)

        comp_clips = [*comp_fwrd, clip, *comp_back]
        n_clips = len(comp_clips)
        offset = (n_clips - 1) // 2

        interleaved = core.std.Interleave(comp_clips)

        if temporal_func:
            processed = temporal_func(interleaved)

            return processed.std.SelectEvery(n_clips, offset)

        return interleaved, (n_clips, offset)

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[True] = True, temporal_func: None = None
    ) -> vs.VideoNode:
        ...

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[True] = True, temporal_func: VSFunction = ...
    ) -> tuple[vs.VideoNode, tuple[int, int]]:
        ...

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | None] | None = None,
        interleave: Literal[False] = False
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        ...

    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | None] | None = None,
        interleave: bool = True, temporal_func: VSFunction | None = None
    ) -> vs.VideoNode | tuple[list[vs.VideoNode], list[vs.VideoNode]] | tuple[vs.VideoNode, tuple[int, int]]:
        """
        Performs motion compensation using pixel-level motion vectors interpolated from block vectors.

        Unlike block-based compensation, this calculates a unique motion vector for each pixel by bilinearly interpolating
        between the motion vectors of the current block and its neighbors based on the pixel's position.
        The pixels in the reference frame are then moved along these interpolated vectors to their estimated positions in the current frame.

        :param clip:             The clip to process.
        :param super:            The multilevel super clip prepared by :py:attr:`super`.
                                 If None, super will be obtained from clip.
        :param vectors:          Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                                 If None, uses the vectors from this instance.
        :param direction:        Motion vector direction to use.
        :param tr:               The temporal radius. This determines how many frames are analyzed before/after the current frame.
        :param time:             Time position between frames as a percentage (0.0-100.0).
                                 Controls the interpolation position between frames.
        :param mode:             Method for positioning pixels during motion compensation.
                                 See :py:class:`FlowMode` for options.
        :param thscd:            Scene change detection thresholds as a tuple of (threshold1, threshold2).
                                 threshold1: SAD difference threshold between frames to consider a block changed
                                 threshold2: Number of changed blocks needed to trigger a scene change
        :param interleave:       Whether to interleave the compensated frames with the input.
        :param temporal_func:    Optional function to process the motion compensated frames.
                                 Takes the interleaved frames as input and returns processed frames.

        :return:                 Motion compensated frames if func is provided, otherwise returns a tuple containing:
                                 - The interleaved compensated frames.
                                 - A tuple of (total_frames, center_offset) for manual frame selection.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        tr = fallback(tr, self.tr)

        vect_b, vect_f = self.get_vectors(self.vectors, direction=direction, tr=tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_args = self.flow_args | KwargsNotNone(
            time=time, mode=mode, fields=self.fieldbased.is_inter,
            thscd1=thscd1, thscd2=thscd2, tff=self.fieldbased.is_tff
        )

        flow_back, flow_fwrd = [
            [self.mvtools.Flow(clip, super_clip, vectors=vect, **flow_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        if not interleave:
            return (flow_back, flow_fwrd)

        flow_clips = [*flow_fwrd, clip, *flow_back]
        n_clips = len(flow_clips)
        offset = (n_clips - 1) // 2

        interleaved = core.std.Interleave(flow_clips)

        if temporal_func:
            processed = temporal_func(interleaved)

            return processed.std.SelectEvery(n_clips, offset)

        return interleaved, (n_clips, offset)

    def degrain(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, tr: int | None = None,
        thsad: int | tuple[int | None, int | None] | None = None,
        thsad2: int | tuple[int | None, int | None] | None = None,
        limit: int | tuple[int | None, int | None] | None = None,
        thscd: int | tuple[int | None, int | None] | None = None,
    ) -> vs.VideoNode:
        """
        Perform temporal denoising using motion compensation.

        Motion compensated blocks from previous and next frames are averaged with the current frame.
        The weighting factors for each block depend on their SAD from the current frame.

        :param clip:       The clip to process.
                           If None, the :py:attr:`workclip` attribute is used.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param tr:         The temporal radius. This determines how many frames are analyzed before/after the current frame.
        :param thsad:      Defines the soft threshold of block sum absolute differences.
                           Blocks with SAD above this threshold have zero weight for averaging (denoising).
                           Blocks with low SAD have highest weight.
                           The remaining weight is taken from pixels of source clip.
        :param thsad2:     Define the SAD soft threshold for frames with the largest temporal distance.
                           The actual SAD threshold for each reference frame is interpolated between thsad (nearest frames)
                           and thsad2 (furthest frames).
                           Only used with the FLOAT MVTools plugin.
        :param limit:      Maximum allowed change in pixel values.
        :param thscd:      Scene change detection thresholds:
                           - First value: SAD threshold for considering a block changed between frames.
                           - Second value: Number of changed blocks needed to trigger a scene change.

        :return:           Motion compensated and temporally filtered clip with reduced noise.
        """

        if self.disable_degrain:
            raise CustomRuntimeError('Motion analysis was performed with a temporal radius of 1!', self.degrain)

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        tr = fallback(tr, self.tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        degrain_args = dict[str, Any](thscd1=thscd1, thscd2=thscd2, plane=self.mv_plane)

        if self.mvtools is MVToolsPlugin.FLOAT:
            degrain_args.update(thsad=thsad, thsad2=thsad2, limit=limit)
        else:
            vect_b, vect_f = self.get_vectors(vectors, tr=tr)

            thsad, thsadc = normalize_seq(thsad, 2)
            limit, limitc = normalize_seq(limit, 2)

            if limit is not None:
                limit = scale_delta(limit, 8, clip)  # type: ignore[assignment]

            if limitc is not None:
                limitc = scale_delta(limitc, 8, clip)  # type: ignore[assignment]

            degrain_args.update(thsad=thsad, thsadc=thsadc, limit=limit, limitc=limitc)

        degrain_args = self.degrain_args | KwargsNotNone(degrain_args)

        if self.mvtools is MVToolsPlugin.FLOAT:
            output = self.mvtools.Degrain()(clip, super_clip, vectors.vmulti, **degrain_args)
        else:
            output = self.mvtools.Degrain(tr)(
                clip, super_clip, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        return output

    def flow_interpolate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, time: float | None = None,
        ml: float | None = None, blend: bool | None = None, thscd: int | tuple[int | None, int | None] | None = None,
        interleave: bool = True
    ) -> vs.VideoNode:
        """
        Motion interpolation function that creates an intermediate frame between two frames.

        Uses both backward and forward motion vectors to estimate motion and create a frame at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:          The clip to process.
        :param super:         The multilevel super clip prepared by :py:attr:`super`.
                              If None, super will be obtained from clip.
        :param vectors:       Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                              If None, uses the vectors from this instance.
        :param time:          Time position between frames as a percentage (0.0-100.0).
                              Controls the interpolation position between frames.
        :param ml:            Mask scale parameter that controls occlusion mask strength.
                              Higher values produce weaker occlusion masks.
                              Used in MakeVectorOcclusionMaskTime for modes 3-5.
                              Used in MakeSADMaskTime for modes 6-8.
        :param blend:         Whether to blend frames at scene changes.
                              If True, frames will be blended. If False, frames will be copied.
        :param thscd:         Scene change detection thresholds.
                              First value is the block change threshold between frames.
                              Second value is the number of changed blocks needed for a scene change.
        :param interleave:    Whether to interleave the interpolated frames with the source clip.

        :return:              Motion interpolated clip with frames created
                              at the specified time position between input frames.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect_b, vect_f = self.get_vectors(self.vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_interpolate_args = self.flow_interpolate_args | KwargsNotNone(
            time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        interpolated = self.mvtools.FlowInter(clip, super_clip, vect_b, vect_f, **flow_interpolate_args)

        if interleave:
            interpolated = core.std.Interleave([clip, interpolated])

        return interpolated

    def flow_fps(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, fps: Fraction | None = None,
        mask: int | None = None, ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | None] | None = None
    ) -> vs.VideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param fps:        Target output framerate as a Fraction.
        :param mask:       Processing mask mode for handling occlusions and motion failures.
        :param ml:         Mask scale parameter that controls occlusion mask strength.
                           Higher values produce weaker occlusion masks.
                           Used in MakeVectorOcclusionMaskTime for modes 3-5.
                           Used in MakeSADMaskTime for modes 6-8.
        :param blend:      Whether to blend frames at scene changes.
                           If True, frames will be blended. If False, frames will be copied.
        :param thscd:      Scene change detection thresholds.
                           First value is the block change threshold between frames.
                           Second value is the number of changed blocks needed for a scene change.

        :return:           Clip with its framerate resampled.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect_b, vect_f = self.get_vectors(self.vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_fps_args: dict[str, Any] = KwargsNotNone(mask=mask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            flow_fps_args.update(num=fps.numerator, den=fps.denominator)

        flow_fps_args = self.flow_fps_args | flow_fps_args

        return self.mvtools.FlowFPS(clip, super_clip, vect_b, vect_f, **flow_fps_args)

    def block_fps(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, fps: Fraction | None = None,
        mode: int | None = None, ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | None] | None = None
    ) -> vs.VideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames
        using block-based motion compensation.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param fps:        Target output framerate as a Fraction.
        :param mode:       Processing mask mode for handling occlusions and motion failures.
        :param ml:         Mask scale parameter that controls occlusion mask strength.
                           Higher values produce weaker occlusion masks.
                           Used in MakeVectorOcclusionMaskTime for modes 3-5.
                           Used in MakeSADMaskTime for modes 6-8.
        :param blend:      Whether to blend frames at scene changes.
                           If True, frames will be blended. If False, frames will be copied.
        :param thscd:      Scene change detection thresholds.
                           First value is the block change threshold between frames.
                           Second value is the number of changed blocks needed for a scene change.

        :return:           Clip with its framerate resampled.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect_b, vect_f = self.get_vectors(self.vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        block_fps_args: dict[str, Any] = KwargsNotNone(mode=mode, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            block_fps_args.update(num=fps.numerator, den=fps.denominator)

        block_fps_args = self.block_fps_args | block_fps_args

        return self.mvtools.BlockFPS(clip, super_clip, vect_b, vect_f, **block_fps_args)

    def flow_blur(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | MVTools | None = None, blur: float | None = None,
        prec: int | None = None, thscd: int | tuple[int | None, int | None] | None = None
    ) -> vs.VideoNode:
        """
        Creates a motion blur effect by simulating finite shutter time, similar to film cameras.

        Uses backward and forward motion vectors to create and overlay multiple copies of motion compensated pixels
        at intermediate time positions within a blurring interval around the current frame.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param blur:       Blur time interval between frames as a percentage (0.0-100.0).
                           Controls the simulated shutter time/motion blur strength.
        :param prec:       Blur precision in pixel units. Controls the accuracy of the motion blur.
        :param thscd:      Scene change detection thresholds.
                           First value is the block change threshold between frames.
                           Second value is the number of changed blocks needed for a scene change.

        :return:           Motion blurred clip.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect_b, vect_f = self.get_vectors(self.vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_blur_args = self.flow_blur_args | KwargsNotNone(blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2)

        return self.mvtools.FlowBlur(clip, super_clip, vect_b, vect_f, **flow_blur_args)

    def mask(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | MVTools | None = None,
        direction: Literal[MVDirection.FWRD] | Literal[MVDirection.BACK] = MVDirection.BACK,
        delta: int = 1, ml: float | None = None, gamma: float | None = None,
        kind: MaskMode | None = None, time: float | None = None, ysc: int | None = None,
        thscd: int | tuple[int | None, int | None] | None = None
    ) -> vs.VideoNode:
        """
        Creates a mask clip from motion vectors data.

        :param clip:         The clip to process.
                             If None, the :py:attr:`workclip` attribute is used.
        :param vectors:      Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                             If None, uses the vectors from this instance.
        :param direction:    Motion vector direction to use.
        :param delta:        Motion vector delta to use.
        :param ml:           Motion length scale factor. When the vector's length (or other mask value)
                             is greater than or equal to ml, the output is saturated to 255.
        :param gamma:        Exponent for the relation between input and output values.
                             1.0 gives a linear relation, 2.0 gives a quadratic relation.
        :param kind:         Type of mask to generate. See :py:class:`MaskMode` for options.
        :param time:         Time position between frames as a percentage (0.0-100.0).
        :param ysc:          Value assigned to the mask on scene changes.
        :param thscd:        Scene change detection thresholds.
                             First value is the block change threshold between frames.
                             Second value is the number of changed blocks needed for a scene change.

        :return:             Motion mask clip.
        """

        clip = fallback(clip, self.clip)

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect = vectors.get_mv(direction, delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        mask_args = self.mask_args | KwargsNotNone(
            ml=ml, gamma=gamma, kind=kind, time=time, ysc=ysc, thscd1=thscd1, thscd2=thscd2
        )

        mask_clip = depth(clip, 8) if self.mvtools is MVToolsPlugin.INTEGER else clip

        mask_clip = self.mvtools.Mask(mask_clip, vect, **mask_args)

        return depth(mask_clip, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)

    def sc_detection(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | MVTools | None = None,
        delta: int = 1, thscd: int | tuple[int | None, int | None] | None = None
    ) -> vs.VideoNode:
        """
        Creates scene detection mask clip from motion vectors data.

        :param clip:       The clip to process.
                           If None, the :py:attr:`workclip` attribute is used.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        :param delta:      Motion vector delta to use.
        :param thscd:      Scene change detection thresholds.
                           First value is the block change threshold between frames.
                           Second value is the number of changed blocks needed for a scene change.

        :return:           Clip with scene change properties set.
        """

        clip = fallback(clip, self.clip)

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        thscd1, thscd2 = normalize_thscd(thscd)

        sc_detection_args = self.sc_detection_args | KwargsNotNone(thscd1=thscd1, thscd2=thscd2)

        detect = clip
        for direction in MVDirection:
            detect = self.mvtools.SCDetection(detect, vectors.get_mv(direction, delta), **sc_detection_args)

        return detect

    def scale_vectors(
            self, scale: int | tuple[int, int], vectors: MotionVectors | MVTools | None = None, strict: bool = True
        ) -> None:
        """
        Scales image_size, block_size, overlap, padding, and the individual motion_vectors contained in Analyse output
        by arbitrary and independent x and y factors.

        :param scale:      Factor to scale motion vectors by.
        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        """

        if self.disable_manipmv:
            raise CustomRuntimeError(
                f'Motion vector manipulation not supported with {self.mvtools}!', self.scale_vectors
            )
        
        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        scalex, scaley = normalize_seq(scale, 2)

        supported_blksize = (
            (4, 4), (8, 4), (8, 8), (16, 2), (16, 8), (16, 16), (32, 16),
            (32, 32), (64, 32), (64, 64), (128, 64), (128, 128)
        )

        if scalex > 1 and scaley > 1:
            self.expand_analysis_data(vectors)

            blksize, blksizev = vectors.analysis_data['Analysis_BlockSize']

            scaled_blksize = (blksize * scalex, blksizev * scaley)

            if strict and scaled_blksize not in supported_blksize:
                raise CustomRuntimeError('Unsupported block size!', self.scale_vectors)

            for i in range(1, self.tr + 1):
                for direction in MVDirection:
                    vector = vectors.get_mv(direction, i).manipmv.ScaleVect(scalex, scaley)
                    vectors.set_mv(direction, i, vector)

            self.clip = self.clip.std.RemoveFrameProps('MSuper')
            self.search_clip = self.search_clip.std.RemoveFrameProps('MSuper')

            vectors.analysis_data.clear()
            vectors.scaled = True

    def show_vector(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | MVTools | None = None,
        direction: Literal[MVDirection.FWRD] | Literal[MVDirection.BACK] = MVDirection.BACK,
        delta: int = 1, scenechange: bool | None = None
    ) -> vs.VideoNode:
        """
        Draws generated vectors onto a clip.

        :param clip:           The clip to overlay the motion vectors on.
        :param vectors:        Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                               If None, uses the vectors from this instance.
        :param direction:      Motion vector direction to use.
        :param delta:          Motion vector delta to use.
        :param scenechange:    Skips drawing vectors if frame props indicate they are from a different scene
                               than the current frame of the clip.

        :return:               Clip with motion vectors overlaid.
        """

        if self.disable_manipmv:
            raise CustomRuntimeError(f'Motion vector manipulation not supported with {self.mvtools}!', self.show_vector)

        clip = fallback(clip, self.clip)

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect = vectors.get_mv(direction, delta)

        return clip.manipmv.ShowVect(vect, scenechange)
    
    def expand_analysis_data(self, vectors: MotionVectors | MVTools | None = None) -> None:
        """
        Expands the binary MVTools_MVAnalysisData frame prop into separate frame props for convenience.

        :param vectors:    Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
                           If None, uses the vectors from this instance.
        """

        if self.disable_manipmv:
            raise CustomRuntimeError(
                f'Motion vector manipulation not supported with {self.mvtools}!', self.expand_analysis_data
            )

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        props_list = (
            'Analysis_BlockSize', 'Analysis_Pel', 'Analysis_LevelCount', 'Analysis_CpuFlags', 'Analysis_MotionFlags',
            'Analysis_FrameSize', 'Analysis_Overlap', 'Analysis_BlockCount', 'Analysis_BitsPerSample',
            'Analysis_ChromaRatio', 'Analysis_Padding'
        )

        if not vectors.analysis_data:
            analysis_props = dict[str, Any]()

            with vectors.get_mv(MVDirection.BACK, 1).manipmv.ExpandAnalysisData().get_frame(0) as clip_props:
                for i in props_list:
                    analysis_props[i] = get_prop(clip_props, i, int | list)  # type: ignore

            vectors.analysis_data = analysis_props

    def get_super(self, clip: vs.VideoNode | None = None) -> vs.VideoNode:
        """
        Get the super clips from the specified clip.

        If :py:attr:`super` wasn't previously called,
        it will do so here with default values or kwargs specified in the constructor.

        :param clip:    The clip to get the super clip from.

        :return:        VideoNode containing the super clip.
        """

        clip = fallback(clip, self.clip)

        try:
            super_clip = clip.std.PropToClip(prop='MSuper')
        except vs.Error:
            clip = self.super(clip)
            super_clip = clip.std.PropToClip(prop='MSuper')

        return super_clip

    def get_vectors(
        self, vectors: MotionVectors, *,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backwards and forward vectors.

        :param vectors:    The motion vectors to get the backwards and forward vectors from.
        :param tr:         The number of frames to get the vectors for.

        :return:           A tuple containing two lists of motion vectors.
                           The first list contains backward vectors and the second contains forward vectors.
        """

        if not vectors.has_vectors:
            raise CustomRuntimeError('You need to run analyze before getting motion vectors!', self.get_vectors)

        tr = fallback(tr, self.tr)

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        if self.mvtools is MVToolsPlugin.FLOAT:
            vmulti = vectors.vmulti

            for i in range(0, tr * 2, 2):
                if direction in [MVDirection.BACK, MVDirection.BOTH]:
                    vectors_backward.append(vmulti.std.SelectEvery(tr * 2, i))
                if direction in [MVDirection.FWRD, MVDirection.BOTH]:
                    vectors_forward.append(vmulti.std.SelectEvery(tr * 2, i + 1))
        else:
            for i in range(1, tr + 1):
                if direction in [MVDirection.BACK, MVDirection.BOTH]:
                    vectors_backward.append(vectors.get_mv(MVDirection.BACK, i))
                if direction in [MVDirection.FWRD, MVDirection.BOTH]:
                    vectors_forward.append(vectors.get_mv(MVDirection.FWRD, i))

        return (vectors_backward, vectors_forward)
