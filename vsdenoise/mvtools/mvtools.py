from __future__ import annotations

from fractions import Fraction
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, MutableMapping, Union, overload

from vstools import (
    ColorRange, ConstantFormatVideoNode, CustomRuntimeError, FieldBased, InvalidColorFamilyError, KwargsNotNone,
    KwargsT, PlanesT, VSFunctionNoArgs, check_variable, check_variable_format, core, depth, fallback, get_prop,
    normalize_planes, normalize_seq, scale_delta, vs, vs_object
)

from .enums import (
    FlowMode, MaskMode, MotionMode, MVDirection, MVToolsPlugin, PenaltyMode, RFilterMode, SADMode, SearchMode,
    SharpMode, SmoothMode
)
from .motion import MotionVectors
from .utils import normalize_thscd, planes_to_mvtools

__all__ = [
    'MVTools'
]


class MVTools(vs_object):
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

    clip: ConstantFormatVideoNode
    """Clip to process."""

    def __init__(
        self, clip: vs.VideoNode, search_clip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
        vectors: MotionVectors | None = None,
        pad: int | tuple[int | None, int | None] | None = None,
        pel: int | None = None, planes: PlanesT = None,
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
        :param vectors:                  Motion vectors to use.
                                         If None, uses the vectors from this instance.
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

        self.mvtools = MVToolsPlugin.from_video(clip)
        self.vectors = fallback(vectors, MotionVectors())

        self.fieldbased = FieldBased.from_video(clip, False, self.__class__)
        self.clip = clip.std.SeparateFields(self.fieldbased.is_tff) if self.fieldbased.is_inter else clip

        self.planes = normalize_planes(self.clip, planes)
        self.mv_plane = planes_to_mvtools(self.planes)
        self.chroma = self.mv_plane != 0
        self.disable_compensate = False

        self.pel = pel
        self.pad = normalize_seq(pad, 2)

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
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | None = None,
        levels: int | None = None, sharp: SharpMode | None = None,
        rfilter: RFilterMode | None = None,
        pelclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Get source clip and prepare special "super" clip with multilevel (hierarchical scaled) frames data.
        The super clip is used by both :py:attr:`analyze` and motion compensation (client) functions.

        You can use different Super clip for generation vectors with :py:attr:`analyze` and a different super clip format for the actual action.
        Source clip is appended to clip's frameprops, :py:attr:`get_super` can be used to extract the super clip if you wish to view it yourself.

        :param clip:       The clip to process. If None, the :py:attr:`clip` attribute is used.
        :param vectors:    Motion vectors to use.
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

        vectors = fallback(vectors, self.vectors)

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
        
        if levels := super_args.pop('levels', None) is None and clip is not self.search_clip:
            levels = 1

        super_clip = self.mvtools.Super(clip, levels=levels, **super_args)

        super_clip = clip.std.ClipToProp(super_clip, prop='MSuper')

        if clip is self.clip:
            self.clip = super_clip

        if clip is self.search_clip:
            self.search_clip = super_clip

        return super_clip

    def analyze(
        self, super: vs.VideoNode | None = None, tr: int = 1,
        blksize: int | tuple[int | None, int | None] | None = None, levels: int | None = None,
        search: SearchMode | None = None, searchparam: int | None = None,
        pelsearch: int | None = None, lambda_: int | None = None, truemotion: MotionMode | None = None,
        lsad: int | None = None, plevel: PenaltyMode | None = None, global_: bool | None = None,
        pnew: int | None = None, pzero: int | None = None, pglobal: int | None = None,
        overlap: int | tuple[int | None, int | None] | None = None, divide: bool | None = None,
        badsad: int | None = None, badrange: int | None = None, meander: bool | None = None,
        trymany: bool | None = None, dct: SADMode | None = None, scale_lambda: bool = True
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

        :param super:           The multilevel super clip prepared by :py:attr:`super`.
                                If None, super will be obtained from clip.
        :param tr:              The temporal radius. This determines how many frames are analyzed before/after the current frame.
                                Default: 1.
        :param blksize:         Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
        :param levels:          Number of levels used in hierarchical motion vector analysis.
                                A positive value specifies how many levels to use.
                                A negative or zero value specifies how many coarse levels to skip.
                                Lower values generally give better results since vectors of any length can be found.
                                Sometimes adding more levels can help prevent false vectors in CGI or similar content.
        :param search:          Search algorithm to use at the finest level. See :py:class:`SearchMode` for options.
        :param searchparam:     Additional parameter for the search algorithm. For NSTEP, this is the step size.
                                For EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
        :param lambda_:         Controls the coherence of the motion vector field.
                                Higher values enforce more coherent/smooth motion between blocks.
                                Too high values may cause the algorithm to miss the optimal vectors.
        :param truemotion:      Preset that controls the default values of motion estimation parameters to optimize for true motion.
                                For more information, see :py:class:`MotionMode`.
        :param lsad:            SAD limit for lambda.
                                When the SAD value of a vector predictor (formed from neighboring blocks) exceeds this limit,
                                the local lambda value is decreased. This helps prevent the use of bad predictors,
                                but reduces motion coherence between blocks.
        :param plevel:          Controls how the penalty factor (lambda) scales with hierarchical levels.
                                For more information, see :py:class:`PenaltyMode`.
        :param global_:         Whether to estimate global motion at each level and use it as an additional predictor.
                                This can help with camera motion.
        :param pnew:            Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                                Higher values make the search more conservative.
        :param pzero:           Penalty multiplier (relative to 256) applied to the SAD cost for the zero motion vector.
                                Higher values discourage using zero motion.
        :param pglobal:         Penalty multiplier (relative to 256) applied to the SAD cost when using the global motion predictor.
        :param overlap:         Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap values.
                                Each value must be even and less than its corresponding block size dimension.
        :param divide:          Whether to divide each block into 4 subblocks during post-processing.
                                This may improve accuracy at the cost of performance.
        :param badsad:          SAD threshold above which a wider secondary search will be performed to find better motion vectors.
                                Higher values mean fewer blocks will trigger the secondary search.
        :param badrange:        Search radius for the secondary search when a block's SAD exceeds badsad.
        :param meander:         Whether to use a meandering scan pattern when processing blocks.
                                If True, alternates between left-to-right and right-to-left scanning between rows to improve motion coherence.
        :param trymany:         Whether to test multiple predictor vectors during the search process at coarser levels.
                                Enabling this can find better vectors but increases processing time.
        :param dct:             SAD calculation mode using block DCT (frequency spectrum) for comparing blocks.
                                For more information, see :py:class:`SADMode`.
        :param scale_lambda:    Whether to scale lambda_ value according to truemotion's default value formula.

        :return:               A :py:class:`MotionVectors` object containing the analyzed motion vectors for each frame.
                               These vectors describe the estimated motion between frames and can be used for motion compensation.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        if scale_lambda and lambda_ and blksize:
            lambda_ = lambda_ * blksize * fallback(blksizev, blksize) // 64

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

        if self.vectors.has_vectors:
            self.vectors.clear()

        self.vectors.tr = tr

        if self.mvtools is MVToolsPlugin.FLOAT:
            self.vectors.mv_multi = self.mvtools.Analyze(super_clip, radius=self.vectors.tr, **analyze_args)
        else:
            if not any((analyze_args.get('overlap'), analyze_args.get('overlapv'))):
                self.disable_compensate = True

            for delta in range(1, self.vectors.tr + 1):
                for direction in MVDirection:
                    self.vectors.set_vector(
                        self.mvtools.Analyze(
                            super_clip, isb=direction is MVDirection.BACKWARD, delta=delta, **analyze_args
                        ),
                        direction, delta,
                    )

    def recalculate(
        self, super: vs.VideoNode | None = None, vectors: MotionVectors | None = None,
        thsad: int | None = None, smooth: SmoothMode | None = None,
        blksize: int | tuple[int | None, int | None] | None = None, search: SearchMode | None = None,
        searchparam: int | None = None, lambda_: int | None = None, truemotion: MotionMode | None = None,
        pnew: int | None = None, overlap: int | tuple[int | None, int | None] | None = None,
        divide: bool | None = None, meander: bool | None = None, dct: SADMode | None = None, scale_lambda: bool = True
    ) -> None:
        """
        Refines and recalculates motion vectors that were previously estimated, optionally using a different super clip or parameters.
        This two-stage approach can provide more stable and robust motion estimation.

        The refinement only occurs at the finest hierarchical level. It uses the interpolated vectors from the original blocks
        as predictors for the new vectors, and recalculates their SAD values.

        Only vectors with poor quality (SAD above threshold) will be re-estimated through a new search.
        The SAD threshold is normalized to an 8x8 block size. Vectors with good quality are preserved,
        though their SAD values are still recalculated and updated.

        :param super:           The multilevel super clip prepared by :py:attr:`super`.
                                If None, super will be obtained from clip.
        :param vectors:         Motion vectors to use.
                                If None, uses the vectors from this instance.
        :param thsad:           Only bad quality new vectors with a SAD above this will be re-estimated by search.
                                thsad value is scaled to 8x8 block size.
        :param blksize:         Size of blocks for motion estimation. Can be an int or tuple of (width, height).
                                Larger blocks are less sensitive to noise and faster to process, but will produce less accurate vectors.
        :param smooth:          This is method for dividing coarse blocks into smaller ones.
                                Only used with the FLOAT MVTools plugin.
        :param search:          Search algorithm to use at the finest level. See :py:class:`SearchMode` for options.
        :param searchparam:     Additional parameter for the search algorithm. For NSTEP, this is the step size.
                                For EXHAUSTIVE, EXHAUSTIVE_H, EXHAUSTIVE_V, HEXAGON and UMH, this is the search radius.
        :param lambda_:         Controls the coherence of the motion vector field.
                                Higher values enforce more coherent/smooth motion between blocks.
                                Too high values may cause the algorithm to miss the optimal vectors.
        :param truemotion:      Preset that controls the default values of motion estimation parameters to optimize for true motion.
                                For more information, see :py:class:`MotionMode`.
        :param pnew:            Penalty multiplier (relative to 256) applied to the SAD cost when evaluating new candidate vectors.
                                Higher values make the search more conservative.
        :param overlap:         Block overlap value. Can be a single integer for both dimensions or a tuple of (horizontal, vertical) overlap values.
                                Each value must be even and less than its corresponding block size dimension.
        :param divide:          Whether to divide each block into 4 subblocks during post-processing.
                                This may improve accuracy at the cost of performance.
        :param meander:         Whether to use a meandering scan pattern when processing blocks.
                                If True, alternates between left-to-right and right-to-left scanning between rows to improve motion coherence.
        :param dct:             SAD calculation mode using block DCT (frequency spectrum) for comparing blocks.
                                For more information, see :py:class:`SADMode`.
        :param scale_lambda:    Whether to scale lambda_ value according to truemotion's default value formula.
        """

        super_clip = self.get_super(fallback(super, self.search_clip))

        vectors = fallback(vectors, self.vectors)

        assert vectors.has_vectors

        blksize, blksizev = normalize_seq(blksize, 2)
        overlap, overlapv = normalize_seq(overlap, 2)

        if scale_lambda and lambda_ and blksize:
            lambda_ = lambda_ * blksize * fallback(blksizev, blksize) // 64

        recalculate_args = self.recalculate_args | KwargsNotNone(
            thsad=thsad, smooth=smooth, blksize=blksize, blksizev=blksizev, search=search, searchparam=searchparam,
            lambda_=lambda_, chroma=self.chroma, truemotion=truemotion, pnew=pnew, overlap=overlap, overlapv=overlapv,
            divide=divide, meander=meander, fields=self.fieldbased.is_inter, tff=self.fieldbased.is_tff, dct=dct
        )

        vectors.analysis_data.clear()

        if self.mvtools is MVToolsPlugin.FLOAT:
            vectors.mv_multi = self.mvtools.Recalculate(super_clip, vectors=vectors.mv_multi, **recalculate_args)
        else:
            if not any((recalculate_args.get('overlap'), recalculate_args.get('overlapv'))):
                self.disable_compensate = True

            for delta in range(1, vectors.tr + 1):
                for direction in MVDirection:
                    vectors.set_vector(
                        self.mvtools.Recalculate(
                            super_clip, self.get_vector(vectors, direction=direction, delta=delta), **recalculate_args
                        ),
                        direction, delta,
                    )

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[True] = True, temporal_func: None = None
    ) -> tuple[ConstantFormatVideoNode, tuple[int, int]]:
        ...

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[True] = True, temporal_func: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = ...
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[False] = False, temporal_func: None = None
    ) -> tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        ...

    def compensate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, scbehavior: bool | None = None,
        thsad: int | None = None, thsad2: int | None = None,
        time: float | None = None, thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = True, temporal_func: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None
    ) -> Union[
        ConstantFormatVideoNode,
        tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]],
        tuple[ConstantFormatVideoNode, tuple[int, int]]
    ]:
        """
        Perform motion compensation by moving blocks from reference frames to the current frame according to motion vectors.
        This creates a prediction of the current frame by taking blocks from neighboring frames and moving them along their estimated motion paths.

        :param clip:             The clip to process.
        :param super:            The multilevel super clip prepared by :py:attr:`super`.
                                 If None, super will be obtained from clip.
        :param vectors:          Motion vectors to use.
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
        :param thscd:            Scene change detection thresholds:
                                  - First value: SAD threshold for considering a block changed between frames.
                                  - Second value: Percentage of changed blocks needed to trigger a scene change.
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

        vect_b, vect_f = self.get_vectors(vectors, direction, tr)

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
        cycle = len(comp_clips)
        offset = (cycle - 1) // 2

        interleaved = core.std.Interleave(comp_clips)

        if temporal_func:
            processed = temporal_func(interleaved)

            assert check_variable_format(processed, self.compensate)

            return core.std.SelectEvery(processed, cycle, offset)

        return interleaved, (cycle, offset)

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[True] = True, temporal_func: None = None
    ) -> tuple[ConstantFormatVideoNode, tuple[int, int]]:
        ...

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[True] = True, temporal_func: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = ...
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[False] = False, temporal_func: None = None
    ) -> tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        ...

    def flow(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, time: float | None = None, mode: FlowMode | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = True, temporal_func: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None
    ) -> Union[
        ConstantFormatVideoNode,
        tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]],
        tuple[ConstantFormatVideoNode, tuple[int, int]]
    ]:
        """
        Performs motion compensation using pixel-level motion vectors interpolated from block vectors.

        Unlike block-based compensation, this calculates a unique motion vector for each pixel by bilinearly interpolating
        between the motion vectors of the current block and its neighbors based on the pixel's position.
        The pixels in the reference frame are then moved along these interpolated vectors to their estimated positions in the current frame.

        :param clip:             The clip to process.
        :param super:            The multilevel super clip prepared by :py:attr:`super`.
                                 If None, super will be obtained from clip.
        :param vectors:          Motion vectors to use.
                                 If None, uses the vectors from this instance.
        :param direction:        Motion vector direction to use.
        :param tr:               The temporal radius. This determines how many frames are analyzed before/after the current frame.
        :param time:             Time position between frames as a percentage (0.0-100.0).
                                 Controls the interpolation position between frames.
        :param mode:             Method for positioning pixels during motion compensation.
                                 See :py:class:`FlowMode` for options.
        :param thscd:            Scene change detection thresholds:
                                  - First value: SAD threshold for considering a block changed between frames.
                                  - Second value: Percentage of changed blocks needed to trigger a scene change.
        :param interleave:       Whether to interleave the compensated frames with the input.
        :param temporal_func:    Optional function to process the motion compensated frames.
                                 Takes the interleaved frames as input and returns processed frames.

        :return:                 Motion compensated frames if func is provided, otherwise returns a tuple containing:
                                  - The interleaved compensated frames.
                                  - A tuple of (total_frames, center_offset) for manual frame selection.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vect_b, vect_f = self.get_vectors(vectors, direction, tr)

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
        cycle = len(flow_clips)
        offset = (cycle - 1) // 2

        interleaved = core.std.Interleave(flow_clips)

        if temporal_func:
            processed = temporal_func(interleaved)

            assert check_variable_format(processed, self.compensate)

            return core.std.SelectEvery(processed, cycle, offset)

        return interleaved, (cycle, offset)

    def degrain(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, tr: int | None = None,
        thsad: int | tuple[int | None, int | None] | None = None,
        thsad2: int | tuple[int | None, int | None] | None = None,
        limit: int | tuple[int | None, int | None] | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
    ) -> ConstantFormatVideoNode:
        """
        Perform temporal denoising using motion compensation.

        Motion compensated blocks from previous and next frames are averaged with the current frame.
        The weighting factors for each block depend on their SAD from the current frame.

        :param clip:       The clip to process.
                           If None, the :py:attr:`workclip` attribute is used.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use.
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
                            - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:           Motion compensated and temporally filtered clip with reduced noise.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vectors = fallback(vectors, self.vectors)
        tr = fallback(tr, vectors.tr)

        thscd1, thscd2 = normalize_thscd(thscd)

        degrain_args = dict[str, Any](thscd1=thscd1, thscd2=thscd2, plane=self.mv_plane)

        if self.mvtools is MVToolsPlugin.FLOAT:
            if tr == 1:
                raise CustomRuntimeError(
                    f'Cannot degrain with a temporal radius of {tr} while using {self.mvtools}!',
                    self.degrain
                )

            degrain_args.update(thsad=thsad, thsad2=thsad2, limit=limit)
        else:
            thsad, thsadc = normalize_seq(thsad, 2)
            nlimit, nlimitc = normalize_seq(limit, 2)

            if nlimit is not None:
                nlimit = scale_delta(nlimit, 8, clip)

            if nlimitc is not None:
                nlimitc = scale_delta(nlimitc, 8, clip)

            degrain_args.update(thsad=thsad, thsadc=thsadc, limit=nlimit, limitc=nlimitc)

        degrain_args = self.degrain_args | KwargsNotNone(degrain_args)

        if self.mvtools is MVToolsPlugin.FLOAT:
            output = self.mvtools.Degrain()(
                clip, super_clip, self.get_vectors(vectors, tr=tr, multi=True), **degrain_args
            )
        else:
            vect_b, vect_f = self.get_vectors(vectors, tr=tr)

            output = self.mvtools.Degrain(tr)(
                clip, super_clip, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        return output

    @overload
    def flow_interpolate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, time: float | None = None,
        ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[True] = ..., multi: int | None = None
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def flow_interpolate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, time: float | None = None,
        ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: Literal[False] = ..., multi: int | None = None
    ) -> list[ConstantFormatVideoNode]:
        ...

    @overload
    def flow_interpolate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, time: float | None = None,
        ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = ..., multi: int | None = None
    ) -> ConstantFormatVideoNode | list[ConstantFormatVideoNode]:
        ...

    def flow_interpolate(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, time: float | None = None,
        ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None,
        interleave: bool = True, multi: int | None = None
    ) -> ConstantFormatVideoNode | list[ConstantFormatVideoNode]:
        """
        Motion interpolation function that creates an intermediate frame between two frames.

        Uses both backward and forward motion vectors to estimate motion and create a frame at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:          The clip to process.
        :param super:         The multilevel super clip prepared by :py:attr:`super`.
                              If None, super will be obtained from clip.
        :param vectors:       Motion vectors to use.
                              If None, uses the vectors from this instance.
        :param time:          Time position between frames as a percentage (0.0-100.0).
                              Controls the interpolation position between frames.
                              Does nothing if multi is specified.
        :param ml:            Mask scale parameter that controls occlusion mask strength.
                              Higher values produce weaker occlusion masks.
                              Used in MakeVectorOcclusionMaskTime for modes 3-5.
                              Used in MakeSADMaskTime for modes 6-8.
        :param blend:         Whether to blend frames at scene changes.
                              If True, frames will be blended. If False, frames will be copied.
        :param thscd:         Scene change detection thresholds:
                               - First value: SAD threshold for considering a block changed between frames.
                               - Second value: Percentage of changed blocks needed to trigger a scene change.
        :param interleave:    Whether to interleave the interpolated frames with the source clip.
        :param multi:         Framerate multiplier.

        :return:              List of the motion interpolated frames if interleave=False else
                              a motion interpolated clip.
        """

        clip = fallback(clip, self.clip)

        assert check_variable_format(clip, self.flow_interpolate)

        super_clip = self.get_super(fallback(super, clip))

        vect_b, vect_f = self.get_vectors(vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_interpolate_args = self.flow_interpolate_args | KwargsNotNone(
            time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2
        )

        interpolated = list[ConstantFormatVideoNode]()

        if multi:
            if multi < 2:
                raise CustomRuntimeError(
                    'Invalid framerate multiplier specified!', self.flow_interpolate, f'{multi} < 2'
                )

            flow_interpolate_args.pop('time', None)

            for pos in range(1, multi):
                time = pos * 100 / multi

                interpolated.append(
                    self.mvtools.FlowInter(clip, super_clip, vect_b, vect_f, time=time, **flow_interpolate_args)
                )
        else:
            interpolated.append(self.mvtools.FlowInter(clip, super_clip, vect_b, vect_f, **flow_interpolate_args))

        if interleave:
            interpolated.insert(0, clip)

            return core.std.Interleave(interpolated)

        return interpolated

    def flow_fps(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, fps: Fraction | None = None,
        mask: int | None = None, ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        :param fps:        Target output framerate as a Fraction.
        :param mask:       Processing mask mode for handling occlusions and motion failures.
        :param ml:         Mask scale parameter that controls occlusion mask strength.
                           Higher values produce weaker occlusion masks.
                           Used in MakeVectorOcclusionMaskTime for modes 3-5.
                           Used in MakeSADMaskTime for modes 6-8.
        :param blend:      Whether to blend frames at scene changes.
                           If True, frames will be blended. If False, frames will be copied.
        :param thscd:      Scene change detection thresholds:
                            - First value: SAD threshold for considering a block changed between frames.
                            - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:           Clip with its framerate resampled.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vect_b, vect_f = self.get_vectors(vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_fps_args: dict[str, Any] = KwargsNotNone(mask=mask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            flow_fps_args.update(num=fps.numerator, den=fps.denominator)

        flow_fps_args = self.flow_fps_args | flow_fps_args

        return self.mvtools.FlowFPS(clip, super_clip, vect_b, vect_f, **flow_fps_args)

    def block_fps(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, fps: Fraction | None = None,
        mode: int | None = None, ml: float | None = None, blend: bool | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Changes the framerate of the clip by interpolating frames between existing frames
        using block-based motion compensation.

        Uses both backward and forward motion vectors to estimate motion and create frames at any time position between
        the current and next frame. Occlusion masks are used to handle areas where motion estimation fails, and time
        weighting ensures smooth blending between frames to minimize artifacts.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        :param fps:        Target output framerate as a Fraction.
        :param mode:       Processing mask mode for handling occlusions and motion failures.
        :param ml:         Mask scale parameter that controls occlusion mask strength.
                           Higher values produce weaker occlusion masks.
                           Used in MakeVectorOcclusionMaskTime for modes 3-5.
                           Used in MakeSADMaskTime for modes 6-8.
        :param blend:      Whether to blend frames at scene changes.
                           If True, frames will be blended. If False, frames will be copied.
        :param thscd:      Scene change detection thresholds:
                            - First value: SAD threshold for considering a block changed between frames.
                            - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:           Clip with its framerate resampled.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vect_b, vect_f = self.get_vectors(vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        block_fps_args: dict[str, Any] = KwargsNotNone(mode=mode, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

        if fps is not None:
            block_fps_args.update(num=fps.numerator, den=fps.denominator)

        block_fps_args = self.block_fps_args | block_fps_args

        return self.mvtools.BlockFPS(clip, super_clip, vect_b, vect_f, **block_fps_args)

    def flow_blur(
        self, clip: vs.VideoNode | None = None, super: vs.VideoNode | None = None,
        vectors: MotionVectors | None = None, blur: float | None = None,
        prec: int | None = None, thscd: int | tuple[int | None, int | float | None] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Creates a motion blur effect by simulating finite shutter time, similar to film cameras.

        Uses backward and forward motion vectors to create and overlay multiple copies of motion compensated pixels
        at intermediate time positions within a blurring interval around the current frame.

        :param clip:       The clip to process.
        :param super:      The multilevel super clip prepared by :py:attr:`super`.
                           If None, super will be obtained from clip.
        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        :param blur:       Blur time interval between frames as a percentage (0.0-100.0).
                           Controls the simulated shutter time/motion blur strength.
        :param prec:       Blur precision in pixel units. Controls the accuracy of the motion blur.
        :param thscd:      Scene change detection thresholds:
                            - First value: SAD threshold for considering a block changed between frames.
                            - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:           Motion blurred clip.
        """

        clip = fallback(clip, self.clip)
        super_clip = self.get_super(fallback(super, clip))

        vect_b, vect_f = self.get_vectors(vectors, tr=1)

        thscd1, thscd2 = normalize_thscd(thscd)

        flow_blur_args = self.flow_blur_args | KwargsNotNone(blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2)

        return self.mvtools.FlowBlur(clip, super_clip, vect_b, vect_f, **flow_blur_args)

    def mask(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | None = None,
        direction: Literal[MVDirection.FORWARD] | Literal[MVDirection.BACKWARD] = MVDirection.FORWARD,
        delta: int = 1, ml: float | None = None, gamma: float | None = None,
        kind: MaskMode | None = None, time: float | None = None, ysc: int | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Creates a mask clip from motion vectors data.

        :param clip:         The clip to process.
                             If None, the :py:attr:`workclip` attribute is used.
        :param vectors:      Motion vectors to use.
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
        :param thscd:        Scene change detection thresholds:
                              - First value: SAD threshold for considering a block changed between frames.
                              - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:             Motion mask clip.
        """

        clip = fallback(clip, self.clip)

        vect = self.get_vector(vectors, direction=direction, delta=delta)

        thscd1, thscd2 = normalize_thscd(thscd)

        mask_args = self.mask_args | KwargsNotNone(
            ml=ml, gamma=gamma, kind=kind, time=time, ysc=ysc, thscd1=thscd1, thscd2=thscd2
        )

        mask_clip = depth(clip, 8) if self.mvtools is MVToolsPlugin.INTEGER else clip

        mask_clip = self.mvtools.Mask(mask_clip, vect, **mask_args)

        return depth(mask_clip, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)

    def sc_detection(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | None = None,
        delta: int = 1, thscd: int | tuple[int | None, int | float | None] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Creates scene change frameprops from motion vectors data.

        :param clip:       The clip to process.
                           If None, the :py:attr:`workclip` attribute is used.
        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        :param delta:      Motion vector delta to use.
        :param thscd:      Scene change detection thresholds:
                            - First value: SAD threshold for considering a block changed between frames.
                            - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:           Clip with scene change properties set.
        """

        clip = fallback(clip, self.clip)

        assert check_variable_format(clip, self.sc_detection)

        thscd1, thscd2 = normalize_thscd(thscd)

        sc_detection_args = self.sc_detection_args | KwargsNotNone(thscd1=thscd1, thscd2=thscd2)

        detect = clip
        for direction in MVDirection:
            detect = self.mvtools.SCDetection(
                detect, self.get_vector(vectors, direction=direction, delta=delta), **sc_detection_args
            )

        return detect

    def scale_vectors(
        self, vectors: MotionVectors | None = None, *, scale: int | tuple[int, int], strict: bool = True
    ) -> None:
        """
        Scales image_size, block_size, overlap, padding, and the individual motion_vectors contained in Analyse output
        by arbitrary and independent x and y factors.

        :param scale:      Factor to scale motion vectors by.
        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        """

        if self.mvtools is MVToolsPlugin.FLOAT:
            raise CustomRuntimeError(
                f'Motion vector manipulation not supported with {self.mvtools}!', self.scale_vectors
            )

        supported_blksize = (
            (4, 4), (8, 4), (8, 8), (16, 2), (16, 8), (16, 16), (32, 16),
            (32, 32), (64, 32), (64, 64), (128, 64), (128, 128)
        )

        vectors = fallback(vectors, self.vectors)

        scalex, scaley = normalize_seq(scale, 2)

        if scalex > 1 and scaley > 1:
            self.expand_analysis_data(vectors)

            blksizex, blksizev = vectors.analysis_data['Analysis_BlockSize']

            scaled_blksize = (blksizex * scalex, blksizev * scaley)

            if strict and scaled_blksize not in supported_blksize:
                raise CustomRuntimeError('Unsupported block size!', self.scale_vectors, scaled_blksize)

            vectors.analysis_data.clear()
            vectors.scaled = True

            self.clip = core.std.RemoveFrameProps(self.clip, 'MSuper')
            self.search_clip = core.std.RemoveFrameProps(self.search_clip, 'MSuper')

            for delta in range(1, vectors.tr + 1):
                for direction in MVDirection:
                    vectors.set_vector(
                        self.get_vector(vectors, direction=direction, delta=delta).manipmv.ScaleVect(scalex, scaley),
                        direction, delta,
                    )

    def show_vector(
        self, clip: vs.VideoNode | None = None, vectors: MotionVectors | None = None,
        direction: Literal[MVDirection.FORWARD] | Literal[MVDirection.BACKWARD] = MVDirection.FORWARD,
        delta: int = 1, scenechange: bool | None = None
    ) -> ConstantFormatVideoNode:
        """
        Draws generated vectors onto a clip.

        :param clip:           The clip to overlay the motion vectors on.
        :param vectors:        Motion vectors to use.
                               If None, uses the vectors from this instance.
        :param direction:      Motion vector direction to use.
        :param delta:          Motion vector delta to use.
        :param scenechange:    Skips drawing vectors if frame props indicate they are from a different scene
                               than the current frame of the clip.

        :return:               Clip with motion vectors overlaid.
        """

        if self.mvtools is MVToolsPlugin.FLOAT:
            raise CustomRuntimeError(f'Motion vector manipulation not supported with {self.mvtools}!', self.show_vector)

        clip = fallback(clip, self.clip)

        vect = self.get_vector(vectors, direction=direction, delta=delta)

        return clip.manipmv.ShowVect(vect, scenechange)

    def expand_analysis_data(self, vectors: MotionVectors | None = None) -> None:
        """
        Expands the binary MVTools_MVAnalysisData frame prop into separate frame props for convenience.

        :param vectors:    Motion vectors to use.
                           If None, uses the vectors from this instance.
        """

        if self.mvtools is MVToolsPlugin.FLOAT:
            raise CustomRuntimeError(
                f'Motion vector manipulation not supported with {self.mvtools}!', self.expand_analysis_data
            )

        vectors = fallback(vectors, self.vectors)

        props_list = (
            'Analysis_BlockSize', 'Analysis_Pel', 'Analysis_LevelCount', 'Analysis_CpuFlags', 'Analysis_MotionFlags',
            'Analysis_FrameSize', 'Analysis_Overlap', 'Analysis_BlockCount', 'Analysis_BitsPerSample',
            'Analysis_ChromaRatio', 'Analysis_Padding'
        )

        if not vectors.analysis_data:
            vect = self.get_vector(vectors, direction=MVDirection.BACKWARD, delta=1).manipmv.ExpandAnalysisData()

            analysis_props = dict[str, Any]()

            for prop in props_list:
                analysis_props[prop] = get_prop(vect, prop, (int, list))

            vectors.analysis_data = analysis_props

    def get_super(self, clip: vs.VideoNode | None = None) -> ConstantFormatVideoNode:
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

    def get_vector(
        self, vectors: MotionVectors | None = None, *, direction: MVDirection, delta: int
    ) -> ConstantFormatVideoNode:
        """
        Get a single motion vector.

        :param vectors:        The motion vectors to get the vector from.
        :param direction:      Motion vector direction to get.
        :param delta:          Motion vector delta to get.

        :return:               A single motion vector VideoNode
        """

        vectors = fallback(vectors, self.vectors)

        assert vectors.has_vectors

        if delta > vectors.tr:
            raise CustomRuntimeError(
                'Tried to get a motion vector delta larger than what exists!', self.get_vector, f'{delta} > {vectors.tr}'
            )

        if self.mvtools is MVToolsPlugin.FLOAT:
            return vectors.mv_multi[(delta - 1) * 2 + direction - 1 :: vectors.tr * 2]

        return vectors.motion_vectors[direction][delta]

    @overload
    def get_vectors(
        self, vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, multi: Literal[False] = ...
    ) -> tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        ...

    @overload
    def get_vectors(
        self, vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, multi: Literal[True] = ...
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def get_vectors(
        self, vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, multi: bool = ...
    ) -> ConstantFormatVideoNode | tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        ...

    def get_vectors(
        self, vectors: MotionVectors | None = None,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None, multi: bool = False
    ) -> ConstantFormatVideoNode | tuple[list[ConstantFormatVideoNode], list[ConstantFormatVideoNode]]:
        """
        Get the backward and forward vectors.

        :param vectors:       The motion vectors to get the backward and forward vectors from.
                              If None, uses the vectors from this instance.
        :param direction:     Motion vector direction to get.
        :param tr:            The number of frames to get the vectors for.
        :param multi:         Whether to return the mv_multi vector clip
                              Only used with the FLOAT MVTools plugin.

        :return:              If multi is false: A tuple containing two lists of motion vectors.
                              The first list contains backward vectors and the second contains forward vectors.
                              If multi is true: The multi vector VideoNode used by the FLOAT MVTools plugin.
        """

        vectors = fallback(vectors, self.vectors)
        tr = fallback(tr, vectors.tr)

        assert vectors.has_vectors

        if tr > vectors.tr:
            raise CustomRuntimeError(
                'Tried to obtain more motion vectors than what exist!', self.get_vectors, f'{tr} > {vectors.tr}'
            )

        if multi and self.mvtools is MVToolsPlugin.FLOAT:
            mv_multi =  vectors.mv_multi

            if tr != vectors.tr:
                trim = vectors.tr - tr
                mv_multi = core.std.SelectEvery(mv_multi, vectors.tr * 2, range(trim, vectors.tr * 2 - trim))

            return mv_multi

        vectors_backward = list[ConstantFormatVideoNode]()
        vectors_forward = list[ConstantFormatVideoNode]()

        for delta in range(1, tr + 1):
            if direction in [MVDirection.BACKWARD, MVDirection.BOTH]:
                vectors_backward.append(self.get_vector(vectors, direction=MVDirection.BACKWARD, delta=delta))
            if direction in [MVDirection.FORWARD, MVDirection.BOTH]:
                vectors_forward.append(self.get_vector(vectors, direction=MVDirection.FORWARD, delta=delta))

        return (vectors_backward, vectors_forward)

    def __vs_del__(self, core_id: int) -> None:
        if not TYPE_CHECKING:
            self.clip = None

        for v in self.__dict__.values():
            if not isinstance(v, MutableMapping):
                continue

            for k2, v2 in v.items():
                if isinstance(v2, vs.VideoNode):
                    v[k2] = None
