from __future__ import annotations

from typing import Iterable, Sequence

import vapoursynth as vs
from jetpytools import FuncExceptT, T, cachedproperty, fallback, iterate, kwargs_fallback, normalize_seq, to_arr

from vstools.exceptions.color import InvalidColorspacePathError

from ..enums import (
    ChromaLocation,
    ChromaLocationT,
    ColorRange,
    ColorRangeT,
    FieldBased,
    FieldBasedT,
    Matrix,
    MatrixT,
    Primaries,
    PrimariesT,
    Transfer,
    TransferT,
)
from ..exceptions import UndefinedMatrixError
from ..types import ConstantFormatVideoNode, HoldsVideoFormatT, PlanesT, VideoFormatT, vs_object
from .check import check_variable
from .normalize import normalize_planes
from .utils import depth, join, plane

__all__ = ["FunctionUtil", "fallback", "iterate", "kwargs_fallback"]


class FunctionUtil(cachedproperty.baseclass, list[int], vs_object):
    """
    Function util to normalize common actions and boilerplate often used in functions.

    Main use is:
        - Automatically dither up and down as required.
        - Automatically check if the input clip has variable formats, resolutions, etc.
        - Fully type safe and removes the need for asserts or typeguards in function code.
        - Handy properties for common code paths, improving code readability and writability.

    Examples:

        >>> func = FunctionUtil(clip, planes=0, color_family=(vs.YUV, vs.GRAY), bitdepth=16)
        >>> wclip = func.work_clip
        >>> txt = wclip.text.Text("This clip has been processed!")
        >>> return func.return_clip(txt)

    For further examples, see: <https://github.com/search?q=org%3AJaded-Encoding-Thaumaturgy+FunctionUtil>
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        func: FuncExceptT,
        planes: PlanesT = None,
        color_family: VideoFormatT
        | HoldsVideoFormatT
        | vs.ColorFamily
        | Iterable[VideoFormatT | HoldsVideoFormatT | vs.ColorFamily]
        | None = None,
        bitdepth: int | range | tuple[int, int] | set[int] | None = None,
        *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        range_in: ColorRangeT | None = None,
        chromaloc: ChromaLocationT | None = None,
        order: FieldBasedT | None = None,
    ) -> None:
        """
        Args:
            clip: Clip to process.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            planes: Planes that get processed in the function. Default: All planes.
            color_family: Accepted color families. If the input does not adhere to these, an exception will be raised.
                Default: All families.
            bitdepth: The bitdepth or range of bitdepths to work with. Can be an int, range, tuple, or set. Range or
                tuple indicates a range of allowed bitdepths, set indicates specific allowed bitdepths. If an int is
                provided, set the clip's bitdepth to that value.

                If a range or set is provided and the work clip's bitdepth is not allowed,
                the work clip's bitdepth will be converted to the lowest bitdepth that is greater than
                or equal to the work clip's current bitdepth.

                `return_clip` automatically restores the clip to the original bitdepth.
                If None, use the input clip's bitdepth. Default: None.
            matrix: Color Matrix to work in. Used for YUV <-> RGB conversions. Default: Get matrix from the input clip.
            transfer: Transfer to work in. Default: Get transfer from the input clip.
            primaries: Color primaries to work in. Default: Get primaries from the input clip.
            range_in: Color Range to work in. Default: Get the color range from the input clip.
            chromaloc: Chroma location to work in. Default: Get the chroma location from the input clip.
            order: Field order to work in. Default: Get the field order from the input clip.
        """
        from ..utils import get_color_family

        assert check_variable(clip, func)

        all_color_family: list[vs.ColorFamily] | None

        if color_family is not None:
            all_color_family = [get_color_family(c) for c in to_arr(color_family)]  # type: ignore[arg-type]
            if not set(all_color_family) & {vs.YUV, vs.RGB}:
                planes = 0
        else:
            all_color_family = color_family

        if isinstance(bitdepth, tuple):
            bitdepth = range(bitdepth[0], bitdepth[1] + 1)

        self.clip = clip
        self.planes = planes
        self.func = func
        self.allowed_cfamilies = all_color_family
        self.cfamily_converted = False
        self.bitdepth = bitdepth

        self._matrix = Matrix.from_param(matrix, self.func)
        self._transfer = Transfer.from_param(transfer, self.func)
        self._primaries = Primaries.from_param(primaries, self.func)
        self._range_in = ColorRange.from_param(range_in, self.func)
        self._chromaloc = ChromaLocation.from_param(chromaloc, self.func)
        self._order = FieldBased.from_param(order, self.func)

        self.norm_planes = normalize_planes(self.norm_clip, self.planes)

        super().__init__(self.norm_planes)

        self.num_planes = self.work_clip.format.num_planes

    @cachedproperty
    def norm_clip(self) -> ConstantFormatVideoNode:
        """
        Get a "normalized" clip. This means color space and bitdepth are converted if necessary.
        """

        if isinstance(self.bitdepth, (range, set)) and self.clip.format.bits_per_sample not in self.bitdepth:
            from .. import get_depth

            src_depth = get_depth(self.clip)
            target_depth = next((bits for bits in self.bitdepth if bits >= src_depth), max(self.bitdepth))

            clip = depth(self.clip, target_depth)
        elif isinstance(self.bitdepth, int):
            clip = depth(self.clip, self.bitdepth)
        else:
            clip = self.clip

        assert check_variable(clip, self.func)

        cfamily = clip.format.color_family

        if not self.allowed_cfamilies or cfamily in self.allowed_cfamilies:
            return clip

        if cfamily is vs.RGB:
            if not self._matrix:
                raise UndefinedMatrixError(
                    "You must specify a matrix for RGB to {} conversions!".format(
                        "/".join(cf.name for cf in sorted(self.allowed_cfamilies, key=lambda x: x.name))
                    ),
                    self.func,
                )

            self.cfamily_converted = True

            clip = clip.resize.Bicubic(format=clip.format.replace(color_family=vs.YUV).id, matrix=self._matrix)

        elif (
            cfamily in (vs.YUV, vs.GRAY) and not set(self.allowed_cfamilies) & {vs.YUV, vs.GRAY}
        ) or self.planes not in (0, [0]):
            self.cfamily_converted = True

            clip = clip.resize.Bicubic(
                format=clip.format.replace(color_family=vs.RGB, subsampling_h=0, subsampling_w=0).id,
                matrix_in=self._matrix,
                chromaloc_in=self._chromaloc,
                range_in=self._range_in.value_zimg if self._range_in else None,
            )

            InvalidColorspacePathError.check(self.func, clip)

        return clip

    @cachedproperty
    def work_clip(self) -> ConstantFormatVideoNode:
        """
        Get the "work clip" as specified from the input planes.
        """

        return plane(self.norm_clip, 0) if self.luma_only else self.norm_clip

    @cachedproperty
    def chroma_planes(self) -> list[vs.VideoNode]:
        """
        Get a list of all chroma planes in the normalised clip.
        """

        if self != [0] or self.norm_clip.format.num_planes == 1:
            return []

        return [plane(self.norm_clip, i) for i in (1, 2)]

    @cachedproperty
    def matrix(self) -> Matrix:
        """
        Get the clip's matrix.
        """

        return Matrix.from_param_or_video(self._matrix, self.clip, True, self.func)

    @cachedproperty
    def transfer(self) -> Transfer:
        """
        Get the clip's transfer.
        """

        return Transfer.from_param_or_video(self._transfer, self.clip, True, self.func)

    @cachedproperty
    def primaries(self) -> Primaries:
        """
        Get the clip's primaries.
        """

        return Primaries.from_param_or_video(self._primaries, self.clip, True, self.func)

    @cachedproperty
    def color_range(self) -> ColorRange:
        """
        Get the clip's color range.
        """

        return ColorRange.from_param_or_video(self._range_in, self.clip, True, self.func)

    @cachedproperty
    def chromaloc(self) -> ChromaLocation:
        """
        Get the clip's chroma location.
        """

        return ChromaLocation.from_param_or_video(self._chromaloc, self.clip, True, self.func)

    @cachedproperty
    def order(self) -> FieldBased:
        """
        Get the clip's field order.
        """

        return FieldBased.from_param_or_video(self._order, self.clip, True, self.func)

    @property
    def is_float(self) -> bool:
        """
        Whether the clip is of a float sample type.
        """

        return self.norm_clip.format.sample_type is vs.FLOAT

    @property
    def is_integer(self) -> bool:
        """
        Whether the clip is of an integer sample type.
        """

        return self.norm_clip.format.sample_type is vs.INTEGER

    @property
    def is_hd(self) -> bool:
        """
        Whether the clip is of an HD resolution (>= 1280x720).
        """

        return self.norm_clip.width >= 1280 or self.norm_clip.height >= 720

    @property
    def luma(self) -> bool:
        """
        Whether the luma gets processed.
        """

        return 0 in self

    @property
    def luma_only(self) -> bool:
        """
        Whether luma is the only channel that gets processed.
        """

        return self == [0]

    @property
    def chroma(self) -> bool:
        """
        Whether any chroma planes get processed.
        """

        return 1 in self or 2 in self

    @property
    def chroma_only(self) -> bool:
        """
        Whether only chroma planes get processed.
        """

        return self == [1, 2]

    @property
    def chroma_pplanes(self) -> list[int]:
        """
        A list of which chroma planes get processed based on the work clip.

        This means that if you pass [0, 1, 2] but passed a GRAY clip, it will only return [0].
        Similarly, if you pass GRAY and it gets converted to RGB, this will return [0, 1, 2].
        """

        return list({*self} - {0})

    def normalize_planes(self, planes: PlanesT) -> list[int]:
        """
        Normalize the given sequence of planes.
        """

        return normalize_planes(self.work_clip, planes)

    def with_planes(self, planes: PlanesT) -> list[int]:
        return self.normalize_planes(sorted(set(self + self.normalize_planes(planes))))

    def without_planes(self, planes: PlanesT) -> list[int]:
        return self.normalize_planes(sorted(set(self) - {*self.normalize_planes(planes)}))

    def return_clip(self, processed: vs.VideoNode) -> ConstantFormatVideoNode:
        """
        Merge back the chroma if necessary and convert the processed clip back to the original clip's format.
        If `bitdepth != None`, the bitdepth will also be converted if necessary.

        Args:
            processed: The clip with all the processing applied to it.

        Returns:
            Processed clip converted back to the original input clip's format.
        """

        assert check_variable(processed, self.func)

        if len(self.chroma_planes):
            processed = join([processed, *self.chroma_planes], self.norm_clip.format.color_family)

        if self.chroma_only:
            processed = join(self.norm_clip, processed)

        if self.bitdepth:
            processed = depth(processed, self.clip)

        if self.cfamily_converted:
            processed = processed.resize.Bicubic(
                format=self.clip.format.id,
                matrix=self.matrix if self.norm_clip.format.color_family is vs.RGB else None,
                chromaloc=self.chromaloc,
                range=self.color_range.value_zimg,
            )

        return processed

    def norm_seq(self, seq: T | Sequence[T], null: T) -> list[T]:
        """
        Normalize a value or sequence to a list mapped to the clip's planes.
        Unprocessed planes will be set to the given "null" value.
        """

        return [x if i in self else null for i, x in enumerate(normalize_seq(seq, self.num_planes))]

    def __vs_del__(self, core_id: int) -> None:
        self.__dict__[cachedproperty.cache_key].clear()
