from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntFlag, auto
from typing import Any, ClassVar, Sequence, TypeAlias, TypeVar, cast

from jetpytools import CustomNotImplementedError, inject_kwargs_params
from typing_extensions import Self, TypeIs

from vsexprtools import ExprOp, ExprToken, norm_expr
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    CustomRuntimeError,
    CustomValueError,
    DitherType,
    FuncExceptT,
    KwargsT,
    PlanesT,
    T,
    check_variable,
    depth,
    get_lowest_values,
    get_peak_value,
    get_peak_values,
    get_subclasses,
    inject_self,
    join,
    normalize_planes,
    plane,
    scale_mask,
    vs,
)

from ..exceptions import UnknownEdgeDetectError, UnknownRidgeDetectError

__all__ = [
    "EdgeDetect",
    "EdgeDetectT",
    "EuclideanDistance",
    "MagDirection",
    "MagnitudeMatrix",
    "MatrixEdgeDetect",
    "Max",
    "RidgeDetect",
    "RidgeDetectT",
    "SingleMatrix",
    "get_all_edge_detects",
    "get_all_ridge_detect",
]


class _Feature(Enum):
    EDGE = auto()
    RIDGE = auto()


class MagDirection(IntFlag):
    """
    Direction of magnitude calculation.
    """

    N = auto()
    NW = auto()
    W = auto()
    SW = auto()
    S = auto()
    SE = auto()
    E = auto()
    NE = auto()

    ALL = N | NW | W | SW | S | SE | E | NE

    AXIS = N | W | S | E
    CORNERS = NW | SW | SE | NE

    NORTH = N | NW | NE
    WEST = W | NW | SW
    EAST = E | NE | SE
    SOUTH = S | SW | SE

    def select_matrices(self, matrices: Sequence[T]) -> Sequence[T]:
        # In Python <3.11, composite flags are included in MagDirection's
        # collection and iteration interfaces.
        primary_flags = [flag for flag in MagDirection if flag != 0 and flag & (flag - 1) == 0]
        assert len(matrices) == len(primary_flags) and self

        return [matrix for flag, matrix in zip(primary_flags, matrices) if self & flag]


def _base_from_param(
    cls: type[EdgeDetectTypeVar],
    value: str | type[EdgeDetectTypeVar] | EdgeDetectTypeVar | None,
    exception_cls: type[CustomValueError],
    excluded: Sequence[type[EdgeDetectTypeVar]] = [],
    func_except: FuncExceptT | None = None,
) -> type[EdgeDetectTypeVar]:
    if isinstance(value, str):
        all_edge_detects = get_subclasses(EdgeDetect, excluded)
        search_str = value.lower().strip()

        for edge_detect_cls in all_edge_detects:
            if edge_detect_cls.__name__.lower() == search_str:
                return cast(type[EdgeDetectTypeVar], edge_detect_cls)

        raise exception_cls(func_except or cls.from_param, value)

    if isinstance(value, type) and issubclass(value, cls):
        return value

    if isinstance(value, cls):
        return value.__class__

    return cls


def _base_ensure_obj(
    cls: type[EdgeDetectTypeVar],
    value: str | type[EdgeDetectTypeVar] | EdgeDetectTypeVar | None,
    exception_cls: type[CustomValueError],
    excluded: Sequence[type[EdgeDetectTypeVar]] = [],
    func_except: FuncExceptT | None = None,
) -> EdgeDetectTypeVar:
    if value is None:
        new_edge_detect = cls()
    elif isinstance(value, cls):
        new_edge_detect = value
    else:
        new_edge_detect = cls.from_param(value, func_except)()

    if new_edge_detect.__class__ in excluded:
        raise exception_cls(
            func_except or cls.ensure_obj,
            new_edge_detect.__class__,
            "This {cls_name} can't be instantiated to be used!",
            cls_name=new_edge_detect.__class__,
        )

    return new_edge_detect


def _is_discard_planes_mode(planes: PlanesT | tuple[PlanesT, bool]) -> TypeIs[tuple[PlanesT, bool]]:
    return bool(
        isinstance(planes, tuple)
        and len(planes) == 2
        and (isinstance(planes[0], (int, Sequence)) or planes[0] is None)
        and isinstance(planes[1], bool)
    )


class EdgeDetect(ABC):
    """
    Abstract edge detection interface.
    """

    kwargs: KwargsT | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.kwargs = kwargs

    @classmethod
    def from_param(
        cls,
        edge_detect: type[Self] | Self | str | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> type[Self]:
        return _base_from_param(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(
        cls,
        edge_detect: type[Self] | Self | str | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> Self:
        return _base_ensure_obj(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @inject_self
    @inject_kwargs_params
    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0,
        hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: PlanesT | tuple[PlanesT, bool] = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Makes edge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        Args:
            clip: Source clip
            lthr: Low threshold. Anything below lthr will be set to 0
            hthr: High threshold. Anything above hthr will be set to the range max
            multi: Multiply all pixels by this before thresholding
            clamp: Clamp to TV or full range if True or specified range `(low, high)`

        Returns:
            Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.EDGE, planes, **kwargs)

    def _mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0,
        hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        feature: _Feature = _Feature.EDGE,
        planes: PlanesT | tuple[PlanesT, bool] = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        peak = get_peak_value(clip)
        hthr = 1.0 if hthr is None else hthr

        lthr = scale_mask(lthr, 32, clip)
        hthr = scale_mask(hthr, 32, clip)

        discard_planes = False
        if _is_discard_planes_mode(planes):
            planes, discard_planes = planes

        planes = normalize_planes(clip, planes)

        wclip = plane(clip, planes[0]) if len(planes) == 1 else clip

        clip_p = self._preprocess(wclip)

        if feature == _Feature.EDGE:
            mask = self._compute_edge_mask(clip_p, **kwargs)
        elif feature == _Feature.RIDGE:
            if not isinstance(self, RidgeDetect):
                raise CustomRuntimeError(
                    f"Ridge feature has been requested but {self.__class__.__name__} is not a subclass of RidgeDetect",
                    self.__class__,
                )

            mask = self._compute_ridge_mask(clip_p, **kwargs)
        else:
            raise CustomNotImplementedError

        mask = self._postprocess(mask, clip.format.bits_per_sample)

        if multi != 1:
            mask = ExprOp.MUL(mask, suffix=str(multi), planes=planes)

        if lthr == hthr:
            mask = norm_expr(mask, f"x {hthr} >= {ExprToken.RangeMax} 0 ?", planes, func=self.__class__)
        elif lthr > 0 and hthr < peak:
            mask = norm_expr(mask, f"x {hthr} > {ExprToken.RangeMax} x {lthr} < 0 x ? ?", planes, func=self.__class__)
        elif lthr > 0:
            mask = norm_expr(mask, f"x {lthr} < 0 x ?", planes, func=self.__class__)
        elif hthr < peak:
            mask = norm_expr(mask, f"x {hthr} > {ExprToken.RangeMax} x ?", planes, func=self.__class__)

        if clamp is True:
            crange = ColorRange.from_video(clip)
            clamp = list(zip(get_lowest_values(mask, crange), get_peak_values(mask, crange)))

        if isinstance(clamp, list):
            mask = norm_expr(mask, [ExprOp.clamp(*c, c="x") for c in clamp], planes, func=self.__class__)
        elif isinstance(clamp, tuple):
            mask = ExprOp.clamp(*clamp, c="x")(mask, planes=planes)

        if mask.format.num_planes != clip.format.num_planes and not discard_planes:
            return join({None: clip.std.BlankClip(color=[0] * clip.format.num_planes, keep=True), planes[0]: mask})

        return mask

    @abstractmethod
    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return ColorRange.FULL.apply(clip)

    def _postprocess(self, clip: ConstantFormatVideoNode, input_bits: int | None = None) -> ConstantFormatVideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Sequence[float] | None] = None
    mode_types: ClassVar[Sequence[str] | None] = None

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return self._merge_edge(
            [
                clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
                for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
            ]
        )

    def _compute_ridge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        def _x(c: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[0], divisor=self._get_divisors()[0])

        def _y(c: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[1], divisor=self._get_divisors()[1])

        x = _x(clip)
        y = _y(clip)
        xx = _x(x)
        yy = _y(y)
        xy = _y(x)
        return self._merge_ridge([xx, yy, xy])

    @abstractmethod
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        raise NotImplementedError

    @abstractmethod
    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        raise NotImplementedError

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ["s"] * len(self._get_matrices())

    def _postprocess(self, clip: ConstantFormatVideoNode, input_bits: int | None = None) -> ConstantFormatVideoNode:
        if self.mode_types is None or self.mode_types[0] == "s":
            cropped = vs.core.std.Crop(
                clip, right=clip.format.subsampling_w * 2 if clip.format.subsampling_w != 0 else 2
            )
            clip = vs.core.resize.Point(cropped, clip.width, src_width=clip.width)

        return clip


class MagnitudeMatrix(MatrixEdgeDetect):
    def __init__(self, mag_directions: MagDirection = MagDirection.ALL, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.mag_directions = mag_directions

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.mag_directions.select_matrices(self.matrices)


class RidgeDetect(MatrixEdgeDetect):
    @classmethod
    def from_param(
        cls,
        edge_detect: type[Self] | Self | str | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> type[Self]:
        return _base_from_param(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(
        cls,
        edge_detect: type[Self] | Self | str | None = None,
        /,
        func_except: FuncExceptT | None = None,
    ) -> Self:
        return _base_ensure_obj(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @inject_self
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0,
        hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: PlanesT | tuple[PlanesT, bool] = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Makes ridge mask based on convolution kernel.

        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.
        Using a 32-bit float clip is recommended.

        Args:
            clip: Source clip
            lthr: Low threshold. Anything below lthr will be set to 0
            hthr: High threshold. Anything above hthr will be set to the range max
            multi: Multiply all pixels by this before thresholding
            clamp: Clamp to TV or full range if True or specified range `(low, high)`

        Returns:
            Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.RIDGE, planes, **kwargs)

    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        return norm_expr(clips, "x 2 pow z 2 pow 4 * + x y * 2 * - y 2 pow + sqrt x y + + 0.5 *", func=self.__class__)

    def _preprocess(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        if self.mode_types is None or self.mode_types[0] == "s":
            clip = vs.core.resize.Point(clip, clip.width + 4, src_width=clip.width + 4)

        return super()._preprocess(depth(clip, 32))

    def _postprocess(self, clip: ConstantFormatVideoNode, input_bits: int | None = None) -> ConstantFormatVideoNode:
        clip = depth(clip, input_bits, dither_type=DitherType.NONE)

        if self.mode_types is None or self.mode_types[0] == "s":
            return clip.std.Crop(right=4)

        return super()._postprocess(clip, input_bits)


class SingleMatrix(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        return clips[0]

    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        raise NotImplementedError


class EuclideanDistance(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        return norm_expr(clips, "x dup * y dup * + sqrt", func=self.__class__)

    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        raise NotImplementedError


class Max(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        return ExprOp.MAX.combine(*clips)

    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode]) -> ConstantFormatVideoNode:
        raise NotImplementedError


EdgeDetectT: TypeAlias = type[EdgeDetect] | EdgeDetect | str
RidgeDetectT: TypeAlias = type[RidgeDetect] | RidgeDetect | str

EdgeDetectTypeVar = TypeVar("EdgeDetectTypeVar", bound=EdgeDetect)
RidgeDetectTypeVar = TypeVar("RidgeDetectTypeVar", bound=RidgeDetect)


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0,
    hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
) -> list[ConstantFormatVideoNode]:
    """
    Returns all the EdgeDetect subclasses

    Args:
        clip: Source clip
        lthr: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        hthr: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        multi: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        clamp: Clamp to TV or full range if True or specified range `(low, high)`

    Returns:
        A list edge masks
    """

    def _all_subclasses(cls: type[EdgeDetect] = EdgeDetect) -> set[type[EdgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s
        for s in _all_subclasses()
        if s.__name__
        not in {
            "MatrixEdgeDetect",
            "RidgeDetect",
            "SingleMatrix",
            "EuclideanDistance",
            "MagnitudeMatrix",
            "Max",
            "Matrix1D",
            "SavitzkyGolay",
            "SavitzkyGolayNormalise",
            "Matrix3x3",
            "Matrix5x5",
        }
    }
    return [
        edge_detect().edgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]


def get_all_ridge_detect(
    clip: vs.VideoNode,
    lthr: float = 0.0,
    hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
) -> list[ConstantFormatVideoNode]:
    """
    Returns all the RidgeDetect subclasses

    Args:
        clip: Source clip
        lthr: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        hthr: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        multi: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        clamp: Clamp to TV or full range if True or specified range `(low, high)`

    Returns:
        A list edge masks
    """

    def _all_subclasses(cls: type[RidgeDetect] = RidgeDetect) -> set[type[RidgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s
        for s in _all_subclasses()
        if s.__name__
        not in {
            "MatrixEdgeDetect",
            "RidgeDetect",
            "SingleMatrix",
            "EuclideanDistance",
            "MagnitudeMatrix",
            "Max",
            "Matrix1D",
            "SavitzkyGolay",
            "SavitzkyGolayNormalise",
            "Matrix3x3",
            "Matrix5x5",
        }
    }
    return [
        edge_detect().ridgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]
