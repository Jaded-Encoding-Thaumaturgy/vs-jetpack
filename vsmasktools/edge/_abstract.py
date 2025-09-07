from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntFlag, auto
from inspect import isabstract
from typing import Any, ClassVar, Sequence, TypeAlias, TypeVar

from jetpytools import inject_kwargs_params
from typing_extensions import Self

from vsexprtools import ExprOp, ExprToken, norm_expr
from vsrgtools import BlurMatrix
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    ConvMode,
    FuncExcept,
    HoldsVideoFormat,
    KwargsT,
    Planes,
    T,
    VideoFormatLike,
    check_variable,
    depth,
    get_lowest_value,
    get_peak_value,
    get_subclasses,
    get_video_format,
    inject_self,
    join,
    limiter,
    normalize_planes,
    scale_mask,
    vs,
)

from ..exceptions import UnknownEdgeDetectError, UnknownRidgeDetectError, _UnknownMaskDetectError

__all__ = [
    "EdgeDetect",
    "EdgeDetectLike",
    "EdgeDetectT",
    "EuclideanDistance",
    "MagDirection",
    "MagnitudeMatrix",
    "MatrixEdgeDetect",
    "Max",
    "RidgeDetect",
    "RidgeDetectLike",
    "RidgeDetectT",
    "SingleMatrix",
    "get_all_edge_detects",
    "get_all_ridge_detect",
]


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
    exception_cls: type[_UnknownMaskDetectError],
    func_except: FuncExcept | None = None,
) -> type[EdgeDetectTypeVar]:
    # If value is an instance returns the class
    if isinstance(value, cls):
        return value.__class__

    # If value is a type and a subclass of the caller returns the value itself
    if isinstance(value, type) and issubclass(value, cls):
        return value

    # Search for the subclasses of the caller and the caller itself
    if isinstance(value, str):
        all_scalers = {s.__name__.lower(): s for s in [*get_subclasses(cls), cls]}

        try:
            return all_scalers[value.lower().strip()]
        except KeyError:
            raise exception_cls(func_except or cls.from_param, value)

    if value is None:
        return cls

    raise exception_cls(func_except or cls.from_param, str(value))


def _base_ensure_obj(
    cls: type[EdgeDetectTypeVar],
    value: str | type[EdgeDetectTypeVar] | EdgeDetectTypeVar | None,
    func_except: FuncExcept | None = None,
) -> EdgeDetectTypeVar:
    if isinstance(value, cls):
        return value

    return cls.from_param(value, func_except)()


class EdgeDetect(ABC):
    """
    Abstract edge detection interface.
    """

    _err_class: ClassVar[type[_UnknownMaskDetectError]] = UnknownEdgeDetectError
    """Custom error class used for validation failures."""

    kwargs: KwargsT | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.kwargs = kwargs

    @inject_self
    @inject_kwargs_params
    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0,
        hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: Planes = None,
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
            clamp: Clamp to legal values if True or specified range `(low, high)`
            planes: Which planes to process.

        Returns:
            Mask clip
        """
        assert check_variable(clip, self.__class__)

        clip_p = self._preprocess(clip)

        mask = self._compute_edge_mask(clip_p, planes=planes, **kwargs)

        mask = self._postprocess(mask, clip)

        return self._finalize_mask(mask, lthr, hthr, multi, clamp, planes)

    @classmethod
    def depth_scale(
        cls, clip: vs.VideoNode, bitdepth: VideoFormatLike | HoldsVideoFormat | int
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, cls)

        fmt = get_video_format(bitdepth)

        if fmt.sample_type == vs.INTEGER:
            return norm_expr(
                clip,
                ("x abs {peak} *", "x abs {peak} * neutral -"),
                format=fmt,
                peak=get_peak_value(bitdepth),
            )
        return clip

    @classmethod
    def from_param(
        cls, value: type[Self] | Self | str | None = None, /, func_except: FuncExcept | None = None
    ) -> type[Self]:
        return _base_from_param(cls, value, cls._err_class, func_except)

    @classmethod
    def ensure_obj(cls, value: type[Self] | Self | str | None = None, /, func_except: FuncExcept | None = None) -> Self:
        return _base_ensure_obj(cls, value, func_except)

    def _finalize_mask(
        self,
        mask: ConstantFormatVideoNode,
        lthr: float,
        hthr: float | None,
        multi: float,
        clamp: bool | tuple[float, float] | list[tuple[float, float]],
        planes: Planes,
    ) -> ConstantFormatVideoNode:
        peak = get_peak_value(mask)

        hthr = 1.0 if hthr is None else hthr

        lthr = scale_mask(lthr, 32, mask)
        hthr = scale_mask(hthr, 32, mask)

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
            clamp = (get_lowest_value(mask, range_in=ColorRange.FULL), get_peak_value(mask, range_in=ColorRange.FULL))

        if isinstance(clamp, list):
            mask = limiter(mask, *zip(*clamp), planes=planes, func=self.__class__)
        elif isinstance(clamp, tuple):
            mask = limiter(mask, *clamp, planes=planes, func=self.__class__)

        if planes is not None:
            return join(
                {
                    None: mask.std.BlankClip(color=[0] * mask.format.num_planes, keep=True),
                    tuple(normalize_planes(mask, planes)): mask,
                }
            )

        return mask

    @abstractmethod
    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return ColorRange.FULL.apply(clip)

    def _postprocess(
        self, clip: ConstantFormatVideoNode, input_bits: HoldsVideoFormat | VideoFormatLike | int
    ) -> ConstantFormatVideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Sequence[float] | None] = None
    mode_types: ClassVar[Sequence[str] | None] = None

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return self._merge_edge(
            [
                BlurMatrix.custom(mat, ConvMode(mode))(clip, divisor=div, saturate=False, func=self.__class__, **kwargs)
                for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
            ]
        )

    @abstractmethod
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        raise NotImplementedError

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ["s"] * len(self._get_matrices())


class NormalizeProcessor(MatrixEdgeDetect):
    def _preprocess(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return super()._preprocess(depth(clip, 32))

    def _postprocess(
        self, clip: ConstantFormatVideoNode, input_bits: HoldsVideoFormat | VideoFormatLike | int
    ) -> ConstantFormatVideoNode:
        return super()._postprocess(self.depth_scale(clip, input_bits), input_bits)


class MagnitudeMatrix(MatrixEdgeDetect):
    def __init__(self, mag_directions: MagDirection = MagDirection.ALL, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.mag_directions = mag_directions

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return [m for m in self.mag_directions.select_matrices(self.matrices) if m]


class SingleMatrix(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        return clips[0]


class EuclideanDistance(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        return norm_expr(clips, "x dup * y dup * + sqrt 0 mask_max clamp", kwargs.get("planes"), func=self.__class__)


class Max(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        return ExprOp.MAX.combine(*clips, planes=kwargs.get("planes"), func=self.__class__)


class RidgeDetect(MatrixEdgeDetect):
    _err_class: ClassVar[type[_UnknownMaskDetectError]] = UnknownRidgeDetectError

    @inject_self
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0,
        hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: Planes = None,
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
            clamp: clamp: Clamp to legal values if True or specified range `(low, high)`
            planes: Which planes to process.

        Returns:
            Mask clip
        """

        assert check_variable(clip, self.__class__)

        clip_p = self._preprocess_ridge(clip)

        mask = self._compute_ridge_mask(clip_p, planes=planes, **kwargs)

        mask = self._postprocess_ridge(mask, clip)

        return self._finalize_mask(mask, lthr, hthr, multi, clamp, planes)

    def _compute_ridge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        def _x(c: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            return c.std.Convolution(
                matrix=self._get_matrices()[0], divisor=self._get_divisors()[0], planes=kwargs.get("planes")
            )

        def _y(c: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
            return c.std.Convolution(
                matrix=self._get_matrices()[1], divisor=self._get_divisors()[1], planes=kwargs.get("planes")
            )

        x = _x(clip)
        y = _y(clip)
        xx = _x(x)
        yy = _y(y)
        xy = _y(x)
        return self._merge_ridge([xx, yy, xy])

    def _merge_ridge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        return norm_expr(
            clips,
            "x dup * z dup * 4 * + x y * 2 * - y dup * + sqrt x y + + 0.5 *",
            kwargs.get("planes"),
            func=self.__class__,
        )

    def _preprocess_ridge(self, clip: ConstantFormatVideoNode) -> ConstantFormatVideoNode:
        return depth(super()._preprocess(clip), 32)

    def _postprocess_ridge(
        self, clip: ConstantFormatVideoNode, input_bits: HoldsVideoFormat | VideoFormatLike | int
    ) -> ConstantFormatVideoNode:
        return self.depth_scale(super()._postprocess(clip, input_bits), input_bits)


EdgeDetectLike: TypeAlias = type[EdgeDetect] | EdgeDetect | str
"""
Type alias for anything that can resolve to a EdgeDetect.

This includes:
 - A string identifier.
 - A class type subclassing `EdgeDetect`.
 - An instance of a `EdgeDetect`.
"""

RidgeDetectLike: TypeAlias = type[RidgeDetect] | RidgeDetect | str
"""
Type alias for anything that can resolve to a RidgeDetect.

This includes:
 - A string identifier.
 - A class type subclassing `RidgeDetect`.
 - An instance of a `RidgeDetect`.
"""

EdgeDetectT = EdgeDetectLike
"""Deprecated alias of EdgeDetectLike."""

RidgeDetectT = RidgeDetectLike
"""Deprecated alias of RidgeDetectLike."""

EdgeDetectTypeVar = TypeVar("EdgeDetectTypeVar", bound=EdgeDetect)
RidgeDetectTypeVar = TypeVar("RidgeDetectTypeVar", bound=RidgeDetect)


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0,
    hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
    planes: Planes = None,
    **kwargs: Any,
) -> list[ConstantFormatVideoNode]:
    """
    Returns all the EdgeDetect subclasses

    Args:
        clip: Source clip
        lthr: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        hthr: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        multi: See [EdgeDetect.get_mask][vsmasktools.edge.EdgeDetect.edgemask]
        clamp: Clamp to legal values if True or specified range `(low, high)`
        planes: Which planes to process.

    Returns:
        A list of edge masks
    """
    # https://github.com/python/mypy/issues/4717
    all_subclasses = {
        s
        for s in get_subclasses(EdgeDetect)  # type: ignore[type-abstract]
        if not isabstract(s) and s.__module__.split(".")[-1] != "_abstract"
    }

    out = list[ConstantFormatVideoNode]()

    for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__):
        try:
            mask = edge_detect().edgemask(clip, lthr, hthr, multi, clamp, planes, **kwargs)  # pyright: ignore[reportAbstractUsage]
        except AttributeError as e:
            from warnings import warn

            warn(str(e), RuntimeWarning)
            continue

        out.append(mask.text.Text(edge_detect.__name__))

    return out


def get_all_ridge_detect(
    clip: vs.VideoNode,
    lthr: float = 0.0,
    hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
    planes: Planes = None,
    **kwargs: Any,
) -> list[ConstantFormatVideoNode]:
    """
    Returns all the RidgeDetect subclasses

    Args:
        clip: Source clip
        lthr: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        hthr: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        multi: See [RidgeDetect.get_mask][vsmasktools.edge.RidgeDetect.ridgemask]
        clamp: Clamp to legal values if True or specified range `(low, high)`
        planes: Which planes to process.

    Returns:
        A list edge masks
    """
    # https://github.com/python/mypy/issues/4717
    all_subclasses = {
        s
        for s in get_subclasses(RidgeDetect)  # type: ignore[type-abstract]
        if not isabstract(s) and s.__module__.split(".")[-1] != "_abstract"
    }

    out = list[ConstantFormatVideoNode]()

    for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__):
        try:
            mask = edge_detect().ridgemask(clip, lthr, hthr, multi, clamp, planes, **kwargs)  # pyright: ignore[reportAbstractUsage]
        except AttributeError as e:
            from warnings import warn

            warn(str(e), RuntimeWarning)
            continue

        out.append(mask.text.Text(edge_detect.__name__))

    return out
