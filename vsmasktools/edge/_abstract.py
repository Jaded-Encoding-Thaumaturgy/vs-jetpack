"""
This module defines core abstract classes for building edge detection and ridge detection operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntFlag, auto
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, Self

from jetpytools import FuncExcept, get_subclasses, inject_kwargs_params, inject_self, to_arr

from vsexprtools import ExprList, ExprOp, norm_expr
from vstools import (
    ColorRange,
    ConvMode,
    Planes,
    core,
    depth,
    get_peak_value,
    get_video_format,
    join,
    normalize_planes,
    scale_mask,
    vs,
)

from ..exceptions import UnknownEdgeDetectError, UnknownRidgeDetectError, _UnknownMaskDetectError

__all__ = [
    "EdgeDetect",
    "EdgeDetectLike",
    "EdgeMasksEdgeDetect",
    "EuclideanDistance",
    "MagDirection",
    "MagnitudeEdgeMasks",
    "MagnitudeMatrix",
    "MatrixEdgeDetect",
    "Max",
    "RidgeDetect",
    "RidgeDetectLike",
    "SingleMatrix",
    "get_all_edge_detects",
    "get_all_ridge_detect",
]


class MagDirection(IntFlag):
    """
    Direction flags for magnitude calculations.

    Includes 8 primary compass directions (N, NE, E, SE, S, SW, W, NW) and composite groups (e.g., ALL, AXIS, CORNERS).
    Supports bitwise operations.
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
    """All eight directions combined."""

    AXIS = N | W | S | E
    """The four cardinal directions: North, West, South, East."""

    CORNERS = NW | SW | SE | NE
    """The four intercardinal (diagonal) directions."""

    NORTH = N | NW | NE
    """All northern directions (North, Northwest, Northeast)."""

    WEST = W | NW | SW
    """All western directions (West, Northwest, Southwest)."""

    EAST = E | NE | SE
    """All eastern directions (East, Northeast, Southeast)."""

    SOUTH = S | SW | SE
    """All southern directions (South, Southwest, Southeast)."""

    def select_matrices[T](self, matrices: Sequence[T]) -> Sequence[T]:
        """
        Return matrices matching the active directions in `self`.

        Args:
            matrices: One item for each primary direction, in definition order.

        Returns:
            The subset of matrices for directions enabled in `self`.
        """
        # In Python <3.11, composite flags are included in MagDirection's
        # collection and iteration interfaces.
        primary_flags = [flag for flag in MagDirection if flag != 0 and flag & (flag - 1) == 0]
        assert len(matrices) == len(primary_flags) and self

        return [matrix for flag, matrix in zip(primary_flags, matrices) if self & flag]


def _base_from_param[EdgeDetectT: EdgeDetect](
    cls: type[EdgeDetectT],
    value: str | type[EdgeDetectT] | EdgeDetectT | None,
    exception_cls: type[_UnknownMaskDetectError],
    func_except: FuncExcept | None = None,
) -> type[EdgeDetectT]:
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


def _base_ensure_obj[EdgeDetectT: EdgeDetect](
    cls: type[EdgeDetectT],
    value: str | type[EdgeDetectT] | EdgeDetectT | None,
    func_except: FuncExcept | None = None,
) -> EdgeDetectT:
    if isinstance(value, cls):
        return value

    return cls.from_param(value, func_except)()


class EdgeDetect(ABC):
    """
    Abstract base class for edge detection operators.
    """

    _err_class: ClassVar[type[_UnknownMaskDetectError]] = UnknownEdgeDetectError
    """Custom error class used for validation failures."""

    kwargs: dict[str, Any]
    """Arguments passed to the edgemask function(s)."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the scaler with optional keyword arguments.

        Args:
            **kwargs: Keyword arguments passed to the edgemask function(s).
        """
        self.kwargs = kwargs

    @inject_self
    @inject_kwargs_params
    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float | Sequence[float] | None = None,
        hthr: float | Sequence[float] | None = None,
        multi: float | Sequence[float] = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
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
        clip_p = self._preprocess(clip, **kwargs)

        mask = self._compute_edge_mask(clip_p, multi=multi, planes=planes, **kwargs)

        mask = self._postprocess(mask, clip)

        return self._finalize_mask(mask, lthr, hthr, clamp, planes)

    @classmethod
    def _depth_scale(cls, clip: vs.VideoNode, bitdepth: vs.VideoNode) -> vs.VideoNode:
        fmt = get_video_format(bitdepth)

        if fmt.sample_type == vs.INTEGER:
            return norm_expr([clip, bitdepth], "x mask_max_y *", format=fmt, func=cls)
        return clip

    @classmethod
    def from_param(
        cls, value: type[Self] | Self | str | None = None, /, func_except: FuncExcept | None = None
    ) -> type[Self]:
        """
        Resolve and return an edgemask type from a given input (string, type, or instance).

        Args:
            value: Edgemask identifier (string, class, or instance).
            func_except: Function returned for custom error handling.

        Returns:
            Resolved edgemask type.
        """
        return _base_from_param(cls, value, cls._err_class, func_except)

    @classmethod
    def ensure_obj(cls, value: type[Self] | Self | str | None = None, /, func_except: FuncExcept | None = None) -> Self:
        """
        Ensure that the input is an edgemask instance, resolving it if necessary.

        Args:
            value: Edgemask identifier (string, class, or instance).
            func_except: Function returned for custom error handling.

        Returns:
            Edgemask instance.
        """
        return _base_ensure_obj(cls, value, func_except)

    def _finalize_mask(
        self,
        mask: vs.VideoNode,
        lthr: float | Sequence[float] | None,
        hthr: float | Sequence[float] | None,
        clamp: bool | tuple[float, float] | list[tuple[float, float]],
        planes: Planes,
    ) -> vs.VideoNode:
        if not any([lthr, hthr, clamp]):
            planes = normalize_planes(mask, planes)

            if planes == normalize_planes(mask, None):
                return mask

            return join({None: mask.std.BlankClip(color=[0] * mask.format.num_planes, keep=True), tuple(planes): mask})

        if lthr is None:
            lthr = 0.0
        if hthr is None:
            hthr = 1.0

        lthr = [scale_mask(lt, 32, mask) for lt in to_arr(lthr)]
        hthr = [scale_mask(ht, 32, mask) for ht in to_arr(hthr)]

        peak = get_peak_value(mask, range_in=ColorRange.FULL)

        thr_expr = ExprList(["x"])

        if lthr == hthr:
            thr_expr.append("{hthr} >= range_max 0 ?")
        elif any(lt > 0 for lt in lthr) and any(ht < peak for ht in hthr):
            thr_expr.append("{hthr} >", "range_max", ["x {lthr} < 0 x ?"], "?")
        elif any(lt > 0 for lt in lthr):
            thr_expr.append("{lthr} < 0 x ?")
        elif any(ht < peak for ht in hthr):
            thr_expr.append("{hthr} > range_max x ?")

        if clamp is True and mask.format.sample_type == vs.FLOAT:
            clamp = [(0, 1)]
        elif isinstance(clamp, bool):
            clamp = [(False, False)]
        elif isinstance(clamp, tuple):
            clamp = [clamp]

        mask = norm_expr(
            mask,
            [thr_expr, "{clamp}"],
            planes,
            func=self.__class__,
            lthr=lthr,
            hthr=hthr,
            clamp=["{} {} clamp".format(*c) if any(c) else "" for c in clamp],
        )

        return self._finalize_mask(mask, None, None, False, planes)

    @abstractmethod
    def _compute_edge_mask(
        self,
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    def _preprocess(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return ColorRange.FULL.apply(clip)

    def _postprocess(self, clip: vs.VideoNode, input_bits: vs.VideoNode) -> vs.VideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect):
    """
    Edge detection based on convolution matrices.
    """

    matrices: ClassVar[Sequence[Sequence[float]]]
    """Convolution kernels used for edge detection."""

    divisors: ClassVar[Sequence[float] | None] = None
    """Optional divisors applied to each kernel. Defaults to zeros (no division)."""

    mode_types: ClassVar[Sequence[str] | None] = None
    """Optional convolution modes (e.g. 's' for square). Defaults to 's'."""

    def _compute_edge_mask(
        self,
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        exprs = [
            ExprOp.convolution("x", mat, divisor=div, saturate=False, mode=ConvMode(mode))[0]
            for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
        ]
        return self._merge_edge(exprs, clip, multi=multi, planes=planes, **kwargs)

    @abstractmethod
    def _merge_edge(
        self,
        exprs: Sequence[ExprList],
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ["s"] * len(self._get_matrices())


class EdgeMasksEdgeDetect(MatrixEdgeDetect):
    """
    Edge detection using VapourSynth's `EdgeMasks` plugin with expression-based convolution fallback.
    """

    @inject_self
    @inject_kwargs_params
    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float | Sequence[float] | None = None,
        hthr: float | Sequence[float] | None = None,
        multi: float | Sequence[float] = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        if not hasattr(core, "edgemasks"):
            kwargs.setdefault("use_expr", True)

        return super().edgemask(clip, lthr, hthr, multi, clamp, planes, **kwargs)

    def _preprocess(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if kwargs.get("use_expr"):
            return super()._preprocess(clip)
        return clip

    def _compute_edge_mask(
        self,
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        if kwargs.pop("use_expr", False):
            return super()._compute_edge_mask(clip, multi=multi, planes=planes, **kwargs)

        return getattr(core.edgemasks, self.__class__.__name__)(clip, planes, multi, **kwargs)


class NormalizeProcessor(MatrixEdgeDetect):
    """
    Edge detection processor with normalized precision.
    """

    def _preprocess(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return super()._preprocess(depth(clip, 32))

    def _postprocess(self, clip: vs.VideoNode, input_bits: vs.VideoNode) -> vs.VideoNode:
        return super()._postprocess(self._depth_scale(clip, input_bits), input_bits)


class MagnitudeMatrix(MatrixEdgeDetect):
    """
    Edge detector using a subset of convolution matrices.

    Allows selecting which matrices to apply based on directional flags. By default, all directions are used.
    """

    def __init__(self, mag_directions: MagDirection = MagDirection.ALL, **kwargs: Any) -> None:
        """
        Initialize with a set of magnitude directions.

        Args:
            mag_directions: Directional flags controlling which matrices are used. Defaults to all directions.
            **kwargs: Additional parameters passed to the base class.
        """
        super().__init__(**kwargs)

        self.mag_directions = mag_directions

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return [m for m in self.mag_directions.select_matrices(self.matrices) if m]


class MagnitudeEdgeMasks(EdgeMasksEdgeDetect, MagnitudeMatrix):
    """
    Edge detector using a subset of convolution matrices.

    Allows selecting which matrices to apply based on directional flags. By default, all directions are used.

    If a subset of directions is selected, the computation automatically switches to the expression-based backend.
    """

    if TYPE_CHECKING:

        def __init__(self, mag_directions: MagDirection = MagDirection.ALL, **kwargs: Any) -> None:
            """
            Initialize the MagnitudeEdgeMasks detector.

            Args:
                mag_directions: Directional flags controlling which matrices are used. Defaults to all directions.
                    If a subset is specified, the expression-based backend is used automatically
                    to support custom directions.
                **kwargs: Additional parameters forwarded to the base EdgeMasksEdgeDetect class.
            """
            ...

    @inject_self
    @inject_kwargs_params
    def edgemask(
        self,
        clip: vs.VideoNode,
        lthr: float | Sequence[float] | None = None,
        hthr: float | Sequence[float] | None = None,
        multi: float | Sequence[float] = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        if self.mag_directions != MagDirection.ALL:
            kwargs["use_expr"] = True

        return super().edgemask(clip, lthr, hthr, multi, clamp, planes, **kwargs)


class SingleMatrix(MatrixEdgeDetect, ABC):
    """
    Edge detector that uses a single convolution matrix.
    """

    def _merge_edge(
        self,
        exprs: Sequence[ExprList],
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return norm_expr(
            clip,
            [exprs[0], "{multi}"],
            planes,
            func=self.__class__,
            multi=[f"{m} *" if m != 1.0 else "" for m in to_arr(multi)],
            **kwargs,
        )


class EuclideanDistance(MatrixEdgeDetect, ABC):
    """
    Edge detector combining two matrices via Euclidean distance.
    """

    def _merge_edge(
        self,
        exprs: Sequence[ExprList],
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return norm_expr(
            clip,
            [exprs[0], "dup *", exprs[1], "dup * + sqrt 0 max", "{multi}"],
            planes,
            func=self.__class__,
            multi=[f"{m} *" if m != 1.0 else "" for m in to_arr(multi)],
            **kwargs,
        )


class Max(MatrixEdgeDetect, ABC):
    """
    Edge detector combining multiple matrices by maximum response.

    Produces the edge mask by selecting the maximum value across all convolution results.
    """

    def _merge_edge(
        self,
        exprs: Sequence[ExprList],
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return norm_expr(
            clip,
            [*exprs, ExprOp.MAX * (len(exprs) - 1), "{multi}"],
            planes,
            func=self.__class__,
            multi=[f"{m} *" if m != 1.0 else "" for m in to_arr(multi)],
        )


class RidgeDetect(MatrixEdgeDetect):
    """
    Edge detector specialized for ridge detection.
    """

    _err_class: ClassVar[type[_UnknownMaskDetectError]] = UnknownRidgeDetectError

    @inject_self
    @inject_kwargs_params
    def ridgemask(
        self,
        clip: vs.VideoNode,
        lthr: float | Sequence[float] | None = None,
        hthr: float | Sequence[float] | None = None,
        multi: float | Sequence[float] = 1.0,
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
        clip_p = self._preprocess_ridge(clip)

        mask = self._compute_ridge_mask(clip_p, multi=multi, planes=planes, **kwargs)

        mask = self._postprocess_ridge(mask, clip)

        return self._finalize_mask(mask, lthr, hthr, clamp, planes)

    def _compute_ridge_mask(
        self,
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        def _x(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[0], divisor=self._get_divisors()[0], planes=planes)

        def _y(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[1], divisor=self._get_divisors()[1], planes=planes)

        return self._merge_ridge([_x(clip), _y(clip)], multi=multi, planes=planes, **kwargs)

    def _merge_ridge(
        self,
        clips: Sequence[vs.VideoNode],
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        (xx,) = ExprOp.convolution("x", self._get_matrices()[0], divisor=self._get_divisors()[0], mode=ConvMode.SQUARE)
        (yy,) = ExprOp.convolution("y", self._get_matrices()[1], divisor=self._get_divisors()[1], mode=ConvMode.SQUARE)
        (xy,) = ExprOp.convolution("x", self._get_matrices()[1], divisor=self._get_divisors()[1], mode=ConvMode.SQUARE)

        expr = [
            xx,
            "XX!",
            yy,
            "YY!",
            xy,
            "XY!",
            "XX@ dup * XY@ dup * 4 * + XX@ YY@ * 2 * - YY@ dup * + sqrt XX@ YY@ + + 0.5 *",
            "{multi}",
        ]

        return norm_expr(
            clips,
            expr,
            planes,
            func=self.__class__,
            multi=[f"{m} *" if m != 1.0 else "" for m in to_arr(multi)],
            **kwargs,
        )

    def _preprocess_ridge(self, clip: vs.VideoNode) -> vs.VideoNode:
        return depth(super()._preprocess(clip), 32)

    def _postprocess_ridge(self, clip: vs.VideoNode, input_bits: vs.VideoNode) -> vs.VideoNode:
        return self._depth_scale(super()._postprocess(clip, input_bits), input_bits)


type EdgeDetectLike = type[EdgeDetect] | EdgeDetect | str
"""
Type alias for anything that can resolve to a EdgeDetect.

This includes:
 - A string identifier.
 - A class type subclassing `EdgeDetect`.
 - An instance of a `EdgeDetect`.
"""

type RidgeDetectLike = type[RidgeDetect] | RidgeDetect | str
"""
Type alias for anything that can resolve to a RidgeDetect.

This includes:
 - A string identifier.
 - A class type subclassing `RidgeDetect`.
 - An instance of a `RidgeDetect`.
"""


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0,
    hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
    planes: Planes = None,
    **kwargs: Any,
) -> list[vs.VideoNode]:
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

    out = list[vs.VideoNode]()

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
) -> list[vs.VideoNode]:
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

    out = list[vs.VideoNode]()

    for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__):
        try:
            mask = edge_detect().ridgemask(clip, lthr, hthr, multi, clamp, planes, **kwargs)  # pyright: ignore[reportAbstractUsage]
        except AttributeError as e:
            from warnings import warn

            warn(str(e), RuntimeWarning)
            continue

        out.append(mask.text.Text(edge_detect.__name__))

    return out
