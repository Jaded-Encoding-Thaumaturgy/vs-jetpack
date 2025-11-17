"""
This module implements utilities for correcting dirty or damaged borders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

from jetpytools import CustomOverflowError, CustomTypeError, FuncExcept, normalize_seq

from vsexprtools import ExprList, ExprOp, ExprToken, norm_expr
from vsjetpack import TypeIs
from vstools import VSObject, get_lowest_values, get_peak_values, get_resolutions, vs

__all__ = ["FixBorderBrightness"]

if TYPE_CHECKING:
    type NoneSlice = slice[None, None, None] | None
    """A slice with all components possibly None, or None itself."""

    type IndexLike = SupportsIndex | slice[SupportsIndex, SupportsIndex, SupportsIndex]
    """Either an integer-like index or a fully-typed slice."""
else:
    type NoneSlice = slice | None
    type IndexLike = SupportsIndex | slice


def _normalize_slice(index: IndexLike | NoneSlice, length: int, func: FuncExcept) -> slice:
    index_i = index

    if _is_slice_none(index):
        index = 0

    if isinstance(index, SupportsIndex):
        index = index.__index__()

        if index < 0:
            index += length

        if not 0 <= index < length:
            raise CustomOverflowError(
                "Passed 'num' is out of bound. Must be in the range (-{last}, {last})",
                func,
                index_i,
                last=length - 1,
            )

        index = slice(index, index + 1, 1)

    return index


def _is_slice_not_none(index: SupportsIndex | slice | None) -> TypeIs[IndexLike]:
    return index is not None and index != slice(None, None, None)


def _is_slice_none(index: SupportsIndex | slice | None) -> TypeIs[NoneSlice]:
    return index is None or index == slice(None, None, None)


class _BorderDict(dict[int, float]):
    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__()


class FixBorderBrightness(VSObject):
    """
    Utility class to adjust or correct brightness inconsistencies along clip borders.

    Example:
        ```py
        # The following example darkens the top and right borders slightly (0.9x)
        # to reduce edge brightness inconsistencies, then applies the correction.

        fbb = FixBorderBrightness(clip)
        fbb.fix_row(-2, 0.9)  # Adjust brightness near the bottom edge
        fbb.fix_row(1, 0.9)  # Adjust brightness near the top edge
        fbb.fix_column(1, 0.9)  # Adjust brightness near the left edge
        fbb.fix_column(-2, 0.9)  # Adjust brightness near the right edge
        out = fbb.process()  # Apply all configured border corrections
        ```
    """

    def __init__(
        self, clip: vs.VideoNode, protect: bool | tuple[float, float] | list[tuple[float, float]] = True
    ) -> None:
        """
        Initializes the class.

        Args:
            clip: Input clip.
            protect: Protection configuration for pixel intensity ranges.

                   - If `True`, automatically determines safe low/high protection thresholds
                     per plane using the clip's minimum and maximum allowed pixel values.
                   - If `False`, disables protection (no clipping protection is applied).
                   - If a tuple `(low, high)` is given, applies it as a protection range.
                   - If a list of tuples is provided, applies individual protection ranges per plane.
        """
        self.clip = clip

        ws, hs = zip(*((w, h) for _, w, h in get_resolutions(clip)))

        self._tofix_columns = {i: _BorderDict(w) for i, w in enumerate(ws)}
        self._tofix_rows = {i: _BorderDict(h) for i, h in enumerate(hs)}

        if protect is True:
            protect = [(low, hight) for low, hight in zip(get_lowest_values(clip), get_peak_values(clip))]
        elif protect is False:
            protect = [(False, False)]
        elif isinstance(protect, tuple):
            protect = [protect]

        self._protect = normalize_seq(protect, clip.format.num_planes)

    def __getitem__(
        self,
        key: SupportsIndex
        | tuple[SupportsIndex, NoneSlice]
        | tuple[NoneSlice, SupportsIndex]
        | tuple[SupportsIndex, NoneSlice, NoneSlice]
        | tuple[NoneSlice, SupportsIndex, NoneSlice]
        | tuple[SupportsIndex, NoneSlice, SupportsIndex]
        | tuple[NoneSlice, SupportsIndex, SupportsIndex],
        /,
    ) -> float:
        """
        Retrieve the correction value for a specific border pixel position.

        Args:
            key: Index or tuple specifying a position:

                   - (column, None[, plane]) to get a column border value
                   - (None, row[, plane]) to get a row border value

        Raises:
            CustomTypeError: If both column and row are specified or both are `None`.

        Returns:
            The correction multiplier for the specified position, or 0.0 if none is set.
        """
        if isinstance(key, SupportsIndex):
            return self.__getitem__((key, None, 0))

        if len(key) == 2:
            (column, row), plane = key, 0
        else:
            column, row, plane = key

        if (_is_slice_not_none(column) and _is_slice_not_none(row)) or (_is_slice_none(column) and _is_slice_none(row)):
            raise CustomTypeError(
                f"Invalid key combination: column={column}, row={row}. "
                "Exactly one of column or row must be a non-slice index.",
                self.__class__,
            )

        if _is_slice_none(plane):
            plane = 0

        plane = plane.__index__()

        if isinstance(column, SupportsIndex):
            return self._tofix_columns[plane].get(column.__index__(), 0.0)

        if isinstance(row, SupportsIndex):
            return self._tofix_rows[plane].get(row.__index__(), 0.0)

        raise CustomTypeError(
            f"Invalid key format: {key}. Expected a valid (column, row[, plane]) combination.", self.__class__
        )

    def __setitem__(
        self,
        key: IndexLike
        | tuple[IndexLike, NoneSlice]
        | tuple[NoneSlice, IndexLike]
        | tuple[IndexLike, NoneSlice, NoneSlice]
        | tuple[NoneSlice, IndexLike, NoneSlice]
        | tuple[IndexLike, NoneSlice, IndexLike]
        | tuple[NoneSlice, IndexLike, IndexLike],
        value: float,
        /,
    ) -> None:
        """
        Define a correction value for a specific row or column border.

        Args:
            key: Index or tuple specifying the border to correct:

                   - (column, None[, plane]) for column correction
                   - (None, row[, plane]) for row correction
            value: Correction multiplier to apply to the specified border region.

        Raises:
            CustomTypeError: If both row and column are specified or both are `None`.
        """
        if isinstance(key, (SupportsIndex, slice)):
            return self.__setitem__((key, None, 0), value)

        if len(key) == 2:
            (columns, rows), plane = key, 0
        else:
            columns, rows, plane = key

        if (_is_slice_not_none(columns) and _is_slice_not_none(rows)) or (
            _is_slice_none(columns) and _is_slice_none(rows)
        ):
            raise CustomTypeError(
                f"Invalid key combination: columns={columns}, rows={rows}. "
                "Exactly one of columns or rows must be a non-slice index.",
                self.__class__,
            )

        if _is_slice_none(plane):
            plane = 0

        for p_i in range(
            *_normalize_slice(plane, self.clip.format.num_planes, self.__class__).indices(self.clip.format.num_planes)
        ):
            if _is_slice_not_none(columns):
                length = self._tofix_columns[p_i].length

                for k in range(*_normalize_slice(columns, length, self.__class__).indices(length)):
                    self._tofix_columns[p_i][k] = value

            if _is_slice_not_none(rows):
                length = self._tofix_rows[p_i].length

                for k in range(*_normalize_slice(rows, length, self.__class__).indices(length)):
                    self._tofix_rows[p_i][k] = value

    def fix_column(self, num: int, value: float, plane_index: int = 0) -> None:
        """
        Apply a correction multiplier to an entire column in a specific plane.

        Args:
            num: Column index to correct.
            value: Correction value.
            plane_index: Plane index to apply the correction to (default is 0).
        """
        self[num, :, plane_index] = value

    def fix_row(self, num: int, value: float, plane_index: int = 0) -> None:
        """
        Apply a correction multiplier to an entire row in a specific plane.

        Args:
            num: Row index to correct.
            value: Correction value.
            plane_index: Plane index to apply the correction to (default is 0).
        """
        self[:, num, plane_index] = value

    def process(self, **kwargs: Any) -> vs.VideoNode:
        """
        Apply all configured border corrections to the clip.

        Args:
            **kwargs: Additional arguments forwarded to [vsexprtools.norm_expr][].

        Returns:
            A new clip with fixed borders applied.
        """
        exprs = list[ExprList]()

        for i, (columns, rows, protect) in enumerate(
            zip(self._tofix_columns.values(), self._tofix_rows.values(), self._protect)
        ):
            expr = ExprList()

            if not rows and not columns:
                exprs.append(expr)
                continue

            norm = ExprToken.PlaneMin if i == 0 or self.clip.format.color_family == vs.RGB else ExprToken.Neutral

            expr.append(f"x {norm} - CLIP!")

            if columns:
                for num, value in columns.items():
                    expr.append("X", num, "=", "CLIP@", value, "*")
                expr.append("CLIP@", ExprOp.TERN * len(columns), "CLIP!")

            if rows:
                for num, value in rows.items():
                    expr.append("Y", num, "=", "CLIP@", value, "*")
                expr.append("CLIP@", ExprOp.TERN * len(rows), "CLIP!")

            expr.append("CLIP@", norm, "+")

            if any(protect):
                expr = ExprList(["x", "{protect_lo}", ">", "x", "{protect_hi}", "<", "and", expr, "x", "?"])

            exprs.append(expr)

        protect_lo, protect_hi = zip(*self._protect)

        return norm_expr(
            self.clip, tuple(exprs), func=self.__class__, **kwargs, protect_lo=protect_lo, protect_hi=protect_hi
        )
