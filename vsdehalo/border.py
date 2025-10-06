from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex, TypeAlias, TypeIs

from jetpytools import CustomTypeError, normalize_seq

from vsexprtools import ExprList, ExprOp, ExprToken, norm_expr
from vstools import check_variable_format, get_lowest_values, get_peak_values, get_resolutions, vs, vs_object

IndexLike: TypeAlias = SupportsIndex | slice
if TYPE_CHECKING:
    NoneSlice: TypeAlias = slice[None, None, None] | None
else:
    NoneSlice: TypeAlias = slice | None


def _normalize_slice(index: IndexLike, length: int) -> slice:
    if index == slice(None, None, None):
        index = 0

    if isinstance(index, SupportsIndex):
        return (
            slice(length + i, length + i + 1)
            if (i := index.__index__()) < 0
            else slice(index.__index__(), index.__index__() + 1)
        )

    return index


def _is_slice_not_none(index: SupportsIndex | slice | None) -> TypeIs[SupportsIndex | slice]:
    return index is not None and index != slice(None, None, None)


def _is_slice_none(index: SupportsIndex | slice | None) -> TypeIs[slice | None]:
    return index is None or index == slice(None, None, None)


class _Border(NamedTuple):
    num: int
    value: float


class _BorderSet(set[_Border]):
    width: int
    height: int

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        super().__init__()


class FixClipBorder(vs_object):
    def __init__(
        self, clip: vs.VideoNode, protect: bool | tuple[float, float] | list[tuple[float, float]] = True
    ) -> None:
        assert check_variable_format(clip, self.__class__)
        self.clip = clip

        res = {k: (w, h) for k, w, h in get_resolutions(clip)}

        self._tofix_columns = {i: _BorderSet(w, h) for i, (w, h) in res.items()}
        self._tofix_rows = {i: _BorderSet(w, h) for i, (w, h) in res.items()}

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
        if isinstance(key, SupportsIndex):
            return self.__getitem__((key, None, 0))

        if len(key) == 2:
            (column, row), plane = key, 0
        else:
            column, row, plane = key

        if (_is_slice_not_none(column) and _is_slice_not_none(row)) or (_is_slice_none(column) and _is_slice_none(row)):
            raise CustomTypeError

        if _is_slice_none(plane):
            plane = 0

        plane = plane.__index__()

        if isinstance(column, SupportsIndex):
            if (column := column.__index__()) < 0:
                length = self._tofix_columns[plane].width
                column += length

            return next((b.value for b in self._tofix_columns[plane] if b.num == column), 0.0)

        if isinstance(row, SupportsIndex):
            if (row := row.__index__()) < 0:
                length = self._tofix_rows[plane].height
                row += length

            return next((b.value for b in self._tofix_rows[plane] if b.num == row), 0.0)

        raise CustomTypeError

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
        if isinstance(key, IndexLike):
            return self.__setitem__((_normalize_slice(key, self._tofix_columns[0].width), None, 0), value)

        if len(key) == 2:
            (columns, rows), plane = key, 0
        else:
            columns, rows, plane = key

        if (_is_slice_not_none(columns) and _is_slice_not_none(rows)) or (
            _is_slice_none(columns) and _is_slice_none(rows)
        ):
            raise CustomTypeError

        if _is_slice_none(plane):
            plane = 0

        plane = _normalize_slice(plane, self.clip.format.num_planes)

        for p_i in range(*plane.indices(self.clip.format.num_planes)):
            if _is_slice_not_none(columns):
                length = self._tofix_columns[p_i].width

                for k in range(*_normalize_slice(columns, length).indices(length)):
                    self._tofix_columns[p_i].add(_Border(k, value))

            if _is_slice_not_none(rows):
                length = self._tofix_rows[p_i].height

                for k in range(*_normalize_slice(rows, length).indices(length)):
                    self._tofix_rows[p_i].add(_Border(k, value))

    def fix_column(self, num: int, value: float, plane_index: int = 0) -> None:
        self[num, :, plane_index] = value

    def fix_row(self, num: int, value: float, plane_index: int = 0) -> None:
        self[:, num, plane_index] = value

    def process(self, **kwargs: Any) -> vs.VideoNode:
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
                for column in columns:
                    expr.append("X", column.num, "=", "CLIP@", column.value, "*")
                expr.append("CLIP@", ExprOp.TERN * len(columns), "CLIP!")

            if rows:
                for row in rows:
                    expr.append("Y", row.num, "=", "CLIP@", row.value, "*")
                expr.append("CLIP@", ExprOp.TERN * len(rows), "CLIP!")

            expr.append("CLIP@", norm, "+")

            if any(protect):
                expr = ExprList(["x", "{protect_lo}", ">", "x", "{protect_hi}", "<", "and", expr, "x", "?"])

            exprs.append(expr)

        protect_lo, protect_hi = zip(*self._protect)

        return norm_expr(
            self.clip, tuple(exprs), func=self.__class__, **kwargs, protect_lo=protect_lo, protect_hi=protect_hi
        )

    def __vs_del__(self, core_id: int) -> None:
        del self.clip
