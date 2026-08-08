from collections.abc import Callable
from functools import partial

from ._rg import (
    remove_grain_expr_1,
    remove_grain_expr_2_4,
    remove_grain_expr_5,
    remove_grain_expr_6,
    remove_grain_expr_7,
    remove_grain_expr_8,
    remove_grain_expr_9,
    remove_grain_expr_10,
    remove_grain_expr_11_12,
    remove_grain_expr_17,
    remove_grain_expr_18,
    remove_grain_expr_19,
    remove_grain_expr_20,
    remove_grain_expr_21_22,
    remove_grain_expr_23,
    remove_grain_expr_24,
    remove_grain_expr_25,
    remove_grain_expr_26,
    remove_grain_expr_27,
    remove_grain_expr_28,
)
from ._rp import (
    repair_expr_1_4,
    repair_expr_5,
    repair_expr_6,
    repair_expr_7,
    repair_expr_8,
    repair_expr_9,
    repair_expr_10,
    repair_expr_11_14,
    repair_expr_15,
    repair_expr_16,
    repair_expr_17,
    repair_expr_18,
    repair_expr_19,
    repair_expr_20,
    repair_expr_21,
    repair_expr_22,
    repair_expr_23,
    repair_expr_24,
    repair_expr_26,
    repair_expr_27,
    repair_expr_28,
)


def _noop_expr() -> str:
    return ""


def _unimpl_expr() -> str:
    raise NotImplementedError("Unimplemented RGTools expr mode!", _unimpl_expr)


removegrain_exprs = list[Callable[[], str]](
    [
        _noop_expr,
        remove_grain_expr_1,
        partial(remove_grain_expr_2_4, 2),
        partial(remove_grain_expr_2_4, 3),
        partial(remove_grain_expr_2_4, 4),
        remove_grain_expr_5,
        remove_grain_expr_6,
        remove_grain_expr_7,
        remove_grain_expr_8,
        remove_grain_expr_9,
        remove_grain_expr_10,
        remove_grain_expr_11_12,
        remove_grain_expr_11_12,
        _unimpl_expr,
        _unimpl_expr,
        _unimpl_expr,
        _unimpl_expr,
        remove_grain_expr_17,
        remove_grain_expr_18,
        remove_grain_expr_19,
        remove_grain_expr_20,
        remove_grain_expr_21_22,
        remove_grain_expr_21_22,
        remove_grain_expr_23,
        remove_grain_expr_24,
        remove_grain_expr_25,
        remove_grain_expr_26,
        remove_grain_expr_27,
        remove_grain_expr_28,
    ]
)

repair_exprs = list[Callable[[], str]](
    [
        _noop_expr,
        partial(repair_expr_1_4, 1),
        partial(repair_expr_1_4, 2),
        partial(repair_expr_1_4, 3),
        partial(repair_expr_1_4, 4),
        repair_expr_5,
        repair_expr_6,
        repair_expr_7,
        repair_expr_8,
        repair_expr_9,
        repair_expr_10,
        partial(repair_expr_11_14, 1),
        partial(repair_expr_11_14, 2),
        partial(repair_expr_11_14, 3),
        partial(repair_expr_11_14, 4),
        repair_expr_15,
        repair_expr_16,
        repair_expr_17,
        repair_expr_18,
        repair_expr_19,
        repair_expr_20,
        repair_expr_21,
        repair_expr_22,
        repair_expr_23,
        repair_expr_24,
        _unimpl_expr,
        repair_expr_26,
        repair_expr_27,
        repair_expr_28,
    ]
)
