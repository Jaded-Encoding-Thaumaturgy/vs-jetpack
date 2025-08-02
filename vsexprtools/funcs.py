from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeAlias, Union

from vstools import (
    ConstantFormatVideoNode,
    CustomRuntimeError,
    FuncExceptT,
    HoldsVideoFormatT,
    PlanesT,
    ProcessVariableResClip,
    StrList,
    SupportsString,
    VideoFormatT,
    VideoNodeIterableT,
    check_variable_format,
    core,
    flatten_vnodes,
    get_video_format,
    to_arr,
    vs,
)

from .exprop import ExprOp, ExprOpBase, TupleExprList
from .util import ExprVars, bitdepth_aware_tokenize_expr, norm_expr_planes

__all__ = ["combine", "expr_func", "norm_expr"]


def expr_func(
    clips: vs.VideoNode | Sequence[vs.VideoNode],
    expr: str | Sequence[str],
    format: HoldsVideoFormatT | VideoFormatT | None = None,
    opt: bool = False,
    boundary: bool = True,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Calls `akarin.Expr` plugin.

    For a higher-level function, see [norm_expr][vsexprtools.norm_expr]

    Args:
        clips: Input clip(s). Supports constant format clips, or one variable resolution clip.
        expr: Expression to be evaluated.
        format: Output format, defaults to the first clip format.
        opt: Forces integer evaluation as much as possible.
        boundary: Specifies the default boundary condition for relative pixel accesses:

               - True (default): Mirrored edges.
               - False: Clamped edges.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Raises:
        CustomRuntimeError: If `akarin` plugin is not found.
        vapoursynth.Error: If the expression could not be evaluated.

    Returns:
        Evaluated clip.
    """
    func = func or expr_func

    clips = to_arr(clips)

    if TYPE_CHECKING:
        assert check_variable_format(clips, func)

    fmt = get_video_format(format).id if format is not None else None

    try:
        return core.akarin.Expr(clips, expr, fmt, opt, boundary)
    except AttributeError as e:
        raise CustomRuntimeError(e)
    except vs.Error as e:
        if len(clips) == 1 and 0 in (clips[0].width, clips[0].height):
            return ProcessVariableResClip[ConstantFormatVideoNode].from_func(
                clips[0], lambda clip: core.akarin.Expr(clip, expr, fmt, opt, boundary)
            )

        raise CustomRuntimeError(e, func, expr) from e


def _combine_norm__ix(ffix: SupportsString | Iterable[SupportsString] | None, n_clips: int) -> list[SupportsString]:
    if ffix is None:
        return [""] * n_clips

    ffix = to_arr(ffix)

    return ffix * max(1, ceil(n_clips / len(ffix)))


def combine(
    clips: VideoNodeIterableT[vs.VideoNode],
    operator: ExprOpBase = ExprOp.MAX,
    suffix: SupportsString | Iterable[SupportsString] | None = None,
    prefix: SupportsString | Iterable[SupportsString] | None = None,
    expr_suffix: SupportsString | Iterable[SupportsString] | None = None,
    expr_prefix: SupportsString | Iterable[SupportsString] | None = None,
    planes: PlanesT = None,
    split_planes: bool = False,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    """
    Combines multiple video clips using a specified expression operator.

    Args:
        clips: Input clip(s).
        operator: An ExprOpBase enum used to join the clips.
        suffix: Optional suffix string(s) to append to each input variable in the expression.
        prefix: Optional prefix string(s) to prepend to each input variable in the expression.
        expr_suffix: Optional expression to append after the combined input expression.
        expr_prefix: Optional expression to prepend before the combined input expression.
        planes: Which planes to process. Defaults to all.
        split_planes: If True, treats each plane of input clips as separate inputs.
        **kwargs: Additional keyword arguments forwarded to [norm_expr][vsexprtools.norm_expr].

    Returns:
        A clip representing the combined result of applying the expression.
    """
    clips = flatten_vnodes(clips, split_planes=split_planes)

    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    args = zip(prefixes, ExprVars(n_clips), suffixes)

    has_op = (n_clips >= operator.n_op) or any(x is not None for x in (suffix, prefix, expr_suffix, expr_prefix))

    operators = operator * max(n_clips - 1, int(has_op))

    return norm_expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **kwargs)


ExprLike: TypeAlias = Union[SupportsString | None, Iterable["ExprLike"]]
"""
A recursive type representing a valid expression input.

Acceptable forms include:
- A single string (or string-like object): Used as the same expression for all planes.
- A list of expressions: Concatenated into a single expression for all planes.
- A tuple of expressions: Interpreted as separate expressions for each plane.
- A TupleExprList: will make a [norm_expr][vsexprtools.norm_expr] call for each expression within this tuple.
"""


def norm_expr(
    clips: VideoNodeIterableT[vs.VideoNode],
    expr: ExprLike,
    planes: PlanesT = None,
    format: HoldsVideoFormatT | VideoFormatT | None = None,
    opt: bool = False,
    boundary: bool = True,
    func: FuncExceptT | None = None,
    split_planes: bool = False,
    debug: bool = False,
    **kwargs: Iterable[SupportsString] | SupportsString,
) -> ConstantFormatVideoNode:
    """
    Evaluate a per-pixel expression on input clip(s), normalize it based on the specified planes,
    and format tokens and placeholders using provided keyword arguments.

    Args:
        clips: Input clip(s). Supports constant format clips, or one variable resolution clip.
        expr: Expression to be evaluated.

               - A single str will be processed for all planes.
               - A list will be concatenated to form a single expr for all planes.
               - A tuple of these types will allow specification of different expr for each planes.
               - A TupleExprList will make a `norm_expr` call for each expression within this tuple.
        planes: Plane to process, defaults to all.
        format: Output format, defaults to the first clip format.
        opt: Forces integer evaluation as much as possible.
        boundary: Specifies the default boundary condition for relative pixel accesses:

               - True (default): Mirrored edges.
               - False: Clamped edges.
        func: Function returned for custom error handling. This should only be set by VS package developers.
        split_planes: Splits the VideoNodes into their individual planes.
        debug: Print out the normalized expr.
        **kwargs: Additional keywords arguments to be passed to the expression function. These arguments are key-value
            pairs, where the keys are placeholders that will be replaced in the expression string. Iterable values
            (except str and bytes types) will be associated with the corresponding plane.

    Returns:
        Evaluated clip.
    """
    if isinstance(expr, str):
        nexpr = ([expr],)
    elif isinstance(expr, tuple):
        if isinstance(expr, TupleExprList):
            return expr(
                clips,
                planes=planes,
                format=format,
                opt=opt,
                boundary=boundary,
                func=func,
                split_planes=split_planes,
                **kwargs,
            )
        else:
            nexpr = tuple([to_arr(x) for x in expr])
    else:
        nexpr = (to_arr(expr),)

    clips = flatten_vnodes(clips, split_planes=split_planes)

    normalized_exprs = [StrList(plane_expr).to_str() for plane_expr in nexpr]

    normalized_expr = norm_expr_planes(clips[0], normalized_exprs, planes, **kwargs)

    tokenized_expr = [
        bitdepth_aware_tokenize_expr(clips, e, bool(is_chroma)) for is_chroma, e in enumerate(normalized_expr)
    ]

    if debug:
        print(tokenized_expr)

    return expr_func(clips, tokenized_expr, format, opt, boundary, func)
