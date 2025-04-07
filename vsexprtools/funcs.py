from __future__ import annotations

from functools import partial
from math import ceil
from typing import Any, Iterable, Sequence, cast

from vstools import (
    ConstantFormatVideoNode, CustomRuntimeError, CustomValueError, FuncExceptT, HoldsVideoFormatT, PlanesT, ProcessVariableResClip, StrArr,
    StrArrOpt, StrList, SupportsString, VideoFormatT, VideoNodeIterableT, VideoNodeT, check_variable_format, core, flatten_vnodes,
    get_video_format, to_arr, vs
)

from .exprop import ExprOp, ExprOpBase, ExprList, TupleExprList
from .util import ExprVars, bitdepth_aware_tokenize_expr, complexpr_available, norm_expr_planes

__all__ = [
    'expr_func', 'combine', 'norm_expr'
]


def expr_func(
    clips: VideoNodeT | Sequence[VideoNodeT], expr: str | Sequence[str],
    format: HoldsVideoFormatT | VideoFormatT | None = None, opt: bool | None = None, boundary: bool = True,
    func: FuncExceptT | None = None
) -> VideoNodeT:
    func = func or expr_func

    clips = list(clips) if isinstance(clips, Sequence) else [clips]
    over_clips = len(clips) > 26

    if not complexpr_available:
        if over_clips:
            raise ExprVars._get_akarin_err('This function only works with akarin plugin!')(func=func)
    elif over_clips and b'src26' not in vs.core.akarin.Version()['expr_features']:
        raise ExprVars._get_akarin_err('You need at least v0.96 of akarin plugin!')(func=func)

    fmt = None if format is None else get_video_format(format).id

    got_var_res = False

    for clip in clips:
        check_variable_format(clip, func)
        got_var_res = got_var_res or (0 in (clip.width, clip.height))

    if complexpr_available and opt is None:
        opt = all(clip.format.sample_type == vs.INTEGER for clip in clips if clip.format)

    if complexpr_available:
        func_impl = partial(core.akarin.Expr, expr=expr, format=fmt, opt=opt, boundary=boundary)
    else:
        func_impl = partial(core.std.Expr, expr=expr, format=fmt)

    if got_var_res:
        if len(clips) == 1:
            return ProcessVariableResClip[VideoNodeT].from_func(clips[0], func_impl, None, clips[0].format)  # type: ignore

        raise CustomValueError('You can run only one var res clip!')

    try:
        return cast(VideoNodeT, func_impl(clips))
    except Exception:
        raise CustomRuntimeError(
            'There was an error when evaluating the expression:\n' + (
                '' if complexpr_available else 'You might need akarin-plugin, and are missing it.'
            ), func, f'\n{expr}\n'
        )


def _combine_norm__ix(ffix: StrArrOpt, n_clips: int) -> list[SupportsString]:
    if ffix is None:
        return [''] * n_clips

    ffix = [ffix] if isinstance(ffix, (str, tuple)) else list(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))  # type: ignore


def combine(
    clips: VideoNodeIterableT[vs.VideoNode], operator: ExprOpBase = ExprOp.MAX, suffix: StrArrOpt = None,
    prefix: StrArrOpt = None, expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None,
    planes: PlanesT = None, split_planes: bool = False, **kwargs: Any
) -> ConstantFormatVideoNode:
    clips = flatten_vnodes(clips, split_planes=split_planes)

    assert check_variable_format(clips, combine)

    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    args = zip(prefixes, ExprVars(n_clips), suffixes)

    has_op = (n_clips >= operator.n_op) or any(x is not None for x in (suffix, prefix, expr_suffix, expr_prefix))

    operators = operator * max(n_clips - 1, int(has_op))

    return norm_expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **kwargs)


def norm_expr(
    clips: VideoNodeIterableT[VideoNodeT],
    expr: str | StrArr | ExprList | tuple[str | StrArr | ExprList, ...] | TupleExprList,
    planes: PlanesT = None, format: HoldsVideoFormatT | VideoFormatT | None = None,
    opt: bool | None = None, boundary: bool = True,
    func: FuncExceptT | None = None,
    split_planes: bool = False,
    **kwargs: Iterable[SupportsString] | SupportsString
) -> VideoNodeT:
    """
    Evaluates an expression per pixel.

    :param clips:           Input clip(s).
    :param expr:            Expression to be evaluated.
                            A single str will be processed for all planes.
                            A list will be concatenated to form a single expr for all planes.
                            A tuple of these types will allow specification of different expr for each planes.
                            A TupleExprList will make a norm_expr call for each expression within this tuple.
    :param planes:          Plane to process, defaults to all.
    :param format:          Output format, defaults to the first clip format.
    :param opt:             Forces integer evaluation as much as possible.
    :param boundary:        Specifies the default boundary condition for relative pixel accesses:
                            - 0 means clamped
                            - 1 means mirrored
    :param split_planes:    Splits the VideoNodes into their individual planes.
    :param **kwargs:        Additional keywords arguments to be passed to the expression function.
                            These arguments are key-value pairs, where the keys are placeholders
                            that will be replaced in the expression string.
                            Iterable values (except str and bytes types) will be associated with the corresponding plane.
    :return:                Evaluated clip.
    """
    clips = flatten_vnodes(clips, split_planes=split_planes)

    if isinstance(expr, str):
        nexpr = tuple([[expr]])
    elif isinstance(expr, tuple):
        if isinstance(expr, TupleExprList):
            if len(expr) < 1:
                raise CustomRuntimeError(
                    "When passing a TupleExprList you need at least one expr in it!", func, expr
                )

            nclips: Sequence[VideoNodeT] | VideoNodeT = clips

            for e in expr:
                nclips = norm_expr(
                    nclips, e, planes, format, opt, boundary, func, split_planes, **kwargs
                )

            return cast(VideoNodeT, nclips)
        else:
            nexpr = tuple([to_arr(x) for x in expr])
    else:
        nexpr = tuple([to_arr(expr)])

    normalized_exprs = [StrList(plane_expr).to_str() for plane_expr in nexpr]

    normalized_expr = norm_expr_planes(clips[0], normalized_exprs, planes, **kwargs)

    tokenized_expr = [
        bitdepth_aware_tokenize_expr(clips, e, bool(is_chroma))
        for is_chroma, e in enumerate(normalized_expr)
    ]

    return expr_func(clips, tokenized_expr, format, opt, boundary, func)
