from __future__ import annotations

import sys
from types import UnionType
from typing import (  # type: ignore[attr-defined]
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    TypeVar,
    _UnionGenericAlias,  # pyright: ignore[reportAttributeAccessIssue]
    get_args,
    get_origin,
    overload,
)
from typing import cast as typing_cast

import vapoursynth as vs
from jetpytools import (
    MISSING,
    FileWasNotFoundError,
    FuncExceptT,
    MissingT,
    SPath,
    SPathLike,
    SupportsString,
    T,
    normalize_seq,
    to_arr,
)

from ..enums import PropEnum
from ..exceptions import FramePropError
from ..types import BoundVSMapValue, ConstantFormatVideoNode, HoldsPropValueT
from ..types.generic import BoundVSMapValue_0, BoundVSMapValue_1
from .cache import NodesPropsCache

__all__ = ["get_clip_filepath", "get_prop", "get_props", "merge_clip_props"]

DT = TypeVar("DT")
CT = TypeVar("CT")


_get_prop_cache = NodesPropsCache[vs.RawNode]()


def _normalize_types(types: type[T] | Iterable[type[T]]) -> tuple[type[T], ...]:
    norm_t = list[type[T]]()

    for tt in to_arr(types):
        t_origin = get_origin(tt)

        if t_origin is not None:
            if isinstance(tt, (_UnionGenericAlias, UnionType)):
                norm_t.extend(_normalize_types(get_args(tt)))
            else:
                norm_t.append(t_origin)
        else:
            norm_t.append(tt)

    return tuple(norm_t)


@overload
def get_prop(
    obj: HoldsPropValueT, key: SupportsString | PropEnum, t: type[BoundVSMapValue], *, func: FuncExceptT | None = None
) -> BoundVSMapValue: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: tuple[type[BoundVSMapValue], type[BoundVSMapValue_0]],
    *,
    func: FuncExceptT | None = None,
) -> BoundVSMapValue | BoundVSMapValue_0: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: tuple[type[BoundVSMapValue], type[BoundVSMapValue_0], type[BoundVSMapValue_1]],
    *,
    func: FuncExceptT | None = None,
) -> BoundVSMapValue | BoundVSMapValue_0 | BoundVSMapValue_1: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: tuple[type[BoundVSMapValue], ...],
    *,
    func: FuncExceptT | None = None,
) -> Any: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: type[BoundVSMapValue] | tuple[type[BoundVSMapValue], ...],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    *,
    func: FuncExceptT | None = None,
) -> CT: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: type[BoundVSMapValue] | tuple[type[BoundVSMapValue], ...],
    *,
    default: DT | MissingT = ...,
    func: FuncExceptT | None = None,
) -> BoundVSMapValue | DT: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: type[BoundVSMapValue] | tuple[type[BoundVSMapValue], ...],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    default: DT | MissingT = ...,
    func: FuncExceptT | None = None,
) -> CT | DT: ...


@overload
def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: type[BoundVSMapValue],
    cast: type[CT] | Callable[[BoundVSMapValue], CT] | None,
    default: DT | MissingT,
    func: FuncExceptT | None = None,
) -> BoundVSMapValue | CT | DT: ...


def get_prop(
    obj: HoldsPropValueT,
    key: SupportsString | PropEnum,
    t: type[BoundVSMapValue] | tuple[type[BoundVSMapValue], ...],
    cast: type[CT] | Callable[[BoundVSMapValue], CT] | None = None,
    default: DT | MissingT = MISSING,
    func: FuncExceptT | None = None,
) -> BoundVSMapValue | CT | DT:
    """
    Get FrameProp ``prop`` from frame ``frame`` with expected type ``t`` to satisfy the type checker.

    Args:
        obj: Clip or frame containing props.
        key: Prop to get.
        t: type of prop.
        cast: Cast value to this type, if specified.
        default: Fallback value.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        frame.prop[key].

    Raises:
        FramePropError: ``key`` is not found in props.
        FramePropError: ``key`` is of the wrong type.
    """
    func = func or get_prop

    if isinstance(obj, vs.RawNode):
        props = _get_prop_cache.get((obj, 0))

        if props is None:
            with obj.get_frame(0) as f:
                props = f.props.copy()

            _get_prop_cache[(obj, 0)] = props

    elif isinstance(obj, vs.RawFrame):
        props = obj.props
    else:
        props = obj

    resolved_key = key.prop_key if isinstance(key, type) and issubclass(key, PropEnum) else str(key)

    try:
        prop = props[resolved_key]
    except KeyError as e:
        if default is not MISSING:
            return default

        raise FramePropError(func, resolved_key, f'Key "{resolved_key}" not present in props!') from e

    norm_t = _normalize_types(t)

    if not isinstance(prop, norm_t):
        if all(issubclass(ty, str) for ty in norm_t) and isinstance(prop, bytes):
            return prop.decode("utf-8")  # type: ignore[return-value]

        if default is not MISSING:
            return default

        raise FramePropError(
            func,
            resolved_key,
            'Key "{key}" did not contain expected type: Expected "{t}" got "{prop_t}"!',
            t=t,
            prop_t=type(prop),
        )

    try:
        return cast(prop) if cast else prop  # type: ignore
    except Exception as e:
        if default is not MISSING:
            return default

        raise FramePropError(func, resolved_key) from e


@overload
def merge_clip_props(*clips: ConstantFormatVideoNode, main_idx: int = 0) -> ConstantFormatVideoNode: ...


@overload
def merge_clip_props(*clips: vs.VideoNode, main_idx: int = 0) -> vs.VideoNode: ...


def merge_clip_props(*clips: vs.VideoNode, main_idx: int = 0) -> vs.VideoNode:
    """
    Merge frame properties from all provided clips.

    The props of the main clip (defined by main_idx) will be overwritten,
    and all other props will be added to it.

    Args:
        *clips: Clips which will be merged.
        main_idx: Index of the main clip to which all other clips props will be merged.

    Returns:
        First clip with all the frameprops of every other given clip merged into it.
    """

    if len(clips) == 1:
        return clips[0]

    def _merge_props(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
        fdst = f[main_idx].copy()

        for i, frame in enumerate(f):
            if i == main_idx:
                continue

            fdst.props.update(frame.props)

        return fdst

    return clips[0].std.ModifyFrame(clips, _merge_props)


@overload
def get_clip_filepath(
    clip: vs.VideoNode,
    fallback: SPathLike | None = ...,
    strict: Literal[False] = ...,
    *,
    func: FuncExceptT | None = ...,
) -> SPath | None: ...


@overload
def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = ..., strict: Literal[True] = ..., *, func: FuncExceptT | None = ...
) -> SPath: ...


@overload
def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = ..., strict: bool = ..., *, func: FuncExceptT | None = ...
) -> SPath | None: ...


def get_clip_filepath(
    clip: vs.VideoNode, fallback: SPathLike | None = None, strict: bool = False, *, func: FuncExceptT | None = None
) -> SPath | None:
    """
    Helper function to get the file path from a clip.

    This functions checks for the `IdxFilePath` frame property.
    It also checks to ensure the file exists, and throws an error if it doesn't.

    Args:
        clip: The clip to get the file path from.
        fallback: Fallback file path to use if the `prop` is not found.
        strict: If True, will raise an error if the `prop` is not found. This makes it so the function will NEVER return
            False. Default: False.
        func: Function returned for error handling. This should only be set by VS package developers.

    Raises:
        FileWasNotFoundError: The file path was not found.
        FramePropError: The property was not found in the clip.
    """

    func = func or get_clip_filepath

    if fallback is not None and not (fallback_path := SPath(fallback)).exists() and strict:
        raise FileWasNotFoundError("Fallback file not found!", func, fallback_path.absolute())

    if not (path := get_prop(clip, "IdxFilePath", str, default=MISSING if strict else False, func=func)):
        return fallback_path if fallback is not None else None

    if not (spath := SPath(str(path))).exists() and not fallback:
        raise FileWasNotFoundError("File not found!", func, spath.absolute())

    if spath.exists():
        return spath

    if fallback is not None and fallback_path.exists():
        return fallback_path

    raise FileWasNotFoundError("File not found!", func, spath.absolute())


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    *,
    func: FuncExceptT | None = None,
) -> dict[str, BoundVSMapValue]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    *,
    func: FuncExceptT | None = None,
) -> dict[str, CT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    *,
    default: DT,
    func: FuncExceptT | None = None,
) -> dict[str, BoundVSMapValue | DT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue],
    cast: type[CT] | Callable[[BoundVSMapValue], CT],
    default: DT,
    func: FuncExceptT | None = None,
) -> dict[str, CT | DT]: ...


@overload
def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: Sequence[type[BoundVSMapValue]],
    cast: Sequence[type[CT] | Callable[[BoundVSMapValue], CT]] | None = None,
    default: DT | Sequence[DT] | MissingT = ...,
    func: FuncExceptT | None = None,
) -> dict[str, Any]: ...


def get_props(
    obj: HoldsPropValueT,
    keys: Sequence[SupportsString | PropEnum],
    t: type[BoundVSMapValue] | Sequence[type[BoundVSMapValue]],
    cast: type[CT]
    | Callable[[BoundVSMapValue], CT]
    | Sequence[type[CT] | Callable[[BoundVSMapValue], CT]]
    | None = None,
    default: DT | Sequence[DT] | MissingT = MISSING,
    func: FuncExceptT | None = None,
) -> dict[str, Any]:
    """
    Get multiple frame properties from a clip.

    Args:
        obj: Clip or frame containing props.
        keys: List of props to get.
        t: Type of prop or list of types of props. If fewer types are provided than props, the last type will be used
            for the remaining props.
        cast: Cast value to this type, if specified.
        default: Fallback value. Can be a single value or a list of values. If a list is provided, it must be the same
            length as keys.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Dictionary mapping property names to their values. Values will be of type specified by cast if provided,
        otherwise of the type(s) specified in ``t`` or a default value if provided.
    """

    func = func or "get_props"

    if not keys:
        return {}

    t = normalize_seq(t, len(keys))
    ncast = typing_cast(list[type[CT | Callable[[BoundVSMapValue], CT]]], normalize_seq(cast, len(keys)))
    ndefault = normalize_seq(default, len(keys))

    props = dict[str, Any]()
    exceptions = list[Exception]()

    for k, t_, cast_, default_ in zip(keys, t, ncast, ndefault):
        try:
            prop = get_prop(obj, k, t_, cast_, default_, func)
        except Exception as e:
            exceptions.append(e)
        else:
            props[str(k)] = prop

    if exceptions:
        if sys.version_info >= (3, 11):
            raise ExceptionGroup("Multiple exceptions occurred!", exceptions)  # noqa: F821

        raise Exception(exceptions)

    return props
