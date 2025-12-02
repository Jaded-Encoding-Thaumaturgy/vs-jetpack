from __future__ import annotations

from collections.abc import Callable, Iterable
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, overload

from jetpytools import MISSING, FileWasNotFoundError, FuncExcept, MissingT, SPath, SPathLike, to_arr

from ..enums import PropEnum
from ..exceptions import FramePropError
from ..types import HoldsPropValue
from ..vs_proxy import vs
from .cache import NodesPropsCache

__all__ = ["get_clip_filepath", "get_prop", "get_props", "merge_clip_props"]


type _PropValue = (
    int
    | float
    | str
    | bytes
    | vs.RawFrame
    | vs.VideoFrame
    | vs.AudioFrame
    | vs.RawNode
    | vs.VideoNode
    | vs.AudioNode
    | Callable[..., Any]
    | list[int]
    | list[float]
    | list[str]
    | list[bytes]
    | list[vs.RawFrame]
    | list[vs.VideoFrame]
    | list[vs.AudioFrame]
    | list[vs.RawNode]
    | list[vs.VideoNode]
    | list[vs.AudioNode]
    | list[Callable[..., Any]]
)


_get_prop_cache = NodesPropsCache[vs.RawNode]()


def _normalize_types[T](types: type[T] | Iterable[type[T]]) -> tuple[type[T], ...]:
    norm_t = list[type[T]]()

    for tt in to_arr(types):
        t_origin = get_origin(tt)

        if t_origin is not None:
            if isinstance(tt, UnionType) or t_origin is Union:
                norm_t.extend(_normalize_types(get_args(tt)))
            else:
                norm_t.append(t_origin)
        else:
            norm_t.append(tt)

    return tuple(norm_t)


# One type signature
@overload
def get_prop[PropValueT: _PropValue, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: type[PropValueT],
    *,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> PropValueT | DT: ...


# Tuple of two types signature
@overload
def get_prop[PropValueT: _PropValue, PropValueT0: _PropValue, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: tuple[type[PropValueT], type[PropValueT0]],
    *,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> PropValueT | PropValueT0 | DT: ...


# Tuple of three types signature
@overload
def get_prop[PropValueT: _PropValue, PropValueT0: _PropValue, PropValueT1: _PropValue, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: tuple[type[PropValueT], type[PropValueT0], type[PropValueT1]],
    *,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> PropValueT | PropValueT0 | PropValueT1 | DT: ...


# Tuple of four types or more signature
@overload
def get_prop[PropValueT: _PropValue, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: tuple[type[PropValueT], ...],
    *,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> Any | DT: ...


# Signature when cast is specified
@overload
def get_prop[PropValueT: _PropValue, CT, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: type[PropValueT] | tuple[type[_PropValue], ...],
    cast: Callable[[PropValueT], CT],
    default: DT = ...,
    func: FuncExcept | None = None,
) -> CT | DT: ...


# Signature for callable
@overload
def get_prop[DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: Literal["Callable"],
    *,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> Callable[..., Any] | DT: ...


# Generic signature
@overload
def get_prop[CT, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: type[Any] | tuple[type[Any], ...] | Literal["Callable"],
    cast: Callable[[Any], CT] = ...,
    default: DT = ...,
    func: FuncExcept | None = None,
) -> Any | CT | DT: ...


def get_prop[CT, DT](
    obj: HoldsPropValue,
    key: str | type[PropEnum],
    t: type[Any] | tuple[type[Any], ...] | Literal["Callable"],
    cast: Callable[[Any], CT] | MissingT = MISSING,
    default: DT | MissingT = MISSING,
    func: FuncExcept | None = None,
) -> Any | CT | DT:
    """
    Get FrameProp ``prop`` from frame ``frame`` with expected type ``t``.

    If the property is stored as bytes and `t` is ``str``, the value will be decoded as UTF-8.

    Example:
        ```py
        assert get_prop(clip.get_frame(0), "_PictType", str) == "I"
        ```

    Args:
        obj: Clip or frame containing props.
        key: Prop to get.
        t: Expected type of the prop (or tuple of types). Use "Callable" if expecting a callable.
        cast: Optional cast to apply to the value.
        default: Fallback value if missing.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        The property value, possibly cast, or the provided default.

    Raises:
        FramePropError: If the property is missing or has the wrong type and no default value is given.
    """
    func = func or get_prop

    if isinstance(obj, vs.RawNode):
        props = _get_prop_cache.get((obj, 0), MISSING)

        if props is MISSING:
            with obj.get_frame(0) as f:
                props = f.props.copy()

            _get_prop_cache[obj, 0] = props

    elif isinstance(obj, vs.RawFrame):
        props = obj.props
    else:
        props = obj

    resolved_key = key.prop_key if isinstance(key, type) and issubclass(key, PropEnum) else str(key)

    prop = props.get(resolved_key, MISSING)

    if prop is MISSING:
        if default is not MISSING:
            return default

        raise FramePropError(func, resolved_key, f'Key "{resolved_key}" not present in props!')

    if t == "Callable":
        if callable(prop):
            return prop
        if default is not MISSING:
            return default

        raise FramePropError(
            func,
            resolved_key,
            f'Key "{resolved_key}" did not contain expected type: Expected "{t}" got "{type(prop)}"!',
        )

    norm_t = _normalize_types(t)

    if isinstance(prop, norm_t):
        if cast is not MISSING:
            return cast(prop)
        return prop

    if all(issubclass(ty, str) for ty in norm_t) and isinstance(prop, bytes):
        return prop.decode("utf-8")

    raise FramePropError(
        func,
        resolved_key,
        f'Key "{resolved_key}" did not contain expected type: Expected "{t}" got "{type(prop)}"!',
    )


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


def get_clip_filepath(clip: vs.VideoNode, fallback: SPathLike | None = None, func: FuncExcept | None = None) -> SPath:
    """
    Helper function to get the file path from a clip.

    This functions checks for the `IdxFilePath` frame property.
    It also checks to ensure the file exists, and throws an error if it doesn't.

    Args:
        clip: The clip to get the file path from.
        fallback: Fallback file path to use if the `prop` is not found.
        func: Function returned for error handling. This should only be set by VS package developers.

    Raises:
        FileWasNotFoundError: The file path was not found.
        FramePropError: The property was not found in the clip.
    """
    func = func or get_clip_filepath

    path = get_prop(
        clip, "IdxFilePath", str, SPath, default=SPath(fallback) if fallback is not None else None, func=func
    )

    if path is None or not path.exists():
        raise FileWasNotFoundError("File not found!", func)

    return path


@overload
def get_props[PropValueT: _PropValue, CT, DT](
    obj: HoldsPropValue,
    keys: Iterable[str | type[PropEnum]],
    t: type[PropValueT],
    cast: Callable[[PropValueT], CT] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    default: DT = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    func: FuncExcept | None = None,
) -> dict[str, PropValueT | CT | DT]: ...


@overload
def get_props[CT, DT](
    obj: HoldsPropValue,
    keys: Iterable[str | type[PropEnum]],
    t: tuple[type[Any], ...],
    cast: Callable[[Any], CT] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    default: DT = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    func: FuncExcept | None = None,
) -> dict[str, Any | DT | CT]: ...


def get_props[CT](
    obj: HoldsPropValue,
    keys: Iterable[str | type[PropEnum]],
    t: type[Any] | tuple[type[Any], ...],
    cast: Callable[[Any], CT] | MissingT = MISSING,
    default: Any | MissingT = MISSING,
    func: FuncExcept | None = None,
) -> dict[str, Any]:
    """
    Get multiple frame properties from a clip.

    Args:
        obj: Clip or frame containing props.
        keys: List of props to get.
        t: Expected type of the prop.
        cast: Optional cast to apply to the value.
        default: Fallback value if missing or invalid.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Dictionary mapping property names to their values. Values will be of type specified by cast if provided,
        otherwise of the type(s) specified in ``t`` or a default value if provided.
    """
    func = func or get_props

    if not keys:
        return {}

    props = dict[str, Any]()
    exceptions = list[Exception]()

    kwargs: dict[str, Any] = {"t": t, "func": func}

    if cast is not MISSING:
        kwargs["cast"] = cast

    if default is not MISSING:
        kwargs["default"] = default

    for k in keys:
        try:
            prop = get_prop(obj, k, **kwargs)
        except Exception as e:
            exceptions.append(e)
        else:
            props[k if isinstance(k, str) else k.prop_key] = prop

    if exceptions:
        raise ExceptionGroup("Multiple exceptions occurred!", exceptions) from None

    return props
