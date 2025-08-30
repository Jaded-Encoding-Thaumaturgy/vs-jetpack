from __future__ import annotations

from typing import Any, Literal, Mapping, overload

import vapoursynth as vs
from jetpytools import KwargsT
from typing_extensions import deprecated

from ..enums import ChromaLocation, ColorRange, Matrix, Primaries, PropEnum, Transfer

__all__ = ["video_heuristics", "video_resample_heuristics"]


@overload
def video_heuristics(
    clip: vs.VideoNode,
    props: Mapping[str, Any] | bool | None = None,
    prop_in: bool = True,
    assumed_return: Literal[False] = False,
) -> dict[str, PropEnum]: ...


@overload
def video_heuristics(
    clip: vs.VideoNode,
    props: Mapping[str, Any] | bool | None = None,
    prop_in: bool = True,
    *,
    assumed_return: Literal[True],
) -> tuple[dict[str, PropEnum], list[str]]: ...


def video_heuristics(
    clip: vs.VideoNode,
    props: Mapping[str, Any] | bool | None = None,
    prop_in: bool = True,
    assumed_return: bool = False,
) -> dict[str, PropEnum] | tuple[dict[str, PropEnum], list[str]]:
    """
    Determine video heuristics from frame properties.

    Args:
        clip: The input clip.
        props: Source properties used to retrieve values.

               - If True: uses the frame properties from the clip when available.
               - If a Mapping is passed: uses the frame properties from it when available.
               - If any other value or frame properties unavailable: values are inferred from the clip's resolution.

        prop_in: If True, returns a dict with keys in the form `{prop_name}_in` (e.g., `matrix_in`
            instead of `matrix`).

            For more details, see the [Resize docs](https://www.vapoursynth.com/doc/functions/video/resize.html).
        assumed_return: If True, returns the assumed props as a list alongside the heuristics.

    Returns:
        A dict containing all determinable video heuristics, optionally using key names derived
        from the resize plugin.
    """
    from ..utils import get_prop

    assumed_props = list[str]()

    prop_enums: dict[str, type[PropEnum]] = {
        "matrix": Matrix,
        "primaries": Primaries,
        "transfer": Transfer,
        "range": ColorRange,
        "chromaloc": ChromaLocation,
    }

    def _get_props(obj: vs.VideoNode | Mapping[str, Any], key: type[PropEnum]) -> PropEnum:
        p = get_prop(obj, key, int, cast=key, default=None, func=video_heuristics)

        if p is not None and not p.is_unknown(p):
            return p

        assumed_props.append(key.prop_key)

        return key.from_video(clip, strict=False, func=video_heuristics)

    if props is False or props is None:
        heuristics = {k: t.from_res(clip) for k, t in prop_enums.items()}
        assumed_props.extend(v.prop_key for v in heuristics.values())
    elif props is True:
        heuristics = {k: _get_props(clip, t) for k, t in prop_enums.items()}
    else:
        heuristics = {k: _get_props(props, t) for k, t in prop_enums.items()}

    if prop_in:
        heuristics = {f"{k}_in": v for k, v in heuristics.items()}

    if assumed_return:
        return heuristics, assumed_props

    return heuristics


@deprecated("This function is deprecated and will be removed in a future version.", category=DeprecationWarning)
def video_resample_heuristics(clip: vs.VideoNode, kwargs: KwargsT | None = None, **fmt_kwargs: Any) -> KwargsT:
    """
    Get a kwargs object for a video's heuristics to pass to the resize plugin or Kernel.resample.

    Args:
        clip: Clip to derive the heuristics from.
        kwargs: Keyword arguments for the _out parameters.
        **fmt_kwargs: Keyword arguments to pass to the output kwargs. These will override any heuristics that were
            derived from the input clip!

    Returns:
        Keyword arguments to pass on to the resize plugin or Kernel.resample.
    """
    from .check import check_variable_format

    assert check_variable_format(clip, video_resample_heuristics)

    video_fmt = clip.format.replace(**fmt_kwargs)

    def_kwargs_in = video_heuristics(clip, False, True)
    def_kwargs_out = video_heuristics(clip.std.BlankClip(format=video_fmt.id), False, False)

    return KwargsT(format=video_fmt.id, **def_kwargs_in, **def_kwargs_out) | (kwargs or KwargsT())
