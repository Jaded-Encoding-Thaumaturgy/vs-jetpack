from __future__ import annotations

from typing import Any, Literal, Mapping, overload

from ..enums import ChromaLocation, ColorRange, Matrix, Primaries, PropEnum, Transfer
from ..vs_proxy import vs

__all__ = ["video_heuristics"]


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
    from ..utils import get_prop, get_video_format

    fmt = get_video_format(clip)

    assumed_props = list[str]()

    prop_enums: dict[str, type[PropEnum]] = {
        "matrix": Matrix,
        "primaries": Primaries,
        "transfer": Transfer,
    }

    if fmt.color_family != vs.RGB:
        prop_enums["range"] = ColorRange

    if fmt.subsampling_h == fmt.subsampling_w != 0:
        prop_enums["chromaloc"] = ChromaLocation

    def _get_props(obj: vs.VideoNode | Mapping[str, Any], key: type[PropEnum]) -> PropEnum:
        p = get_prop(obj, key, int, cast=key, default=None, func=video_heuristics)

        if p is not None and not p.is_unspecified():
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
