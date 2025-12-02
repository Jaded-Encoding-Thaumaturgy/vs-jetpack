from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jetpytools import normalize_seq

from vstools import GenericVSFunction, Planes, join, normalize_planes, split, vs

__all__ = ["normalize_radius"]


def normalize_radius(
    __clip: vs.VideoNode,
    __function: GenericVSFunction,
    __radius: Sequence[float | int] | dict[str, Sequence[float | int]],
    /,
    planes: Planes,
    **kwargs: Any,
) -> vs.VideoNode:
    if isinstance(__radius, dict):
        name, radius = __radius.popitem()
    else:
        name, radius = "radius", __radius

    radius = normalize_seq(radius, __clip.format.num_planes)
    planes = normalize_planes(__clip, planes)

    if len(set(radius)) > 1:
        pplanes = [
            __function(p, **kwargs | {name: rad, "planes": 0}) if i in planes else p
            for i, (rad, p) in enumerate(zip(radius, split(__clip)))
        ]
        return join(pplanes)

    return __function(__clip, **kwargs | {name: radius[0], "planes": planes})
