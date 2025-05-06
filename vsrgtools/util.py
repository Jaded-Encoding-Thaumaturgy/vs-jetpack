from __future__ import annotations

from typing import Any, Sequence

from vstools import (
    ConstantFormatVideoNode, GenericVSFunction, PlanesT, check_variable, check_variable_format,
    join, normalize_planes, normalize_seq, split, vs
)

__all__ = [
    'norm_rmode_planes',
    'normalize_radius'
]


def norm_rmode_planes(
    clip: vs.VideoNode, mode: int | Sequence[int], planes: PlanesT = None
) -> list[int]:
    assert check_variable(clip, norm_rmode_planes)

    modes_array = normalize_seq(mode, clip.format.num_planes)

    planes = normalize_planes(clip, planes)

    return [rep if i in planes else 0 for i, rep in enumerate(modes_array, 0)]


def normalize_radius(
    clip: vs.VideoNode,
    func: GenericVSFunction[ConstantFormatVideoNode],
    radius: Sequence[float | int] | dict[str, Sequence[float | int]],
    planes: PlanesT,
    **kwargs: Any
) -> ConstantFormatVideoNode:
    assert check_variable_format(clip, normalize_radius)

    if isinstance(radius, dict):
        name, radius = radius.popitem()
    else:
        name, radius = "radius", radius

    radius = normalize_seq(radius, clip.format.num_planes)
    planes = normalize_planes(clip, planes)

    if len(set(radius)) > 1:
        pplanes = [
            func(p, **kwargs | {name: rad, 'planes': 0})
            if i in planes else p
            for i, (rad, p) in enumerate(zip(radius, split(clip)))
        ]
        return join(pplanes)

    return func(clip, **kwargs | {name: radius[0], 'planes': planes})
