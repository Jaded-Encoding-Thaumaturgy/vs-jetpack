from __future__ import annotations

from typing import Callable, Union

from vstools import CustomEnum, vs

from .abstract import GeneralMask
from .edge._abstract import EdgeDetectT, RidgeDetectT

__all__ = ["Coordinates", "GenericMaskT", "XxpandMode"]


class XxpandMode(CustomEnum):
    """
    Expand/inpand mode
    """

    RECTANGLE = object()
    """
    Rectangular shape
    """

    ELLIPSE = object()
    """
    Elliptical shape
    """

    LOSANGE = object()
    """
    Diamond shape
    """


class Coordinates(tuple[int, ...], CustomEnum):
    VERTICAL = (0, 1, 0, 0, 0, 0, 1, 0)
    HORIZONTAL = (0, 0, 0, 1, 1, 0, 0, 0)
    RECTANGLE = (1, 1, 1, 1, 1, 1, 1, 1)
    DIAMOND = (0, 1, 0, 1, 1, 0, 1, 0)
    CORNERS = (1, 0, 1, 0, 0, 1, 0, 1)

    @classmethod
    def from_iter(cls, iter: int) -> Coordinates:
        return cls.DIAMOND if (iter % 3) != 1 else cls.RECTANGLE

    @classmethod
    def from_xxpand_mode(cls, xxpand_mode: XxpandMode, iter: int = 1) -> Coordinates:
        if xxpand_mode == XxpandMode.LOSANGE or (xxpand_mode is XxpandMode.ELLIPSE and iter % 3 != 1):
            return cls.DIAMOND

        return cls.RECTANGLE


GenericMaskT = Union[
    vs.VideoNode, Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode], EdgeDetectT, RidgeDetectT, GeneralMask, str
]
