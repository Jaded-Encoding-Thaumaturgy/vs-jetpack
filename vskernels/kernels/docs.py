from __future__ import annotations

from typing import Any

from vstools import (
    ConstantFormatVideoNode, HoldsVideoFormatT, Matrix, MatrixT, VideoFormatT, core, depth, get_video_format,
    vs
)

from ..types import ShiftT
from .abstract import Kernel

__all__ = [
    'Example'
]


class Example(Kernel):
    """Example Kernel class for documentation purposes."""

    def __init__(self, b: float = 0, c: float = 1 / 2, **kwargs: Any) -> None:
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def scale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular scaling operation.

        :param clip:        Input clip
        :param width:       Output width
        :param height:      Output height
        :param shift:       Shift clip during the operation.
                            Expects a tuple of (src_top, src_left).

        :rtype:             ``VideoNode``
        """
        width, height = self._wh_norm(clip, width, height)
        return core.resize.Bicubic(
            clip, width, height, src_top=shift[0], src_left=shift[1],
            filter_param_a=self.b, filter_param_b=self.c, **self.kwargs, **kwargs
        )

    def descale(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: ShiftT = (0, 0), **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Perform a regular descaling operation.

        :param clip:        Input clip
        :param width:       Output width
        :param height:      Output height
        :param shift:       Shift clip during the operation.
                            Expects a tuple of (src_top, src_left).

        :rtype:             ``ConstantFormatVideoNode``
        """
        width, height = self._wh_norm(clip, width, height)
        shift = self._shift_norm(shift, True)
        return depth(core.descale.Debicubic(
            depth(clip, 32), width, height, b=self.b, c=self.c, src_top=shift[0], src_left=shift[1], **kwargs
        ), clip)

    def resample(
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None = None, matrix_in: MatrixT | None = None, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Perform a regular resampling operation.

        :param clip:        Input clip
        :param format:      Output format
        :param matrix:      Output matrix. If `None`, will take the matrix from the input clip's frameprops.
        :param matrix_in:   Input matrix. If `None`, will take the matrix from the input clip's frameprops.

        :rtype:             ``ConstantFormatVideoNode``
        """
        return core.resize.Bicubic(
            clip, format=get_video_format(format).id,
            filter_param_a=self.b, filter_param_b=self.c,
            matrix=Matrix.from_param(matrix),
            matrix_in=Matrix.from_param(matrix_in), **self.kwargs | kwargs
        )

    def shift(
        self,
        clip: vs.VideoNode,
        shifts_or_shift_top: tuple[float, float] | float | list[float],
        shift_left: float | list[float], /,
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular shifting operation.

        :param clip:        Input clip
        :param shift:       Shift clip during the operation.\n
                            Expects a tuple of (src_top, src_left)\n
                            or two top, left arrays for shifting planes individually.

        :rtype:             ``VideoNode``
        """
        return super().shift(clip, shifts_or_shift_top, shift_left, **kwargs)
