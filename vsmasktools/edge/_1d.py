"""1D matrices"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Sequence

from vstools import ColorRange, ConstantFormatVideoNode, depth, vs

from ._abstract import EdgeDetect, EuclideanDistance

# ruff: noqa: RUF022

__all__ = [
    "Matrix1D",
    "TEdge",
    "TEdgeTedgemask",
    #
    "SavitzkyGolay",
    #
    "SavitzkyGolayDeriv1Quad5",
    "SavitzkyGolayDeriv1Quad7",
    "SavitzkyGolayDeriv1Quad9",
    "SavitzkyGolayDeriv1Quad11",
    "SavitzkyGolayDeriv1Quad13",
    "SavitzkyGolayDeriv1Quad15",
    "SavitzkyGolayDeriv1Quad17",
    "SavitzkyGolayDeriv1Quad19",
    "SavitzkyGolayDeriv1Quad21",
    "SavitzkyGolayDeriv1Quad23",
    "SavitzkyGolayDeriv1Quad25",
    #
    "SavitzkyGolayDeriv1Cubic5",
    "SavitzkyGolayDeriv1Cubic7",
    "SavitzkyGolayDeriv1Cubic9",
    "SavitzkyGolayDeriv1Cubic11",
    "SavitzkyGolayDeriv1Cubic13",
    "SavitzkyGolayDeriv1Cubic15",
    "SavitzkyGolayDeriv1Cubic17",
    "SavitzkyGolayDeriv1Cubic19",
    "SavitzkyGolayDeriv1Cubic21",
    "SavitzkyGolayDeriv1Cubic23",
    "SavitzkyGolayDeriv1Cubic25",
    #
    "SavitzkyGolayDeriv1Quint7",
    "SavitzkyGolayDeriv1Quint9",
    "SavitzkyGolayDeriv1Quint11",
    "SavitzkyGolayDeriv1Quint13",
    "SavitzkyGolayDeriv1Quint15",
    "SavitzkyGolayDeriv1Quint17",
    "SavitzkyGolayDeriv1Quint19",
    "SavitzkyGolayDeriv1Quint21",
    "SavitzkyGolayDeriv1Quint23",
    "SavitzkyGolayDeriv1Quint25",
    #
    "SavitzkyGolayDeriv2Quad5",
    "SavitzkyGolayDeriv2Quad7",
    "SavitzkyGolayDeriv2Quad9",
    "SavitzkyGolayDeriv2Quad11",
    "SavitzkyGolayDeriv2Quad13",
    "SavitzkyGolayDeriv2Quad15",
    "SavitzkyGolayDeriv2Quad17",
    "SavitzkyGolayDeriv2Quad19",
    "SavitzkyGolayDeriv2Quad21",
    "SavitzkyGolayDeriv2Quad23",
    "SavitzkyGolayDeriv2Quad25",
    "SavitzkyGolayDeriv2Quart7",
    "SavitzkyGolayDeriv2Quart9",
    "SavitzkyGolayDeriv2Quart11",
    "SavitzkyGolayDeriv2Quart13",
    "SavitzkyGolayDeriv2Quart15",
    "SavitzkyGolayDeriv2Quart17",
    "SavitzkyGolayDeriv2Quart19",
    "SavitzkyGolayDeriv2Quart21",
    "SavitzkyGolayDeriv2Quart23",
    "SavitzkyGolayDeriv2Quart25",
    #
    "SavitzkyGolayDeriv3Cub5",
    "SavitzkyGolayDeriv3Cub7",
    "SavitzkyGolayDeriv3Cub9",
    "SavitzkyGolayDeriv3Cub11",
    "SavitzkyGolayDeriv3Cub13",
    "SavitzkyGolayDeriv3Cub15",
    "SavitzkyGolayDeriv3Cub17",
    "SavitzkyGolayDeriv3Cub19",
    "SavitzkyGolayDeriv3Cub21",
    "SavitzkyGolayDeriv3Cub23",
    "SavitzkyGolayDeriv3Cub25",
    #
    "SavitzkyGolayDeriv3Quint7",
    "SavitzkyGolayDeriv3Quint9",
    "SavitzkyGolayDeriv3Quint11",
    "SavitzkyGolayDeriv3Quint13",
    "SavitzkyGolayDeriv3Quint15",
    "SavitzkyGolayDeriv3Quint17",
    "SavitzkyGolayDeriv3Quint19",
    "SavitzkyGolayDeriv3Quint21",
    "SavitzkyGolayDeriv3Quint23",
    "SavitzkyGolayDeriv3Quint25",
    #
    "SavitzkyGolayDeriv4Quart7",
    "SavitzkyGolayDeriv4Quart9",
    "SavitzkyGolayDeriv4Quart11",
    "SavitzkyGolayDeriv4Quart13",
    "SavitzkyGolayDeriv4Quart15",
    "SavitzkyGolayDeriv4Quart17",
    "SavitzkyGolayDeriv4Quart19",
    "SavitzkyGolayDeriv4Quart21",
    "SavitzkyGolayDeriv4Quart23",
    "SavitzkyGolayDeriv4Quart25",
    #
    "SavitzkyGolayDeriv5Quint7",
    "SavitzkyGolayDeriv5Quint9",
    "SavitzkyGolayDeriv5Quint11",
    "SavitzkyGolayDeriv5Quint13",
    "SavitzkyGolayDeriv5Quint15",
    "SavitzkyGolayDeriv5Quint17",
    "SavitzkyGolayDeriv5Quint19",
    "SavitzkyGolayDeriv5Quint21",
    "SavitzkyGolayDeriv5Quint23",
    "SavitzkyGolayDeriv5Quint25",
]


class Matrix1D(EdgeDetect, ABC): ...


class TEdge(EuclideanDistance, Matrix1D):
    """(TEdgeMasktype=2) Avisynth plugin."""

    matrices: ClassVar[Sequence[Sequence[float]]] = [[12, -74, 0, 74, -12], [-12, 74, 0, -74, 12]]
    divisors: ClassVar[Sequence[float] | None] = [62, 62]
    mode_types: ClassVar[Sequence[str] | None] = ["h", "v"]


class TEdgeTedgemask(Matrix1D, EdgeDetect):
    """(tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin."""

    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return clip.tedgemask.TEdgeMask(threshold=0, type=2)


class SavitzkyGolay(EuclideanDistance, Matrix1D):
    mode_types: ClassVar[Sequence[str] | None] = ["h", "v"]


class SavitzkyGolayNormalise(SavitzkyGolay):
    def _preprocess(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        return depth(clip, 32)

    def _postprocess(self, clip: vs.VideoNode, input_bits: int | None = None) -> ConstantFormatVideoNode:
        return depth(clip, input_bits, range_in=ColorRange.FULL, range_out=ColorRange.FULL)

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        assert self.divisors
        return [[c / div for c in mat] for mat, div in zip(self.matrices, self.divisors)]

    def _get_divisors(self) -> Sequence[float]:
        return [0.0] * len(self._get_matrices())


class SavitzkyGolayDeriv1Quad5(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 5"""

    matrices = [[-2, -1, 0, 1, 2]] * 2
    divisors = [10] * 2


class SavitzkyGolayDeriv1Quad7(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 7"""

    matrices = [[-3, -2, -1, 0, 1, 2, 3]] * 2
    divisors = [28] * 2


class SavitzkyGolayDeriv1Quad9(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 9"""

    matrices = [[-4, -3, -2, -1, 0, 1, 2, 3, 4]] * 2
    divisors = [60] * 2


class SavitzkyGolayDeriv1Quad11(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 11"""

    matrices = [[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]] * 2
    divisors = [110] * 2


class SavitzkyGolayDeriv1Quad13(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 13"""

    matrices = [[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]] * 2
    divisors = [182] * 2


class SavitzkyGolayDeriv1Quad15(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 15"""

    matrices = [[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]] * 2
    divisors = [280] * 2


class SavitzkyGolayDeriv1Quad17(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 17"""

    matrices = [[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]] * 2
    divisors = [408] * 2


class SavitzkyGolayDeriv1Quad19(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 19"""

    matrices = [[-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2
    divisors = [570] * 2


class SavitzkyGolayDeriv1Quad21(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 21"""

    matrices = [[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 2
    divisors = [770] * 2


class SavitzkyGolayDeriv1Quad23(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 23"""

    matrices = [[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 2
    divisors = [1012] * 2


class SavitzkyGolayDeriv1Quad25(SavitzkyGolay):
    """Savitzky-Golay first quadratic derivative operator of size 25"""

    matrices = [[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] * 2
    divisors = [1300] * 2


class SavitzkyGolayDeriv1Cubic5(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 5"""

    matrices = [[1, -8, 0, 8, -1]] * 2
    divisors = [12] * 2


class SavitzkyGolayDeriv1Cubic7(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic derivative operator of size 7"""

    matrices = [[22, -67, -58, 0, 58, 67, -22]] * 2
    divisors = [252] * 2


class SavitzkyGolayDeriv1Cubic9(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 9"""

    matrices = [[86, -142, -193, -126, 0, 126, 193, 142, -86]] * 2
    divisors = [1188] * 2


class SavitzkyGolayDeriv1Cubic11(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 11"""

    matrices = [[300, -294, -532, -503, -296, 0, 296, 503, 532, 294, -300]] * 2
    divisors = [5148] * 2


class SavitzkyGolayDeriv1Cubic13(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 13"""

    matrices = [[1133, -660, -1578, -1796, -1489, -832, 0, 832, 1489, 1796, 1578, 660, -1133]] * 2
    divisors = [24024] * 2


class SavitzkyGolayDeriv1Cubic15(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 15"""

    matrices = [
        [12922, -4121, -14150, -18334, -17842, -13843, -7506, 0, 7506, 13843, 17842, 18334, 14150, 4121, -12922]
    ] * 2
    divisors = [334152] * 2


class SavitzkyGolayDeriv1Cubic17(SavitzkyGolay):
    """Savitzky-Golay first cubic/quartic operator of size 17"""

    matrices = [[748, -98, -643, -930, -1002, -902, -673, -358, 0, 358, 673, 902, 1002, 930, 643, 98, -748]] * 2
    divisors = [23256] * 2


class SavitzkyGolayDeriv1Cubic19(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 19"""

    matrices = [
        [
            6936,
            68,
            -4648,
            -7481,
            -8700,
            -8574,
            -7372,
            -5363,
            -2816,
            0,
            2816,
            5363,
            7372,
            8574,
            8700,
            7481,
            4648,
            -68,
            -6936,
        ]
    ] * 2
    divisors = [255816] * 2


class SavitzkyGolayDeriv1Cubic21(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 21"""

    matrices = [
        [
            84075,
            10032,
            -43284,
            -78176,
            -96947,
            -101900,
            -95338,
            -79564,
            -56881,
            -29592,
            0,
            29592,
            56881,
            79564,
            95338,
            101900,
            96947,
            78176,
            43284,
            -10032,
            -84075,
        ]
    ] * 2
    divisors = [3634092] * 2


class SavitzkyGolayDeriv1Cubic23(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 23"""

    matrices = [
        [
            3938,
            815,
            -1518,
            -3140,
            -4130,
            -4567,
            -4530,
            -4098,
            -3350,
            -2365,
            -1222,
            0,
            1222,
            2365,
            3350,
            4098,
            4530,
            4567,
            4130,
            3140,
            1518,
            -815,
            -3938,
        ]
    ] * 2
    divisors = [197340] * 2


class SavitzkyGolayDeriv1Cubic25(SavitzkyGolayNormalise):
    """Savitzky-Golay first cubic/quartic operator of size 25"""

    matrices = [
        [
            30866,
            8602,
            -8525,
            -20982,
            -29236,
            -33754,
            -35003,
            -33450,
            -29562,
            -23806,
            -16649,
            -8558,
            0,
            8558,
            16649,
            23806,
            29562,
            33450,
            35003,
            33754,
            29236,
            20982,
            8525,
            -8602,
            -30866,
        ]
    ] * 2
    divisors = [1776060] * 2


class SavitzkyGolayDeriv1Quint7(SavitzkyGolay):
    """Savitzky-Golay first quintic/sextic derivative operator of size 7"""

    matrices = [[-1, 9, -45, 0, 45, -9, 1]] * 2
    divisors = [60] * 2


class SavitzkyGolayDeriv1Quint9(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 9"""

    matrices = [[-254, 1381, -2269, -2879, 0, 2879, 2269, -1381, 254]] * 2
    divisors = [8580] * 2


class SavitzkyGolayDeriv1Quint11(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 11"""

    matrices = [[-573, 2166, -1249, -3774, -3084, 0, 3084, 3774, 1249, -2166, 573]] * 2
    divisors = [17160] * 2


class SavitzkyGolayDeriv1Quint13(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 13"""

    matrices = [[-9647, 27093, -12, -33511, -45741, -31380, 0, 31380, 45741, 33511, 12, -27093, 9647]] * 2
    divisors = [291720] * 2


class SavitzkyGolayDeriv1Quint15(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 15"""

    matrices = [
        [
            -78351,
            169819,
            65229,
            -130506,
            -266401,
            -279975,
            -175125,
            0,
            175125,
            279975,
            266401,
            130506,
            -65229,
            -169819,
            78351,
        ]
    ] * 2
    divisors = [2519400] * 2


class SavitzkyGolayDeriv1Quint17(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 17"""

    matrices = [
        [
            -14404,
            24661,
            16679,
            -8671,
            -32306,
            -43973,
            -40483,
            -23945,
            0,
            23945,
            40483,
            43973,
            32306,
            8671,
            -16679,
            -24661,
            14404,
        ]
    ] * 2
    divisors = [503880] * 2


class SavitzkyGolayDeriv1Quint19(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 19"""

    matrices = [
        [
            -255102,
            349928,
            322378,
            9473,
            -348823,
            -604484,
            -686099,
            -583549,
            -332684,
            0,
            332684,
            583549,
            686099,
            604484,
            348823,
            -9473,
            -322378,
            -349928,
            255102,
        ]
    ] * 2
    divisors = [9806280] * 2


class SavitzkyGolayDeriv1Quint21(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 21"""

    matrices = [
        [
            -15033066,
            16649358,
            19052988,
            6402438,
            -10949942,
            -26040033,
            -34807914,
            -35613829,
            -28754154,
            -15977364,
            0,
            15977364,
            28754154,
            35613829,
            34807914,
            26040033,
            10949942,
            -6402438,
            -19052988,
            -16649358,
            15033066,
        ]
    ] * 2
    divisors = [637408200] * 2


class SavitzkyGolayDeriv1Quint23(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 23"""

    matrices = [
        [
            -400653,
            359157,
            489687,
            265164,
            -106911,
            -478349,
            -752859,
            -878634,
            -840937,
            -654687,
            -357045,
            0,
            357045,
            654687,
            840937,
            878634,
            752859,
            478349,
            106911,
            -265164,
            -489687,
            -359157,
            400653,
        ]
    ] * 2
    divisors = [18747300] * 2


class SavitzkyGolayDeriv1Quint25(SavitzkyGolayNormalise):
    """Savitzky-Golay first quintic/sextic derivative operator of size 25"""

    matrices = [
        [
            -8322182,
            6024183,
            9604353,
            6671883,
            544668,
            -6301491,
            -12139321,
            -15896511,
            -17062146,
            -15593141,
            -11820675,
            -6356625,
            0,
            6356625,
            11820675,
            15593141,
            17062146,
            15896511,
            12139321,
            6301491,
            -544668,
            -6671883,
            -9604353,
            -6024183,
            8322182,
        ]
    ] * 2
    divisors = [429214500] * 2


class SavitzkyGolayDeriv2Quad5(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 5"""

    matrices = [[2, -1, -2, -1, 2]] * 2
    divisors = [7] * 2


class SavitzkyGolayDeriv2Quad7(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 7"""

    matrices = [[5, 0, -3, -4, -3, 0, 5]] * 2
    divisors = [42] * 2


class SavitzkyGolayDeriv2Quad9(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 9"""

    matrices = [[28, 7, -8, -17, -20, -17, -8, 7, 28]] * 2
    divisors = [462] * 2


class SavitzkyGolayDeriv2Quad11(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 11"""

    matrices = [[15, 6, -1, -6, -9, -10, -9, -6, -1, 6, 15]] * 2
    divisors = [429] * 2


class SavitzkyGolayDeriv2Quad13(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 13"""

    matrices = [[22, 11, 2, -5, -10, -13, -14, -13, -10, -5, 2, 11, 22]] * 2
    divisors = [1001] * 2


class SavitzkyGolayDeriv2Quad15(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 15"""

    matrices = [[91, 52, 19, -8, -29, -44, -53, -56, -53, -44, -29, -8, 19, 52, 91]] * 2
    divisors = [6188] * 2


class SavitzkyGolayDeriv2Quad17(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 17"""

    matrices = [[40, 25, 12, 1, -8, -15, -20, -23, -24, -23, -20, -15, -8, 1, 12, 25, 40]] * 2
    divisors = [3876] * 2


class SavitzkyGolayDeriv2Quad19(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 19"""

    matrices = [[51, 34, 19, 6, -5, -14, -21, -26, -29, -30, -29, -26, -21, -14, -5, 6, 19, 34, 51]] * 2
    divisors = [6783] * 2


class SavitzkyGolayDeriv2Quad21(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 21"""

    matrices = [
        [190, 133, 82, 37, -2, -35, -62, -83, -98, -107, -110, -107, -98, -83, -62, -35, -2, 37, 82, 133, 190]
    ] * 2
    divisors = [33649] * 2


class SavitzkyGolayDeriv2Quad23(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 23"""

    matrices = [
        [77, 56, 37, 20, 5, -8, -19, -28, -35, -40, -43, -44, -43, -40, -35, -28, -19, -8, 5, 20, 37, 56, 77]
    ] * 2
    divisors = [17710] * 2


class SavitzkyGolayDeriv2Quad25(SavitzkyGolay):
    """Savitzky-Golay second quadratic/cubic derivative operator of size 25"""

    matrices = [
        [
            92,
            69,
            48,
            29,
            12,
            -3,
            -16,
            -27,
            -36,
            -43,
            -48,
            -51,
            -52,
            -51,
            -48,
            -43,
            -36,
            -27,
            -16,
            -3,
            12,
            29,
            48,
            69,
            92,
        ]
    ] * 2
    divisors = [26910] * 2


class SavitzkyGolayDeriv2Quart7(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 7"""

    matrices = [[-13, 67, -19, -70, -19, 67, -13]] * 2
    divisors = [132] * 2


class SavitzkyGolayDeriv2Quart9(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 9"""

    matrices = [[-126, 371, 151, -211, -370, -211, 151, 371, -126]] * 2
    divisors = [1716] * 2


class SavitzkyGolayDeriv2Quart11(SavitzkyGolay):
    """Savitzky-Golay second quartic/quintic derivative operator of size 11"""

    matrices = [[-90, 174, 146, 1, -136, -190, -136, 1, 146, 174, -90]] * 2
    divisors = [1716] * 2


class SavitzkyGolayDeriv2Quart13(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 13"""

    matrices = [[-2211, 2970, 3504, 1614, -971, -3016, -3780, -3016, -971, 1614, 3504, 2970, -2211]] * 2
    divisors = [58344] * 2


class SavitzkyGolayDeriv2Quart15(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 15"""

    matrices = [
        [-31031, 29601, 44495, 31856, 6579, -19751, -38859, -45780, -38859, -19751, 6579, 31856, 44495, 29601, -31031]
    ] * 2
    divisors = [1108536] * 2


class SavitzkyGolayDeriv2Quart17(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 17"""

    matrices = [
        [-2132, 1443, 2691, 2405, 1256, -207, -1557, -2489, -2820, -2489, -1557, -207, 1256, 2405, 2691, 1443, -2132]
    ] * 2
    divisors = [100776] * 2


class SavitzkyGolayDeriv2Quart19(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 19"""

    matrices = [
        [
            -32028,
            15028,
            35148,
            36357,
            25610,
            8792,
            -9282,
            -24867,
            -35288,
            -38940,
            -35288,
            -24867,
            -9282,
            8792,
            25610,
            36357,
            35148,
            15028,
            -32028,
        ]
    ] * 2
    divisors = [1961256] * 2


class SavitzkyGolayDeriv2Quart21(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 21"""

    matrices = [
        [
            -12597,
            3876,
            11934,
            13804,
            11451,
            6578,
            626,
            -5226,
            -10061,
            -13224,
            -14322,
            -13224,
            -10061,
            -5226,
            626,
            6578,
            11451,
            13804,
            11934,
            3876,
            -12597,
        ]
    ] * 2
    divisors = [980628] * 2


class SavitzkyGolayDeriv2Quart23(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivative operator of size 23"""

    matrices = [
        [
            -115577,
            20615,
            93993,
            119510,
            110545,
            78903,
            34815,
            -13062,
            -57645,
            -93425,
            -116467,
            -124410,
            -116467,
            -93425,
            -57645,
            -13062,
            34815,
            78903,
            110545,
            119510,
            93993,
            20615,
            -115577,
        ]
    ] * 2
    divisors = [11248380] * 2


class SavitzkyGolayDeriv2Quart25(SavitzkyGolayNormalise):
    """Savitzky-Golay second quartic/quintic derivativeoperator of size 25"""

    matrices = [
        [
            -143198,
            10373,
            99385,
            137803,
            138262,
            112067,
            69193,
            18285,
            -33342,
            -79703,
            -116143,
            -139337,
            -147290,
            -139337,
            -116143,
            -79703,
            -33342,
            18285,
            69193,
            112067,
            138262,
            137803,
            99385,
            10373,
            -143198,
        ]
    ] * 2
    divisors = [17168580] * 2


class SavitzkyGolayDeriv3Cub5(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 5"""

    matrices = [[-1, 2, 0, -2, 1]] * 2
    divisors = [2] * 2


class SavitzkyGolayDeriv3Cub7(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 7"""

    matrices = [[-1, 1, 1, 0, -1, -1, 1]] * 2
    divisors = [6] * 2


class SavitzkyGolayDeriv3Cub9(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 9"""

    matrices = [[-14, 7, 13, 9, 0, -9, -13, -7, 14]] * 2
    divisors = [198] * 2


class SavitzkyGolayDeriv3Cub11(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 11"""

    matrices = [[-30, 6, 22, 23, 14, 0, -14, -23, -22, -6, 30]] * 2
    divisors = [858] * 2


class SavitzkyGolayDeriv3Cub13(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 13"""

    matrices = [[-11, 0, 6, 8, 7, 4, 0, -4, -7, -8, -6, 0, 11]] * 2
    divisors = [572] * 2


class SavitzkyGolayDeriv3Cub15(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 15"""

    matrices = [[-91, -13, 35, 58, 61, 49, 27, 0, -27, -49, -61, -58, -35, 13, 91]] * 2
    divisors = [7956] * 2


class SavitzkyGolayDeriv3Cub17(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 17"""

    matrices = [[-28, -7, 7, 15, 18, 17, 13, 7, 0, -7, -13, -17, -18, -15, -7, 7, 28]] * 2
    divisors = [3876] * 2


class SavitzkyGolayDeriv3Cub19(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 19"""

    matrices = [[-204, -68, 28, 89, 120, 126, 112, 83, 44, 0, -44, -83, -112, -126, -120, -89, -28, 68, 204]] * 2
    divisors = [42636] * 2


class SavitzkyGolayDeriv3Cub21(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 21"""

    matrices = [
        [-285, -114, 12, 98, 149, 170, 166, 142, 103, 54, 0, -54, -103, -142, -166, -170, -149, -98, -12, 114, 285]
    ] * 2
    divisors = [86526] * 2


class SavitzkyGolayDeriv3Cub23(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 23"""

    matrices = [
        [-77, -35, -3, 20, 35, 43, 45, 42, 35, 25, 13, 0, -13, -25, -35, -42, -45, -43, -35, -20, 3, 35, 77]
    ] * 2
    divisors = [32890] * 2


class SavitzkyGolayDeriv3Cub25(SavitzkyGolay):
    """Savitzky-Golay third cubic/quartic derivative operator of size 25"""

    matrices = [
        [
            -506,
            -253,
            -55,
            93,
            196,
            259,
            287,
            285,
            258,
            211,
            149,
            77,
            0,
            -77,
            -149,
            -211,
            -258,
            -285,
            -287,
            -259,
            -196,
            -93,
            55,
            253,
            506,
        ]
    ] * 2
    divisors = [296010] * 2


class SavitzkyGolayDeriv3Quint7(SavitzkyGolay):
    """Savitzky-Golay third quintic/sexic derivative operator of size 7"""

    matrices = [[1, -8, 13, 0, -13, 8, -1]] * 2
    divisors = [8] * 2


class SavitzkyGolayDeriv3Quint9(SavitzkyGolay):
    """Savitzky-Golay third quintic/sexic derivative operator of size 9"""

    matrices = [[100, -457, 256, 459, 0, -459, -256, 457, -100]] * 2
    divisors = [1144] * 2


class SavitzkyGolayDeriv3Quint11(SavitzkyGolay):
    """Savitzky-Golay third quintic/sexic derivative operator of size 11"""

    matrices = [[129, -402, -11, 340, 316, 0, -316, -340, 11, 402, -129]] * 2
    divisors = [2288] * 2


class SavitzkyGolayDeriv3Quint13(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 13"""

    matrices = [[1430, -3267, -1374, 1633, 3050, 2252, 0, -2252, -3050, -1633, 1374, 3267, -1430]] * 2
    divisors = [38896] * 2


class SavitzkyGolayDeriv3Quint15(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 15"""

    matrices = [
        [8281, -14404, -10379, 1916, 11671, 14180, 9315, 0, -9315, -14180, -11671, -1916, 10379, 14404, -8281]
    ] * 2
    divisors = [335920] * 2


class SavitzkyGolayDeriv3Quint17(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 17"""

    matrices = [
        [1144, -1547, -1508, -351, 876, 1595, 1604, 983, 0, -983, -1604, -1595, -876, 351, 1508, 1547, -1144]
    ] * 2
    divisors = [67184] * 2


class SavitzkyGolayDeriv3Quint19(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 19"""

    matrices = [
        [
            15810,
            -16796,
            -20342,
            -9818,
            4329,
            15546,
            20525,
            18554,
            10868,
            0,
            -10868,
            -18554,
            -20525,
            -15546,
            -4329,
            9818,
            20342,
            16796,
            -15810,
        ]
    ] * 2
    divisors = [1307504] * 2


class SavitzkyGolayDeriv3Quint21(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 21"""

    matrices = [
        [
            748068,
            -625974,
            -908004,
            -598094,
            -62644,
            448909,
            787382,
            887137,
            749372,
            425412,
            0,
            -425412,
            -749372,
            -887137,
            -787382,
            -448909,
            62644,
            598094,
            908004,
            625974,
            -748068,
        ]
    ] * 2
    divisors = [84987760] * 2


class SavitzkyGolayDeriv3Quint23(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 23"""

    matrices = [
        [
            49115,
            -32224,
            -55233,
            -43928,
            -16583,
            13632,
            38013,
            51684,
            52959,
            42704,
            23699,
            0,
            -23699,
            -42704,
            -52959,
            -51684,
            -38013,
            -13632,
            16583,
            43928,
            55233,
            32224,
            -49115,
        ]
    ] * 2
    divisors = [7498920] * 2


class SavitzkyGolayDeriv3Quint25(SavitzkyGolayNormalise):
    """Savitzky-Golay third quintic/sexic derivative operator of size 25"""

    matrices = [
        [
            284372,
            -144463,
            -293128,
            -266403,
            -146408,
            5131,
            144616,
            244311,
            290076,
            279101,
            217640,
            118745,
            0,
            -118745,
            -217640,
            -279101,
            -290076,
            -244311,
            -144616,
            -5131,
            146408,
            266403,
            293128,
            144463,
            -284372,
        ]
    ] * 2
    divisors = [57228600] * 2


class SavitzkyGolayDeriv4Quart7(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 7"""

    matrices = [[3, -7, 1, 6, 1, -7, 3]] * 2
    divisors = [11] * 2


class SavitzkyGolayDeriv4Quart9(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 9"""

    matrices = [[14, -21, -11, 9, 18, 9, -11, -21, 14]] * 2
    divisors = [143] * 2


class SavitzkyGolayDeriv4Quart11(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 11"""

    matrices = [[6, -6, -6, -1, 4, 6, 4, -1, -6, -6, 6]] * 2
    divisors = [143] * 2


class SavitzkyGolayDeriv4Quart13(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 13"""

    matrices = [[99, -66, -96, -54, 11, 64, 84, 64, 11, -54, -96, -66, 99]] * 2
    divisors = [4862] * 2


class SavitzkyGolayDeriv4Quart15(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 15"""

    matrices = [[1001, -429, -869, -704, -249, 251, 621, 756, 621, 251, -249, -704, -869, -429, 1001]] * 2
    divisors = [92378] * 2


class SavitzkyGolayDeriv4Quart17(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 17"""

    matrices = [[52, -13, -39, -39, -24, -3, 17, 31, 36, 31, 17, -3, -24, -39, -39, -13, 52]] * 2
    divisors = [8398] * 2


class SavitzkyGolayDeriv4Quart19(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 19"""

    matrices = [
        [612, -68, -388, -453, -354, -168, 42, 227, 352, 396, 352, 227, 42, -168, -354, -453, -388, -68, 612]
    ] * 2
    divisors = [163438] * 2


class SavitzkyGolayDeriv4Quart21(SavitzkyGolay):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 21"""

    matrices = [
        [969, 0, -510, -680, -615, -406, -130, 150, 385, 540, 594, 540, 385, 150, -130, -406, -615, -680, -510, 0, 969]
    ] * 2
    divisors = [408595] * 2


class SavitzkyGolayDeriv4Quart23(SavitzkyGolayNormalise):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 23"""

    matrices = [
        [
            1463,
            133,
            -627,
            -950,
            -955,
            -747,
            -417,
            -42,
            315,
            605,
            793,
            858,
            793,
            605,
            315,
            -42,
            -417,
            -747,
            -955,
            -950,
            -627,
            133,
            1463,
        ]
    ] * 2
    divisors = [937365] * 2


class SavitzkyGolayDeriv4Quart25(SavitzkyGolayNormalise):
    """Savitzky-Golay fourth quartic/quintic derivative operator of size 25"""

    matrices = [
        [
            1518,
            253,
            -517,
            -897,
            -982,
            -857,
            -597,
            -267,
            78,
            393,
            643,
            803,
            858,
            803,
            643,
            393,
            78,
            -267,
            -597,
            -857,
            -982,
            -897,
            -517,
            253,
            1518,
        ]
    ] * 2
    divisors = [1430715] * 2


class SavitzkyGolayDeriv5Quint7(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 7"""

    matrices = [[-1, 4, -5, 0, 5, -4, 1]] * 2
    divisors = [2] * 2


class SavitzkyGolayDeriv5Quint9(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 9"""

    matrices = [[-4, 11, -4, -9, 0, 9, 4, -11, 4]] * 2
    divisors = [26] * 2


class SavitzkyGolayDeriv5Quint11(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 11"""

    matrices = [[-3, 6, 1, -4, -4, 0, 4, 4, -1, -6, 3]] * 2
    divisors = [52] * 2


class SavitzkyGolayDeriv5Quint13(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 13"""

    matrices = [[-22, 33, 18, -11, -26, -20, 0, 20, 26, 11, -18, -33, 22]] * 2
    divisors = [884] * 2


class SavitzkyGolayDeriv5Quint15(SavitzkyGolayNormalise):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 15"""

    matrices = [[-1001, 1144, 979, 44, -751, -1000, -675, 0, 675, 1000, 751, -44, -979, -1144, 1001]] * 2
    divisors = [83980] * 2


class SavitzkyGolayDeriv5Quint17(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 17"""

    matrices = [[-104, 91, 104, 39, -36, -83, -88, -55, 0, 55, 88, 83, 36, -39, -104, -91, 104]] * 2
    divisors = [16796] * 2


class SavitzkyGolayDeriv5Quint19(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 19"""

    matrices = [[-102, 68, 98, 58, -3, -54, -79, -74, -44, 0, 44, 74, 79, 54, 3, -58, -98, -68, 102]] * 2
    divisors = [29716] * 2


class SavitzkyGolayDeriv5Quint21(SavitzkyGolayNormalise):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 21"""

    matrices = [
        [
            -3876,
            1938,
            3468,
            2618,
            788,
            -1063,
            -2354,
            -2819,
            -2444,
            -1404,
            0,
            1404,
            2444,
            2819,
            2354,
            1063,
            -788,
            -2618,
            -3468,
            -1938,
            3876,
        ]
    ] * 2
    divisors = [1931540] * 2


class SavitzkyGolayDeriv5Quint23(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 23"""

    matrices = [
        [
            -209,
            76,
            171,
            152,
            77,
            -12,
            -87,
            -132,
            -141,
            -116,
            -65,
            0,
            65,
            116,
            141,
            132,
            87,
            12,
            -77,
            -152,
            -171,
            -76,
            209,
        ]
    ] * 2
    divisors = [170430] * 2


class SavitzkyGolayDeriv5Quint25(SavitzkyGolay):
    """Savitzky-Golay fifth quintic/sexic derivative operator of size 25"""

    matrices = [
        [
            -1012,
            253,
            748,
            753,
            488,
            119,
            -236,
            -501,
            -636,
            -631,
            -500,
            -275,
            0,
            275,
            500,
            631,
            636,
            501,
            236,
            -119,
            -488,
            -753,
            -748,
            -253,
            1012,
        ]
    ] * 2
    divisors = [1300650] * 2
