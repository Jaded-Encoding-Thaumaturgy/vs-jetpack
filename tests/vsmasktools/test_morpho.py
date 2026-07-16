import itertools

from vsexprtools import ExprOp, norm_expr
from vsmasktools import Morpho
from vstools import ConvMode, core, vs


def test_morpho_equivalence() -> None:
    clip = core.std.BlankClip(format=vs.GRAY16, width=64, height=64, length=1).akarin.Expr("X Y + 500 *")

    morpho = Morpho()

    dilated_r3_it1 = morpho.dilation(clip, radius=3, iterations=1)
    dilated_r1_it3 = morpho.dilation(clip, radius=1, iterations=3)

    # They should produce the exact same output since thr/multiply/coords are None
    f_opt, f_ref = dilated_r3_it1.get_frame(0), dilated_r1_it3.get_frame(0)
    for y, x in itertools.product(range(clip.height), range(clip.width)):
        assert f_opt[0][y, x] == f_ref[0][y, x]

    eroded_r3_it1 = morpho.erosion(clip, radius=3, iterations=1)
    eroded_r1_it3 = morpho.erosion(clip, radius=1, iterations=3)

    for f_opt, f_ref in zip(eroded_r3_it1.frames(close=True), eroded_r1_it3.frames(close=True)):
        for y, x in itertools.product(range(clip.height), range(clip.width)):
            assert f_opt[0][y, x] == f_ref[0][y, x]


def test_morpho_equivalence_with_thr() -> None:
    clip = core.std.BlankClip(format=vs.GRAY16, width=64, height=64, length=1).akarin.Expr("X Y + 500 *")
    morpho = Morpho()

    # Dilation with thr=0.05
    opt = morpho.dilation(clip, radius=3, iterations=1, thr=0.05)
    ref = norm_expr(
        clip,
        morpho._morpho_xx_imum(
            clip, (3, ConvMode.SQUARE), 0.05, None, None, False, op=ExprOp.MAX, func=morpho.dilation
        ),
    )

    for f_opt, f_ref in zip(opt.frames(close=True), ref.frames(close=True)):
        for y, x in itertools.product(range(clip.height), range(clip.width)):
            assert f_opt[0][y, x] == f_ref[0][y, x]


def test_morpho_equivalence_with_multiply() -> None:
    clip = core.std.BlankClip(format=vs.GRAY16, width=64, height=64, length=1).akarin.Expr("X Y + 500 *")
    morpho = Morpho()

    # Dilation with multiply=1.5
    opt = morpho.dilation(clip, radius=3, iterations=1, multiply=1.5)
    ref = norm_expr(
        clip,
        morpho._morpho_xx_imum(clip, (3, ConvMode.SQUARE), None, None, 1.5, False, op=ExprOp.MAX, func=morpho.dilation),
    )

    for f_opt, f_ref in zip(opt.frames(close=True), ref.frames(close=True)):
        for y, x in itertools.product(range(clip.height), range(clip.width)):
            assert f_opt[0][y, x] == f_ref[0][y, x]
