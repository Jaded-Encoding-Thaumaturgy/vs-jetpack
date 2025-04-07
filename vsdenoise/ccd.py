"""
This module contains a CCD implementation
"""

from __future__ import annotations

from math import sin, sqrt
from typing import Any

from vsaa import Nnedi3
from vsexprtools import complexpr_available, norm_expr
from vskernels import Bicubic, Point
from vsscale import SSIM
from vstools import (
    EXPR_VARS, CustomIndexError, CustomIntEnum, InvalidColorFamilyError, Matrix, MatrixT, PlanesT,
    UnsupportedSubsamplingError, check_ref_clip, fallback, get_peak_value, join, normalize_planes, plane, shift_clip,
    split, vs
)

__all__ = [
    'ccd', 'CCDMode', 'CCDPoints'
]


class CCDMode(CustomIntEnum):
    """Processing mode for CCD."""

    CHROMA_ONLY = 0
    """Only process chroma."""

    BICUBIC_CHROMA = 1
    """Process in 4:4:4, downscaling the luma to chroma size."""

    BICUBIC_LUMA = 2
    """Process in 4:4:4, upscaling the chroma to luma size with :py:class:`Bicubic`."""

    NNEDI_BICUBIC = 3
    """
    Process in 4:4:4, upscaling the chroma to luma size with :py:class:`NNedi3`,
    finally downscaling it to the original size with :py:class:`Bicubic`.
    """

    NNEDI_SSIM = 4
    """
    Process in 4:4:4, upscaling the chroma to luma size with :py:class:`NNedi3`,
    finally downscaling it to original size with :py:class:`SSIM`.
    """


class CCDPoints(CustomIntEnum):
    """
    Sample points of reference taken into account when processing with CCD.

    Graph of all the points:

    x => center pixel
    ^ => CCDPoints.LOW
    ' => CCDPoints.MEDIUM
    ° => CCDPoints.HIGH

    °     °     °     °
       '     '     '
    °     ^     ^     °
       '     x     '
    °     ^     ^     °
       '     '     '
    °     °     °     °
    """

    LOW = 11
    """
    Vertices of the square with l = scale * 4.\n
    ^ in the main docstrings.
    """

    MEDIUM = 22
    """
    Vertices and middle points of the sides of the square with l = scale * 8.\n
    ' in the main docstrings.
    """

    HIGH = 44
    """
    Vertices, 2/3 and 3/4 points of the sides of the square with l = scale * 12.\n
    ' in the main docstrings.
    """

    ALL = 63
    """All points combined."""


def ccd(
    src: vs.VideoNode, thr: float = 4, tr: int = 0, ref: vs.VideoNode | None = None,
    mode: int | CCDMode | None = None, scale: float | None = None, matrix: MatrixT | None = None,
    ref_points: int | CCDPoints | None = CCDPoints.LOW | CCDPoints.MEDIUM,
    i444: bool = False, planes: PlanesT = None, **ssim_kwargs: Any
) -> vs.VideoNode:
    """
    Camcorder Color Denoise is an original VirtualDub filter made by Sergey Stolyarevsky.
    It's a chroma denoiser that works great on old sources such as VHSes and DVDs.

    It works as a convolution of near pixels determined by ``ref_points``.
    If the euclidean distance between the RGB values of the center pixel and a given pixel in the convolution
    matrix is less than the threshold, then this pixel is considered in the average.

    :param src:         Source clip.
    :param thr:         Euclidean distance threshold for including pixel in the matrix.
                        Higher values results in stronger denoising.
    :param tr:          Temporal radius of the processing.
    :param ref:         Ref clip to use for calculating the processing to perform on the main clip.
    :param mode:        Processing mode for CCD. See :py:attr:`vsdenoise.ccd.CCDMode`.
    :param scale:       Relative scale of the analyzed matrix of points decided by ``ref_points``.
    :param matrix:      Enum for the matrix of the Clip to process.
                        See :py:attr:`vstools.enums.color.Matrix` for more info.
                        If `None`, gets matrix from the "_Matrix" prop of the clip unless it's an RGB clip,
                        in which case it stays as `None`.
    :param ref_points:  Sample points of reference for processing.
                        See :py:attr:`vsdenoise.ccd.CCDPoints`.
    :param i444:        Output the clip as 4:4:4.
    :param planes:      Planes to process.
    :param ssim_kwargs: Keyword arguments to pass to :py:class:`vsscale.SSIM`.

    :return:            Denoised clip.
    """

    assert src.format

    check_ref_clip(src, ref)

    InvalidColorFamilyError.check(src, (vs.YUV, vs.RGB), ccd)

    if complexpr_available:
        if tr < 0 or tr > 3:
            raise CustomIndexError('Temporal radius must be between 0 and 3 (inclusive)!', ccd, tr)
        elif tr > src.num_frames // 2:
            raise CustomIndexError('Temporal radius must be less than half of the clip length!', ccd, tr)
    elif tr < 0:
        raise CustomIndexError('Temporal radius must be more than 0!', ccd, tr)

    is_yuv = src.format.color_family is vs.YUV
    is_subsampled = src.format.subsampling_h or src.format.subsampling_w

    if mode is not None and not is_subsampled:
        raise UnsupportedSubsamplingError(f'{mode} is available only for subsampled video!', ccd)

    mode = fallback(CCDMode.from_param(mode), CCDMode.CHROMA_ONLY)
    if not isinstance(ref_points, int):
        ref_points = fallback(CCDPoints.from_param(ref_points), CCDPoints.MEDIUM).value

    src_width, src_height = src.width, src.height
    src444_format = src.format.replace(subsampling_w=0, subsampling_h=0)

    if planes is None and mode in {CCDMode.CHROMA_ONLY, CCDMode.BICUBIC_CHROMA}:
        planes = [1, 2]

    planes = normalize_planes(src, planes)

    def _ccd_expr(src: vs.VideoNode, rgb: vs.VideoNode) -> vs.VideoNode:
        nonlocal scale

        rgb_clips = [
            vs.core.std.ShufflePlanes([rgb, rgb, rgb], [i, i, i], vs.RGB) for i in range(3)
        ]

        peak = get_peak_value(src, False)

        thrs = thr ** 2 / (255 ** 2 * 3)

        expr_clips = [src, *rgb_clips]

        for i in range(1, tr + 1):
            for clip in rgb_clips:
                expr_clips.extend([shift_clip(clip, -i), shift_clip(clip, i)])

        if not scale or scale <= 0:
            scale = 1.0
        elif scale == 1:
            scale = src_height / 240

        l_d, m_d, h_d = round(scale * 4), round(scale * 8), round(scale * 12)

        low_points = {
            'F': (-l_d, -l_d), 'G': (+l_d, -l_d),
            'J': (-l_d, +l_d), 'K': (+l_d, +l_d),
        }

        med_points = {
            'Q': (-m_d, -m_d), 'R': (0, -m_d), 'S': (+m_d, -m_d),
            'T': (-m_d, 0), '                   U': (+m_d, 0),
            'V': (-m_d, +m_d), 'W': (0, +m_d), 'X': (+m_d, +m_d),
        }

        high_points = {
            'A': (-h_d, -h_d), 'B': (-l_d, -h_d), 'C': (+l_d, -h_d), 'D': (+h_d, -h_d),
            'E': (-h_d, -l_d), '                                      H': (+h_d, -l_d),
            'I': (-h_d, +l_d), '                                      L': (+h_d, +l_d),
            'M': (-h_d, +h_d), 'N': (-l_d, +h_d), 'O': (+l_d, +h_d), 'P': (+h_d, +h_d),
        }

        if ref_points == CCDPoints.ALL:
            expr_points = low_points | med_points | high_points
        elif ref_points == (CCDPoints.LOW | CCDPoints.HIGH):
            expr_points = low_points | high_points
        elif ref_points == (CCDPoints.MEDIUM | CCDPoints.HIGH):
            expr_points = med_points | high_points
        else:
            expr_points = low_points | med_points

        tr_nclips = tr * 2 + 1
        num_points = len(expr_points.keys())

        plusses_plane = '+ ' * (tr_nclips - 1)
        plusses_points = '+ ' * (num_points - 1)

        def _get_weight_expr(x: int, y: int, c: str, weight: float | None = None) -> str:
            scale_str = peak != 1 and f'{peak} / ' or ''
            weigth_str = weight is not None and f'{weight_b} *' or ''

            return f'{c}[{x},{y}] {c} - {scale_str} 2 pow {weigth_str}'

        expression = list[str]()

        for char, (x, y) in expr_points.items():
            char = char.strip()
            rgb_expr = []

            for i, c in enumerate(EXPR_VARS[1:4], 1):
                rgb_expr.append(_get_weight_expr(x, y, c))

                if tr:
                    for j in range(0, tr):
                        offset = i + 3 + j * 6
                        bc, fc = EXPR_VARS[offset], EXPR_VARS[offset + 1]
                        weight_f, weight_b = sqrt((4 - j) / 8), sin((5 - j) / 8)
                        rgb_expr.append(_get_weight_expr(x, y, bc, weight_b))
                        rgb_expr.append(_get_weight_expr(x, y, fc, weight_f))

                    rgb_expr.append(f'{plusses_plane} {tr_nclips} /')

            expression.append(f"{' '.join(rgb_expr)} + + {char}!")

        for char in expr_points:
            expression.append(f'{char}@ {thrs} < 1 0 ?')

        expression.append(f'{plusses_points} 1 + WQ!')

        for char, (x, y) in expr_points.items():
            expression.append(f'{char}@ {thrs} < x[{x},{y}] 0 ?')

        expression.append(f'{plusses_points} x + WQ@ /')

        return norm_expr(expr_clips, expression, planes, src444_format, func=ccd)

    if not is_yuv:
        return _ccd_expr(ref or src, src)

    if matrix is None:
        matrix = Matrix.from_video(src, True)

    divw, divh = 1 << src.format.subsampling_w, 1 << src.format.subsampling_h

    if mode == CCDMode.BICUBIC_LUMA or not is_subsampled:
        yuvw, yuvh = src_width, src_height
        src_left = 0.0
    else:
        yuvw, yuvh = src_width // divw, src_height // divh

        src_left = 0.25 - 0.25 * divw

    yuv = yuvref = None

    if not is_subsampled:
        yuv, yuvref = src, ref
    elif mode in {CCDMode.NNEDI_BICUBIC, CCDMode.NNEDI_SSIM}:
        ref_clips = list[list[vs.VideoNode] | None]([split(src), ref and split(ref) or None])

        src_left += 0.125 * divw

        yuv, yuvref = [
            join(planes[:1] + [
                Nnedi3().scale(p, p.width * divw, p.height * divh) for p in planes[1:]
            ]) if planes else None for planes in ref_clips
        ]
    else:
        yuv = Bicubic.scale(src, yuvw, yuvh, format=src444_format)
        yuvref = ref and Bicubic.scale(ref, yuvw, yuvh, format=src444_format)

    assert yuv and yuv.format

    rgb = Point.resample(yuv, yuv.format.replace(color_family=vs.RGB), None, matrix)

    denoised = _ccd_expr(yuvref or yuv, rgb)

    down_format = src444_format

    if not i444:
        if mode == CCDMode.NNEDI_BICUBIC:
            down_format = src.format
        elif mode == CCDMode.NNEDI_SSIM:
            down_format = down_format.replace(sample_type=vs.FLOAT, bits_per_sample=32)

    denoised = Bicubic.resample(denoised, down_format, src_left=src_left)

    if not is_subsampled and 0 in planes:
        return denoised

    if mode == CCDMode.NNEDI_SSIM and not i444:
        u = SSIM.scale(plane(denoised, 1), yuvw, yuvh, **ssim_kwargs)
        v = SSIM.scale(plane(denoised, 2), yuvw, yuvh, **ssim_kwargs)

        denoised = join(denoised if 0 in planes else src, u, v, vs.YUV)
    else:
        denoised = join(src, denoised, vs.YUV)

    return denoised if i444 else Bicubic.resample(denoised, src.format)
