from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vskernels import (
    Bicubic,
    BicubicAuto,
    Bilinear,
    BorderHandling,
    Lanczos,
    Point,
    Spline16,
    Spline36,
    Spline64,
    ZimgComplexKernel,
)
from vstools import core, vs

kernels = [Bicubic(), BicubicAuto(), Bilinear(), Lanczos(), Point(), Spline16(), Spline36(), Spline64()]


@pytest.mark.parametrize("kernel", kernels)
def test_blur(kernel: ZimgComplexKernel) -> None:
    clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
    clip_descaled = core.std.BlankClip(format=vs.YUV420P16, width=1600, height=900)

    with patch.object(kernel, "scale_function") as mock_scale_function:
        kernel.scale(clip, 1600, 900, blur=1.15)

        mock_scale_function.assert_called_once()
        assert mock_scale_function.call_args.kwargs["blur"] == 1.15

    with patch.object(kernel, "descale_function", new=MagicMock(return_value=clip_descaled)) as mock_descale_function:
        kernel.descale(clip, 1600, 900, blur=1.15)

        mock_descale_function.assert_called_once()
        assert mock_descale_function.call_args.kwargs["blur"] == 1.15


@pytest.mark.parametrize("kernel", kernels)
def test_force(kernel: ZimgComplexKernel) -> None:
    clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
    with patch.object(kernel, "scale_function") as mock_scale_function:
        kernel.scale(clip, 1600, 900, force=True)

        mock_scale_function.assert_called_once()
        assert mock_scale_function.call_args.kwargs["force"]

    with patch.object(kernel, "scale_function") as mock_scale_function:
        kernel.scale(clip, 1600, 900, force_h=True)

        mock_scale_function.assert_called_once()
        assert mock_scale_function.call_args.kwargs["force_h"]

    with patch.object(kernel, "scale_function") as mock_scale_function:
        kernel.scale(clip, 1600, 900, force_w=True)

        mock_scale_function.assert_called_once()
        assert mock_scale_function.call_args.kwargs["force_w"]


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("border_handling", set(BorderHandling))
def test_scale_border_handling(kernel: ZimgComplexKernel, border_handling: BorderHandling) -> None:
    clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
    with patch.object(kernel, "scale_function") as mock_scale_function:
        kernel.scale(clip, 3840, 2160, border_handling=border_handling)
        mock_scale_function.assert_called_once()
        assert "border_handling" not in mock_scale_function.call_args.kwargs


def test_add_init_kwargs_preserves_on_exception() -> None:
    kernel = Bicubic(blur=1.15)
    clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
    assert kernel.kwargs == {"blur": 1.15}

    with (
        patch.object(kernel, "scale_function", side_effect=RuntimeError("Transient error")),
        pytest.raises(RuntimeError, match="Transient error"),
    ):
        kernel.scale(clip, 1600, 900)

    assert kernel.kwargs == {"blur": 1.15}
