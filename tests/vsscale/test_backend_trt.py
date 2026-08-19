from pathlib import Path

import pytest

from vsscale.mlrt.backend.trt import TRT


def test_trt_skips_fp16_conversion_for_preconverted_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model_fp16.onnx"
    model.write_bytes(b"model")
    engine = tmp_path / "model.engine"
    converted = False

    def convert_fp16(self: TRT, network_path: Path) -> Path:
        nonlocal converted
        converted = True
        return network_path

    monkeypatch.setattr(TRT, "_convert_onnx_fp16", convert_fp16)
    monkeypatch.setattr(TRT, "get_identity", lambda self, *args: 1)
    monkeypatch.setattr(
        "vsscale.mlrt.backend.trt.get_artifact_path",
        lambda *args, **kwargs: engine,
    )
    monkeypatch.setattr(TRT, "build", lambda self, **kwargs: engine.write_bytes(b"engine" * 512))

    backend = TRT(fp16=True, skip_fp16_conversion=True)
    assert backend.build_engine(model, channels=3, tilesize=(64, 64)) == engine
    assert converted is False
