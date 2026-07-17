from pathlib import Path
from typing import Self

import pytest
from jetpytools import FileNotExistsError

from vsscale.mlrt.settings import (
    get_artifact_path,
    get_artifacts_folder,
    get_cache,
    get_global_cache,
    get_local_cache,
    get_model_folder,
    get_model_path,
    get_onnx_folder,
    get_toml_config,
    is_fallback_enabled,
    is_global_true,
)


class DummyFile:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None: ...


def test_is_global_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Global True
    with monkeypatch.context() as m:
        m.setenv("VSSCALE_GLOBAL", "true")
        assert is_global_true("onnx") is True

    # Specific True
    with monkeypatch.context() as m:
        m.setenv("VSSCALE_ONNX_GLOBAL", "1")
        assert is_global_true("onnx") is True

    # Global False
    with monkeypatch.context() as m:
        m.setenv("VSSCALE_GLOBAL", "false")
        assert is_global_true("onnx") is False

    # Empty env
    assert is_global_true("onnx") is False


def test_is_fallback_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Env Vars
    with monkeypatch.context() as m:
        m.setenv("VSSCALE_FALLBACK", "false")
        assert is_fallback_enabled("onnx") is False

    with monkeypatch.context() as m:
        m.setenv("VSSCALE_ONNX_FALLBACK", "0")
        assert is_fallback_enabled("onnx") is False

    with monkeypatch.context() as m:
        m.setenv("VSSCALE_FALLBACK", "true")
        assert is_fallback_enabled("onnx") is True

    # TOML configs
    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {"fallback": False})
        assert is_fallback_enabled("onnx") is False

    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {"fallback": True})
        assert is_fallback_enabled("onnx") is True

    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {"onnx": {"fallback": False}})
        assert is_fallback_enabled("onnx") is False

    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {"onnx": {"fallback": True}})
        assert is_fallback_enabled("onnx") is True

    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {})
        assert is_fallback_enabled("onnx") is True


def test_get_model_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    onnx_dir = tmp_path / "onnx"
    provider_dir = onnx_dir / "artcnn"
    provider_dir.mkdir(parents=True)

    monkeypatch.setattr("vsscale.mlrt.settings.get_onnx_folder", lambda **kwargs: onnx_dir)
    with pytest.raises(FileNotExistsError):
        get_model_folder("artcnn")

    v1 = provider_dir / "v1.0.0"
    v2 = provider_dir / "v2.0.0"
    v1.mkdir()
    v2.mkdir()

    assert get_model_folder("artcnn") == v2
    assert get_model_folder("artcnn", "v1.0.0") == v1

    with pytest.raises(FileNotExistsError):
        get_model_folder("artcnn", "v3.0.0")


def test_get_model_path_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_dir = tmp_path / "local"
    global_dir = tmp_path / "global"

    local_provider = local_dir / "artcnn" / "v1.0.0"
    global_provider = global_dir / "artcnn" / "v2.0.0"

    local_provider.mkdir(parents=True)
    global_provider.mkdir(parents=True)

    global_model = global_provider / "ArtCNN_R8F64.onnx"
    global_model.touch()

    monkeypatch.setattr(
        "vsscale.mlrt.settings.get_onnx_folder",
        lambda global_=False: global_dir if global_ else local_dir,
    )

    # Fallback enabled (default): should find the global model file (since local doesn't exist)
    path = get_model_path("artcnn", "ArtCNN_R8F64")
    assert path == global_model

    # Local exists: should find the local model file first
    local_model = local_provider / "ArtCNN_R8F64.onnx"
    local_model.touch()
    path = get_model_path("artcnn", "ArtCNN_R8F64")
    assert path == local_model
    local_model.unlink()

    # Fallback disabled: should return the default local path (which doesn't exist)
    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.is_fallback_enabled", lambda thing: False)
        path = get_model_path("artcnn", "ArtCNN_R8F64")
        assert path == local_provider / "ArtCNN_R8F64.onnx"
        assert not path.exists()


def test_get_artifact_path_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_dir = tmp_path / "local_art"
    global_dir = tmp_path / "global_art"

    local_dir.mkdir()
    global_dir.mkdir()

    global_file = global_dir / "test.engine"
    global_file.touch()

    monkeypatch.setattr(
        "vsscale.mlrt.settings.get_artifacts_folder",
        lambda global_=False: global_dir if global_ else local_dir,
    )

    # Fallback enabled (default) and fallback requested: finds global engine
    path = get_artifact_path("test.engine", fallback=True)
    assert path == global_file

    # Local exists: finds local engine first
    local_file = local_dir / "test.engine"
    local_file.touch()
    path = get_artifact_path("test.engine", fallback=True)
    assert path == local_file
    local_file.unlink()

    # Fallback parameter false: returns local path (which doesn't exist)
    path = get_artifact_path("test.engine", fallback=False)
    assert path == local_dir / "test.engine"

    # Fallback disabled in config: returns local path
    with monkeypatch.context() as m:
        m.setattr("vsscale.mlrt.settings.is_fallback_enabled", lambda thing: False)
        path = get_artifact_path("test.engine", fallback=True)
        assert path == local_dir / "test.engine"


def test_get_toml_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test when no config file exists
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", lambda self: False)
        assert get_toml_config() == {}

    # Test when vsjet.toml exists
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", lambda self: self.name == "vsjet.toml")
        m.setattr(Path, "open", lambda self, *args, **kwargs: DummyFile())
        m.setattr("tomllib.load", lambda f: {"vsscale": {"global": True}})

        assert get_toml_config() == {"global": True}


def test_cache_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    global_cache = get_global_cache()
    local_cache = get_local_cache()

    assert isinstance(global_cache, Path)
    assert isinstance(local_cache, Path)

    monkeypatch.setattr("vsscale.mlrt.settings.get_cache", lambda thing, global_=False: Path("/dummy"))
    assert get_onnx_folder() == Path("/dummy/onnx")
    assert get_artifacts_folder() == Path("/dummy/artifact")


def test_get_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    get_cache.cache_clear()

    dummy_global = Path("/dummy_global")
    dummy_local = Path("/dummy_local")

    monkeypatch.setattr("vsscale.mlrt.settings.get_global_cache", lambda: dummy_global)
    monkeypatch.setattr("vsscale.mlrt.settings.get_local_cache", lambda: dummy_local)

    # Test global_=True parameter
    get_cache.cache_clear()
    assert get_cache("onnx", global_=True) == dummy_global

    # Test is_global_true is True
    with monkeypatch.context() as m:
        get_cache.cache_clear()
        m.setattr("vsscale.mlrt.settings.is_global_true", lambda thing: True)
        assert get_cache("onnx") == dummy_global

    # Test get_toml_config specifies global=True
    with monkeypatch.context() as m:
        get_cache.cache_clear()
        m.setattr("vsscale.mlrt.settings.is_global_true", lambda thing: False)
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {"global": True})
        assert get_cache("onnx") == dummy_global

    # Test fallback to local cache
    with monkeypatch.context() as m:
        get_cache.cache_clear()
        m.setattr("vsscale.mlrt.settings.is_global_true", lambda thing: False)
        m.setattr("vsscale.mlrt.settings.get_toml_config", lambda: {})
        assert get_cache("onnx") == dummy_local
