import os
import tomllib
from contextlib import suppress
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Any

from jetpytools import FileNotExistsError

from vstools import PackageStorage

APP_AUTHOR = "vsjet"
APP_NAME = "vsscale"
TRUTHY = frozenset({"1", "true", "yes", "on"})
FALSY = frozenset({"0", "false", "no", "off"})
ENV_KEYS = ("VSSCALE_GLOBAL", "VSSCALE_{}_GLOBAL")
TOML_CONFIG = ("vsjet.toml", "pyproject.toml")
TOML_KEYS: tuple[list[str], ...] = (["vsscale"], ["tool", "vsscale"])

logger = getLogger(__name__)


def get_toml_config() -> dict[str, Any]:
    for config_file, keys in zip(TOML_CONFIG, TOML_KEYS):
        file = Path(config_file).expanduser().resolve().absolute()
        if file.exists():
            with file.open("rb") as f:
                config = tomllib.load(f)

            for key in keys:
                config = config.get(key, {})
            return config

    return {}


@cache
def get_global_cache() -> Path:
    import platformdirs

    return platformdirs.user_cache_path(APP_NAME, APP_AUTHOR)


@cache
def get_local_cache() -> Path:
    return PackageStorage(package_name=f"{__name__}").folder


@cache
def get_cache(thing: str, *, global_: bool = False) -> Path:
    if global_ or is_global_true(thing):
        return get_global_cache()

    if get_toml_config().get("global", False):
        return get_global_cache()

    return get_local_cache()


@cache
def get_onnx_folder(*, global_: bool = False) -> Path:
    r"""
    If `global_=True`:
        - Linux: ~/.cache/vsscale/onnx
        - macOS: ~/Library/Caches/vsscale/onnx
        - Windows: ...\AppData\Local\vsjet\vsscale\Cache\onnx

    Else returns a local `.vsjet` package storage.
    """
    return get_cache("onnx", global_=global_) / "onnx"


@cache
def get_artifacts_folder(*, global_: bool = False) -> Path:
    r"""
    If `global_=True`:
        - Linux: ~/.cache/vsscale/artifact
        - macOS: ~/Library/Caches/vsscale/artifact
        - Windows: ...\AppData\Local\vsjet\vsscale\Cache\artifact

    Else returns a local `.vsjet` package storage.
    """
    return get_cache("artifact", global_=global_) / "artifact"


def is_global_true(filetype: str) -> bool:
    return any(env.lower() in TRUTHY for env in (os.getenv(k.format(filetype.upper()), "") for k in ENV_KEYS))


def is_fallback_enabled(thing: str) -> bool:
    keys = ["VSSCALE_FALLBACK", f"VSSCALE_{thing.upper()}_FALLBACK"]

    for k in keys:
        if val := os.getenv(k, ""):
            return val.lower() not in FALSY

    config = get_toml_config()
    if (val := config.get("fallback")) is not None:
        return bool(val)

    if isinstance(val := config.get(thing, {}), dict) and "fallback" in val:
        return bool(val["fallback"])

    return True


def get_model_folder(provider: str, version: str | None = None, *, global_: bool = False) -> Path:
    folder = get_onnx_folder(global_=global_) / provider.lower()

    if version is None:
        latest = sorted(folder.glob("*"), reverse=True)

        if not latest:
            raise FileNotExistsError("The folder doesn't exist", get_model_folder)

        return latest[0]

    folder /= version

    if not folder.exists():
        raise FileNotExistsError("The folder doesn't exist", get_model_folder)

    return folder


def get_model_path(provider: str, model_name: str, version: str | None = None) -> Path:
    with suppress(FileNotExistsError):
        if (path := get_model_folder(provider, version) / f"{model_name}.onnx").exists():
            logger.info("Found model %s at %s", model_name, path)
            return path

    if is_fallback_enabled("onnx") and get_onnx_folder() != get_onnx_folder(global_=True):
        with suppress(FileNotExistsError):
            if (path := get_model_folder(provider, version, global_=True) / f"{model_name}.onnx").exists():
                logger.info("Found model %s in global fallback %s", model_name, path)
                return path

    default_path = get_model_folder(provider, version) / f"{model_name}.onnx"
    logger.info("Model %s not found. Returning default path: %s", model_name, default_path)
    return default_path


def get_artifact_path(filename: str, *, fallback: bool = True) -> Path:
    dirname = get_artifacts_folder()
    path = dirname / filename

    if path.exists():
        logger.info("Found artifact %s at %s", filename, path)
        return path

    if fallback and is_fallback_enabled("artifact") and dirname != (g_dirname := get_artifacts_folder(global_=True)):
        g_path = g_dirname / filename
        if g_path.exists():
            logger.info("Found artifact %s in global fallback %s", filename, g_path)
            return g_path

    logger.info("Artifact %s not found. Returning default path: %s", filename, path)
    return path
