import os
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

import platformdirs
from jetpytools import FileNotExistsError

from vstools import PackageStorage

APP_AUTHOR = "vsjet"
APP_NAME = "vsscale"
TRUTHY = frozenset({"1", "true", "yes", "on"})
ENV_KEYS = ("VSSCALE_GLOBAL", "VSSCALE_{}_GLOBAL")
TOML_CONFIG = ("vsjet.toml", "pyproject.toml")
TOML_KEYS: tuple[list[str], ...] = (["vsscale"], ["tool", "vsscale"])


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
    return platformdirs.user_cache_path(APP_NAME, APP_AUTHOR)


@cache
def get_local_cache() -> Path:
    return PackageStorage(package_name=f"{__name__}").folder


@cache
def get_cache(thing: str, *, global_: bool = False) -> Path:
    if global_ or is_global_in_env(thing):
        return get_global_cache()

    if get_toml_config().get("global", False):
        return get_global_cache()

    return get_local_cache()


@cache
def get_provider_folder(*, global_: bool = False) -> Path:
    r"""
    Linux: ~/.cache/vsscale/provider

    macOS: ~/Library/Caches/vsscale/provider

    Windows: ...\AppData\Local\vsjet\vsscale\Cache\provider
    """
    return get_cache("provider", global_=global_) / "provider"


@cache
def get_artifacts_folder(*, global_: bool = False) -> Path:
    r"""
    Linux: ~/.cache/vsscale/artifact

    macOS: ~/Library/Caches/vsscale/artifact

    Windows: ...\AppData\Local\vsjet\vsscale\Cache\artifact
    """
    return get_cache("artifact", global_=global_) / "artifact"


def is_global_in_env(filetype: str) -> bool:
    return any(env.lower() in TRUTHY for env in (os.getenv(k.format(filetype), "") for k in ENV_KEYS))


def get_model_folder(provider: str, version: str | None = None) -> Path:
    folder = get_provider_folder() / provider.lower()

    if version is None:
        latest = sorted(folder.glob("*"), reverse=True)

        if not latest:
            raise FileNotExistsError("The folder doesn't exist", get_model_folder)

        return latest[0]

    folder /= version

    if not folder.exists():
        raise FileNotExistsError("The folder doesn't exist", get_model_folder)

    return folder
