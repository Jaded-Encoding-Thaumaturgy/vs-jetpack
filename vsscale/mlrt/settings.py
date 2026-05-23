import os
import sys
import tomllib
from functools import cache
from pathlib import Path

import platformdirs

from vstools import PackageStorage

APP_AUTHOR = "vsjet"
APP_NAME = "vsscale"
TRUTHY = frozenset({"1", "true", "yes", "on"})
ENV_KEYS = ("VSSCALE_LOCAL", "VSSCALE_{}_LOCAL")
TOML_CONFIG = ("vsjet.toml", "pyproject.toml")
TOML_KEYS: tuple[list[str], ...] = (["vsscale"], ["tool", "vsscale"])


@cache
def get_global_cache() -> Path:
    return platformdirs.user_cache_path(APP_NAME, APP_AUTHOR)


@cache
def get_local_cache() -> Path:
    return PackageStorage(package_name=f"{__name__}").folder


@cache
def get_cache(thing: str, *, local: bool = False) -> Path:
    if local or is_local_in_env(thing):
        return get_local_cache()

    if running_via_cli():
        return get_global_cache()

    for config_file, keys in zip(TOML_CONFIG, TOML_KEYS):
        file = Path(config_file).expanduser().resolve().absolute()
        if file.exists():
            with file.open("rb") as f:
                config = tomllib.load(f)

            for key in keys:
                config = config.get(key, {})

            if config.get("local", False):
                return get_local_cache()

    return get_global_cache()


@cache
def get_onnx_folder(*, local: bool = False) -> Path:
    r"""
    Linux: ~/.cache/vsscale/onnx
    macOS: ~/Library/Caches/vsscale/onnx
    Windows: ...\AppData\Local\vsjet\vsscale\Cache\onnx
    """
    return get_cache("onnx", local=local) / "onnx"


@cache
def get_engines_folder(*, local: bool = False) -> Path:
    r"""
    Linux: ~/.cache/vsscale/engines
    macOS: ~/Library/Caches/vsscale/engines
    Windows: ...\AppData\Local\vsjet\vsscale\Cache\engines
    """
    return get_cache("engine", local=local) / "engines"


def is_local_in_env(filetype: str) -> bool:
    return any(env.lower() in TRUTHY for env in (os.getenv(k.format(filetype), "") for k in ENV_KEYS))


def running_via_cli() -> bool:
    if Path(sys.argv[0]).stem == "vsscale":
        return True

    main_mod = sys.modules.get("__main__")
    return bool(main_mod and main_mod.__package__ == "vsscale")
