import sys
from importlib.util import LazyLoader, find_spec, module_from_spec
from types import ModuleType


def lazy_load(name: str, package: str | None = None) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]

    spec = find_spec(name, package)

    if spec is None:
        raise ModuleNotFoundError(f"No module named {name}", name=name, path=__file__)

    module = module_from_spec(spec)

    if spec.loader is None:
        raise NotImplementedError

    loader = LazyLoader(spec.loader)
    loader.exec_module(module)

    return module
