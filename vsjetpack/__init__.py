from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import vsaa as aa
    import vsdeband as deband
    import vsdehalo as dehalo
    import vsdeinterlace as deinterlace
    import vsdenoise as denoise
    import vsexprtools as expr  # Alias
    import vsexprtools as exprtools
    import vskernels as kernels
    import vsmasktools as mask  # Alias
    import vsmasktools as masktools
    import vsrgtools as rg  # Alias
    import vsrgtools as rgtools
    import vsscale as scale
    import vssource as source
    import vstools as tools

    from .logging import *

    __version__: str


def __getattr__(name: str) -> Any:
    from ._modules import JETPACK_MODULES_ALIASES

    if name in JETPACK_MODULES_ALIASES:
        from ._loader import lazy_load

        return lazy_load(JETPACK_MODULES_ALIASES[name])

    from .logging import setup_logging

    if name == "setup_logging":
        from .logging import setup_logging

        return setup_logging

    if name == "__version__":
        from importlib import import_module

        try:
            return import_module("._version", package=__package__).__version__
        except ModuleNotFoundError:
            return "unknown"

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
