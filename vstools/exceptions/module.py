from __future__ import annotations

from typing import Any

from jetpytools import CustomImportError, DependencyNotFoundError, FuncExceptT, SupportsString

__all__ = ["CustomImportError", "DependencyNotFoundError", "OutdatedPluginError"]


class OutdatedPluginError(CustomImportError):
    """
    Raised when a plugin is outdated and needs to be updated.
    """

    def __init__(
        self,
        func: FuncExceptT,
        package: str | ImportError,
        message: SupportsString = "Plugin '{package}' version is too old. Please update to a more recent version.",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, package, message, **kwargs)
