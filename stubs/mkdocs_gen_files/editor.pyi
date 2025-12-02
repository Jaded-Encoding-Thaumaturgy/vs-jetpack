from collections.abc import MutableMapping
from os import PathLike
from typing import IO, Any, Self

from mkdocs.config import Config as Config
from mkdocs.structure.files import File, Files

def file_sort_key(f: File) -> tuple[str, ...]: ...

class FilesEditor:
    config: Config
    directory: str
    edit_paths: MutableMapping[str, str | None]
    def open(
        self,
        name: str | PathLike[str],
        mode: str,
        buffering: int = -1,
        encoding: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> IO[Any]: ...
    def set_edit_path(self, name: str, edit_name: str | None) -> None: ...
    def __init__(self, files: Files, config: Config, directory: str | None = None) -> None: ...
    @classmethod
    def current(cls) -> FilesEditor: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> None: ...
    @property
    def files(self) -> Files: ...
