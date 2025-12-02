from collections.abc import MutableMapping
from logging import Logger
from os import PathLike
from typing import IO, Any, Final

from mkdocs.config import Config
from mkdocs.structure.files import Files

from .editor import FilesEditor
from .nav import Nav as Nav

__version__: str
log: Logger

config: Config
directory: str
edit_paths: MutableMapping[str, str | None]

def open(
    name: str | PathLike[str], mode: str, buffering: int = -1, encoding: str | None = None, *args: Any, **kwargs: Any
) -> IO[Any]: ...
def _get_file(name: str, new: bool = False) -> str: ...
def set_edit_path(name: str | PathLike[str], edit_name: str | PathLike[str] | None) -> None: ...
def current() -> FilesEditor: ...

files: Final[Files]
