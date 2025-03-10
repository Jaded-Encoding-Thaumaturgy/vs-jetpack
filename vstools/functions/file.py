from __future__ import annotations

import inspect

from pathlib import Path
from typing import TYPE_CHECKING

from jetpytools import CustomRuntimeError, SPath, SPathLike, get_script_path

__all__ = [
    'PackageStorage'
]


class PackageStorage:
    BASE_FOLDER = SPath('.vsjet')

    def __init__(
        self, cwd: str | Path | SPath | None = None, *, mode: int = 0o777, package_name: str | None = None
    ) -> None:
        if not package_name:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])

            if module:
                package_name = module.__name__

            if not TYPE_CHECKING:
                frame = module = None

        if not package_name:
            raise CustomRuntimeError('Can\'t determine package name!')

        package_name = package_name.strip('.').split('.')[0]

        if not cwd:
            cwd = SPath(get_script_path())
        elif not isinstance(cwd, SPath):
            cwd = SPath(str(cwd))

        cwd = cwd.get_folder()
        base_folder = cwd / self.BASE_FOLDER

        for old_names in ('.vsstg', '.vsiew'):
            old_base_folder = (cwd / '.vsstg')

            if old_base_folder.exists():
                old_base_folder.move_dir(base_folder)

        old_folder = cwd / f'.{package_name}'
        new_folder = base_folder / package_name

        if old_folder.exists():
            old_folder.move_dir(new_folder)

        self.mode = mode
        self.folder = new_folder

    def ensure_folder(self) -> None:
        self.folder.mkdir(self.mode, True, True)

    def get_file(self, filename: SPathLike, *, ext: SPathLike | None = None) -> SPath:
        filename = SPath(filename)

        if ext:
            filename = filename.with_suffix(SPath(ext).suffix)

        self.ensure_folder()

        return (self.folder / filename.name).resolve()
