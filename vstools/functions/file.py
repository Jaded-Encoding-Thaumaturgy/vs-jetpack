from __future__ import annotations

from typing import TYPE_CHECKING

from jetpytools import CustomRuntimeError, SPath, SPathLike, get_script_path

__all__ = ["PackageStorage"]


class PackageStorage:
    BASE_FOLDER = SPath(".vsjet")

    def __init__(self, cwd: SPathLike | None = None, *, mode: int = 0o777, package_name: str | None = None) -> None:
        if not package_name:
            from inspect import getmodule, stack

            frame = stack()[1]
            module = getmodule(frame[0])

            if module:
                package_name = module.__name__

            if not TYPE_CHECKING:
                frame = module = None

        if not package_name:
            raise CustomRuntimeError("Can't determine package name!")

        package_name = package_name.strip(".").split(".")[0]

        if not cwd:
            cwd = get_script_path()
        elif not isinstance(cwd, SPath):
            cwd = SPath(cwd)

        self.mode = mode
        self.folder = cwd / self.BASE_FOLDER / package_name

    def ensure_folder(self) -> None:
        self.folder.mkdir(self.mode, True, True)

    def get_file(self, filename: SPathLike, *, ext: SPathLike | None = None) -> SPath:
        filename = SPath(filename)

        if ext:
            filename = filename.with_suffix(SPath(ext).suffix)

        self.ensure_folder()

        return (self.folder / filename.name).resolve()
