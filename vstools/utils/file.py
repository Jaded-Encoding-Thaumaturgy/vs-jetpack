from __future__ import annotations

from typing import TYPE_CHECKING

from jetpytools import CustomRuntimeError, SPath, SPathLike, add_script_path_hook, get_script_path

__all__ = ["PackageStorage"]


def _vspreview_script_path() -> SPath | None:
    # TODO: move to vspreview
    try:
        from vspreview import is_preview

        if is_preview():
            from vspreview.core import main_window

            return SPath(main_window().script_path)
    except ModuleNotFoundError:
        ...

    return None


add_script_path_hook(_vspreview_script_path)


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
        self.folder = cwd.get_folder() / self.BASE_FOLDER / package_name

    def ensure_folder(self) -> None:
        self.folder.mkdir(self.mode, True, True)

    def get_file(self, filename: SPathLike, *, ext: SPathLike | None = None) -> SPath:
        filename = SPath(filename)

        if ext:
            filename = filename.with_suffix(SPath(ext).suffix or str(ext))

        self.ensure_folder()

        return (self.folder / filename.name).resolve()
