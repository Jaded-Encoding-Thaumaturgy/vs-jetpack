from __future__ import annotations

from jetpytools import SPath, SPathLike, get_script_path

__all__ = ["PackageStorage"]


class PackageStorage:
    BASE_FOLDER = SPath(".vsjet")

    def __init__(self, cwd: SPathLike | None = None, *, package_name: str, mode: int = 0o777) -> None:
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
