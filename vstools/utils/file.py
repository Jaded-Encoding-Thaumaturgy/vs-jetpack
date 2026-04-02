from __future__ import annotations

from logging import getLogger

from jetpytools import CustomRuntimeError, SPath, SPathLike, get_script_path

__all__ = ["PackageStorage"]


logger = getLogger(__name__)


class PackageStorage:
    BASE_FOLDER = SPath(".vsjet")

    def __init__(self, cwd: SPathLike | None = None, *, package_name: str, mode: int = 0o777) -> None:
        package_name = package_name.strip(".").split(".")[0]

        if cwd is not None:
            cwd = SPath(cwd)
        else:
            try:
                cwd = get_script_path()
            except CustomRuntimeError:
                cwd = SPath.cwd()

        self.mode = mode
        self.folder = cwd.get_folder() / self.BASE_FOLDER / package_name

        logger.debug("PackageStorage(%s) initialized in the %r directory ", package_name, self.folder)

    def ensure_folder(self) -> None:
        self.folder.mkdir(self.mode, True, True)

    def get_file(self, filename: SPathLike, *, ext: SPathLike | None = None) -> SPath:
        filename = SPath(filename)

        if ext:
            filename = filename.with_suffix(SPath(ext).suffix or str(ext))

        self.ensure_folder()

        return (self.folder / filename.name).resolve()
