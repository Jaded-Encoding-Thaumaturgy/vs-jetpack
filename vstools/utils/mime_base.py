from __future__ import annotations

from typing import TYPE_CHECKING

from jetpytools import CustomStrEnum

__all__ = ["FileTypeBase", "FileTypeIndex", "FileTypeIndexWithType"]

if TYPE_CHECKING:
    from .mime import FileType

    class FileTypeIndexBase:
        INDEX: FileTypeIndex
        """File type representing an indexing file."""

    class FileTypeBase(FileTypeIndexBase, CustomStrEnum):
        def __new__(cls, value_or_mime: str | FileType | None = None) -> FileType:
            """
            Instantiate the FileType with a string or mime ex video,index/video.
            """
            ...

    class FileTypeIndex(FileType):  # type: ignore
        def __call__(self, file_type: str | FileType) -> FileTypeIndexWithType:
            """
            Instantiate FileType.INDEX with its own sub-FileType.
            """
            ...

    class FileTypeIndexWithType(FileTypeIndex):  # type: ignore
        file_type: FileType
        """Sub-FileType that the index file indexes."""
else:
    FileTypeBase = CustomStrEnum
    FileTypeIndex = CustomStrEnum
    FileTypeIndexWithType = CustomStrEnum
