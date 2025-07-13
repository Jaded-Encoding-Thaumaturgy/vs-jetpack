from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from hashlib import md5
from os import name as os_name
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Literal, Protocol, Sequence

from vstools import (
    MISSING,
    ChromaLocationT,
    ColorRangeT,
    CustomRuntimeError,
    DataType,
    FieldBasedT,
    MatrixT,
    MissingT,
    PackageStorage,
    PrimariesT,
    SPath,
    SPathLike,
    TransferT,
    core,
    initialize_clip,
    inject_self,
    to_arr,
    vs,
)

from ..dataclasses import IndexFileType

if TYPE_CHECKING:
    from ..formats.dvd.parsedvd import IFOX, IFO0Title


PackageStorage(package_name="vssource")


__all__ = ["DVDExtIndexer", "DVDIndexer", "ExternalIndexer", "Indexer", "VSSourceFunc"]


class VSSourceFunc(Protocol):
    def __call__(self, path: DataType, *args: Any, **kwargs: Any) -> vs.VideoNode: ...


class Indexer(ABC):
    """
    Abstract indexer interface.
    """

    _source_func: ClassVar[Callable[..., vs.VideoNode]]

    def __init__(self, *, force: bool = True, **kwargs: Any) -> None:
        super().__init__()

        self.force = force
        self.indexer_kwargs = kwargs

    @classmethod
    def _split_lines(cls, buff: list[str]) -> tuple[list[str], list[str]]:
        return buff[: (split_idx := buff.index(""))], buff[split_idx + 1 :]

    @classmethod
    def get_joined_names(cls, files: list[SPath]) -> str:
        return "_".join([file.name for file in files])

    @classmethod
    def get_videos_hash(cls, files: list[SPath]) -> str:
        length = sum(file.stat().st_size for file in files)
        to_hash = length.to_bytes(32, "little") + cls.get_joined_names(files).encode()
        return md5(to_hash).hexdigest()

    @classmethod
    def source_func(cls, path: DataType | SPathLike, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return cls._source_func(str(path), *args, **kwargs)

    @classmethod
    def normalize_filenames(cls, file: SPathLike | Sequence[SPathLike]) -> list[SPath]:
        files = list[SPath]()

        for f in to_arr(file):
            if str(f).startswith("file:///"):
                f = str(f)[8::]

            files.append(SPath(f))

        return files

    def _source(
        self,
        clips: Iterable[vs.VideoNode],
        bits: int | None = None,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
    ) -> vs.VideoNode:
        clips = list(clips)

        clip = clips[0] if len(clips) == 1 else core.std.Splice(clips)

        return initialize_clip(clip, bits, matrix, transfer, primaries, chroma_location, color_range, field_based)

    @inject_self
    def source(
        self,
        file: SPathLike | Sequence[SPathLike],
        bits: int | None = None,
        *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return self._source(
            [self.source_func(f.to_str(), **self.indexer_kwargs | kwargs) for f in self.normalize_filenames(file)],
            bits,
            matrix,
            transfer,
            primaries,
            chroma_location,
            color_range,
            field_based,
        )


class ExternalIndexer(Indexer):
    _bin_path: ClassVar[str]
    _ext: ClassVar[str]

    _default_args: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        bin_path: SPathLike | MissingT = MISSING,
        ext: str | MissingT = MISSING,
        force: bool = True,
        default_out_folder: SPathLike | Literal[False] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(force=force, **kwargs)

        if bin_path is MISSING:
            bin_path = self._bin_path

        if ext is MISSING:
            ext = self._ext

        self.bin_path = SPath(bin_path)
        self.ext = ext
        self.default_out_folder = default_out_folder

    @abstractmethod
    def get_cmd(self, files: list[SPath], output: SPath) -> list[str]:
        """
        Returns the indexer command
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self, index_path: SPath, file_idx: int = 0) -> IndexFileType:
        """
        Returns info about the indexing file
        """
        raise NotImplementedError

    @abstractmethod
    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        raise NotImplementedError

    def _get_bin_path(self) -> SPath:
        if not (bin_path := shutil.which(str(self.bin_path))):
            raise FileNotFoundError(f"Indexer: `{self.bin_path}` was not found{' in PATH' if os_name == 'nt' else ''}!")
        return SPath(bin_path)

    def _run_index(self, files: list[SPath], output: SPath, cmd_args: Sequence[str]) -> None:
        output.mkdirp()

        proc = subprocess.Popen(
            list(map(str, (*self.get_cmd(files, output), *cmd_args, *self._default_args))),
            text=True,
            encoding="utf-8",
            shell=os_name == "nt",
            cwd=output.get_folder().to_str(),
        )

        status = proc.wait()

        if status:
            stderr = stdout = ""

            if proc.stderr:
                stderr = proc.stderr.read().strip()
                if stderr:
                    stderr = f"\n\t{stderr}"

            if proc.stdout:
                stdout = proc.stdout.read().strip()
                if stdout:
                    stdout = f"\n\t{stdout}"

            raise CustomRuntimeError(f"There was an error while running the {self.bin_path} command!: {stderr}{stdout}")

    def get_out_folder(
        self, output_folder: SPathLike | Literal[False] | None = None, file: SPath | None = None
    ) -> SPath:
        if output_folder is None:
            return SPath(file).get_folder() if file else self.get_out_folder(False)

        if not output_folder:
            return SPath(tempfile.gettempdir())

        return SPath(output_folder)

    def get_idx_file_path(self, path: SPath) -> SPath:
        return path.with_suffix(f".{self.ext}")

    def file_corrupted(self, index_path: SPath) -> None:
        if self.force:
            try:
                index_path.unlink()
            except OSError:
                raise CustomRuntimeError("Index file corrupted, tried to delete it and failed.", self.__class__)
        else:
            raise CustomRuntimeError("Index file corrupted! Delete it and retry.", self.__class__)

    def index(
        self,
        files: Sequence[SPath],
        force: bool = False,
        split_files: bool = False,
        output_folder: SPathLike | Literal[False] | None = None,
        *cmd_args: str,
    ) -> list[SPath]:
        if len(unique_folders := list({f.get_folder().to_str() for f in files})) > 1:
            return [
                c
                for s in (
                    self.index(
                        [f for f in files if f.get_folder().to_str() == folder], force, split_files, output_folder
                    )
                    for folder in unique_folders
                )
                for c in s
            ]

        dest_folder = self.get_out_folder(output_folder, files[0])

        files = sorted(set(files))

        hash_str = self.get_videos_hash(files)

        def _index(files: list[SPath], output: SPath) -> None:
            if output.is_file():
                if output.stat().st_size == 0 or force:
                    output.unlink()
                else:
                    return self.update_video_filenames(output, files)
            return self._run_index(files, output, cmd_args)

        if not split_files:
            output = self.get_video_idx_path(dest_folder, hash_str, "JOINED" if len(files) > 1 else "SINGLE")
            _index(files, output)
            return [output]

        outputs = [self.get_video_idx_path(dest_folder, hash_str, file.name) for file in files]

        for file, output in zip(files, outputs):
            _index([file], output)

        return outputs

    def get_video_idx_path(self, folder: SPath, file_hash: str, video_name: SPathLike) -> SPath:
        vid_name = SPath(video_name).stem
        current_indxer = os.path.basename(self._bin_path)
        filename = "_".join([file_hash, vid_name, current_indxer])

        return self.get_idx_file_path(PackageStorage(folder).get_file(filename))

    @inject_self
    def source(
        self,
        file: SPathLike | Sequence[SPathLike],
        bits: int | None = None,
        *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        index_files = self.index(self.normalize_filenames(file))

        return self._source(
            (self.source_func(idx_filename.to_str(), **kwargs) for idx_filename in index_files),
            bits,
            matrix,
            transfer,
            primaries,
            chroma_location,
            color_range,
            field_based,
        )


class DVDIndexer:
    iso_path: SPath

    def parse_vts(
        self,
        title: IFO0Title,
        disable_rff: bool,
        vobidcellids_to_take: list[tuple[int, int]],
        target_vts: IFOX,
        output_folder: SPath,
        vob_input_files: Sequence[SPath],
    ) -> tuple[vs.VideoNode, list[int], list[tuple[int, int]], list[int]]:
        raise NotImplementedError


class DVDExtIndexer(ExternalIndexer, DVDIndexer): ...
