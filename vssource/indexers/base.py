from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from logging import getLogger
from os import name as os_name
from typing import Any, ClassVar, Literal, Protocol, Self

from jetpytools import (
    MISSING,
    CustomRuntimeError,
    CustomValueError,
    FuncExcept,
    SPath,
    SPathLike,
    get_subclasses,
    inject_self,
    to_arr,
)

from vstools import (
    ChromaLocation,
    ColorRangeLike,
    FieldBasedLike,
    MatrixLike,
    MissingT,
    PackageStorage,
    PrimariesLike,
    TransferLike,
    core,
    initialize_clip,
    match_clip,
    vs,
)

from ..dataclasses import IndexFileType

__all__ = ["CacheIndexer", "ExternalIndexer", "Indexer", "IndexerLike", "VSSourceFunc"]

log = getLogger(__name__)


def _base_from_param[IndexerT: Indexer](
    cls: type[IndexerT], value: str | type[IndexerT] | IndexerT | None, func_except: FuncExcept | None = None
) -> type[IndexerT]:
    # If value is an instance returns the class
    if isinstance(value, cls):
        return value.__class__

    # If value is a type and a subclass of the caller returns the value itself
    if isinstance(value, type) and issubclass(value, cls):
        return value

    # Search for the subclasses of the caller and the caller itself
    # + plugin namespace
    if isinstance(value, str):
        all_indexers = dict[str, type[IndexerT]]()

        for s in [*get_subclasses(cls), cls]:
            all_indexers[s.__name__.lower()] = s

            source_func = getattr(s, "_source_func", None)
            plugin = getattr(source_func, "plugin", None)
            plugin_ns = getattr(plugin, "namespace", None)

            if plugin_ns:
                all_indexers[plugin_ns] = s

        try:
            return all_indexers[value.lower().strip()]
        except KeyError:
            raise CustomValueError("Unknown indexer", func_except or cls.from_param, value) from None

    if value is None:
        return cls

    raise CustomValueError("Unknown indexer", func_except or cls.from_param, value)


def _base_ensure_obj[IndexerT: Indexer](
    cls: type[IndexerT],
    value: str | type[IndexerT] | IndexerT | None,
    func_except: FuncExcept | None = None,
) -> IndexerT:
    if isinstance(value, cls):
        return value

    return cls.from_param(value, func_except)()


class VSSourceFunc(Protocol):
    def __call__(self, path: str | bytes | bytearray, *args: Any, **kwargs: Any) -> vs.VideoNode: ...


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
    def from_param(
        cls, indexer: str | type[Self] | Self | None = None, /, func_except: FuncExcept | None = None
    ) -> type[Self]:
        """
        Resolve and return an Indexer type from a given input (string, type, or instance).

        Args:
            indexer: Indexer identifier (string, class, or instance). Plugin namespace is also supported.
            func_except: Function returned for custom error handling.

        Returns:
            Resolved indexer type.
        """
        return _base_from_param(cls, indexer, func_except)

    @classmethod
    def ensure_obj(
        cls, indexer: str | type[Self] | Self | None = None, /, func_except: FuncExcept | None = None
    ) -> Self:
        """
        Ensure that the input is a indexer instance, resolving it if necessary.

        Args:
            indexer: Indexer identifier (string, class, or instance). Plugin namespace is also supported.
            func_except: Function returned for custom error handling.

        Returns:
            Indexer instance.
        """
        return _base_ensure_obj(cls, indexer, func_except)

    @classmethod
    def _split_lines(cls, buff: list[str]) -> tuple[list[str], list[str]]:
        return buff[: (split_idx := buff.index(""))], buff[split_idx + 1 :]

    @classmethod
    def get_joined_names(cls, files: list[SPath]) -> str:
        return "_".join([file.name for file in files])

    @classmethod
    def get_videos_hash(cls, files: list[SPath]) -> str:
        from hashlib import md5

        length = sum(file.stat().st_size for file in files)
        to_hash = length.to_bytes(32, "little") + cls.get_joined_names(files).encode()
        return md5(to_hash).hexdigest()

    @classmethod
    def source_func(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        log.debug("%s: indexing %r; arguments: %r", cls, path, kwargs)
        return cls._source_func(str(path), **kwargs)

    @classmethod
    def normalize_filenames(cls, file: SPathLike | Iterable[SPathLike]) -> list[SPath]:
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
        matrix: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        chroma_location: ChromaLocation | None = None,
        color_range: ColorRangeLike | None = None,
        field_based: FieldBasedLike | None = None,
    ) -> vs.VideoNode:
        clips = list(clips)

        clip = clips[0] if len(clips) == 1 else core.std.Splice(clips)

        return initialize_clip(clip, bits, matrix, transfer, primaries, chroma_location, color_range, field_based)

    @inject_self
    def source(
        self,
        file: SPathLike | Iterable[SPathLike],
        bits: int | None = None,
        *,
        matrix: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        chroma_location: ChromaLocation | None = None,
        color_range: ColorRangeLike | None = None,
        field_based: FieldBasedLike | None = None,
        idx_props: bool = True,
        ref: vs.VideoNode | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Load one or more input files using the indexer and return a processed clip.

        The returned clip is passed through [initialize_clip][vstools.initialize_clip] to apply bit depth conversion
        and frame props initialization.
        """

        nfiles = self.normalize_filenames(file)
        clip = self._source(
            [self.source_func(f.to_str(), **self.indexer_kwargs | kwargs) for f in nfiles],
            bits,
            matrix,
            transfer,
            primaries,
            chroma_location,
            color_range,
            field_based,
        )
        if idx_props:
            clip = clip.std.SetFrameProps(IdxFilePath=[f.to_str() for f in nfiles], Idx=self.__class__.__name__)

        if name:
            clip = clip.std.SetFrameProps(Name=name)

        if ref:
            clip = match_clip(clip, ref, length=True)

        return clip


@cache
def _get_indexer_cache_storage() -> PackageStorage:
    return PackageStorage(package_name=f"{__name__}")


class CacheIndexer(Indexer):
    """Indexer interface with cache storage logic."""

    _cache_arg_name: ClassVar[str]
    _ext: ClassVar[str | None]

    @staticmethod
    def get_cache_path(file_name: SPathLike, ext: str | None = None) -> SPath:
        storage = _get_indexer_cache_storage()

        return storage.get_file(file_name, ext=ext)

    @classmethod
    def source_func(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        path = SPath(path)

        if cls._cache_arg_name not in kwargs:
            kwargs[cls._cache_arg_name] = cls.get_cache_path(path.name, cls._ext)

        return super().source_func(path, **kwargs)


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
        from shutil import which

        if not (bin_path := which(str(self.bin_path))):
            raise FileNotFoundError(f"Indexer: `{self.bin_path}` was not found{' in PATH' if os_name == 'nt' else ''}!")
        return SPath(bin_path)

    def _run_index(self, files: list[SPath], output: SPath, cmd_args: Sequence[str]) -> None:
        from subprocess import Popen

        output.mkdirp()

        proc = Popen(
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
            from tempfile import gettempdir

            return SPath(gettempdir())

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
        current_indxer = SPath(self._bin_path).name
        filename = "_".join([file_hash, vid_name, current_indxer])

        return self.get_idx_file_path(PackageStorage(folder).get_file(filename))

    @inject_self
    def source(
        self,
        file: SPathLike | Iterable[SPathLike],
        bits: int | None = None,
        *,
        matrix: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        chroma_location: ChromaLocation | None = None,
        color_range: ColorRangeLike | None = None,
        field_based: FieldBasedLike | None = None,
        idx_props: bool = True,
        **kwargs: Any,
    ) -> vs.VideoNode:
        index_files = self.index(self.normalize_filenames(file))

        return super().source(
            index_files,
            bits,
            matrix=matrix,
            transfer=transfer,
            primaries=primaries,
            chroma_location=chroma_location,
            color_range=color_range,
            field_based=field_based,
            idx_props=idx_props,
            **kwargs,
        )


type IndexerLike = str | type[Indexer] | Indexer
"""
Type alias for anything that can resolve to an Indexer.

This includes:

- A string identifier or plugin namespace of this indexer.
- A class type subclassing [Indexer][vssource.Indexer].
- An instance of a [Indexer][vssource.Indexer].
"""
