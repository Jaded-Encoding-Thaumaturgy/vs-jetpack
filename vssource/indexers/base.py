from __future__ import annotations

import hashlib
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from logging import getLogger
from typing import Any, ClassVar, Literal, Self, override

from jetpytools import (
    MISSING,
    CustomRuntimeError,
    CustomValueError,
    FuncExcept,
    SPath,
    SPathLike,
    classproperty,
    get_subclasses,
    inject_self,
    to_arr,
)

from vsjetpack import deprecated
from vstools import (
    ChromaLocation,
    FieldBasedLike,
    MatrixLike,
    MissingT,
    PackageStorage,
    PrimariesLike,
    RangeLike,
    TransferLike,
    core,
    initialize_clip,
    match_clip,
    vs,
)

from ..dataclasses import IndexFileType

__all__ = ["CacheIndexer", "ExternalIndexer", "Indexer", "IndexerLike"]

log = getLogger(__name__)


class Indexer(ABC):
    """
    Abstract indexer interface for video and audio node sourcing.
    """

    _source_func: ClassVar[Callable[..., vs.VideoNode] | None] = None
    _asource_func: ClassVar[Callable[..., vs.AudioNode] | None] = None
    _audio_track_arg_name: ClassVar[str] = "track"

    def __init__(self, *, force: bool = True, **kwargs: Any) -> None:
        super().__init__()

        self.force = force
        self.indexer_kwargs = kwargs

    @classproperty
    @classmethod
    def has_video(cls) -> bool:
        """Whether this indexer supports video sourcing."""
        return getattr(cls, "_source_func", None) is not None

    @classproperty
    @classmethod
    def has_audio(cls) -> bool:
        """Whether this indexer supports audio sourcing."""
        return getattr(cls, "_asource_func", None) is not None

    @inject_self
    def source(
        self,
        file: SPathLike | Iterable[SPathLike],
        bits: int | None = 32,
        *,
        matrix: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        chroma_location: ChromaLocation | None = None,
        color_range: RangeLike | None = None,
        field_based: FieldBasedLike | None = None,
        idx_props: bool = True,
        ref: vs.VideoNode | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Load one or more input files as a [VideoNode][vs.VideoNode] using the indexer.

        The returned clip is passed through [initialize_clip][vstools.initialize_clip] to apply bit depth conversion
        and frame props initialization.
        """
        if not self.has_video:
            raise CustomRuntimeError(f"{self.__class__.__name__} does not support video indexing!", self.source)

        nfiles = self.normalize_filenames(file)
        clips = [self._source_file(f.to_str(), **self.indexer_kwargs | kwargs) for f in nfiles]
        clip = clips[0] if len(clips) == 1 else core.std.Splice(clips)
        clip = initialize_clip(clip, bits, matrix, transfer, primaries, chroma_location, color_range, field_based)

        if idx_props:
            clip = clip.std.SetFrameProps(IdxFilePath=[f.to_str() for f in nfiles], Idx=self.__class__.__name__)

        if name:
            clip = clip.std.SetFrameProps(Name=name)

        if ref:
            clip = match_clip(clip, ref, length=True)

        return clip

    @inject_self
    def asource(
        self,
        file: SPathLike | Iterable[SPathLike],
        *,
        track: int = -1,
        sample_rate: int | None = None,
        channels: Sequence[int] | None = None,
        bits: int | None = None,
        **kwargs: Any,
    ) -> vs.AudioNode:
        """
        Load one or more input files as an [AudioNode][vs.AudioNode] using the indexer.
        """
        if not self.has_audio:
            raise CustomRuntimeError(f"{self.__class__.__name__} does not support audio indexing!", self.asource)

        nfiles = self.normalize_filenames(file)
        call_kwargs = self.indexer_kwargs | kwargs
        if self._audio_track_arg_name not in call_kwargs and track is not None:
            call_kwargs[self._audio_track_arg_name] = track

        clips = [self._asource_file(f.to_str(), **call_kwargs) for f in nfiles]
        clip = clips[0] if len(clips) == 1 else core.std.AudioSplice(clips)

        if sample_rate is not None or channels is not None or bits is not None:
            clip = core.std.AudioResample(clip, samplerate=sample_rate, channels=channels, bits=bits)

        return clip

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
    @deprecated(
        "`source_func` is deprecated and will be removed in a future version. Use `source` instead.",
        category=DeprecationWarning,
    )
    def source_func(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        return cls._source_file(path, **kwargs)

    @classmethod
    def normalize_filenames(cls, file: SPathLike | Iterable[SPathLike]) -> list[SPath]:
        files = list[SPath]()

        for f in to_arr(file):
            if str(f).startswith("file:///"):
                f = str(f)[8::]

            files.append(SPath(f))

        return files

    @classmethod
    def _source_file(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        if cls._source_func is None:
            raise CustomRuntimeError(f"{cls.__name__} does not support video indexing!", cls._source_file)
        log.debug("%s: indexing video %r; arguments: %r", cls, path, kwargs)
        return cls._source_func(str(path), **kwargs)

    @classmethod
    def _asource_file(cls, path: SPathLike, **kwargs: Any) -> vs.AudioNode:
        if cls._asource_func is None:
            raise CustomRuntimeError(f"{cls.__name__} does not support audio indexing!", cls._asource_file)
        log.debug("%s: indexing audio %r; arguments: %r", cls, path, kwargs)
        return cls._asource_func(str(path), **kwargs)


@cache
def _get_indexer_cache_storage() -> PackageStorage:
    return PackageStorage(package_name=f"{__name__}")


class CacheIndexer(Indexer):
    """Indexer interface with cache storage logic."""

    _cache_arg_name: ClassVar[str]
    _ext: ClassVar[str | None]

    @staticmethod
    def get_cache_path(
        source_path: SPathLike,
        ext: str | None = None,
        track: int | None = None,
        is_audio: bool = False,
    ) -> SPath:
        source_file = SPath(source_path).resolve()
        hashed_path = hashlib.blake2s(source_file.to_str().encode("utf-8"), digest_size=4).hexdigest()
        cache_filename = f"{source_file.name}_{hashed_path}"

        if is_audio and ext:
            tr_suffix = f"_a{track if track is not None else 0}"
            cache_filename = f"{cache_filename}{tr_suffix}"

        if ext:
            cache_filename = f"{cache_filename}.{ext.lstrip('.')}"

        storage = _get_indexer_cache_storage()
        return storage.get_file(cache_filename)

    @classmethod
    @override
    def _source_file(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        path = SPath(path)

        if cls._cache_arg_name not in kwargs:
            kwargs[cls._cache_arg_name] = cls.get_cache_path(path, cls._ext, is_audio=False)

        return super()._source_file(path, **kwargs)

    @classmethod
    @override
    def _asource_file(cls, path: SPathLike, **kwargs: Any) -> vs.AudioNode:
        path = SPath(path)

        if cls._cache_arg_name not in kwargs:
            track = kwargs.get(cls._audio_track_arg_name)
            kwargs[cls._cache_arg_name] = cls.get_cache_path(path, cls._ext, track=track, is_audio=True)

        return super()._asource_file(path, **kwargs)


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

    @abstractmethod
    def get_info(self, index_path: SPath, file_idx: int = 0) -> IndexFileType:
        """
        Returns info about the indexing file
        """

    @inject_self
    @override
    def source(
        self,
        file: SPathLike | Iterable[SPathLike],
        bits: int | None = 32,
        *,
        matrix: MatrixLike | None = None,
        transfer: TransferLike | None = None,
        primaries: PrimariesLike | None = None,
        chroma_location: ChromaLocation | None = None,
        color_range: RangeLike | None = None,
        field_based: FieldBasedLike | None = None,
        idx_props: bool = True,
        **kwargs: Any,
    ) -> vs.VideoNode:
        if not self.has_video:
            raise CustomRuntimeError(f"{self.__class__.__name__} does not support video indexing!", self.source)

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

    @inject_self
    @override
    def asource(
        self,
        file: SPathLike | Iterable[SPathLike],
        *,
        track: int = -1,
        sample_rate: int | None = None,
        channels: Sequence[int] | None = None,
        bits: int | None = None,
        **kwargs: Any,
    ) -> vs.AudioNode:
        if not self.has_audio:
            raise CustomRuntimeError(f"{self.__class__.__name__} does not support audio indexing!", self.asource)

        index_files = self.index(self.normalize_filenames(file))

        return super().asource(
            index_files,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
            bits=bits,
            **kwargs,
        )

    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        """
        Update filepaths recorded inside index file if moved. Default is a no-op.
        """

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

        hash_str = self.get_files_hash(files)

        def _index(files: list[SPath], output: SPath) -> None:
            if output.is_file():
                if output.stat().st_size == 0 or force:
                    output.unlink()
                else:
                    return self.update_video_filenames(output, files)
            return self._run_index(files, output, cmd_args)

        if not split_files:
            output = self.get_file_idx_path(dest_folder, hash_str, "JOINED" if len(files) > 1 else "SINGLE")
            _index(files, output)
            return [output]

        outputs = [self.get_file_idx_path(dest_folder, hash_str, file.name) for file in files]

        for file, output in zip(files, outputs):
            _index([file], output)

        return outputs

    @deprecated(
        "`get_video_idx_path` is deprecated and will be removed in a future version. Use `get_file_idx_path` instead.",
        category=DeprecationWarning,
    )
    def get_video_idx_path(self, folder: SPath, file_hash: str, file_name: SPathLike) -> SPath:
        return self.get_file_idx_path(folder, file_hash, file_name)

    def get_file_idx_path(self, folder: SPath, file_hash: str, file_name: SPathLike) -> SPath:
        f_name = SPath(file_name).stem
        current_indexer = SPath(self._bin_path).name
        filename = f"{file_hash}_{f_name}_{current_indexer}"

        return self.get_idx_file_path(PackageStorage(folder, package_name=__name__).get_file(filename))

    @classmethod
    def get_joined_names(cls, files: list[SPath]) -> str:
        return "_".join([file.name for file in files])

    @classmethod
    @deprecated(
        "`get_videos_hash` is deprecated and will be removed in a future version. Use `get_files_hash` instead.",
        category=DeprecationWarning,
    )
    def get_videos_hash(cls, files: list[SPath]) -> str:
        return cls.get_files_hash(files)

    @classmethod
    def get_files_hash(cls, files: list[SPath]) -> str:
        length = sum(file.stat().st_size for file in files)
        to_hash = length.to_bytes(32, "little") + cls.get_joined_names(files).encode()
        return hashlib.md5(to_hash).hexdigest()

    def _run_index(self, files: list[SPath], output: SPath, cmd_args: Sequence[str]) -> None:
        output.mkdirp()

        proc = subprocess.Popen(
            list(map(str, (*self.get_cmd(files, output), *cmd_args, *self._default_args))),
            text=True,
            encoding="utf-8",
            shell=os.name == "nt",
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


type IndexerLike = str | type[Indexer] | Indexer
"""
Type alias for anything that can resolve to an Indexer.

This includes:

- A string identifier or plugin namespace of this indexer.
- A class type subclassing [Indexer][vssource.Indexer].
- An instance of an [Indexer][vssource.Indexer].
"""


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
