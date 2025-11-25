from __future__ import annotations

from collections import deque
from functools import partial
from logging import getLogger
from os import PathLike
from typing import Any, Callable, Iterable, Literal, overload

from jetpytools import CustomRuntimeError, FileTypeMismatchError, SPath, SPathLike, check_perms, to_arr

from vstools import (
    ChromaLocation,
    ColorRangeLike,
    FieldBasedLike,
    FileType,
    IndexingType,
    MatrixLike,
    ParsedFile,
    PrimariesLike,
    TransferLike,
    match_clip,
    vs,
)

from .indexers import FFMS2, IMWRI, LSMAS, BestSource, D2VWitch, Indexer

__all__ = ["parse_video_filepath", "source"]


log = getLogger(__name__)


def parse_video_filepath(filepath: SPathLike | Iterable[SPathLike]) -> tuple[SPath, ParsedFile]:
    if not isinstance(filepath, (str, PathLike)):
        filepath = list(filepath)

    try:
        filepath = next(iter(Indexer.normalize_filenames(filepath)))
    except StopIteration:
        raise CustomRuntimeError("No files provided!", source, filepath)

    check_perms(filepath, "r", strict=True, func=source)

    file = FileType.parse(filepath) if filepath.exists() else None

    def _check_file_type(file_type: FileType) -> bool:
        return file_type in (FileType.VIDEO, FileType.IMAGE) or file_type.is_index()

    if not file or not _check_file_type(FileType(file.file_type)):
        for itype in IndexingType:
            if (newpath := filepath.with_suffix(f"{filepath.suffix}{itype.value}")).exists():
                file = FileType.parse(newpath)

    if not file or not _check_file_type(FileType(file.file_type)):
        raise FileTypeMismatchError('The file "{file}" isn\'t a video or image file!', source, file=filepath)

    return filepath, file


@overload
def source(
    filepath: SPathLike | Iterable[SPathLike],
    bits: int | None = None,
    *,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocation | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    ref: vs.VideoNode | None = None,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> vs.VideoNode: ...


@overload
def source(
    *,
    bits: int | None = None,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocation | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    ref: vs.VideoNode | None = None,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> Callable[[str], vs.VideoNode]: ...


@overload
def source(
    filepath: None,
    bits: int | None = None,
    *,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocation | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    ref: vs.VideoNode | None = None,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> Callable[[str], vs.VideoNode]: ...


def source(
    filepath: SPathLike | Iterable[SPathLike] | None = None,
    bits: int | None = None,
    *,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocation | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    ref: vs.VideoNode | None = None,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> vs.VideoNode | Callable[[str], vs.VideoNode]:
    """
    Automatically selects and uses an appropriate indexer to load a video or image file into VapourSynth.

    If `filepath` is not provided, a partially-applied version of this function is returned,
    allowing delayed path specification.

    - If `filepath` is an image, the indexer is hardcoded as `IMWRI`.
    - If `filepath` is a video, the indexer order is: `BestSource` -> `LSMAS` -> `FFMS2` -> `D2VWitch`.
      `BestSource` is moved to the end when in preview mode.

    Args:
        filepath: The path to the video file or image sequence. Can be a string path or an iterable of paths.
            If set to `None`, returns a callable that takes the file path later.
        bits: See [initialize_clip][vstools.initialize_clip] documentation.
        matrix: See [initialize_clip][vstools.initialize_clip] documentation.
        transfer: See [initialize_clip][vstools.initialize_clip] documentation.
        primaries: See [initialize_clip][vstools.initialize_clip] documentation.
        chroma_location: See [initialize_clip][vstools.initialize_clip] documentation.
        color_range: See [initialize_clip][vstools.initialize_clip] documentation.
        field_based: See [initialize_clip][vstools.initialize_clip] documentation.
        ref: An optional reference clip to match format, resolution, and frame count.
        name: A custom name to tag the clip with, stored in the `Name` frame prop.
        **kwargs: Additional keyword arguments forwarded to the selected source indexer.

    Raises:
        CustomRuntimeError: If no available indexer could open the provided file.

    Returns:
        If `filepath` is provided, returns a `VideoNode` representing the loaded clip.
        If `filepath` is `None`, returns a callable that accepts a file path and returns the corresponding clip.
    """

    kwargs.update(
        bits=bits,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
        chroma_location=chroma_location,
        color_range=color_range,
        field_based=field_based,
    )

    if filepath is None:
        return partial(source, **kwargs)

    filepath, file = parse_video_filepath(filepath)
    to_skip = to_arr(kwargs.get("_to_skip", []))
    clip = None

    if file.ext is IndexingType.LWI:
        clip = LSMAS.source(filepath, **kwargs)
    elif file.file_type is FileType.IMAGE:
        clip = IMWRI.source(filepath, **kwargs)
    elif clip is None:
        try:
            from vspreview import is_preview

        except (ImportError, ModuleNotFoundError):

            def is_preview() -> bool:
                return False

        indexers = deque[type[Indexer]]((BestSource, LSMAS, FFMS2, D2VWitch))

        if is_preview():
            indexers.rotate(-1)

        for indexer in filter(lambda i: i not in to_skip, indexers):
            try:
                clip = indexer.source(filepath, **kwargs)
                break
            except AttributeError:
                continue
            except vs.Error as e:
                if "bgr0 is not supported" in str(e) and indexer is LSMAS:
                    clip = indexer.source(filepath, format="rgb24", **kwargs)
                    break
                log.debug("Exception:", exc_info=False)
            except Exception:
                log.debug("Exception:", exc_info=False)
        else:
            raise CustomRuntimeError(f'None of the indexers you have installed work on this file! "{filepath}"', source)

    if name:
        clip = clip.std.SetFrameProps(Name=name)

    if ref:
        clip = match_clip(clip, ref, length=file.file_type is FileType.IMAGE)

    return clip
