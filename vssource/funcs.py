from __future__ import annotations

from functools import partial
from os import PathLike
from typing import Any, Callable, Iterable, Literal, overload

from vstools import (
    ChromaLocation,
    ColorRangeLike,
    CustomRuntimeError,
    FieldBasedLike,
    FileType,
    FileTypeMismatchError,
    IndexingType,
    MatrixLike,
    ParsedFile,
    PrimariesLike,
    SPath,
    SPathLike,
    TransferLike,
    check_perms,
    initialize_clip,
    match_clip,
    to_arr,
    vs,
)

from .indexers import IMWRI, LSMAS, BestSource, D2VWitch, Indexer

__all__ = ["parse_video_filepath", "source"]


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
    if filepath is None:
        return partial(
            source,
            bits=bits if bits is not None else filepath,
            matrix=matrix,
            transfer=transfer,
            primaries=primaries,
            chroma_location=chroma_location,
            color_range=color_range,
            field_based=field_based,
            ref=ref,
            name=name,
            **kwargs,
        )

    filepath, file = parse_video_filepath(filepath)
    to_skip = to_arr(kwargs.get("_to_skip", []))
    clip = None

    if file.ext is IndexingType.LWI:
        indexer = LSMAS
        clip = indexer.source_func(filepath, **kwargs)
    elif file.file_type is FileType.IMAGE:
        indexer = IMWRI
        clip = indexer.source_func(filepath, **kwargs)

    if clip is None:
        try:
            from vspreview import is_preview
        except (ImportError, ModuleNotFoundError):
            best_last = False
        else:
            best_last = is_preview()

        indexers = [i for i in list[type[Indexer]]([LSMAS, D2VWitch]) if i not in to_skip]

        if best_last:
            indexers.append(BestSource)
        else:
            indexers.insert(0, BestSource)

        for indexer in indexers:
            try:
                clip = indexer.source(filepath, bits=bits)
                break
            except Exception as e:
                if "bgr0 is not supported" in str(e):
                    clip = indexer.source(filepath, format="rgb24", bits=bits)
                    break
        else:
            raise CustomRuntimeError(f'None of the indexers you have installed work on this file! "{filepath}"')

    if name:
        clip = clip.std.SetFrameProps(Name=name)

    if ref:
        clip = match_clip(clip, ref, length=file.file_type is FileType.IMAGE)

    return initialize_clip(
        clip,
        bits,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
        chroma_location=chroma_location,
        color_range=color_range,
        field_based=field_based,
    )
