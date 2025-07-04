from __future__ import annotations

from functools import partial
from os import PathLike
from typing import Any, Callable, Iterable, Literal, overload

from vstools import (
    ChromaLocationT,
    ColorRangeT,
    CustomRuntimeError,
    FieldBasedT,
    FileType,
    FileTypeMismatchError,
    IndexingType,
    MatrixT,
    ParsedFile,
    PrimariesT,
    SPath,
    SPathLike,
    TransferT,
    check_perms,
    initialize_clip,
    match_clip,
    to_arr,
    vs,
)

from .indexers import IMWRI, LSMAS, BestSource, D2VWitch, DGIndex, DGIndexNV, Indexer

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
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    ref: vs.VideoNode | None = None,
    film_thr: float = 99.0,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> vs.VideoNode: ...


@overload
def source(
    *,
    bits: int | None = None,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    ref: vs.VideoNode | None = None,
    film_thr: float = 99.0,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> Callable[[str], vs.VideoNode]: ...


@overload
def source(
    filepath: None,
    bits: int | None = None,
    *,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    ref: vs.VideoNode | None = None,
    film_thr: float = 99.0,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> Callable[[str], vs.VideoNode]: ...


def source(
    filepath: SPathLike | Iterable[SPathLike] | None = None,
    bits: int | None = None,
    *,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    ref: vs.VideoNode | None = None,
    film_thr: float = 99.0,
    name: str | Literal[False] = False,
    **kwargs: Any,
) -> vs.VideoNode | Callable[[str], vs.VideoNode]:
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
            film_thr=film_thr,
            name=name,
            **kwargs,
        )

    clip = None
    film_thr = float(min(100, film_thr))

    filepath, file = parse_video_filepath(filepath)

    props = dict[str, Any]()

    to_skip = to_arr(kwargs.get("_to_skip", []))

    if file.ext is IndexingType.LWI:
        clip = LSMAS.source_func(filepath, **kwargs)
    elif file.file_type is FileType.IMAGE:
        clip = IMWRI.source_func(filepath, **kwargs)
    else:
        try:
            if DGIndexNV in to_skip:
                raise RuntimeError
            else:
                from pymediainfo import MediaInfo

                tracks = MediaInfo.parse(filepath, parse_speed=0.25).video_tracks
                if tracks:
                    trackmeta = tracks[0].to_data()

                    video_format = trackmeta.get("format")

                    if video_format is not None:
                        video_fmt = str(video_format).strip().lower()

                        if video_fmt == "ffv1":
                            raise RuntimeError

                        bitdepth = trackmeta.get("bit_depth")

                        if bitdepth is not None and video_fmt == "avc" and int(bitdepth) > 8:
                            raise RuntimeError

            indexer, filepath_dgi = DGIndexNV(), SPath(filepath)

            if filepath_dgi.suffix != ".dgi":
                filepath_dgi = next(iter(indexer.index([filepath_dgi], False, False)))

            idx_info = indexer.get_info(filepath_dgi, 0).footer

            props |= {"DgiFieldOp": 0, "DgiOrder": idx_info.order, "DgiFilm": idx_info.film}

            indexer_kwargs = dict[str, Any]()
            if idx_info.film >= film_thr:
                indexer_kwargs |= {"fieldop": 1}
                props |= {"DgiFieldOp": 1, "_FieldBased": 0}

            clip = indexer.source_func(filepath_dgi, **indexer_kwargs)
        except (RuntimeError, AttributeError, FileNotFoundError):
            indexers = list[type[Indexer]]([LSMAS, D2VWitch, DGIndex])

            try:
                from vspreview import is_preview

                best_last = is_preview()
            except BaseException:
                best_last = False

            if best_last:
                indexers.append(BestSource)
            else:
                indexers.insert(0, BestSource)

            for indexerr in indexers:
                if indexerr in to_skip:
                    continue

                try:
                    clip = indexerr.source(filepath, bits=bits)
                    break
                except Exception as e:
                    if "bgr0 is not supported" in str(e):
                        try:
                            clip = indexerr.source(filepath, format="rgb24", bits=bits)
                            break
                        except Exception:
                            ...

    if clip is None:
        raise CustomRuntimeError(f'None of the indexers you have installed work on this file! "{filepath}"')

    props |= {"IdxFilePath": str(filepath)}

    if name:
        props |= {"Name": name}

    clip = clip.std.SetFrameProps(**props)

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
