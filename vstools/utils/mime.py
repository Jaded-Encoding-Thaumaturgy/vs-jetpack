from __future__ import annotations

import json
from itertools import chain
from mimetypes import encodings_map
from mimetypes import guess_type as guess_mime_type
from os import path
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeGuard

from jetpytools import (
    CustomRuntimeError,
    CustomStrEnum,
    CustomValueError,
    FilePathType,
    FuncExceptT,
    complex_hash,
    inject_self,
)

from .mime_base import FileTypeBase, FileTypeIndex, FileTypeIndexWithType

__all__ = ["FileSignature", "FileSignatures", "FileType", "IndexingType", "ParsedFile"]


class IndexingType(CustomStrEnum):
    """
    Enum of common indexing file extensions.
    """

    DGI = ".dgi"
    """
    DGIndexNV index file, mostly used for interlaced/telecined content.
    """

    LWI = ".lwi"
    """
    LSMAS index file.
    """


class ParsedFile(NamedTuple):
    """
    Structure for file info.
    """

    path: Path
    """Resolved path of the file."""

    ext: str
    """Extension of the file, from the binary data, not path."""

    encoding: str | None
    """Present for text files."""

    file_type: FileType
    """Type of the file. It will hold other useful information."""

    mime: str
    """Standard MIME type of the filetype."""


@complex_hash
class FileSignature(NamedTuple):
    """
    Child structure of FileSignatures, holding info of certain types of files and their signatures.
    """

    file_type: str
    """FileType as a str."""

    ext: str
    """Extension from the signature."""

    mime: str
    """MIME type of the signature."""

    offset: int
    """Offset from the start of the file of the signatures."""

    signatures: list[bytes]
    """Byte data signatures, unique for this file type."""

    def check_signature(self, file_bytes: bytes | bytearray, /, *, ignore: int = 0) -> int:
        """
        Verify the signature of the file.

        Args:
            file_bytes: Header bytes of the file to be checked.
            ignore: If a found signature is shorter than this length, it will be ignored.

        Returns:
            Length of the found signature.
        """

        max_signature_length = 0

        for signature in self.signatures:
            signature_len = len(signature)

            if signature_len < ignore or signature_len <= max_signature_length:
                continue

            if signature == file_bytes[self.offset : signature_len + self.offset]:
                max_signature_length = signature_len

        return max_signature_length


class FileSignatures(list[FileSignature]):
    """
    Structure wrapping a json file holding all file signatures.
    """

    _file_headers_data: list[FileSignature] | None = None
    file_headers_path = Path(path.join(path.dirname(path.abspath(__file__)), "__file_headers.json"))
    """
    Custom path for the json containing file headers.
    """

    def __init__(self, *, custom_header_data: str | Path | list[FileSignature] | None = None, force: bool = False):
        """
        Fetch all the file signatures, optionally with added custom signatures.
        """

        self.extend(self.load_headers_data(custom_header_data=custom_header_data, force=force))

        self.max_signature_len = max(
            chain.from_iterable([len(signature) for signature in mime.signatures] for mime in self)
        )

    def load_headers_data(
        self, *, custom_header_data: str | Path | list[FileSignature] | None = None, force: bool = False
    ) -> list[FileSignature]:
        """
        Load file signatures from json file. This is cached unless ``custom_header_data`` is set.

        Args:
            custom_header_data: Custom header data path file or custom list of already parsed FileSignature.
            force: Ignore cache and reload header data from disk.

        Returns:
            List of parsed FileSignature from json file.
        """

        if self._file_headers_data is None or force or custom_header_data:
            header_data: list[dict[str, Any]] = []

            filenames = {self.file_headers_path}

            if custom_header_data and not isinstance(custom_header_data, list):
                filenames.add(Path(custom_header_data))

            for filename in filenames:
                header_data.extend(json.loads(filename.read_text()))

            _file_headers_data = list(
                {
                    FileSignature(
                        info["file_type"],
                        info["ext"],
                        info["mime"],
                        info["offset"],
                        # This is so when checking a file head we first compare the most specific and long signatures
                        sorted([bytes.fromhex(signature) for signature in info["signatures"]], reverse=True),
                    ): 0
                    for info in header_data
                }.keys()
            )

            self._file_headers_data = _file_headers_data

            if isinstance(custom_header_data, list):
                return custom_header_data + _file_headers_data

        return self._file_headers_data

    @inject_self
    def parse(self, filename: Path) -> FileSignature | None:
        """
        Parse a given file.

        Args:
            filename: Path to file.

        Returns:
            The input file's mime signature.
        """

        with open(filename, "rb") as file:
            file_bytes = file.read(self.max_signature_len)

        max_signature_len = 0
        found_signatures = list[FileSignature]()

        for mimetype in self:
            found_signature = mimetype.check_signature(file_bytes, ignore=max_signature_len)

            if not found_signature:
                continue

            if found_signature > max_signature_len:
                max_signature_len = found_signature
                found_signatures = [mimetype]
            elif found_signature == max_signature_len:
                found_signatures.append(mimetype)

        if not found_signatures:
            return None

        signature_match_ext = [mime for mime in found_signatures if f".{mime.ext}" == filename.suffix]

        if signature_match_ext:
            return signature_match_ext[0]

        return found_signatures[0]


class FileType(FileTypeBase):
    """
    Enum for file types and mime types.
    """

    AUTO = "auto"
    """
    Special file type for [FileType.parse][vstools.FileType.parse].
    """

    VIDEO = "video"
    """
    File type for video files.
    """

    AUDIO = "audio"
    """
    File type for audio files.
    """

    CHAPTERS = "chapters"
    """
    File type for chapters files.
    """

    if not TYPE_CHECKING:
        INDEX = "index"
        INDEX_AUDIO = f"{INDEX}_{AUDIO}"
        INDEX_VIDEO = f"{INDEX}_{VIDEO}"

    IMAGE = "image"
    """
    File type for image files.
    """

    ARCHIVE = "archive"
    """
    File type for archive files.
    """

    FONT = "font"
    """
    File type for font files.
    """

    DOCUMENT = "document"
    """
    File type for documents.
    """

    OTHER = "other"
    """
    File type for generic files, like applications.
    """

    @classmethod
    def _missing_(cls, value: Any) -> FileType:
        if value is None:
            return FileType.AUTO

        if isinstance(value, str) and "/" in value:
            fbase, ftype, *_ = value.split("/")

            if fbase == "index":
                return FileType.INDEX(ftype)  # type: ignore[misc]

            if value.endswith("-image"):
                return FileType.IMAGE

            return FileType(ftype)

        return FileType.OTHER

    @inject_self.with_args(AUTO)
    def parse(
        self, path: FilePathType, *, func: FuncExceptT | None = None, force_ffprobe: bool | None = None
    ) -> ParsedFile:
        """
        Parse infos from a file. If the FileType is different than AUTO, this function will throw if the file
        is a different FileType than what this method was called on.

        Args:
            path: Path of the file to be parsed.
            func: Function returned for custom error handling. This should only be set by VS package developers.
        :force_ffprobe:     Only rely on ffprobe to parse the file info.

        Returns:
            ParsedFile object, holding the file's info.
        """

        from .ffprobe import FFProbe, FFProbeStream

        filename = Path(str(path)).absolute()

        file_type: FileType | None = None
        mime: str | None = None
        ext: str | None = None

        header = None if force_ffprobe else FileSignatures.parse(filename)

        if header is not None:
            file_type = FileType(header.file_type)
            mime = header.mime
            ext = f".{header.ext}"
        else:
            stream: FFProbeStream | None = None
            ffprobe = FFProbe(func=func)

            try:
                stream = ffprobe.get_stream(filename, FileType.VIDEO)

                if stream is None:
                    stream = ffprobe.get_stream(filename, FileType.AUDIO)

                if not stream:
                    raise CustomRuntimeError(f"No usable video/audio stream found in {filename}", func)

                file_type = FileType(stream.codec_type)
                mime = f"{file_type.value}/{stream.codec_name}"
            except Exception as e:
                if force_ffprobe:
                    raise e
                elif force_ffprobe is None:
                    return self.parse(path, force_ffprobe=False)

            if stream is None:
                mime, encoding = guess_mime_type(filename)

                file_type = FileType(mime)

        if ext is None:
            ext = filename.suffix

        encoding = encodings_map.get(filename.suffix, None)

        if not file_type or not mime:
            return ParsedFile(filename, ext, encoding, FileType.OTHER, "file/unknown")

        if self is not FileType.AUTO and self is not file_type:
            raise CustomValueError(
                "FileType mismatch! self is {good}, file is {bad}!", FileType.parse, good=self, bad=file_type
            )

        return ParsedFile(filename, ext, encoding, file_type, mime)

    def is_index(self) -> TypeGuard[FileTypeIndexWithType]:  # type: ignore
        """
        Verify whether the FileType is an INDEX that holds its own FileType (e.g. mime: index/video).
        """

        return self in {FileType.INDEX, FileType.INDEX_AUDIO, FileType.INDEX_VIDEO}  # type: ignore

    def __call__(self: FileTypeIndex, file_type: str | FileType) -> FileTypeIndexWithType:  # type: ignore
        """
        Get an INDEX FileType of another FileType (Video, Audio, Other).
        """

        if self is not FileType.INDEX:
            raise NotImplementedError

        file_type = FileType(file_type)

        if file_type in {FileType.AUDIO, FileType.VIDEO}:
            if file_type is FileType.AUDIO:
                return FileType.INDEX_AUDIO  # type: ignore

            if file_type is FileType.VIDEO:
                return FileType.INDEX_VIDEO  # type: ignore

        raise CustomValueError("You can only have Video, Audio or Other index file types!", str(FileType.INDEX))


for _fty, _ftyp in [
    (FileType.AUDIO, FileType.INDEX_AUDIO),  # type: ignore
    (FileType.VIDEO, FileType.INDEX_VIDEO),  # type: ignore
]:
    setattr(_ftyp, "file_type", _fty)
