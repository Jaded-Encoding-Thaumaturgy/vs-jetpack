from typing import Any, overload

__version__: Any

class Track:
    def __eq__(self, other: object) -> bool: ...
    def __getattribute__(self, name: str) -> Any: ...
    track_type: Any
    def __init__(self, xml_dom_fragment: Any) -> None: ...
    def to_data(self) -> dict[str, Any]: ...

class MediaInfo:
    def __eq__(self, other: object) -> bool: ...
    tracks: Any
    def __init__(self, xml: str, encoding_errors: str = "strict") -> None: ...
    @property
    def general_tracks(self) -> list[Track]: ...
    @property
    def video_tracks(self) -> list[Track]: ...
    @property
    def audio_tracks(self) -> list[Track]: ...
    @property
    def text_tracks(self) -> list[Track]: ...
    @property
    def other_tracks(self) -> list[Track]: ...
    @property
    def image_tracks(self) -> list[Track]: ...
    @property
    def menu_tracks(self) -> list[Track]: ...
    @classmethod
    def can_parse(cls, library_file: str | None = None) -> bool: ...
    @overload
    @classmethod
    def parse(
        cls,
        filename: Any,
        *,
        library_file: str | None = None,
        cover_data: bool = False,
        encoding_errors: str = "strict",
        parse_speed: float = 0.5,
        full: bool = True,
        legacy_stream_display: bool = False,
        mediainfo_options: dict[str, str] | None = None,
        output: str,
        buffer_size: int | None = ...,
    ) -> str: ...
    @overload
    @classmethod
    def parse(
        cls,
        filename: Any,
        *,
        library_file: str | None = None,
        cover_data: bool = False,
        encoding_errors: str = "strict",
        parse_speed: float = 0.5,
        full: bool = True,
        legacy_stream_display: bool = False,
        mediainfo_options: dict[str, str] | None = None,
        output: None = None,
        buffer_size: int | None = ...,
    ) -> MediaInfo: ...
    def to_data(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...
