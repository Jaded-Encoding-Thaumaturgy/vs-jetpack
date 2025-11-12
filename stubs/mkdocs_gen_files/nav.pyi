import dataclasses
from typing import Iterable

class Nav:
    def __init__(self) -> None: ...
    def __setitem__(self, keys: str | tuple[str, ...], value: str) -> None: ...
    @dataclasses.dataclass
    class Item:
        level: int
        title: str
        filename: str | None

    def items(self) -> Iterable[Item]: ...
    def build_literate_nav(self, indentation: int | str = "") -> Iterable[str]: ...
