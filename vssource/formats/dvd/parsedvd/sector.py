from __future__ import annotations

from io import BufferedReader, BytesIO
from os import SEEK_SET
from struct import unpack
from typing import TYPE_CHECKING, ClassVar

from jetpytools import SPath

if TYPE_CHECKING:
    from jetpytools import SPathLike

__all__ = ["SectorReadHelper"]


class SectorReadHelper:
    _byte_size_lut: ClassVar[dict[int, str]] = {1: "B", 2: "H", 4: "I", 8: "Q"}
    file: SPath | None = None

    def __init__(self, ifo: bytes | SPathLike | BufferedReader) -> None:
        if isinstance(ifo, (bytes, bytearray, memoryview)):
            ifo = BufferedReader(BytesIO(ifo))

        if not isinstance(ifo, BufferedReader):
            self.file = SPath(ifo)
            ifo = self.file.open("rb")

        self.ifo = ifo

    def __del__(self) -> None:
        if self.file is not None and self.ifo and not self.ifo.closed:
            self.ifo.close()

    def _goto_sector_ptr(self, pos: int) -> None:
        self.ifo.seek(pos, SEEK_SET)

        (ptr,) = self._unpack_byte(4)

        self.ifo.seek(ptr * 2048, SEEK_SET)

    def _seek_unpack_byte(self, addr: int, *n: int) -> tuple[int, ...]:
        self.ifo.seek(addr, SEEK_SET)
        return self._unpack_byte(*n)

    def _unpack_byte(self, *n: int, repeat: int = 1) -> tuple[int, ...]:
        n_list = list(n) * repeat

        bytecnt = sum(n_list)

        stra = ">" + "".join(self._byte_size_lut.get(a, "B") for a in n_list)

        buf = self.ifo.read(bytecnt)

        assert len(buf) == bytecnt

        return unpack(stra, buf)

    def __repr__(self) -> str:
        from pprint import pformat

        return pformat(vars(self), sort_dicts=False)
