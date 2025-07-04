from __future__ import annotations

import os
from io import BufferedReader, BytesIO
from pprint import pformat
from struct import unpack
from typing import ClassVar

from vstools import SPath, SPathLike

__all__ = ["SectorReadHelper"]


class SectorReadHelper:
    _byte_size_lut: ClassVar[dict[int, str]] = {1: "B", 2: "H", 4: "I", 8: "Q"}
    file: SPath | None = None

    def __init__(self, ifo: bytes | SPathLike | BufferedReader) -> None:
        if isinstance(ifo, bytes):
            ifo = BufferedReader(BytesIO(ifo))  # type: ignore

        if not isinstance(ifo, BufferedReader):
            self.file = SPath(ifo)
            ifo = self.file.open("rb")

        self.ifo = ifo

    def __del__(self) -> None:
        if self.file is not None and self.ifo and not self.ifo.closed:
            self.ifo.close()

    def _goto_sector_ptr(self, pos: int) -> None:
        self.ifo.seek(pos, os.SEEK_SET)

        (ptr,) = self._unpack_byte(4)

        self.ifo.seek(ptr * 2048, os.SEEK_SET)

    def _seek_unpack_byte(self, addr: int, *n: int) -> tuple[int, ...]:
        self.ifo.seek(addr, os.SEEK_SET)
        return self._unpack_byte(*n)

    def _unpack_byte(self, *n: int, repeat: int = 1) -> tuple[int, ...]:
        n_list = list(n) * repeat

        bytecnt = sum(n_list)

        stra = ">" + "".join(self._byte_size_lut.get(a, "B") for a in n_list)

        buf = self.ifo.read(bytecnt)

        assert len(buf) == bytecnt

        return unpack(stra, buf)

    def __repr__(self) -> str:
        return pformat(vars(self), sort_dicts=False)
