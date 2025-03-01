from __future__ import annotations

import os
import subprocess

from vstools import SPath, core

from .D2VWitch import D2VWitch

__all__ = [
    'DGIndex'
]


class DGIndex(D2VWitch):
    _bin_path = 'dgindex'
    _ext = 'd2v'
    _source_func = core.lazy.d2v.Source

    def get_cmd(
        self, files: list[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> list[str]:
        is_linux = os.name != 'nt'

        if is_linux:
            output = SPath(f'Z:\\{str(output)[1:]}')
            paths = list(subprocess.check_output(['winepath', '-w', f]).decode('utf-8').strip() for f in files)
        else:
            paths = list(map(str, files))

        return list(map(str, [
            self._get_bin_path(),
            '-i', *paths, '-ia', idct_algo, '-fo', field_op, '-yr', yuv_to_rgb,
            '-om', '0', '-o', output.with_suffix(""), '-hide', '-exit'
        ]))
