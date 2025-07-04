# noqa: N999

from __future__ import annotations

import atexit
import json
import os
import subprocess
from os import name as os_name
from typing import Any

from vstools import SPath

from .IsoFileCore import IsoFileCore

__all__ = ["IsoFile"]


class _WinIsoFile(IsoFileCore):
    def _run_disc_util(self, iso_path: SPath, util: str) -> SPath | None:
        pbjson, err = subprocess.Popen(
            [
                SPath(os.environ["WINDIR"]) / "System32" / "WindowsPowerShell" / "v1.0" / "PowerShell.exe",
                rf'{util}-DiskImage -ImagePath "{iso_path!s}" | Get-Volume | ConvertTo-Json',
            ],
            text=True,
            stdout=subprocess.PIPE,
            shell=True,
            encoding="utf-8",
        ).communicate()

        if err or pbjson[: len(util)] == util or pbjson == "":
            return None

        if util.lower() == "dismount":
            return SPath("")

        bjson: dict[str, str] = json.loads(pbjson)

        return SPath(f"{bjson['DriveLetter']}:\\")

    def _get_mounted_disc(self) -> SPath | None:
        return self._run_disc_util(self.iso_path, "Get")

    def _mount(self) -> SPath | None:
        if mount := self._run_disc_util(self.iso_path, "Mount"):
            atexit.register(self._unmount)
        return mount

    def _unmount(self) -> SPath | None:
        return self._run_disc_util(self.iso_path, "Dismount")


class _LinuxIsoFile(IsoFileCore):
    loop_path: SPath | None = None
    cur_mount: SPath | None = None

    def _subprun(self, *args: Any) -> str:
        return subprocess.run(list(map(str, args)), capture_output=True, universal_newlines=True).stdout

    def _get_mounted_disc(self) -> SPath | None:
        if not (loop_path := self._subprun("losetup", "-j", self.iso_path).strip().split(":")[0]):
            return self.cur_mount

        self.loop_path = SPath(loop_path)

        if "MountPoints:" in (device_info := self._run_disc_util(self.loop_path, ["info", "-b"], True)):
            cur_mount = device_info.split("MountPoints:")[1].split("\n")[0].strip()

            if cur_mount:
                self.cur_mount = SPath(cur_mount)

        return self.cur_mount

    def _run_disc_util(self, path: SPath, params: list[str], strip: bool = False) -> str:
        output = self._subprun("udisksctl", *params, str(path))

        return output.strip() if strip else output

    def _mount(self) -> SPath | None:
        if not self.loop_path:
            loop_path = self._run_disc_util(self.iso_path, ["loop-setup", "-f"], True)

            if "mapped file" not in loop_path.lower():
                raise RuntimeError("IsoFile: Couldn't map the ISO file!")

            loop_splits = loop_path.split(" as ")

            self.loop_path = SPath(loop_splits[-1][:-1])

        if "mounted" not in (cur_mount := self._run_disc_util(self.loop_path, ["mount", "-b"], True)).lower():
            return None

        mount_splits = cur_mount.split(" at ")

        self.cur_mount = SPath(mount_splits[-1])

        atexit.register(self._unmount)

        return self.cur_mount

    def _unmount(self) -> bool:
        if not self.loop_path:
            return True
        self._run_disc_util(self.loop_path, ["unmount", "-b"])
        return bool(self._run_disc_util(self.loop_path, ["loop-delete", "-b"]))


IsoFile: type[IsoFileCore] = _WinIsoFile if os_name == "nt" else _LinuxIsoFile
