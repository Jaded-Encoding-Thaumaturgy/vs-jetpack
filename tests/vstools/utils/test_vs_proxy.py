from __future__ import annotations

from unittest import TestCase

import vapoursynth

import vstools


class TestVsProxy(TestCase):
    def test_core_proxy(self) -> None:
        assert vapoursynth.core.core == vstools.core.core
        assert vapoursynth.core.core == vstools.vs.core.core
