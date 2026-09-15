import pytest
from jetpytools import CustomRuntimeError, CustomValueError

from vssource import FFMS2, BestSource, D2VWitch, Indexer, ZipSource


def test_from_param() -> None:
    assert Indexer.from_param("bEsTsoUrCE") is BestSource
    assert Indexer.from_param("bs") is BestSource
    assert Indexer.from_param("ffms2") is FFMS2
    assert Indexer.from_param("d2vwitch") is D2VWitch
    assert Indexer.from_param("zipsource") is ZipSource
    assert Indexer.from_param(BestSource) is BestSource
    assert Indexer.from_param(BestSource()) is BestSource

    with pytest.raises(CustomValueError):
        Indexer.from_param("beastsource")


def test_ensure_obj() -> None:
    assert isinstance(BestSource.ensure_obj("bEsTsoUrCE"), BestSource)
    assert isinstance(BestSource.ensure_obj("bs"), BestSource)
    assert isinstance(BestSource.ensure_obj("bs"), BestSource)
    assert isinstance(BestSource.ensure_obj(BestSource), BestSource)
    assert isinstance(BestSource.ensure_obj(BestSource()), BestSource)

    with pytest.raises(CustomValueError):
        BestSource.ensure_obj("beastsource")


def test_unsupported_asource() -> None:
    with pytest.raises(CustomRuntimeError, match="does not support audio indexing"):
        D2VWitch.asource("dummy.mkv")

    with pytest.raises(CustomRuntimeError, match="does not support audio indexing"):
        ZipSource.asource("dummy.mkv")
