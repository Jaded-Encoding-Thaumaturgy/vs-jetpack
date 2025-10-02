from __future__ import annotations

from collections.abc import Iterator
from pathlib import PurePath

import pytest

from vstools import PackageStorage


@pytest.fixture(scope="session")
def storage(request: pytest.FixtureRequest) -> Iterator[PackageStorage]:
    storage = PackageStorage(cwd=PurePath(__file__).parent)
    yield storage
    storage.folder.parent.rmdirs()


def test_package_storage_get_file(storage: PackageStorage) -> None:
    result = storage.get_file("test")
    assert result.name == "test"
    assert result.suffix == ""

    result = storage.get_file("test", ext=".txt")
    assert result.name == "test.txt"
    assert result.suffix == ".txt"

    result = storage.get_file("test", ext="should-use-my-extension.mkv")
    assert result.name == "test.mkv"
    assert result.suffix == ".mkv"

    result = storage.get_file("test", ext=PurePath("should-use-my-extension.mkv"))
    assert result.name == "test.mkv"
    assert result.suffix == ".mkv"


def test_package_storage_default_folder(storage: PackageStorage) -> None:
    # Have `cwd` be a file just to pretend it's the result of `get_script_path()`.
    storage = PackageStorage(cwd=PurePath(__file__))

    assert storage.folder.is_dir()
