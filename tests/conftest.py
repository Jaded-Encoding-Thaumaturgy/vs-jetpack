import pytest


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically apply the `vpy("initial-core")` marker to all test items that don't already have a `vpy` marker."""
    for item in items:
        if item.get_closest_marker("vpy") is None:
            item.add_marker(pytest.mark.vpy("initial-core"))
