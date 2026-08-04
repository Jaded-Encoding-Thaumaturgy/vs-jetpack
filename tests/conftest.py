import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Automatically apply the `vpy("initial-core")` marker to all test items that don't already have a `vpy` marker."""
    if metafunc.definition.get_closest_marker("vpy") is None:
        metafunc.definition.add_marker(pytest.mark.vpy("initial-core"))
