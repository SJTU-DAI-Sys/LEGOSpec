"""Keep generated test files inside the checkout."""

from pathlib import Path
import tempfile


def pytest_configure(config):
    directory = Path(__file__).parent / "artifacts" / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(directory.resolve())
    if config.option.basetemp is None:
        config.option.basetemp = str(directory / "pytest")
