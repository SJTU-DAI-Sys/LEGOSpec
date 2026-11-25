"""Keep generated test files inside the checkout."""

from pathlib import Path
import os
import tempfile


def pytest_configure(config):
    directory = Path(__file__).resolve().parents[1] / "tmp_exp" / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(directory.resolve())
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(directory / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(directory / "triton"))
    if config.option.basetemp is None:
        config.option.basetemp = str(directory / "pytest")
