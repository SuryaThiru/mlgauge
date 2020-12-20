from pathlib import Path

from mlgauge.analysis import Analysis
from mlgauge.method import Method, SklearnMethod

__all__ = ["Analysis", "Method", "SklearnMethod"]

VERSION_PATH = Path(__file__).resolve().parent / "VERSION"

with open(VERSION_PATH) as version_file:
    __version__ = version_file.read().strip()
