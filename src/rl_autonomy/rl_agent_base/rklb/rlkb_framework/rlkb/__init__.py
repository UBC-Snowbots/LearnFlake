from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlkb")
except PackageNotFoundError:  # pragma: no cover - during editable installs before metadata exists
    __version__ = "0.0.0"

__all__ = ["__version__"]
