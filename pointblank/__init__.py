from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pointblank")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default version if metadata is not found

# Import objects from the module
from pointblank.test import Test

__all__ = ["Test"]
