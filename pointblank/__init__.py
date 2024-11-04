from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pointblank")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default version if metadata is not found

# Import objects from the module
from pointblank.test import Test
from pointblank.validate import Validate
from pointblank.thresholds import Thresholds

__all__ = ["Test", "Validate", "Thresholds"]
