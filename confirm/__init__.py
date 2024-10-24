from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("confirm")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default version if metadata is not found

# Import objects from the module
from .test_col_vals_gt import test_col_vals_gt

__all__ = ["test_col_vals_gt"]
