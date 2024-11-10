from importlib_metadata import version, PackageNotFoundError

try:  # pragma: no cover
    __version__ = version("pointblank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import objects from the module
from pointblank.tf import TF
from pointblank.validate import Validate
from pointblank.thresholds import Thresholds

__all__ = ["TF", "Validate", "Thresholds"]
