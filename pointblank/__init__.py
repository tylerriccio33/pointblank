try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError

try:  # pragma: no cover
    __version__ = version("pointblank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import objects from the module
from pointblank.tf import TF
from pointblank.column import col
from pointblank.validate import Validate, load_dataset, config
from pointblank.schema import Schema
from pointblank.thresholds import Thresholds

__all__ = ["TF", "Validate", "Thresholds", "Schema", "col", "load_dataset", "config"]
