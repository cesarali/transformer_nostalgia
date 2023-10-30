import os
import sys
from pathlib import Path

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


base_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(base_path, "..", ".."))
data_path = os.path.join(project_path, "data")
test_data_path = Path(os.path.join(project_path, "tests", "resources"))
vectors_path = os.path.join(project_path, "vectors")
sample_data_path = os.path.join(project_path, "sample_data")
logging_config_filename = os.path.join(project_path, "logging_config.yaml")
