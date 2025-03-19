from pathlib import Path

import pytest
import xarray as xr

# Define the data directory
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def load_input_model() -> xr.Dataset:
    """Load the golden input dataset."""
    file_path = DATA_DIR / "test_data.nc"
    if not file_path.exists():
        pytest.fail(f"Missing test file: {file_path}")

    with xr.open_dataset(file_path) as ds:
        return ds.load()  # Ensure data is loaded into memory


@pytest.fixture(scope="session")
def load_golden_output() -> xr.Dataset:
    """Load the golden output dataset."""
    file_path = DATA_DIR / "output_data.nc"
    if not file_path.exists():
        pytest.fail(f"Missing test file: {file_path}")

    with xr.open_dataset(file_path) as ds:
        return ds.load()  # Ensure data is loaded into memory
