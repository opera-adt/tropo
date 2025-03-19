from pathlib import Path

import pytest
import xarray as xr

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def load_input_model():
    # Load the golden input
    return xr.open_dataset(DATA_DIR / "test_data.nc")


@pytest.fixture(scope="session")
def load_golden_output():
    # Load the golden output
    return xr.open_dataset(DATA_DIR / "output_data.nc")
