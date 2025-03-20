import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from RAiDER.models import HRES

# Define the data directory
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def load_input_model() -> xr.Dataset:
    """Load the golden input dataset."""
    file_path = DATA_DIR / "test_dataset.nc"
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


@pytest.fixture
def load_hres_model():
    # Load the golden hres model dict from the file once
    with open(DATA_DIR / "hres_model.pkl", "rb") as file:
        golden_dict = pickle.load(file)

    # Initialize the HRES model
    hres_model = HRES()
    model_dict = hres_model.__dict__

    return golden_dict, model_dict


@pytest.fixture()
def init_raider(load_input_model):
    """Load RAIDER HRES model."""
    # load test data
    da = load_input_model
    da = da.isel(time=0)

    # Init RAIDER
    hres_model = HRES()

    # Extract temperature and specific humidity at the first time step
    hres_model._t = da.t.values
    hres_model._q = da.q.values

    # Extract longitude and latitude values
    longitude = da.longitude.values
    latitude = da.latitude.values

    # Create latitude and longitude grid
    hres_model._lons, hres_model._lats = np.meshgrid(longitude, latitude)
    return hres_model
