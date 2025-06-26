from __future__ import annotations

import logging

import xarray as xr

from .log.loggin_setup import log_runtime

logger = logging.getLogger(__name__)

EXPECTED_COORDS = frozenset(["longitude", "latitude", "level", "time"])
EXPECTED_VARS = frozenset(["z", "t", "q", "lnsp"])

# Valid range with a buffer
VALID_RANGE = {
    "t": [80, 400],  # Temperature (K)
    "q": [-0.1, 0.5],  # Specific humidity (kg/kg)
    "z": [-5000, 65000],  # Geopotential (m²/s²)
    "lnsp": [10, 12],  # Log of surface pressure (unitless)
}


@log_runtime
def validate_input(ds: xr.Dataset) -> None:
    """Validate an xarray Dataset.

    This function performs the following checks on the provided dataset:

    - Ensures that the dataset contains all required coordinates and data variables.
    - Verifies that latitude, longitude, and level values
      fall within their expected ranges.
    - Checks for the absence of NaN values in the expected data variables.
    - Validates that data values fall within their predefined valid ranges.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to validate.

    Raises
    ------
    ValueError
        If any validation check fails, a `ValueError` is raised
        (e.g., missing variables, out-of-range values, NaNs in data variables).

    """
    logger.info("Performing checkup of input file")
    checks = []

    # Check Coordinates
    coords = set(ds.coords.keys())
    if coords != EXPECTED_COORDS:
        missing_coords = EXPECTED_COORDS - coords
        extra_coords = coords - EXPECTED_COORDS
        checks.append(
            (
                f"Unexpected coordinates."
                f" Missing: {missing_coords}, Extra: {extra_coords}"
            )
        )

    if (ds.latitude.min() < -90) | (ds.latitude.max() > 90):
        checks.append("Latitude values must be within (-90, 90)")

    if (ds.longitude.min() < 0) | (ds.longitude.max() > 360):
        checks.append("Longitude values must be within (0, 360)")

    if (ds.level.min() < 0) | (ds.level.max() > 137):
        checks.append("Level values must be within (0, 137)")

    # Check Data Variable
    data_vars = set(ds.data_vars.keys())
    if data_vars != EXPECTED_VARS:
        missing_vars = EXPECTED_VARS - data_vars
        extra_vars = data_vars - EXPECTED_VARS
        checks.append(
            f"Unexpected data variables. Missing: {missing_vars}, Extra: {extra_vars}"
        )

    # Check NaN values and valid range
    for var in EXPECTED_VARS:
        var_data = ds[var].isel(
            time=0, level=0 if var in ["z", "lnsp"] else slice(None)
        )
        var_name = getattr(ds[var], "long_name", var)

        if var_data.isnull().any():
            checks.append(f'Data Variable "{var}" ({var_name}) contains NaN values.')

        min_val, max_val = var_data.min().values, var_data.max().values

        value = VALID_RANGE.get(var)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            valid_min, valid_max = value
        else:
            raise ValueError(f"Invalid range for {var}")

        if (min_val < valid_min) | (max_val > valid_max):
            checks.append(
                (
                    f'Data Variable "{var}" ({var_name}) is out of valid range'
                    f" {VALID_RANGE[var]}. Found min: {min_val}, max: {max_val}"
                )
            )

    # Raise error if any check fails
    if checks:
        raise ValueError("Failed validation checks:\n" + "\n".join(checks))
