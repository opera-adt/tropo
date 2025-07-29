from __future__ import annotations

import logging
from typing import Tuple

import dask as da
import numpy as np
import xarray as xr

from .log.loggin_setup import log_runtime

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for input validation errors."""

    pass


EXPECTED_COORDS = frozenset(["longitude", "latitude", "level", "time"])
EXPECTED_VARS = frozenset(["z", "t", "q", "lnsp"])

# Extended Valid range
VALID_RANGE: dict[str, tuple[float, float]] = {
    "t": (140.0, 360.0),  # Temperature (K), Physical [165, 330]
    "q": (0.0, 0.3),  # Specific humidity (kg/kg), Physical [0, 0,08]
    "z": (-5000.0, 70000.0),  # Geopotential (m²/s²)
    "lnsp": (10.2, 11.75),  # Log of surface pressure (unitless)
}


def get_min_max_nan(var_data: xr.DataArray) -> Tuple[float, float, int]:
    """Get min/max and nan_count."""
    min_da = var_data.min()
    max_da = var_data.max()
    nan_da = var_data.isnull().sum()
    data_arrays = [min_da.data, max_da.data, nan_da.data]

    # Compute all at once
    min_val, max_val, nan_count = da.compute(*data_arrays)
    min_result = float(min_val.item()) if not np.isnan(min_val) else float("nan")
    max_result = float(max_val.item()) if not np.isnan(max_val) else float("nan")
    nan_result = int(nan_count.item())

    return min_result, max_result, nan_result


def check_coords_and_variables(ds: xr.Dataset) -> None:
    """Check dataset coordinates and expected variables."""
    issues: list[str] = []

    # Coordinates
    coords = set(ds.coords.keys())
    if coords != EXPECTED_COORDS:
        missing = EXPECTED_COORDS - coords
        extra = coords - EXPECTED_COORDS
        issues.append(f"Unexpected coordinates. Missing: {missing}, Extra: {extra}")

    if (ds.latitude.min() < -90) | (ds.latitude.max() > 90):
        issues.append("Latitude values must be within (-90, 90)")

    if (ds.longitude.min() < 0) | (ds.longitude.max() > 360):
        issues.append("Longitude values must be within (0, 360)")

    if (ds.level.min() < 0) | (ds.level.max() > 137):
        issues.append("Level values must be within (0, 137)")

    # Variables
    data_vars = set(ds.data_vars.keys())
    if data_vars != EXPECTED_VARS:
        missing = EXPECTED_VARS - data_vars
        extra = data_vars - EXPECTED_VARS
        issues.append(f"Unexpected data variables. Missing: {missing}, Extra: {extra}")

    if issues:
        raise ValidationError("\n".join(issues))


def check_nans_valid_range(ds: xr.Dataset) -> dict[str, list[float]]:
    """Check for NaNs and validate data ranges for key variables."""
    issues: list[str] = []
    out_range_vars: dict[str, list[float]] = {}

    for var in EXPECTED_VARS:
        var_name = getattr(ds[var], "long_name", var)

        try:
            # Select slice for checking (time=0, level=0 for z/lnsp, else full)
            var_data = ds[var].isel(
                time=0, level=0 if var in ["z", "lnsp"] else slice(None)
            )

            # Get min, max, NaN count
            min_val, max_val, nan_count = get_min_max_nan(var_data)

            # Get valid range
            valid_range = VALID_RANGE.get(var)
            if not valid_range or len(valid_range) != 2:
                raise ValueError(f"Invalid or missing valid range for variable '{var}'")

            valid_min, valid_max = valid_range

            # Log and warn if values are out of range
            if (min_val < valid_min) or (max_val > valid_max):
                warning_msg = (
                    f'   Variable "{var}" is out of valid range {valid_range}: '
                    f"min = {min_val:.5f} [<{valid_min}],"
                    f"max = {max_val:.5f} [>{valid_max}]"
                )
                logger.warning(warning_msg)
                out_range_vars[var] = [min_val, max_val]
            else:
                logger.info(
                    f'   Variable "{var}" stats:'
                    f"min = {min_val:.5f}, max = {max_val:.5f}, NaNs = {nan_count}"
                )

            # Append issue if NaNs exist
            if nan_count > 0:
                issues.append(
                    f'Data Variable "{var}" ({var_name}) contains {nan_count} NaNs.'
                )

        except ValueError as e:
            if "only NaN values" in str(e):
                issues.append(
                    f'Variable "{var}" ({var_name}) contains only NaN values.'
                )
            else:
                issues.append(f'Error processing variable "{var}": {str(e)}')

        except Exception as e:
            issues.append(f'Unexpected error processing variable "{var}": {str(e)}')

    if issues:
        raise ValidationError("Failed validation checks:\n" + "\n".join(issues))

    return out_range_vars


@log_runtime
def validate_input(ds: xr.Dataset) -> xr.Dataset:
    """Validate and sanitize an xarray Dataset.

    This function performs a series of validation checks on the input dataset:

    - Confirms the presence of required coordinates and data variables.
    - Verifies that coordinate values (e.g., latitude, longitude, level)
       fall within expected ranges.
    - Detects and reports NaN values in key data variables.
    - Checks that all data variables are within their predefined valid ranges.
    - Clips values falling outside valid ranges to the nearest acceptable bound.

    Note
    ----
    Known ECMWF artifacts may result in small negative humidity values due to
    numerical or interpolation effects. These are clipped during validation.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset to be validated.

    Returns
    -------
    xr.Dataset
        A validated and sanitized xarray Dataset with out-of-range values clipped.

    Raises
    ------
    ValueError
        If required variables are missing, or if coordinate or data checks fail.

    """
    logger.info("Performing checkup of input file")

    # Check Coordinates and Data variables
    logger.info("  Checking coordinate ranges and data variables.")
    check_coords_and_variables(ds)

    # Check Nans and valid range
    logger.info("  Checking nans and data valid range.")
    vars_out = check_nans_valid_range(ds)

    # Clipping value outside of predifiend range
    for var in vars_out.keys():
        if vars_out[var][0] < VALID_RANGE[var][0]:
            logger.info(f"Clipping {var} below {VALID_RANGE[var][0]}")
            ds[var] = ds[var].where(ds[var] >= VALID_RANGE[var][0], VALID_RANGE[var][0])
        if vars_out[var][1] > VALID_RANGE[var][1]:
            logger.info(f"Clipping {var} above {VALID_RANGE[var][1]}")
            ds[var] = ds[var].where(ds[var] <= VALID_RANGE[var][1], VALID_RANGE[var][1])
    return ds
