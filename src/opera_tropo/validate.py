import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


def compare_attrs(ds1: xr.Dataset, ds2: xr.Dataset):
    """Compare dataset attributes while ignoring the 'history' key.

    Parameters
    ----------
    ds1 : xr.Dataset
        First dataset to compare.
    ds2 : xr.Dataset
        Second dataset to compare.

    Returns
    -------
    bool
        True if the datasets have identical attributes
        (excluding 'history'), False otherwise.

    """
    filtered_dict1 = {k: v for k, v in ds1.attrs.items() if k != "history"}
    filtered_dict2 = {k: v for k, v in ds2.attrs.items() if k != "history"}

    return filtered_dict1 == filtered_dict2


def compare_coord_attrs(ds1: xr.Dataset, ds2: xr.Dataset):
    """Compare coordinate attributes between two datasets.

    Parameters
    ----------
    ds1 : xr.Dataset
        First dataset to compare.
    ds2 : xr.Dataset
        Second dataset to compare.

    Returns
    -------
    bool
        True if coordinate attributes match, False otherwise.

    """
    dict1 = {coord: ds1[coord].attrs for coord in ds1.coords}
    dict2 = {coord: ds2[coord].attrs for coord in ds2.coords}

    return dict1 == dict2


def compare_two_datasets(xr_file1: str | Path, xr_file2: str | Path) -> None:
    """Compare two xarray Dataset.

    Parameters
    ----------
    xr_file1 : str or Path
        Path to the first dataset file.
    xr_file2 : str or Path
        Path to the second dataset file.

    Raises
    ------
    AssertionError
        If any dataset properties (dimensions, variable names,
        attributes, or data values) do not match.

    """
    # Load two xarray Datasets
    ds1 = xr.open_dataset(xr_file1)
    ds2 = xr.open_dataset(xr_file2)

    # Compare dataset dimensions
    logger.info("Test Dataset dimensions")
    assert ds1.sizes == ds2.sizes, "Dataset dimensions do not match"

    # Compare variable names
    logger.info("Test Dataset Variable names")
    assert set(ds1.data_vars) == set(ds2.data_vars), "Variable names do not match"

    # Check global attrs
    logger.info("Test Dataset Global attributes")
    assert compare_attrs(
        ds1, ds2
    ), "Global Attribute keys should match (ignoring 'history')"

    # Check coord attrs
    logger.info("Test Dataset Coordinate attributes")
    assert compare_coord_attrs(ds1, ds2), "Coords Attribute keys should match"

    # Raises an AssertionError if two objects are not equal.
    #  This will match data values, dimensions and coordinates,
    # but not names or attributes
    logger.info("Test Dataset Variable Data values")
    xr.testing.assert_equal(ds1, ds2)

    logger.info(f"✅ Datasets {Path(xr_file1).name} and {Path(xr_file2).name} match!")
