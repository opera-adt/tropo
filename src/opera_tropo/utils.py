from __future__ import annotations

import resource
import sys

import numpy as np
import xarray as xr


# This is obsolete
def get_chunks_indices(xr_array: xr.Dataset) -> list:
    """Retrieve indices for chunked slices of an xarray Dataset.

    This function returns a list of slice objects that define the chunked
    sections of the dataset based on its dask chunking structure.

    Parameters
    ----------
    xr_array : xr.Dataset
        The input xarray Dataset.

    Returns
    -------
    list of tuple of slices
        A list where each element is a tuple of slice objects representing
        the chunked indices for each dimension.

    """
    chunks = xr_array.chunks

    iy, ix = chunks["latitude"], chunks["longitude"]

    idx = [sum(ix[:i]) for i in range(len(ix) + 1)]
    idy = [sum(iy[:i]) for i in range(len(iy) + 1)]

    slices = []

    for i in range(len(idy) - 1):  # Y-axis slices for idy
        for j in range(len(idx) - 1):  # X-axis slices for idx
            # Create a slice using the ranges of idt, idy, and idx
            slice_ = {
                "time": np.s_[:],
                "level": np.s_[:],
                "latitude": np.s_[idy[i] : idy[i + 1]],
                "longitude": np.s_[idx[j] : idx[j + 1]],
            }
            slices.append(slice_)
    return slices


def get_max_memory_usage(units: str = "GB", children: bool = True) -> float:
    """Get the maximum memory usage of the current process.

    Parameters
    ----------
    units : str, optional, choices=["GB", "MB", "KB", "byte"]
        The units to return, by default "GB".
    children : bool, optional
        Whether to include the memory usage of child processes, by default True

    Returns
    -------
    float
        The maximum memory usage in the specified units.

    Raises
    ------
    ValueError
        If the units are not recognized.

    References
    ----------
    1. https://stackoverflow.com/a/7669279/4174466
    2. https://unix.stackexchange.com/a/30941/295194
    3. https://manpages.debian.org/bullseye/manpages-dev/getrusage.2.en.html

    """
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if children:
        max_mem += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if units.lower().startswith("g"):
        factor = 1e9
    elif units.lower().startswith("m"):
        factor = 1e6
    elif units.lower().startswith("k"):
        factor = 1e3
    elif units.lower().startswith("byte"):
        factor = 1.0
    else:
        msg = f"Unknown units: {units}"
        raise ValueError(msg)
    if sys.platform.startswith("linux"):
        # on linux, ru_maxrss is in kilobytes, while on mac, ru_maxrss is in bytes
        factor /= 1e3

    return max_mem / factor


def get_hres_datetime(file_path: str):
    """Extract high-resolution date and hour from an xarray Dataset.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.

    Returns
    -------
    tuple[str, int]
        A tuple containing the date in 'YYYYMMDD' format
        and the hour as an integer.

    """
    with xr.open_dataset(file_path) as ds:
        hres_date = ds.time.dt.date.data[0].strftime("%Y%m%d")
        hres_hour = ds.time.dt.hour.data[0]
    return hres_date, hres_hour


# Round_mantissa function from
# https://github.com/isce-framework/dolphin/blob/ee4271fa6e085168587cb96f977b1617a75304e1/src/dolphin/io/_utils.py#L244
def round_mantissa(z: np.ndarray, keep_bits=10):
    """Zero out mantissa bits of elements of array in place.

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression.

    Parameters
    ----------
    z : numpy.ndarray
        Real or complex array whose mantissas are to be zeroed out
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to 10.
        Lower numbers will truncate the mantissa more and enable
        more compression.

    References
    ----------
    https://numcodecs.readthedocs.io/en/v0.12.1/_modules/numcodecs/bitround.html

    """
    max_bits = {
        "float16": 10,
        "float32": 23,
        "float64": 52,
    }

    # MG: If input is xarray.DataArray, process its values
    is_xarray = isinstance(z, xr.DataArray)
    z = z.values if is_xarray else z

    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, keep_bits)
        round_mantissa(z.imag, keep_bits)
        return

    if not z.dtype.kind == "f" or z.dtype.itemsize > 8:
        raise TypeError("Only float arrays (16-64bit) can be bit-rounded")

    bits = max_bits[str(z.dtype)]
    # cast float to int type of same width (preserve endianness)
    a_int_dtype = np.dtype(z.dtype.str.replace("f", "i"))
    all_set = np.array(-1, dtype=a_int_dtype)
    if keep_bits == bits:
        return z
    if keep_bits > bits:
        raise ValueError("keep_bits too large for given dtype")

    b = z.view(a_int_dtype)
    maskbits = bits - keep_bits
    mask = (all_set >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask


def round_mantissa_xr(data, keep_bits=10):
    """Round the mantissa of a floating-point xarray DataArray.

    This function modifies only floating-point arrays, ensuring that integer
    arrays (e.g., spatial reference data) remain unchanged.

    Parameters
    ----------
    data : xr.DataArray
        The input xarray DataArray.
    keep_bits : int, optional
        The number of bits to keep in the mantissa (default is 10).

    Returns
    -------
    xr.DataArray
        The processed DataArray with rounded mantissa for floating-point values.

    """
    # Ensure processing only happens for floating-point arrays
    #  Skip int, e.g. spatial_ref in data_var gives error
    if np.issubdtype(data.dtype, np.floating):
        round_mantissa(data, keep_bits)
    return data
