import numpy as np
import xarray as xr

from .product_info import GLOBAL_ATTRS, TROPO_PRODUCTS
from .utils import round_mantissa

# NOTE: check if it is better to add attributes at the end to
#       leave this empty and more light


def pack_ztd(
    wet_ztd: np.ndarray,
    hydrostatic_ztd: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    zs: np.ndarray,
    model_time: np.ndarray,
    chunk_size={"longitude": 128, "latitude": 128, "height": -1, "time": 1},
    keep_bits: bool = True,
):
    """Package Zenith Total Delay (ZTD) data into an xarray Dataset.

    This function packages wet and hydrostatic Zenith Total Delay (ZTD) data
    along with corresponding geographical coordinates (longitude, latitude, height),
    and model time into an `xarray.Dataset`. The data is chunked for efficient
    processing, and optionally, the bit-level precision of the data is preserved.

    Parameters
    ----------
    wet_ztd : np.ndarray
        A NumPy array containing the wet zenith total delays.
    hydrostatic_ztd : np.ndarray
        A NumPy array containing the hydrostatic zenith total delays.
    lons : np.ndarray
        A NumPy array containing the longitude values (in degrees).
    lats : np.ndarray
        A NumPy array containing the latitude values (in degrees).
    zs : np.ndarray
        A NumPy array containing the height (altitude) values (in meters).
    model_time : np.ndarray
        A NumPy array containing the timestamps of the model data.
    chunk_size : dict, optional
        A dictionary specifying the chunk sizes for the dataset dimensions.
        Defaults to `{"longitude": 128, "latitude": 128, "height": -1, "time": 1}`.
    keep_bits : bool, optional
        If `True`, preserves the bit-level precision of the data. Default is `True`.

    Returns
    -------
    xarray.Dataset
        An `xarray.Dataset` containing the packaged ZTD data with associated
        coordinates and chunking applied.

    """
    dim = ["height", "latitude", "longitude"]
    reference_time = model_time.astype("datetime64[s]").astype("O")[0]
    reference_time = reference_time.strftime("%Y-%m-%d %H:%M:%S")

    # total_zenith_delay = hydrostatic_ztd + wet_ztd
    wet_ztd = wet_ztd.astype(TROPO_PRODUCTS.wet_delay.dtype)
    hydrostatic_ztd = hydrostatic_ztd.astype(TROPO_PRODUCTS.hydrostatic_delay.dtype)
    zs = zs.astype("float64")

    # Rounding
    if keep_bits:
        if TROPO_PRODUCTS.wet_delay.keep_bits:
            round_mantissa(wet_ztd, keep_bits=int(TROPO_PRODUCTS.wet_delay.keep_bits))
        if TROPO_PRODUCTS.hydrostatic_delay.keep_bits:
            round_mantissa(
                hydrostatic_ztd,
                keep_bits=int(TROPO_PRODUCTS.hydrostatic_delay.keep_bits),
            )

    ds = xr.Dataset(
        data_vars={
            "wet_delay": (
                dim,
                wet_ztd.transpose(2, 0, 1),
                TROPO_PRODUCTS.wet_delay.to_dict(),
            ),
            "hydrostatic_delay": (
                dim,
                hydrostatic_ztd.transpose(2, 0, 1),
                TROPO_PRODUCTS.hydrostatic_delay.to_dict(),
            ),
        },
        # normalizing longitudes to the range [-180, 180] from [0, 360]
        # GDAL expects coordinates to be float64
        coords={
            "height": zs,
            "latitude": np.float64(lats),
            "longitude": (np.float64(lons) + 180) % 360 - 180,
        },
        attrs=GLOBAL_ATTRS | {"reference_time": reference_time},
    )

    # Add coords attrs
    ds["height"].attrs.update(TROPO_PRODUCTS.coords.height.get_attr)
    ds["latitude"].attrs.update(TROPO_PRODUCTS.coords.latitude.get_attr)
    ds["longitude"].attrs.update(TROPO_PRODUCTS.coords.longitude.get_attr)

    # Add time
    ds = ds.expand_dims({"time": model_time})
    ds["time"].attrs.update(TROPO_PRODUCTS.coords.time.get_attr)
    # Remove time units due to conflicts with encoding
    del ds["time"].attrs["units"]
    ds["time"].encoding.update(TROPO_PRODUCTS.coords.time.encoding)

    # Add spatial reference
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # Data Variables
    wet_fill = TROPO_PRODUCTS.wet_delay.fillvalue
    hydro_fill = TROPO_PRODUCTS.hydrostatic_delay.fillvalue
    ds["wet_delay"].attrs["_FillValue"] = wet_fill
    ds["hydrostatic_delay"].attrs["_FillValue"] = hydro_fill

    # Add chunks to data variables
    if chunk_size is not None:
        for key in ["wet_delay", "hydrostatic_delay"]:
            ds[key] = ds[key].chunk(chunk_size)

        # Ensure that chunking is applied to the entire dataset
        ds = ds.chunk(chunk_size)
    return ds
