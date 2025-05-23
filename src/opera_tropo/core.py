import logging
from typing import Optional

import numpy as np
import xarray as xr
from RAiDER.models import HRES

from opera_tropo._pack import pack_ztd
from opera_tropo.log.loggin_setup import log_runtime, remove_raider_logs

logger = logging.getLogger(__name__)
remove_raider_logs()


def get_ztd(
    lat: np.ndarray,
    lon: np.ndarray,
    temperature: np.ndarray,
    humidity: np.ndarray,
    z: np.ndarray,
    lnsp: np.ndarray,
) -> xr.Dataset:
    """Compute Zenith Total Delay (ZTD) using the HRES weather model.

    Parameters
    ----------
    lat : np.ndarray
        1D array of latitude values in degrees.
    lon : np.ndarray
        1D array of longitude values in degrees.
    temperature : np.ndarray
        3D array of temperature values (K) with dimensions
        (height, latitude, longitude).
    humidity : np.ndarray
        3D array of specific humidity values (kg/kg) with dimensions
        (height, latitude, longitude).
    z : np.ndarray
        2D array of geopotential values (m²/s²) with dimensions
        (latitude, longitude).
    lnsp : np.ndarray
        2D array of the natural logarithm of surface pressure (Pa) with
        dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        A dataset containing:
        - 'wet_ztd': Zenith Wet Delay (m).
        - 'hydrostatic_ztd': Zenith Hydrostatic Delay (m).
        - Coordinates: 'latitude', 'longitude', 'height'.

    Notes
    -----
    - Uses the HRES weather model for atmospheric profiling.
    - Applies RAiDER processing for delay computations.

    """
    # Initialize HRES model
    hres_model = HRES()

    # Assign temperature and specific humidity
    hres_model._t = temperature
    hres_model._q = humidity

    # Compute pressure and geopotential height from geopotential and log pressure
    hres_model._p, hgt = hres_model._calculategeoh(z, lnsp)[1:]

    # Create latitude and longitude grid
    hres_model._lons, hres_model._lats = np.meshgrid(lon, lat)

    # Compute altitudes
    hres_model._get_heights(hres_model._lats, hgt.transpose(1, 2, 0))
    del hgt  # Free memory

    # Reorder dimensions from (height, lat, lon) to (lon, lat, height)
    hres_model._p = np.flip(hres_model._p.transpose(1, 2, 0), axis=2)
    hres_model._t = np.flip(hres_model._t.transpose(1, 2, 0), axis=2)
    hres_model._q = np.flip(hres_model._q.transpose(1, 2, 0), axis=2)
    hres_model._zs = np.flip(hres_model._zs, axis=2)

    # Perform RAiDER computations
    hres_model._find_e()  # Compute partial pressure of water vapor
    hres_model._uniform_in_z(_zlevels=None)  # Interpolate to common heights
    hres_model._checkForNans()  # Handle NaNs at boundaries
    hres_model._get_wet_refractivity()
    hres_model._get_hydro_refractivity()
    hres_model._adjust_grid(hres_model.get_latlon_bounds())

    # Compute Zenith Total Delay (ZTD)
    hres_model._getZTD()

    # Construct output dataset
    dims = ["latitude", "longitude", "height"]
    out_ds = xr.Dataset(
        data_vars={
            "wet_ztd": (dims, hres_model._wet_ztd),
            "hydrostatic_ztd": (dims, hres_model._hydrostatic_ztd),
        },
        coords={
            "height": ("height", hres_model._zs),
            "latitude": ("latitude", hres_model._lats[:, 0]),
            "longitude": ("longitude", hres_model._lons[0, :]),
        },
    )

    return out_ds


@log_runtime
def calculate_ztd(
    ds: xr.Dataset,
    out_heights: Optional[list] = None,
    chunk_size: Optional[list] = None,
    keep_bits: bool = True,
) -> xr.Dataset:
    """Compute the Zenith Total Delay (ZTD) from an input weather model dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing atmospheric variables, including:
        - 'latitude': 1D array of latitude values (degrees).
        - 'longitude': 1D array of longitude values (degrees).
        - 't': 4D array of temperature values (K) with dimensions
          (time, level, latitude, longitude).
        - 'q': 4D array of specific humidity values (kg/kg) with dimensions
          (time, level, latitude, longitude).
        - 'z': 4D array of geopotential values (m²/s²) with dimensions
          (time, level, latitude, longitude).
        - 'lnsp': 3D array of log surface pressure (Pa) with dimensions
          (time, latitude, longitude).
        - 'time': 1D array of timestamps.

    out_heights : Optional[list], default=None
        List of desired output height levels for interpolation (meters).

    chunk_size : Optional[list], default=None
        List specifying the chunk size for output dataset processing.

    keep_bits : bool, default=True
        Do mantissa rounding with bit range defind in product_info.

    Returns
    -------
    xr.Dataset
        A dataset containing:
        - 'wet_ztd': Zenith Wet Delay (m).
        - 'hydrostatic_ztd': Zenith Hydrostatic Delay (m).
        - Coordinates: 'latitude', 'longitude', 'height'.

    """
    # Get ZTD from weather model dataset
    ztd_ds = get_ztd(
        lat=ds.latitude.values,
        lon=ds.longitude.values,
        temperature=ds.t.isel(time=0).values,
        humidity=ds.q.isel(time=0).values,
        z=ds.z.isel(time=0, level=0).values,
        lnsp=ds.lnsp.isel(time=0, level=0).values,
    )

    # Interpolate to specified output heights if provided
    if out_heights is not None:
        ztd_ds = ztd_ds.interp(height=out_heights, method="cubic")

    # Package and round results using `pack_ztd` using
    # product_info.TropoProducts
    ztd_ds = pack_ztd(
        wet_ztd=ztd_ds.wet_ztd.values,
        hydrostatic_ztd=ztd_ds.hydrostatic_ztd.values,
        lons=ztd_ds.longitude,
        lats=ztd_ds.latitude,
        zs=ztd_ds.height,
        model_time=ds.time.data,
        chunk_size=chunk_size,
        keep_bits=keep_bits,
    )

    return ztd_ds
