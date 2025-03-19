from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from RAiDER.models import HRES

from ._pack import pack_ztd
from .log.loggin_setup import log_runtime

logger = logging.getLogger(__name__)


# NOTE: I could add interpolation to specific height levels here
#       to lower the resolution of the data, and its memory footprint during processing
#       but I will leave that for a future performance tests
@log_runtime
def calculate_ztd(
    da: xr.Dataset, out_heights: list = [], chunk_size=None, keep_bits: bool = True
) -> xr.Dataset:
    """Calculate Zenith Total Delay (ZTD) using HRES weather model data."""
    hres_model = HRES()

    # Extract temperature and specific humidity at the first time step
    hres_model._t = da.t.isel(time=0).values
    hres_model._q = da.q.isel(time=0).values

    # Extract longitude and latitude values
    longitude = da.longitude.values
    latitude = da.latitude.values

    # Use geopotential heights and log of surface pressure
    # to get pressure, geopotential, and geopotential height
    _, pres, hgt = hres_model._calculategeoh(
        da.z.isel(time=0, level=0).values, da.lnsp.isel(time=0, level=0).values
    )
    hres_model._p = pres

    # Create latitude and longitude grid
    hres_model._lons, hres_model._lats = np.meshgrid(longitude, latitude)

    # Get altitudes
    hres_model._get_heights(hres_model._lats, hgt.transpose(1, 2, 0))
    h = hres_model._zs.copy()

    # Re-structure arrays from (heights, lats, lons) to (lons, lats, heights)
    hres_model._p = np.flip(hres_model._p.transpose(1, 2, 0), axis=2)
    hres_model._t = np.flip(hres_model._t.transpose(1, 2, 0), axis=2)
    hres_model._q = np.flip(hres_model._q.transpose(1, 2, 0), axis=2)
    hres_model._zs = np.flip(h, axis=2)
    hres_model._xs, hres_model._ys = hres_model._lons.copy(), hres_model._lats.copy()

    # Perform RAiDER computations
    hres_model._find_e()  # Compute partial pressure of water vapor
    hres_model._uniform_in_z(_zlevels=None)

    hres_model._checkForNans()
    hres_model._get_wet_refractivity()
    hres_model._get_hydro_refractivity()
    hres_model._adjust_grid(hres_model.get_latlon_bounds())

    # Compute zenith delays at the weather model grid nodes
    hres_model._getZTD()

    # Package ztd
    ztd_xr = pack_ztd(
        hres_model._wet_ztd,
        hres_model._hydrostatic_ztd,
        longitude,
        latitude,
        hres_model._zs,
        da.time.data,
        chunk_size=chunk_size,
        keep_bits=keep_bits,
    )
    # Do some cleanup
    del hres_model, longitude, latitude, h, pres, hgt

    # Interpolate
    if out_heights is not None and len(out_heights) > 1:
        ztd_xr = ztd_xr.interp(height=out_heights, method="cubic")

    return ztd_xr
