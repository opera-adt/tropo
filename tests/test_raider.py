from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_prepare_hres_model(load_input_model, init_raider):
    # Init RAiDER HRES instance
    model = init_raider

    # load test data
    da = load_input_model
    da = da.isel(time=0)

    # Step 1: Calculate surface_pressure, geopotenial heights
    geop, pres, hgt = model._calculategeoh(
        da.z.isel(level=0).values, da.lnsp.isel(level=0).values
    )

    err_msg = "f:_calculategeoh, %s values do not match!"
    assert_allclose(da.geop, geop, err_msg=err_msg % "geopotential")
    assert_allclose(da.p, pres, err_msg=err_msg % "surface_pressure")
    assert_allclose(da.ght, hgt, err_msg=err_msg % "geo_height")

    # Step 2: Convert geoheight to ellipsoidal heights
    model._p = pres
    model._get_heights(model._lats, hgt.transpose(1, 2, 0))
    h = model._zs.copy()

    err_msg = "f:_get_heights, %s values do not match!"
    assert_allclose(
        da.hgt, h.transpose(2, 0, 1), err_msg=err_msg % "ellipsoidal_height"
    )

    # Step 3: Get partial water vapor pressure
    model._p = np.flip(model._p.transpose(1, 2, 0), axis=2)
    model._t = np.flip(model._t.transpose(1, 2, 0), axis=2)
    model._q = np.flip(model._q.transpose(1, 2, 0), axis=2)
    model._zs = np.flip(h, axis=2)
    model._find_e()  # Compute partial pressure of water vapor

    err_msg = "f:_find_e, %s values do not match!"
    assert_allclose(
        da.e, model._e.transpose(2, 0, 1), err_msg=err_msg % "partial_water_vapor"
    )


def test_raider_calculate_ztd(load_input_model, load_golden_output, init_raider):
    # Init RAiDER HRES instance
    model = init_raider

    # load test data
    da = load_input_model
    da = da.isel(time=0)

    # load golden output
    out = load_golden_output
    out = out.isel(time=0)

    # Use geopotential heights and log of surface pressure
    # to get pressure, geopotential, and geopotential height
    _, pres, hgt = model._calculategeoh(
        da.z.isel(level=0).values, da.lnsp.isel(level=0).values
    )
    model._p = pres

    # Get altitudes
    model._get_heights(model._lats, hgt.transpose(1, 2, 0))
    h = model._zs.copy()

    # Re-structure arrays from (heights, lats, lons) to (lons, lats, heights)
    model._p = np.flip(model._p.transpose(1, 2, 0), axis=2)
    model._t = np.flip(model._t.transpose(1, 2, 0), axis=2)
    model._q = np.flip(model._q.transpose(1, 2, 0), axis=2)
    model._zs = np.flip(h, axis=2)
    model._xs, model._ys = model._lons.copy(), model._lats.copy()

    # Perform RAiDER computations
    model._find_e()  # Compute partial pressure of water vapor
    model._uniform_in_z(_zlevels=None)

    model._checkForNans()
    model._get_wet_refractivity()

    err_msg = "f:_get_wet_refractivity, %s values do not match!"
    assert_allclose(
        out.wet_refractivity,
        model._wet_refractivity,
        err_msg=err_msg % "wet_refractivity",
    )

    model._get_hydro_refractivity()
    err_msg = "f:_get_hydro_refractivity, %s values do not match!"
    assert_allclose(
        out.hydrostatic_refractivity,
        model._hydrostatic_refractivity,
        err_msg=err_msg % "hydrostatic_refractivity",
    )
    model._adjust_grid(model.get_latlon_bounds())

    # Compute zenith delays at the weather model grid nodes
    model._getZTD()
    err_msg = "f:_getZTD, %s values do not match!"
    assert_allclose(out.wet_ztd, model._wet_ztd, err_msg=err_msg % "wet_ztd")
    assert_allclose(
        out.hydrostatic_ztd, model._hydrostatic_ztd, err_msg=err_msg % "hydrostatic_ztd"
    )


@pytest.mark.parametrize(
    "key",
    [
        "_k1",
        "_k2",
        "_k3",
        "_humidityType",
        "_a",
        "_b",
        "_model_level_type",
        "_R_v",
        "_R_d",
        "_g0",
        "_zmin",
        "_zmax",
        "_zlevels",
    ],
)
def test_hres_class(load_hres_model, key):
    golden_dict, model_dict = load_hres_model

    # Check if key exists in both dictionaries
    assert key in golden_dict, f"Key '{key}' not found in golden_dict"
    assert key in model_dict, f"Key '{key}' not found in model_dict"

    # Get the values for the key
    golden_value = golden_dict[key]
    model_value = model_dict[key]

    # Handle comparison if values are arrays or lists
    if isinstance(golden_value, np.ndarray) and isinstance(model_value, np.ndarray):
        assert np.array_equal(golden_value, model_value), (
            f"Values for key '{key}' do not match: golden_dict = {golden_value},"
            f" model_dict = {model_value}"
        )
    else:
        assert golden_value == model_value, (
            f"Values for key '{key}' do not match: golden_dict = {golden_value},"
            f" model_dict = {model_value}"
        )
