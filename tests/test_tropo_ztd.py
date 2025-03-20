import numpy as np
from numpy.testing import assert_allclose

from opera_tropo.core import calculate_ztd
from opera_tropo.product_info import TropoProducts
from opera_tropo.run import _rounding_mantissa_blocks


def test_wet_delay(load_input_model, load_golden_output):
    # Load test dataset
    ds = load_input_model

    # Load golden dataset
    golden_out = load_golden_output
    golden_out = golden_out.transpose("time", "height", "latitude", "longitude")

    # Calculate ztd
    out_ds = calculate_ztd(ds)

    # Take into account manitissa rounding
    keep_bits = TropoProducts().wet_delay.keep_bits
    golden_out = _rounding_mantissa_blocks(
        golden_out.astype(np.float32), keep_bits=int(keep_bits)
    )

    assert_allclose(out_ds.wet_delay, golden_out.wet_ztd)


def test_hydrostatic_delay(load_input_model, load_golden_output):
    # Load test dataset
    ds = load_input_model

    # Load golden dataset
    golden_out = load_golden_output
    golden_out = golden_out.transpose("time", "height", "latitude", "longitude")

    # Calculate ztd
    out_ds = calculate_ztd(ds)

    # Take into account manitissa rounding
    keep_bits = TropoProducts().hydrostatic_delay.keep_bits
    golden_out = _rounding_mantissa_blocks(
        golden_out.astype(np.float32), keep_bits=int(keep_bits)
    )

    assert_allclose(out_ds.hydrostatic_delay, golden_out.hydrostatic_ztd)
