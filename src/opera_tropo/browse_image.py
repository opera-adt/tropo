from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Union

import cmap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy import ndimage

if TYPE_CHECKING:
    from builtins import ellipsis

    Index = ellipsis | slice | int
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

PathOrStr = Union[str, PathLikeStr]
Filename = PathOrStr

DEFAULT_CMAP = cmap.Colormap("arctic_r").to_mpl()


def _resize_to_max_pixel_dim(arr: ArrayLike, max_dim_allowed=2048) -> np.ndarray:
    """Scale shape of a given array."""
    if max_dim_allowed < 1:
        raise ValueError(f"{max_dim_allowed} is not a valid max image dimension")
    input_shape = arr.shape
    scaling_ratio = max([max_dim_allowed / xy for xy in input_shape])
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0
    arr = ndimage.zoom(arr, scaling_ratio)
    arr[ndimage.zoom(nan_mask, scaling_ratio, order=0)] = np.nan
    return arr


def _save_to_disk_as_color(
    arr: ArrayLike, fname: Filename, cmap: str, vmin: float, vmax: float
) -> None:
    """Save image array as color to file."""
    plt.imsave(fname, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def make_browse_image_from_arr(
    output_filename: Filename,
    arr: ArrayLike,
    max_dim_allowed: int = 2048,
    cmap: str = DEFAULT_CMAP,
    vmin: float = -0.10,
    vmax: float = 0.10,
) -> None:
    """Create a PNG browse image for the output product from given array."""
    arr = _resize_to_max_pixel_dim(arr, max_dim_allowed)
    _save_to_disk_as_color(arr, output_filename, cmap, vmin, vmax)


def make_browse_image_from_nc(
    output_filename: Filename,
    input_filename: Filename,
    max_dim_allowed: int = 2048,
    cmap: str = DEFAULT_CMAP,
    vmin: float = 1.9,
    vmax: float = 2.5,
    height: float = 800,
) -> None:
    """Create a PNG browse image for the output product from product in NetCDF file."""
    # Extract ZTD at zero height for browse image
    with xr.open_dataset(input_filename) as ds:
        wet = ds.wet_delay.isel(time=0).sel(height=0).data
        hydrostatic = (
            ds.hydrostatic_delay.isel(time=0).sel(height=height, method="nearest").data
        )

    ztd = wet + hydrostatic

    make_browse_image_from_arr(output_filename, ztd, max_dim_allowed, cmap, vmin, vmax)
