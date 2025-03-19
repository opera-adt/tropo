from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import xarray as xr
from dask.distributed import Client

from ._pack import pack_ztd
from .checks import validate_input
from .core import calculate_ztd
from .product_info import TROPO_PRODUCTS
from .utils import round_mantissa_xr

try:
    from RAiDER.models.model_levels import A_137_HRES, LEVELS_137_HEIGHTS
except ImportError as e:
    print(f"RAiDER is not properly installed or accessible. Error: {e}")

logger = logging.getLogger(__name__)

BLOCK_SIZE = [128, 256]  # lat, lon
OUTPUT_CHUNKS = [1, 8, 512, 512]  # time, height, lat, lon


def _rounding_mantissa_blocks(ds: xr.DataArray, keep_bits: int):
    """Round the mantissa blocks of a given xr.DataArray.

    Parameters
    ----------
    ds : xr.DataArray
        The input xr.DataArray to process.
    keep_bits : int
        The number of bits to keep in the mantissa blocks.

    Returns
    -------
    xr.DataArray
        The processed xr.DataArray with rounded mantissa blocks.

    """
    return ds.map(round_mantissa_xr, keep_bits=keep_bits)


def tropo(
    file_path: str,
    output_file: str,
    *,
    max_height: int = 81000,
    out_heights: Optional[list[float]] = None,
    block_size: list[int] = BLOCK_SIZE,
    out_chunk_size: list[int] = OUTPUT_CHUNKS,
    num_workers: int = 4,
    num_threads: int = 2,
    max_memory: int | str = "16GB",
    compression_options: dict = {},
    temp_dir: Optional[str] = None,
    pre_check: bool = True,
    keep_bits: bool = True,
) -> None:
    """Run troposphere workflow.

    Parameters
    ----------
    file_path : str
        Path to the input dataset file.
    output_file : str
        Path to the output NetCDF file.
    max_height : int, optional
        Maximum height in meters. Default is 81,000.
    out_heights : list of int, optional
        List of output heights. Default is None (using model heights).
    block_size : list of int, optional
        Block size for processing. Default is [128, 128].
    out_chunk_size : list of int, optional
        Chunk size for output data. Default is [1, 8, 512, 512].
    num_workers : int, optional
        Number of parallel workers. Default is 4.
    num_threads : int, optional
        Number of threads per worker. Default is 2.
    max_memory : int or str, optional
        Maximum memory allocation. Default is '16GB'.
    compression_options : dict, optional
        Compression options for the output NetCDF file. Default is None.
    temp_dir : str, optional
        Directory for temporary files. Default is None.
    pre_check : bool, optional
        Whether to perform pre-check of input data. Default is True.
    keep_bits : bool, optional
        Whether to preserve bit depth in the output. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input dataset file cannot be opened or processed.

    """
    process = psutil.Process()
    logger.info("Calculating TROPO delay")
    # Set default compression options if not provided
    encoding_defaults = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": out_chunk_size,
    }
    encoding = {**encoding_defaults, **compression_options}

    # Setup Dask Client and temp. directory
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    client = Client(
        n_workers=num_workers,
        threads_per_worker=num_threads,
        memory_limit=max_memory,
        local_directory=temp_dir,
    )
    logger.debug(f"Dask server link: {client.dashboard_link}")

    # Open the dataset
    try:
        ds = xr.open_dataset(file_path, chunks={"level": -1})
    except Exception:
        raise ValueError(
            f"Failed to open the dataset file: {file_path}."
            "Make sure the file exists and is a valid dataset."
            "Original error: {e}"
        )

    # Validate input, check valid range,
    #  nan values and exp. var and coords
    if pre_check:
        validate_input(ds)

    # Rechunk for parallel processing
    logger.debug(f"Rechunking {file_path}")
    chunks = {
        "longitude": block_size[1],
        "latitude": block_size[0],
        "time": 1,
        "level": len(A_137_HRES) - 1,
    }
    ds = ds.chunk(chunks)

    chunksizes = {key: value[0] for key, value in ds.chunksizes.items()}
    logger.debug(f"Chunk sizes: {chunksizes}")

    # Get output size
    cols = ds.sizes.get("latitude")
    rows = ds.sizes.get("longitude")

    if out_heights is not None and len(out_heights) > 0:
        zlevels = np.array(out_heights)
    else:
        zlevels = np.flipud(LEVELS_137_HEIGHTS)

    out_size = np.empty((cols, rows, len(zlevels)), dtype=np.float32)

    # To skip interpolation if out_heights are same as default
    if np.array_equal(out_heights, np.flipud(LEVELS_137_HEIGHTS)):
        out_heights = None

    # Get output template
    template = pack_ztd(
        wet_ztd=out_size,
        hydrostatic_ztd=out_size,
        lons=ds.longitude.values,
        lats=ds.latitude.values,
        zs=zlevels,
        model_time=ds.time.values,
        chunk_size={
            "longitude": int(chunksizes["longitude"]),
            "latitude": int(chunksizes["latitude"]),
            "height": -1,
            "time": 1,
        },
        keep_bits=False,
    )

    # Calculate ZTD
    # NOTE: Peak in RAM is 40GB after ingesting map_block output
    # with default 145 height level, specifying out_heights can
    # lower mem usage
    mem = process.memory_info().rss / 1e9
    logger.info(f"Estimating ZTD delay, mem: {mem:.2f}GB")
    t1 = time.time()
    out_ds = ds.map_blocks(
        calculate_ztd, kwargs={"out_heights": out_heights}, template=template
    ).compute()
    t2 = time.time()
    mem = process.memory_info().rss / 1e9
    logger.info(f"ZTD took {t2 - t1:.2f}s, mem: {mem:.2f}GB")

    # Clean up
    ds.close()
    del template, ds

    # Note, apply again rounding as interpolation can change
    # output, double check if needed
    if out_heights is not None and len(out_heights) > 0 and keep_bits:
        # use one keep_bits setting, need to figure how to apply
        # different rounding for each data_var in xr.Dataset
        keep_bit_kwargs = {"keep_bits": TROPO_PRODUCTS.wet_delay.keep_bits}
        out_ds = out_ds.map_blocks(_rounding_mantissa_blocks, kwargs=keep_bit_kwargs)

    # Save and compress output
    t1 = time.time()
    msg = "and Compressing" if encoding["zlib"] else ""
    encoding = {var: encoding for var in out_ds.data_vars}
    # Reoder longitude indexes to adjust for 0-360  transform to -180-180
    out_ds = out_ds.sortby("longitude")
    logger.debug(f"Saving file: {output_file}")
    out_ds.sel(height=slice(None, max_height)).to_netcdf(
        output_file, encoding=encoding, mode="w"
    )
    t2 = time.time()
    mem = process.memory_info().rss / 1e9
    logger.info(f"Saving {msg} took {t2 - t1:.2f}s, mem: {mem:.2f}GB")
    client.close()
    shutil.rmtree(str(temp_dir))
