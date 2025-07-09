from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from dask.distributed import Client

from opera_tropo._pack import pack_ztd
from opera_tropo.checks import validate_input
from opera_tropo.core import calculate_ztd
from opera_tropo.log.loggin_setup import remove_raider_logs

try:
    from RAiDER.models.model_levels import A_137_HRES, LEVELS_137_HEIGHTS

    remove_raider_logs()
except ImportError as e:
    raise ImportError(f"RAiDER is not properly installed or accessible. Error: {e}")

logger = logging.getLogger(__name__)

BLOCK_SIZE = [128, 256]  # lat, lon
DEFAULT_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}
OUTPUT_CHUNKS = [1, 8, 512, 512]  # time, height, lat, lon


def tropo(
    file_path: str,
    output_file: str,
    *,
    max_height: int = 81000,
    out_heights: Optional[list[float] | np.ndarray] = None,
    block_size: list[int] = BLOCK_SIZE,
    out_chunk_size: list[int] = OUTPUT_CHUNKS,
    num_workers: int = 4,
    num_threads: int = 2,
    max_memory: int | str = "16GB",
    compression_options: dict = DEFAULT_COMPRESSION,
    temp_dir: Optional[str] = None,
    pre_check: bool = True,
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

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input dataset file cannot be opened or processed.

    """
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
        ds = xr.open_dataset(file_path, chunks={"level": -1}, engine="h5netcdf")
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

    # Clip negative humidity values
    # due to known ECMWF numerical computation
    # and interpolation artifacts
    ds["q"] = ds.q.where(ds.q >= 0, 0)

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

    out_size = da.empty((cols, rows, len(zlevels)), dtype=np.float32)

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
    model_time_str = ds.time.dt.strftime("%Y%m%dT%H").values[0]
    logger.info(f"Estimating ZTD delay for {model_time_str}.")
    out_ds = ds.map_blocks(
        calculate_ztd, kwargs={"out_heights": out_heights}, template=template
    )

    # Define output encoding: compression and chunk size
    encoding = dict.fromkeys(out_ds.data_vars, encoding)

    # Reorder longitude indexes to adjust for 0-360  transform to -180-180
    out_ds = out_ds.sortby("longitude")
    logger.debug(f"Output file: {output_file}")
    logger.debug(
        f"Output chunksize (time, height, latitude, longitude): {out_chunk_size}"
    )

    # Save output to local file
    out_ds.sel(height=slice(None, max_height)).to_netcdf(
        output_file, encoding=encoding, mode="w"
    )
    # Close dask Client and remove dask temp. spill directory
    logger.debug(f"Closing dask server: {client.dashboard_link.split('/')[2]}.")
    client.close()
    logger.debug(f"Removing dask tmp dir: {temp_dir}")
    shutil.rmtree(str(temp_dir))
