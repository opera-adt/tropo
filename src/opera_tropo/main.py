from __future__ import annotations

import logging
import os
from pathlib import Path

from RAiDER import __version__ as raider_version
from RAiDER.logger import logger as raider_log

from opera_tropo import __version__
from opera_tropo.browse_image import make_browse_image_from_nc
from opera_tropo.config import pge_runconfig, runconfig
from opera_tropo.log.loggin_setup import log_runtime, setup_logging
from opera_tropo.run import tropo
from opera_tropo.utils import get_hres_datetime, get_max_memory_usage

# Logger setup
logger = logging.getLogger(__name__)


@log_runtime
def run(
    cfg: runconfig.TropoWorkflow,
    pge_runconfig: pge_runconfig.RunConfig,
    debug: bool = False,
) -> None:
    """Run the troposphere ZTD on Global Weather Model input.

    Parameters
    ----------
    cfg : TropoWorkflow
        `TropoWorkflow` object for controlling the workflow.
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.
    debug : bool, optional
        Enable debug logging.
        Default is False.

    """
    setup_logging(logger_name="opera_tropo", debug=debug, filename=cfg.log_file)  # type: ignore

    # Save the start for a metadata field
    # processing_start_datetime = datetime.now(timezone.utc)
    cfg.work_directory.mkdir(exist_ok=True, parents=True)
    cfg.output_directory.mkdir(exist_ok=True, parents=True)

    # Change to work directory
    logger.debug(f"Work directory: {cfg.work_directory}")
    os.chdir(cfg.work_directory)

    # Get output filename
    hres_date, hres_hour = get_hres_datetime(cfg.input_options.input_file_path)  # type: ignore
    output_filename = cfg.output_options.get_output_filename(hres_date, hres_hour)

    # Run troposphere workflow
    tropo(
        file_path=cfg.input_options.input_file_path,  # type: ignore
        output_file=Path(cfg.output_directory) / output_filename,  # type: ignore
        max_height=cfg.output_options.max_height,
        out_heights=cfg.output_options.output_heights,  # type: ignore
        out_chunk_size=cfg.output_options.chunk_size,  # type: ignore
        block_size=cfg.worker_settings.block_shape,  # type: ignore
        num_workers=cfg.worker_settings.n_workers,
        num_threads=cfg.worker_settings.threads_per_worker,
        max_memory=cfg.worker_settings.max_memory,
        compression_options=cfg.output_options.compression_kwargs,  # type: ignore
        temp_dir=cfg.worker_settings.dask_temp_dir,  # type: ignore
    )

    # Remove RAIDER empty log files
    for handler in raider_log.handlers:
        if isinstance(handler, logging.FileHandler):
            Path(handler.baseFilename).unlink(missing_ok=True)

    # Generate output browse image
    logger.info(f"Output file: {Path(cfg.output_directory) / output_filename}")
    output_png = Path(cfg.output_directory) / output_filename
    output_png = output_png.with_suffix(".png")
    logger.info(f"Output browse image: {output_png}")
    make_browse_image_from_nc(output_png, Path(cfg.output_directory) / output_filename)

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"RAIDER version: {raider_version}")
    logger.info(f"Current running opera_tropo version: {__version__}")
