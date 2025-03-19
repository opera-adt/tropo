from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
)

from ._yaml import YamlModel

try:
    from RAiDER.models.model_levels import LEVELS_137_HEIGHTS
except ImportError as e:
    print(f"RAiDER is not properly installed or accessible. Error: {e}")


logger = logging.getLogger(__name__)

__all__ = [
    "InputOptions",
    "OutputOptions",
    "WorkerSettings",
    "TropoWorkflow",
]

PRODUCT_VERSION = "0.1"
DEFAULT_ENCODING_OPTIONS = {"zlib": True, "complevel": 5, "shuffle": True}


# Base model
class InputOptions(BaseModel, extra="forbid"):
    """Options specifying input datasets for workflow."""

    _directory: Path = PrivateAttr(Path("data"))

    input_file_path: str | Path = Field(
        default_factory=str,
        description="Path to the input HRES model hres_model.nc",
    )

    date_fmt: str = Field(
        "%Y%m%d",
        description="Format of dates contained in s3 HRES folder",
    )


class OutputOptions(BaseModel, extra="forbid"):
    """Options specifying input datasets for workflow."""

    _directory: Path = PrivateAttr(Path("output"))

    date_fmt: str = Field(
        "%Y%m%dT%H%M%S",
        description="Output Date Format for OPERA TROPO",
    )

    creation_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time the config file was created",
    )

    max_height: int = Field(
        81000,
        description="Clip heights above specified maximum height.",
    )

    output_heights: Optional[List[float]] = Field(
        default=list(reversed(LEVELS_137_HEIGHTS)),
        description=(
            "Output height level to hydrostatic and wet delay,"
            " default: HRES native 145 height levels."
        ),
    )

    chunk_size: tuple[int, int, int, int] = Field(
        (1, 8, 512, 512),
        description="Ouput chunks (time, height, lat, lon).",
    )

    compression_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: DEFAULT_ENCODING_OPTIONS,
        description="Product output compression options for netcdf",
    )

    product_version: str = Field(
        PRODUCT_VERSION,
        description="OPERA TROPO product version",
    )

    def get_output_filename(self, date: str | datetime, hour: str | int):
        """Get product output filename convention."""
        # Ensure date is a string in the expected format
        if isinstance(date, datetime):
            date = date.strftime("%Y%m%d")

        # Ensure hour is a string
        hour = str(hour).zfill(2)

        # Parse date and hour into datetime format
        date_time = datetime.strptime(f"{date}T{hour}", "%Y%m%dT%H")

        # Format output datetime strings
        date_time_str = date_time.strftime(self.date_fmt)
        proc_datetime = self.creation_time.strftime(self.date_fmt)

        datetime_str = f"{date_time_str}Z_{proc_datetime}Z"
        return f"OPERA_L4_TROPO-ZENITH_{datetime_str}_HRES_v{self.product_version}.nc"


class WorkerSettings(BaseModel, extra="forbid"):
    """Settings for controlling CPU settings and parallelism."""

    n_workers: int = Field(
        4,
        ge=1,
        description=("Number of workers to use in dask.Client."),
    )
    threads_per_worker: int = Field(
        2,
        ge=1,
        description=("Number of threads to use per worker in dask.Client"),
    )
    max_memory: int | str = Field(
        default="16GB",
        description=("Workers are given a target memory limit in dask.Client"),
    )
    dask_temp_dir: str | Path = Field(
        "tmp",
        description=("Dask local spill directory."),
    )
    block_shape: tuple[int, int] = Field(
        (128, 256),
        description="Size (rows, columns) of blocks of data to load at a time.",
    )


class TropoWorkflow(YamlModel, extra="forbid"):
    """Troposphere delay calculation configuration models."""

    # Paths to input/output files
    input_options: InputOptions = Field(default_factory=InputOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    work_directory: Path = Field(
        Path(),
        description="Name of directory to use for processing",
        validate_default=True,
    )
    output_directory: Path = Field(
        Path(),
        description="Name of directory to use for writing output files",
        validate_default=True,
    )
    keep_paths_relative: bool = Field(
        False,
        description=(
            "Don't resolve filepaths that are given as relative to be absolute."
        ),
    )

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)
    log_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to output log file (in addition to logging to `stderr`)."
            " Default logs to `tropo.log` within `work_directory`"
        ),
    )

    def model_post_init(self, context: Any, /) -> None:
        """After validation, set up properties for use during workflow run."""
        super().model_post_init(context)
        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Save all directories as absolute paths
            self.work_directory = self.work_directory.resolve(strict=False)
