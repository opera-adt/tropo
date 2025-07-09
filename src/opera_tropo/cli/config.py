import functools
import re
from pathlib import Path
from typing import Final, Union

import click

from opera_tropo.config import pge_runconfig

# Configure click to show defaults for all options
click.option = functools.partial(click.option, show_default=True)

# Configuration constants
DEFAULT_CONFIG_NAME: Final[str] = "runconfig.yaml"
DEFAULT_MAX_HEIGHT: Final[int] = 81000
DEFAULT_N_WORKERS: Final[int] = 4
DEFAULT_N_THREADS: Final[int] = 2
DEFAULT_WORKER_MEMORY: Final[str] = "8GB"
DEFAULT_BLOCK_SHAPE: Final[tuple[int, int]] = (128, 128)
DEFAULT_LOG_FILE: Final[str] = "tropo_run.log"

MEMORY_PATTERN = re.compile(r"^(\d+)(GB|MB|KB)?$", re.IGNORECASE)


def validate_memory_format(memory: Union[int, str]) -> str:
    """Validate and normalize memory format."""
    if isinstance(memory, int):
        return f"{memory}GB"

    if isinstance(memory, str):
        if MEMORY_PATTERN.match(memory.strip()):
            return memory.strip()
        raise ValueError(
            f"Invalid memory format: {memory}. "
            "Expected format: '8GB', '512MB', '2048KB', or integer (GB assumed)"
        )

    raise TypeError(f"Memory must be int or str, got {type(memory)}")


def validate_inputs(
    input_file: Path,
    output_dir: Path,
    work_dir: Path,
    config_name: str,
    n_workers: int,
    n_threads: int,
    worker_memory: str,
    block_shape: tuple[int, int],
) -> None:
    """Validate all input parameters."""
    # Validate file paths
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")

    # Validate config name
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        raise ValueError(
            f"Config file must have .yaml or .yml extension: {config_name}"
        )

    # Validate worker settings
    if n_workers < 1:
        raise ValueError(f"Number of workers must be >= 1, got {n_workers}")

    if n_threads < 1:
        raise ValueError(f"Number of threads must be >= 1, got {n_threads}")

    # Validate block shape
    if len(block_shape) != 2 or any(dim < 1 for dim in block_shape):
        raise ValueError(
            f"Block shape must be tuple of 2 positive integers, got {block_shape}"
        )

    # Validate memory format
    validate_memory_format(worker_memory)

    # Create directories if they don't exist
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def create_config(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    work_dir: Union[str, Path] = ".",
    config_name: str = DEFAULT_CONFIG_NAME,
    max_height: int = DEFAULT_MAX_HEIGHT,
    n_workers: int = DEFAULT_N_WORKERS,
    n_threads: int = DEFAULT_N_THREADS,
    worker_memory: Union[int, str] = DEFAULT_WORKER_MEMORY,
    block_shape: tuple[int, int] = DEFAULT_BLOCK_SHAPE,
    log_file: str = DEFAULT_LOG_FILE,
    keep_relative_paths: bool = False,
) -> Path:
    """Generate and save a run configuration file for tropospheric processing.

    Args:
        input_file: Path to input file for processing
        output_dir: Directory for output files
        work_dir: Working directory for temporary files
        config_name: Name of the configuration file to create
        max_height: Maximum output height in meters
        n_workers: Number of parallel workers
        n_threads: Number of threads per worker
        worker_memory: Memory limit per worker (e.g., '8GB', '512MB')
        block_shape: Processing block dimensions (height, width)
        log_file: Name of the log file
        keep_relative_paths: Whether to keep paths relative instead of absolute

    Returns:
        Path to the created configuration file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If configuration parameters are invalid
        PermissionError: If unable to write configuration file

    """
    # Convert to Path objects
    input_file_path = Path(input_file)
    output_dir_path = Path(output_dir)
    work_dir_path = Path(work_dir)
    log_file_path = Path(log_file)

    # Normalize memory format
    normalized_memory = validate_memory_format(worker_memory)

    # Validate all inputs
    validate_inputs(
        input_file_path,
        output_dir_path,
        work_dir_path,
        config_name,
        n_workers,
        n_threads,
        normalized_memory,
        block_shape,
    )

    # Determine config file path
    config_path = work_dir_path / config_name

    # Resolve paths if requested
    if not keep_relative_paths:
        input_file_path = input_file_path.resolve()
        output_dir_path = output_dir_path.resolve()
        work_dir_path = work_dir_path.resolve()
        log_file_path = output_dir_path / log_file_path.name
    else:
        # For relative paths, make log file relative to output directory
        log_file_path = Path(log_file_path.name)

    # Create and configure runconfig
    runconfig = pge_runconfig.RunConfig(
        input_file={"input_file_path": input_file_path},
        output_options={"max_height": max_height},
        product_path_group={
            "scratch_path": work_dir_path,
            "sas_output_path": output_dir_path,
        },
        worker_settings={
            "n_workers": n_workers,
            "threads_per_worker": n_threads,
            "max_memory": normalized_memory,
            "block_shape": block_shape,
        },
        log_file=log_file_path,
    )

    # Save configuration
    try:
        runconfig.to_yaml(config_path)
    except Exception as e:
        raise PermissionError(f"Failed to write config file {config_path}: {e}")

    return config_path


@click.command("config")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(path_type=Path),
    default=Path.cwd() / DEFAULT_CONFIG_NAME,
    help="Path to output configuration file.",
)
@click.option(
    "--tropo-input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input file for tropospheric processing.",
)
@click.option(
    "--tropo-output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for output files.",
)
@click.option(
    "--max-height",
    "-mh",
    type=int,
    default=DEFAULT_MAX_HEIGHT,
    help="Maximum output height in meters.",
)
@click.option(
    "--n-workers",
    "-w",
    type=int,
    default=DEFAULT_N_WORKERS,
    help="Number of parallel workers.",
)
@click.option(
    "--n-threads",
    "-t",
    type=int,
    default=DEFAULT_N_THREADS,
    help="Number of threads per worker.",
)
@click.option(
    "--worker-memory",
    "-m",
    type=str,
    default=DEFAULT_WORKER_MEMORY,
    help="Memory limit per worker (e.g., '8GB', '512MB').",
)
@click.option(
    "--block-shape",
    type=(int, int),
    default=DEFAULT_BLOCK_SHAPE,
    help="Processing block shape (height, width).",
)
@click.option(
    "--log-file",
    "-l",
    type=str,
    default=DEFAULT_LOG_FILE,
    help="Log filename.",
)
@click.option(
    "--keep-relative-paths/--absolute-paths",
    default=False,
    help="Keep paths relative in config instead of resolving to absolute paths.",
)
def run_create_config(
    config_file: Path,
    tropo_input: Path,
    tropo_output: Path,
    max_height: int,
    n_workers: int,
    n_threads: int,
    worker_memory: str,
    block_shape: tuple[int, int],
    log_file: str,
    keep_relative_paths: bool,
) -> None:
    """Create a tropospheric processing configuration file.

    This command generates a YAML configuration file with all the necessary
    parameters for tropospheric processing, including input/output paths,
    worker settings, and processing parameters.
    """
    print(config_file.parent.name)
    try:
        config_path = create_config(
            input_file=tropo_input,
            output_dir=tropo_output,
            work_dir=config_file.parent.relative_to(Path.cwd()),
            config_name=config_file.name,
            max_height=max_height,
            n_workers=n_workers,
            n_threads=n_threads,
            worker_memory=worker_memory,
            block_shape=block_shape,
            log_file=log_file,
            keep_relative_paths=keep_relative_paths,
        )

        click.echo(f"Configuration file created: {config_path}")

    except (FileNotFoundError, ValueError, PermissionError) as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        raise click.Abort()
