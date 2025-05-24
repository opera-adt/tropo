import functools
from pathlib import Path

import click

from opera_tropo.config import pge_runconfig

click.option = functools.partial(click.option, show_default=True)


def create_config(
    input_file: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "./runconfig.yaml",
    max_height: int = 81000,
    n_workers: int = 4,
    n_threads: int = 2,
    worker_memory: int | str = "8GB",
    block_shape: tuple[int, int] = (128, 128),
    log_file: str = "tropo_run.log",
    keep_relative_paths: bool = False,
) -> None:
    """Generate and save a run configuration file for tropospheric processing."""
    config_path = Path(config_path).resolve()

    if config_path.suffix != ".yaml":
        raise ValueError("config_path must be a YAML file (e.g., runconfig.yaml)")

    runconfig = pge_runconfig.RunConfig()

    input_file_path = Path(input_file)
    output_dir_path = Path(output_dir)
    log_file_path = Path(log_file)

    if not keep_relative_paths:
        input_file_path = input_file_path.resolve()
        output_dir_path = output_dir_path.resolve()
        log_file_path = output_dir_path / log_file_path

    runconfig.input_file.input_file_path = input_file_path
    runconfig.output_options.max_height = max_height
    runconfig.product_path_group.scratch_path = output_dir_path
    runconfig.worker_settings.n_workers = n_workers
    runconfig.worker_settings.threads_per_worker = n_threads
    runconfig.worker_settings.max_memory = worker_memory
    runconfig.worker_settings.block_shape = block_shape
    runconfig.log_file = (
        log_file_path if keep_relative_paths else output_dir_path / log_file_path.name
    )

    runconfig.to_yaml(config_path)


@click.command("config")
@click.option(
    "--config-file",
    "-c",
    type=Path,
    default=Path.cwd() / "runconfig.yaml",
    help="Path to output config file.",
)
@click.option(
    "--tropo-input",
    "-input",
    type=Path,
    required=True,
    help="Path to input file for tropospheric processing.",
)
@click.option(
    "--tropo-output",
    "-out",
    type=Path,
    required=True,
    help="Directory for output files.",
)
@click.option(
    "--max-height", "-mh", type=int, default=81000, help="Maximum output height."
)
@click.option(
    "--worker-settings",
    type=(int, int, int),
    default=(4, 2, 8),
    help="Worker settings: (n_workers, n_threads, worker_memory[GB]).",
)
@click.option(
    "--chunks",
    type=(int, int),
    default=(128, 128),
    help="Block shape for worker processing.",
)
@click.option("--log", type=str, default="run_tropo.log", help="Log filename")
@click.option(
    "--keep-relative-paths/--no-keep-relative-paths",
    default=True,
    help="Keep paths relative in the config instead of resolving to absolute",
)
def run_create_config(
    config_file: Path,
    tropo_input: Path,
    tropo_output: Path,
    max_height: int,
    worker_settings: tuple[int, int, int],
    chunks: tuple[int, int],
    log: str,
    keep_relative_paths: bool,
) -> None:
    """CLI wrapper to create a tropospheric runconfig file."""
    create_config(
        input_file=tropo_input,
        output_dir=tropo_output,
        config_path=config_file,
        max_height=max_height,
        n_workers=worker_settings[0],
        n_threads=worker_settings[1],
        worker_memory=f"{worker_settings[2]}GB",
        block_shape=chunks,
        log_file=log,
        keep_relative_paths=keep_relative_paths,
    )
