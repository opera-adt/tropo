import functools
from pathlib import Path

import click

from opera_tropo.download import HRES_HOURS

__all__ = ["download"]

click.option = functools.partial(click.option, show_default=True)


@click.command("download")
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    default=Path.cwd(),
    help="Directory to save downloaded files",
)
@click.option("--s3-bucket", "-s3", type=str, help="s3 bucket")
@click.option("--date", type=str, help="Model date YYYYMMDD, eg. 20190101")
@click.option("--hour", type=click.Choice(list(HRES_HOURS)), help="Model hour")
def download(
    output_dir: Path, s3_bucket: str, date: str, hour: str, version: int = 1
) -> None:
    """Download the HRES model from s3 bucket."""
    # rest of imports here so --help doesn't take forever
    from opera_tropo.download import HRESConfig, download_hres

    hres = HRESConfig(output_dir, date, hour, version, s3_bucket)
    download_hres(hres)
