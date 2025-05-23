import functools
from pathlib import Path
from typing import Optional

import click

from opera_tropo.download import HRES_HOURS

__all__ = ["download", "list_dates"]

click.option = functools.partial(click.option, show_default=True)


@click.command("download")
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    default=Path.cwd(),
    help="Directory to save downloaded files",
)
@click.option("--s3-bucket", "-s3", type=str, required=True, help="S3 bucket name")
@click.option("--date", type=str, help="Model date YYYYMMDD, e.g., 20190101")
@click.option("--hour", type=click.Choice(list(HRES_HOURS)), help="Model hour")
@click.option(
    "--version", type=int, default=1, help="Version number of the model (default: 1)"
)
def download(
    output_dir: Path,
    s3_bucket: str,
    date: Optional[str],
    hour: Optional[str],
    version: str,
):
    """Download an HRES file from S3 bucket for a given date and hour."""
    from opera_tropo.download import HRESDownloader

    if not (date and hour):
        raise click.UsageError(
            "Both --date and --hour must be specified to download a file."
        )

    client = HRESDownloader(s3_bucket=s3_bucket)
    client.download_hres(output_dir, date, hour, version)


@click.command("list")
@click.option("--s3-bucket", "-s3", type=str, required=True, help="S3 bucket name")
@click.argument("start_date", required=False)
@click.argument("end_date", required=False)
@click.option(
    "--output-file", type=Path, help="Path to write results instead of stdout"
)
def list_dates(
    s3_bucket: str, start_date: str, end_date: str, output_file: Optional[Path]
):
    """List available HRES files in an S3 bucket between two dates (YYYYMMDD)."""
    from opera_tropo.download import HRESDownloader

    client = HRESDownloader(s3_bucket=s3_bucket)
    client.list_matching_keys(
        start_date=start_date, end_date=end_date, output_file=output_file
    )
