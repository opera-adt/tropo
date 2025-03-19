from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

logging.getLogger("backoff").addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

S3_HRES_BUCKET = "opera-dev-lts-fwd-hyunlee"  # this will change
HRES_HOURS = ["00", "06", "12", "18"]


@dataclass
class HRESConfig:
    """Configuration for HRES processing."""

    output_path: Path
    date: str
    hour: str
    version: int = 1
    s3_bucket: str = S3_HRES_BUCKET
    region_name: str = "us-west-2"


def download_from_s3(
    bucket_name: str, s3_key: str, local_path: str, region_name: str = "us-west-2"
) -> None:
    """Download a file from an S3 bucket to the local file system.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    s3_key : str
        The key (path) of the file in the S3 bucket.
    local_path : str
        The local path where the file will be saved.
    region_name : str, optional
        The AWS region where the S3 bucket is located. Defaults to 'us-west-2'.

    """
    # Initialize the S3 client
    s3_client = boto3.client(
        "s3", region_name=region_name, config=Config(signature_version=UNSIGNED)
    )

    try:
        # Download the file from S3
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Download successful: {local_path}")
    # Keep no credential error there for now even as credentials are not checked
    except NoCredentialsError:
        logger.error("Error: AWS credentials not found.")
    except PartialCredentialsError:
        logger.error("Error: Incomplete AWS credentials.")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")


def _get_s3_key(date_input: str, hour: str, version: str = "1", date_format="%Y%m%d"):
    """Generate an S3 key for a specific date and hour.

    Args:
        date_input (str): The date in the format specified
                          by date_format (default is '%Y%m%d').
        hour (str): The hour, which must be one of the allowed values
                    ('00', '06', '12', '18').
        date_format (str): The format of the input date string.
                            Defaults to '%Y%m%d'.
        version (str): The version of the file. Defaults to '01'.

    Returns:
        str: The S3 key for the given date and hour.
        str: HRES nc filename

    Raises:
        ValueError: If the hour is not in the allowed values.

    """
    if hour not in HRES_HOURS:
        raise TypeError(f"Specifed hour input is not in {HRES_HOURS}")

    date_obj = datetime.strptime(date_input, date_format)
    file_ext = f"{date_obj.month:02}{date_obj.day:02}{hour}00"
    filename = f"D{file_ext}{file_ext}{version}.zz.nc"
    return f"{date_input}/{filename}", filename


def download_hres(config: HRESConfig) -> None:
    """Download HRES data from an S3 bucket.

    This function constructs the S3 key for the requested HRES data based on
    the provided configuration, then downloads the file to the specified
    output path.

    Parameters
    ----------
    config : HRESConfig
        Configuration object containing the date, hour, version, S3 bucket
        name, output path, and AWS region.

    Raises
    ------
    RuntimeError
        If the download from S3 fails.

    """
    # Get the s3 key
    s3_key, filename = _get_s3_key(config.date, config.hour, f"{config.version}")
    logger.info(f"Downloading HRES {s3_key} from {config.s3_bucket}")
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(config.output_path) / filename
    print(output_file)

    try:
        download_from_s3(config.s3_bucket, s3_key, output_file, config.region_name)
    except RuntimeError:
        raise
