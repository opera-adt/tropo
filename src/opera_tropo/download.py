from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

logging.getLogger("backoff").addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

S3_HRES_BUCKET = "opera-ecmwf"  # not public
HRES_HOURS = ["00", "06", "12", "18"]


@dataclass
class HRESDownloader:
    """HRES downloader using S3 client."""

    s3_bucket: str = S3_HRES_BUCKET
    region_name: str = "us-west-2"
    profile: str = "saml-pub"
    use_unsigned: bool = False
    s3_client: boto3.client = field(init=False, repr=False)

    def __post_init__(self):
        self.s3_client = self._auth()

    def _auth(self):
        """Authenticate and create the S3 client."""
        try:
            if self.use_unsigned:
                logger.info("Using unsigned (public) S3 access.")
                s3_client = boto3.client(
                    "s3",
                    region_name=self.region_name,
                    config=Config(signature_version=UNSIGNED),
                )
            else:
                logger.info(
                    f"Using profile-based S3 access with profile: {self.profile}"
                )
                session = boto3.Session(profile_name=self.profile)
                s3_client = session.client("s3", region_name=self.region_name)

            # Test permissions
            s3_client.head_bucket(Bucket=self.s3_bucket)
            return s3_client
        except Exception as e:
            logger.error(f"Failed to authenticate to S3: {e}")
            raise RuntimeError(f"Failed to authenticate to S3: {e}") from e

    def list_matching_keys(
        self,
        prefix: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: Optional[Path] = None,
    ) -> list[tuple[Any, str]]:
        """List S3 keys in the bucket that match ECMWF_TROP_*.nc.

        Parameters
        ----------
        prefix : str, optional
            The prefix filter for keys in the S3 bucket (default is "").

        start_date : str, optional
            Filter to include only files with dates >= start_date.
            Format must be 'YYYYMMDD'. If None, no lower bound is applied.

        end_date : str, optional
            Filter to include only files with dates <= end_date.
            Format must be 'YYYYMMDD'. If None, no upper bound is applied.

        output_file : Path, optional
            If specified, save the list of available HRES product to a local file.

        Returns
        -------
        list[str]
            A list of tuples where each tuple contains (S3 key, S3 URI)
            matching the filters.

        """
        logger.info(
            f"Listing files in bucket '{self.s3_bucket}' with prefix '{prefix}'"
        )
        matching_keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]

                if filename.startswith("ECMWF_TROP_") and filename.endswith(".nc"):
                    match = re.search(r"ECMWF_TROP_(\d{12})", filename)
                    if match:
                        datetime_str = match.group(1)
                        date = datetime_str[:8]

                        if start_date and date < start_date:
                            continue
                        if end_date and date > end_date:
                            continue

                        s3_uri = f"s3://{self.s3_bucket}/{key}"
                        matching_keys.append((key, s3_uri))

        logger.info(f"Found {len(matching_keys)} matching files.")

        if output_file:
            lines = [f"Number of Dates: {len(matching_keys)}", "#" * 30]

            for key, uri in matching_keys:
                match = re.search(r"ECMWF_TROP_(\d{12})", key)
                if match:
                    datetime_str = match.group(1)
                    date = datetime_str[:8]
                    hour = datetime_str[8:10]
                    lines.append(f"Date: {date}, Hour: {hour}, URI: {uri}")
                else:
                    lines.append(f"Pattern not found in key: {key}")

            with open(output_file, "w") as f:
                for line in lines:
                    f.write(f"{line}\n")

        return matching_keys

    def _download_file(self, s3_key: str, local_path: str) -> None:
        """Download a file from the S3 bucket to the local file system.

        Parameters
        ----------
        s3_key : str
            The key (path) of the file in the S3 bucket.
        local_path : str
            The local path where the file will be saved.

        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"Download successful: {local_path}")
        except NoCredentialsError:
            logger.error("Error: AWS credentials not found.")
        except PartialCredentialsError:
            logger.error("Error: Incomplete AWS credentials.")
        except Exception as e:
            logger.error(f"Error downloading file: {e}")

    def download_hres(
        self,
        output_path: str | Path,
        date_input: str,
        hour: str,
        version: str = "1",
        date_format: str = "%Y%m%d",
    ) -> None:
        """Download an HRES file from S3 based on date, hour, and version.

        Parameters
        ----------
        output_path : str
            Local directory where the file will be saved.

        date_input : str
            Date string to locate the file, formatted according to `date_format`.

        hour : str
            Hour string representing the time of the file.

        version : str, optional
            Version string of the HRES data (default is "1").

        date_format : str, optional
            The format of the date_input string (default is "%Y%m%d").

        Raises
        ------
        RuntimeError
            If the download fails.

        """
        s3_key, filename = _get_s3_key(date_input, hour, version, date_format)
        logger.info(f"Downloading HRES {s3_key} from bucket {self.s3_bucket}")

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / filename

        try:
            self._download_file(s3_key, output_file)
        except RuntimeError:
            logger.error(f"Failed to download file {s3_key}")
            raise


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
    file_ext = f"{date_obj.year:04}{date_obj.month:02}{date_obj.day:02}{hour}00"
    filename = f"ECMWF_TROP_{file_ext}_{file_ext}_{version}.nc"
    return f"{date_input}/{filename}", filename
