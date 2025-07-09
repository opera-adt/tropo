#!/usr/bin/env python3
"""Weather Data Analysis Tool
for analyzing weather data from NetCDF files stored in S3.
"""

import argparse
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import dask

# Core analysis imports
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from botocore.exceptions import ClientError
from dask.distributed import Client
from tqdm import tqdm

# Import the download module (assuming it's available)
try:
    from opera_tropo import download
except ImportError:
    print(
        "Warning: opera_tropo module not found. Download functionality will be limited."
    )
    download = None


class WeatherDataAnalyzer:
    """A class for analyzing weather data from NetCDF files and generating statistical reports."""

    def __init__(
        self,
        output_dir: str = "./weather_stats",
        n_workers: int = 4,
        s3_profile: str = "saml-pub",
        verbose: bool = True,
    ):
        """Initialize the WeatherDataAnalyzer.

        Parameters
        ----------
        output_dir : str
            Directory to save output files
        n_workers : int
            Number of Dask workers
        s3_profile : str
            S3 profile name for authentication
        verbose : bool
            Whether to print progress messages

        """
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.s3_profile = s3_profile
        self.verbose = verbose
        self.logger = self._setup_logging()

        # Variable configurations
        self.variable_configs = {
            "z": {
                "levels": [0],
                "desc": "Geopotential (surface)",
                "check_negative": False,
            },
            "t": {
                "levels": "all",
                "desc": "Temperature (all levels)",
                "check_negative": False,
            },
            "q": {
                "levels": "all",
                "desc": "Specific humidity (all levels)",
                "check_negative": True,
            },
            "lnsp": {
                "levels": [0],
                "desc": "Log surface pressure",
                "check_negative": False,
            },
        }

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def _extract_time_info(self, dataset: xr.Dataset) -> Tuple[str, str]:
        """Extract model date and time information from dataset."""
        try:
            if "time" in dataset.coords:
                model_time = pd.to_datetime(dataset.time.values[0])
                return (
                    model_time.strftime("%Y-%m-%d"),
                    model_time.strftime("%H:%M:%S UTC"),
                )
            else:
                self.logger.warning("No time coordinate found in dataset")
                return "Unknown", "Unknown"
        except Exception as e:
            self.logger.warning(f"Error extracting time info: {e}")
            return "Unknown", "Unknown"

    def _get_variable_data(
        self, dataset: xr.Dataset, var: str, config: Dict[str, Any]
    ) -> Optional[xr.DataArray]:
        """Extract variable data based on configuration."""
        if var not in dataset.data_vars:
            self.logger.warning(f"Variable '{var}' not found in dataset")
            return None

        data = dataset[var]

        # Apply level selection if specified
        if config["levels"] != "all" and "level" in data.dims:
            try:
                data = data.isel(level=config["levels"])
            except (IndexError, KeyError) as e:
                self.logger.warning(f"Error selecting levels for {var}: {e}")

        return data

    def _compute_statistics(
        self, data: xr.DataArray, var: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute statistics for a variable."""
        stats = {
            f"{var}_min": data.min(),
            f"{var}_max": data.max(),
            f"{var}_mean": data.mean(),
            f"{var}_std": data.std(),
            f"{var}_total_size": data.size,
            f"{var}_nan_count": data.isnull().sum(),
            f"{var}_zero_count": (data == 0.0).sum(),
            f"{var}_inf_count": np.isinf(data).sum(),
            f"{var}_finite_count": np.isfinite(data).sum(),
        }

        # Add negative count only for variables where it's relevant
        if config["check_negative"]:
            stats[f"{var}_negative_count"] = (data < 0).sum()
        else:
            stats[f"{var}_negative_count"] = xr.DataArray(0)

        return stats

    def _create_quality_flags(
        self, var: str, results: Dict[str, Any], config: Dict[str, Any]
    ) -> List[str]:
        """Create quality flags based on computed statistics."""
        flags = []

        nan_count = results[f"{var}_nan_count"].item()
        inf_count = results[f"{var}_inf_count"].item()
        zero_count = results[f"{var}_zero_count"].item()
        negative_count = results[f"{var}_negative_count"].item()

        if nan_count > 0:
            flags.append(f"NaN({nan_count})")
        if inf_count > 0:
            flags.append(f"Inf({inf_count})")
        if zero_count > 0:
            flags.append(f"Zero({zero_count})")
        if negative_count > 0 and config["check_negative"]:
            flags.append(f"Neg({negative_count})")

        return flags

    def _create_summary_dataframe(self, all_results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame from computed results."""
        summary_data = []

        for var, config in self.variable_configs.items():
            if f"{var}_min" not in all_results:
                continue

            total_size = all_results[f"{var}_total_size"]
            nan_count = all_results[f"{var}_nan_count"].item()
            zero_count = all_results[f"{var}_zero_count"].item()
            negative_count = all_results[f"{var}_negative_count"].item()
            inf_count = all_results[f"{var}_inf_count"].item()
            finite_count = all_results[f"{var}_finite_count"].item()

            quality_flags = self._create_quality_flags(var, all_results, config)

            summary_data.append(
                {
                    "Variable": var,
                    "Description": config["desc"],
                    "Min": all_results[f"{var}_min"].item(),
                    "Max": all_results[f"{var}_max"].item(),
                    "Mean": all_results[f"{var}_mean"].item(),
                    "Std": all_results[f"{var}_std"].item(),
                    "Total_Size": total_size,
                    "Finite_Count": finite_count,
                    "NaN_Count": nan_count,
                    "Inf_Count": inf_count,
                    "Zero_Count": zero_count,
                    "Negative_Count": negative_count,
                    "NaN_%": f"{nan_count/total_size*100:.3f}%",
                    "Zero_%": f"{zero_count/total_size*100:.3f}%",
                    "Completeness_%": f"{finite_count/total_size*100:.2f}%",
                    "Data_Quality": "; ".join(quality_flags) if quality_flags else "OK",
                }
            )

        return pd.DataFrame(summary_data)

    def _generate_output_filenames(
        self, file_url: str, model_date_str: str, model_time_str: str
    ) -> Tuple[Path, Path]:
        """Generate expected output filenames for a given input file."""
        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_part = (
            model_date_str.replace("-", "")
            if model_date_str != "Unknown"
            else "unknown"
        )

        try:
            hour_part = pd.to_datetime(f"{model_date_str} {model_time_str}").strftime(
                "%H"
            )
        except:
            hour_part = "unknown"

        base_filename = f"weather_stats_{date_part}_{hour_part}_{analysis_timestamp}"

        txt_path = self.output_dir / f"{base_filename}.txt"
        csv_path = self.output_dir / f"{base_filename}.csv"

        return txt_path, csv_path

    def _check_existing_outputs(
        self, file_url: str, overwrite: bool = False
    ) -> Tuple[bool, List[str]]:
        """Check if output files already exist for this input file.

        Parameters
        ----------
        file_url : str
            URL of the input file
        overwrite : bool
            If True, existing files will be overwritten

        Returns
        -------
        Tuple[bool, List[str]]
            (should_skip, existing_files_list)

        """
        if overwrite:
            return False, []

        try:
            # We need to extract time info to predict the output filename pattern
            # This requires opening the file briefly
            fs = s3fs.S3FileSystem(
                profile=self.s3_profile, config_kwargs={"max_pool_connections": 50}
            )

            with fs.open(file_url, mode="rb") as f:
                dataset = xr.open_dataset(f, engine="h5netcdf")
                model_date_str, model_time_str = self._extract_time_info(dataset)
                dataset.close()

            # Generate the expected filename pattern
            date_part = (
                model_date_str.replace("-", "")
                if model_date_str != "Unknown"
                else "unknown"
            )
            try:
                hour_part = pd.to_datetime(
                    f"{model_date_str} {model_time_str}"
                ).strftime("%H")
            except:
                hour_part = "unknown"

            # Look for existing files with this date_hour pattern
            pattern = f"weather_stats_{date_part}_{hour_part}_*"
            existing_files = list(self.output_dir.glob(pattern + ".txt"))
            existing_csv_files = list(self.output_dir.glob(pattern + ".csv"))

            # If we have both txt and csv files, consider it already processed
            if existing_files and existing_csv_files:
                return True, [str(f) for f in existing_files + existing_csv_files]

            return False, []

        except Exception as e:
            self.logger.warning(f"Error checking existing outputs for {file_url}: {e}")
            # If we can't check, assume it doesn't exist and process it
            return False, []

    def _export_results(
        self,
        summary_df: pd.DataFrame,
        file_url: str,
        model_date_str: str,
        model_time_str: str,
        dataset: xr.Dataset,
    ) -> Tuple[str, str]:
        """Export results to text and CSV files."""
        # Generate output filenames using the new method
        txt_path, csv_path = self._generate_output_filenames(
            file_url, model_date_str, model_time_str
        )

        # Export text report
        self._export_text_report(
            txt_path, summary_df, file_url, model_date_str, model_time_str, dataset
        )

        # Export CSV
        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._export_csv_report(
            csv_path,
            summary_df,
            file_url,
            model_date_str,
            model_time_str,
            analysis_timestamp,
        )

        return str(txt_path), str(csv_path)

    def _export_text_report(
        self,
        file_path: Path,
        summary_df: pd.DataFrame,
        file_url: str,
        model_date_str: str,
        model_time_str: str,
        dataset: xr.Dataset,
    ):
        """Export detailed text report."""
        header = f"""
{'='*80}
WEATHER DATA STATISTICAL ANALYSIS REPORT
{'='*80}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source File: {file_url}
Model Date: {model_date_str}
Model Time: {model_time_str}

Dataset Information:
- Dimensions: {dict(dataset.sizes)}
- Coordinates: {list(dataset.coords.keys())}
- Data Variables: {list(dataset.data_vars.keys())}
- Global Attributes: {len(dataset.attrs)} attributes

Variable Processing Configuration:
{self._format_variable_configs()}
{'='*80}

STATISTICAL SUMMARY:
"""

        with open(file_path, "w") as f:
            f.write(header)

            # Write summary table
            f.write("\nDETAILED STATISTICS TABLE:\n")
            f.write("=" * 140 + "\n")

            display_df = summary_df.copy()
            numerical_cols = ["Min", "Max", "Mean", "Std"]
            for col in numerical_cols:
                display_df[col] = display_df[col].round(6)

            f.write(display_df.to_string(index=False, max_colwidth=30))

            # Write data quality summary
            f.write(f"\n\n{'='*50}\n")
            f.write("DATA QUALITY SUMMARY:\n")
            f.write(f"{'='*50}\n")

            for _, row in summary_df.iterrows():
                status = (
                    "No issues detected"
                    if row["Data_Quality"] == "OK"
                    else row["Data_Quality"]
                )
                f.write(f"{row['Variable']}: {status}\n")

    def _format_variable_configs(self) -> str:
        """Format variable configurations for display."""
        config_lines = []
        for var, config in self.variable_configs.items():
            levels_str = (
                "all levels"
                if config["levels"] == "all"
                else f"levels {config['levels']}"
            )
            config_lines.append(f"- {var}: {levels_str}")
        return "\n".join(config_lines)

    def _export_csv_report(
        self,
        file_path: Path,
        summary_df: pd.DataFrame,
        file_url: str,
        model_date_str: str,
        model_time_str: str,
        analysis_timestamp: str,
    ):
        """Export CSV report with metadata."""
        csv_df = summary_df.copy()
        csv_df.insert(0, "Source_File", file_url)
        csv_df.insert(1, "Model_Date", model_date_str)
        csv_df.insert(2, "Model_Time", model_time_str)
        csv_df.insert(3, "Analysis_Timestamp", analysis_timestamp)

        csv_df.to_csv(file_path, index=False)

    def analyze_file(
        self, file_url: str, client: Optional[Client] = None, overwrite: bool = False
    ) -> Tuple[pd.DataFrame, str, str]:
        """Analyze a single weather data file and export statistics.

        Parameters
        ----------
        file_url : str
            URL or path to the file to analyze
        client : Optional[Client]
            Existing Dask client to use. If None, creates a new one.
        overwrite : bool
            If False, skip files that have already been processed

        Returns
        -------
        Tuple[pd.DataFrame, str, str]
            Summary DataFrame, text report path, and CSV report path

        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if file has already been processed
        should_skip, existing_files = self._check_existing_outputs(file_url, overwrite)
        if should_skip:
            self.logger.info(f"Skipping already processed file: {file_url}")
            self.logger.info(f"Existing files: {', '.join(existing_files)}")
            # Return the most recent existing files
            txt_files = [f for f in existing_files if f.endswith(".txt")]
            csv_files = [f for f in existing_files if f.endswith(".csv")]
            if txt_files and csv_files:
                # Load summary from existing CSV for consistency
                try:
                    summary_df = pd.read_csv(csv_files[0])
                    # Remove metadata columns to match analyze_file output format
                    metadata_cols = [
                        "Source_File",
                        "Model_Date",
                        "Model_Time",
                        "Analysis_Timestamp",
                    ]
                    for col in metadata_cols:
                        if col in summary_df.columns:
                            summary_df = summary_df.drop(columns=[col])
                    return summary_df, txt_files[0], csv_files[0]
                except Exception as e:
                    self.logger.warning(f"Could not load existing summary: {e}")
            # Return empty DataFrame if can't load existing results
            return (
                pd.DataFrame(),
                txt_files[0] if txt_files else "",
                csv_files[0] if csv_files else "",
            )

        self.logger.info(f"Processing: {file_url}")

        def _analyze_with_client(dask_client):
            """Internal function to perform analysis with a given client."""
            try:
                # Initialize S3 filesystem
                if self.verbose:
                    print("  → Connecting to S3...")
                    sys.stdout.flush()

                fs = s3fs.S3FileSystem(
                    profile=self.s3_profile, config_kwargs={"max_pool_connections": 50}
                )

                # Open dataset
                if self.verbose:
                    print("  → Opening dataset...")
                    sys.stdout.flush()

                with fs.open(file_url, mode="rb") as f:
                    dataset = xr.open_dataset(f, engine="h5netcdf", chunks="auto")

                # Extract time information
                model_date_str, model_time_str = self._extract_time_info(dataset)

                # Compute statistics for all variables
                all_operations = {}
                processed_vars = []

                if self.verbose:
                    print("  → Preparing computations...")
                    sys.stdout.flush()

                for var, config in self.variable_configs.items():
                    data = self._get_variable_data(dataset, var, config)
                    if data is not None:
                        stats = self._compute_statistics(data, var, config)
                        all_operations.update(stats)
                        processed_vars.append(var)

                if not all_operations:
                    raise ValueError("No valid variables found in dataset")

                if self.verbose:
                    print("  → Computing statistics...")
                    sys.stdout.flush()
                else:
                    self.logger.info("Computing statistics...")

                results = dask.compute(all_operations)[0]

                if self.verbose:
                    print("  → Generating reports...")
                    sys.stdout.flush()

                # Create summary DataFrame
                summary_df = self._create_summary_dataframe(results)

                # Export results
                txt_path, csv_path = self._export_results(
                    summary_df, file_url, model_date_str, model_time_str, dataset
                )

                if self.verbose:
                    print("  → Analysis completed successfully")
                    sys.stdout.flush()
                else:
                    self.logger.info("Analysis completed successfully")

                return summary_df, txt_path, csv_path

            except Exception as e:
                if self.verbose:
                    print(f"  → Error: {e}")
                    sys.stdout.flush()
                raise

        try:
            if client is not None:
                # Use provided client
                return _analyze_with_client(client)
            else:
                # Create new client for single file analysis
                with Client(
                    n_workers=self.n_workers,
                    threads_per_worker=2,
                    silence_logs=not self.verbose,
                ) as new_client:
                    return _analyze_with_client(new_client)

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise


class WeatherDataWorkflow:
    """Manages the complete workflow for weather data analysis."""

    def __init__(self, s3_profile: str = "saml-pub"):
        self.s3_profile = s3_profile
        self.downloader = None

    def check_s3_access(self) -> bool:
        """Check if S3 access is working."""
        try:
            session = boto3.Session(profile_name=self.s3_profile)
            s3 = session.client("s3")
            s3.head_object(
                Bucket="opera-ecmwf",
                Key="20250318/ECMWF_TROP_202503180600_202503180600_1.nc",
            )
            print("S3 access verified")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                print("Test file not found (but access is working)")
                return True
            elif error_code == "403":
                print("Access denied - check AWS credentials and permissions")
                return False
            else:
                print(f"S3 access error: {e}")
                return False

    def get_file_list(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get list of available files for the date range.

        Parameters
        ----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format

        Returns
        -------
        pd.DataFrame
            DataFrame with file information

        """
        if download is None:
            raise ImportError("opera_tropo module not available")

        if self.downloader is None:
            self.downloader = download.HRESDownloader()

        print(f"Getting file list from {start_date} to {end_date}...")
        hres_dates = self.downloader.list_matching_keys(
            start_date=start_date, end_date=end_date
        )

        df = pd.DataFrame(hres_dates, columns=["s3_key", "url"])
        df["dates"] = df["s3_key"].apply(lambda x: x.split("/")[0])
        df["filename"] = df["s3_key"].apply(lambda x: x.split("/")[1])
        df = df[["dates", "filename", "s3_key", "url"]]

        print(f"Found {len(df)} files")
        return df

    def analyze_all_files(
        self,
        file_list: pd.DataFrame,
        output_dir: str,
        n_workers: int = 4,
        worker_mem: int = 2,
        verbose: bool = False,
        use_progress_bar: bool = True,
        parallel_clients: int = 1,
        overwrite: bool = False,
    ) -> List[str]:
        """Analyze all files in the list using single or multiple Dask clients.

        Parameters
        ----------
        file_list : pd.DataFrame
            DataFrame with file information
        output_dir : str
            Output directory for results
        n_workers : int
            Total number of Dask workers to use
        verbose : bool
            Whether to show detailed progress
        use_progress_bar : bool
            Whether to use tqdm progress bar (if False, uses simple print statements)
        parallel_clients : int
            Number of parallel clients to run (1 = sequential, >1 = parallel)
            If >1, workers will be split across clients
        overwrite : bool
            If False, skip files that have already been processed

        Returns
        -------
        List[str]
            List of generated report files

        """
        if parallel_clients > 1:
            return self._analyze_files_parallel(
                file_list,
                output_dir,
                n_workers,
                worker_mem,
                verbose,
                use_progress_bar,
                parallel_clients,
                overwrite,
            )
        else:
            return self._analyze_files_sequential(
                file_list,
                output_dir,
                n_workers,
                worker_mem,
                verbose,
                use_progress_bar,
                overwrite,
            )

    def _analyze_files_sequential(
        self,
        file_list: pd.DataFrame,
        output_dir: str,
        n_workers: int,
        worker_mem: int,
        verbose: bool,
        use_progress_bar: bool,
        overwrite: bool,
    ) -> List[str]:
        """Sequential file analysis using a single Dask client."""
        analyzer = WeatherDataAnalyzer(
            output_dir=output_dir,
            n_workers=n_workers,
            s3_profile=self.s3_profile,
            verbose=verbose,
        )

        report_files = []
        failed_files = []
        skipped_files = []

        print(f"Processing {len(file_list)} files sequentially...")
        if not overwrite:
            print(
                "Skipping already processed files (use --overwrite to force reprocessing)"
            )
        print(f"Starting Dask client with {n_workers} workers...")

        # Create one client for all files
        with Client(
            n_workers=n_workers,
            threads_per_worker=2,
            memory_limit=f"{worker_mem}GB",
            silence_logs=not verbose,
        ) as client:

            if verbose:
                print(f"Dask dashboard available at: {client.dashboard_link}")

            if use_progress_bar:
                # Use tqdm progress bar
                # Configure tqdm for better responsiveness
                # Force terminal detection and configure for better compatibility
                force_terminal = (
                    os.environ.get("FORCE_COLOR", "0") == "1" or sys.stdout.isatty()
                )

                with tqdm(
                    total=len(file_list),
                    desc="Analyzing files",
                    unit="file",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    disable=not force_terminal and not verbose,
                    file=sys.stdout,
                ) as pbar:

                    for i, idx in enumerate(file_list.index):
                        try:
                            file_url = file_list.iloc[idx].url
                            filename = (
                                file_list.iloc[idx].filename
                                if "filename" in file_list.columns
                                else f"file_{idx}"
                            )

                            # Update progress bar description with current file
                            pbar.set_description(f"Processing {filename}")

                            # Check if file should be skipped (for progress tracking)
                            should_skip, existing_files = (
                                analyzer._check_existing_outputs(file_url, overwrite)
                            )

                            if should_skip:
                                if verbose:
                                    print(
                                        f"\n[{i+1}/{len(file_list)}] Skipping: {filename} (already processed)"
                                    )
                                skipped_files.append(idx)
                                # Still add existing files to report_files list
                                txt_files = [
                                    f for f in existing_files if f.endswith(".txt")
                                ]
                                csv_files = [
                                    f for f in existing_files if f.endswith(".csv")
                                ]
                                report_files.extend(txt_files + csv_files)
                            else:
                                if verbose:
                                    print(
                                        f"\n[{i+1}/{len(file_list)}] Starting: {filename}"
                                    )

                                # Pass the shared client and overwrite flag to analyze_file
                                summary_df, txt_file, csv_file = analyzer.analyze_file(
                                    file_url, client=client, overwrite=overwrite
                                )
                                report_files.extend([txt_file, csv_file])

                                if verbose:
                                    print(
                                        f"[{i+1}/{len(file_list)}] Completed: {filename}"
                                    )

                        except Exception as e:
                            failed_files.append((idx, str(e)))
                            if verbose:
                                print(
                                    f"\n[{i+1}/{len(file_list)}] Failed: {filename} - {e}"
                                )
                            else:
                                print(f"\nFailed to process file {i+1}: {e}")

                        finally:
                            # Always update progress bar
                            pbar.update(1)
                            pbar.refresh()
            else:
                # Use simple print statements
                for i, idx in enumerate(file_list.index):
                    try:
                        file_url = file_list.iloc[idx].url
                        filename = (
                            file_list.iloc[idx].filename
                            if "filename" in file_list.columns
                            else f"file_{idx}"
                        )

                        # Check if file should be skipped
                        should_skip, existing_files = analyzer._check_existing_outputs(
                            file_url, overwrite
                        )

                        if should_skip:
                            print(
                                f"[{i+1}/{len(file_list)}] ⏭ Skipped: {filename} (already processed)"
                            )
                            skipped_files.append(idx)
                            # Still add existing files to report_files list
                            txt_files = [
                                f for f in existing_files if f.endswith(".txt")
                            ]
                            csv_files = [
                                f for f in existing_files if f.endswith(".csv")
                            ]
                            report_files.extend(txt_files + csv_files)
                        else:
                            print(f"[{i+1}/{len(file_list)}] Processing: {filename}")
                            sys.stdout.flush()

                            # Pass the shared client and overwrite flag to analyze_file
                            summary_df, txt_file, csv_file = analyzer.analyze_file(
                                file_url, client=client, overwrite=overwrite
                            )
                            report_files.extend([txt_file, csv_file])

                            print(f"[{i+1}/{len(file_list)}] ✓ Completed: {filename}")

                        sys.stdout.flush()

                    except Exception as e:
                        failed_files.append((idx, str(e)))
                        print(f"[{i+1}/{len(file_list)}] ✗ Failed: {filename} - {e}")
                        sys.stdout.flush()

        # Summary
        processed_files = len(file_list) - len(failed_files) - len(skipped_files)
        print("\nSequential Analysis Complete:")
        print(f"   Successfully processed: {processed_files} files")
        print(f"   Skipped (already done): {len(skipped_files)} files")
        print(f"   Failed: {len(failed_files)} files")
        print(f"   Reports generated/found: {len(report_files)} files")
        print(f"   Output directory: {output_dir}")

        if failed_files and verbose:
            print("\nFailed files:")
            for idx, error in failed_files:
                print(f"   File {idx}: {error}")

        if skipped_files and verbose:
            print("\nSkipped files:")
            for idx in skipped_files:
                filename = (
                    file_list.iloc[idx].filename
                    if "filename" in file_list.columns
                    else f"file_{idx}"
                )
                print(f"   File {idx}: {filename}")

        return report_files

    def _analyze_files_parallel(
        self,
        file_list: pd.DataFrame,
        output_dir: str,
        n_workers: int,
        worker_mem: int,
        verbose: bool,
        use_progress_bar: bool,
        parallel_clients: int,
        overwrite: bool,
    ) -> List[str]:
        """Parallel file analysis using multiple Dask clients."""
        # Validation and warnings
        if parallel_clients > len(file_list):
            parallel_clients = len(file_list)
            print(
                f"Warning: Reduced parallel clients to {parallel_clients} (number of files)"
            )

        if parallel_clients > 16:
            print(
                f"Warning: Using {parallel_clients} parallel clients may consume significant resources"
            )

        # Calculate workers per client
        workers_per_client = max(1, n_workers // parallel_clients)
        actual_total_workers = workers_per_client * parallel_clients

        print(f"Processing {len(file_list)} files in parallel...")
        if not overwrite:
            print(
                "Skipping already processed files (use --overwrite to force reprocessing)"
            )
        print(
            f"Using {parallel_clients} parallel clients with {workers_per_client} workers each"
        )
        print(f"Total workers: {actual_total_workers} (requested: {n_workers})")

        if actual_total_workers > n_workers:
            print(
                f"Warning: Using more workers ({actual_total_workers}) than requested ({n_workers})"
            )

        if parallel_clients > 4:
            print(
                "Note: High parallelism may cause S3 connection limits or memory issues"
            )
            print(
                "Consider using fewer parallel clients with more workers each if issues occur"
            )

        # Split files into chunks for parallel processing
        file_chunks = []
        chunk_size = max(1, len(file_list) // parallel_clients)

        for i in range(0, len(file_list), chunk_size):
            chunk = file_list.iloc[i : i + chunk_size].copy()
            if len(chunk) > 0:
                file_chunks.append(chunk)

        # Limit to actual number of chunks needed
        file_chunks = file_chunks[:parallel_clients]
        actual_clients = len(file_chunks)

        print(
            f"Split into {actual_clients} chunks: {[len(chunk) for chunk in file_chunks]}"
        )

        # Shared data structures for collecting results
        all_report_files = []
        all_failed_files = []
        all_skipped_files = []
        progress_lock = threading.Lock()
        completed_files = 0

        def process_chunk(
            chunk_id: int, chunk_files: pd.DataFrame
        ) -> Tuple[List[str], List[Tuple[int, str]], List[int]]:
            """Process a chunk of files with its own Dask client."""
            nonlocal completed_files  # Declare nonlocal at the beginning

            analyzer = WeatherDataAnalyzer(
                output_dir=output_dir,
                n_workers=workers_per_client,
                s3_profile=self.s3_profile,
                verbose=verbose,
            )

            chunk_report_files = []
            chunk_failed_files = []
            chunk_skipped_files = []

            try:
                with Client(
                    n_workers=workers_per_client,
                    threads_per_worker=2,
                    memory_limit=f"{worker_mem}GB",
                    silence_logs=not verbose,
                ) as client:

                    if verbose:
                        print(f"Client {chunk_id+1} dashboard: {client.dashboard_link}")

                    for i, idx in enumerate(chunk_files.index):
                        try:
                            file_url = chunk_files.iloc[i].url
                            filename = (
                                chunk_files.iloc[i].filename
                                if "filename" in chunk_files.columns
                                else f"file_{idx}"
                            )

                            # Check if file should be skipped
                            should_skip, existing_files = (
                                analyzer._check_existing_outputs(file_url, overwrite)
                            )

                            if should_skip:
                                if verbose:
                                    print(
                                        f"Client {chunk_id+1}: Skipping {filename} (already processed)"
                                    )
                                chunk_skipped_files.append(idx)
                                # Still add existing files to report_files list
                                txt_files = [
                                    f for f in existing_files if f.endswith(".txt")
                                ]
                                csv_files = [
                                    f for f in existing_files if f.endswith(".csv")
                                ]
                                chunk_report_files.extend(txt_files + csv_files)

                                # Update global progress for skipped files
                                with progress_lock:
                                    completed_files += 1
                                    if use_progress_bar and progress_bar:
                                        progress_bar.update(1)
                                        progress_bar.set_description(
                                            f"Skipped: {filename}"
                                        )
                                        progress_bar.refresh()
                                    else:
                                        print(
                                            f"[{completed_files}/{len(file_list)}] ⏭ Client {chunk_id+1}: {filename} (skipped)"
                                        )
                                        sys.stdout.flush()
                            else:
                                if verbose:
                                    print(f"Client {chunk_id+1}: Processing {filename}")

                                summary_df, txt_file, csv_file = analyzer.analyze_file(
                                    file_url, client=client, overwrite=overwrite
                                )
                                chunk_report_files.extend([txt_file, csv_file])

                                # Update global progress for processed files
                                with progress_lock:
                                    completed_files += 1
                                    if use_progress_bar and progress_bar:
                                        progress_bar.update(1)
                                        progress_bar.set_description(
                                            f"Completed: {filename}"
                                        )
                                        progress_bar.refresh()
                                    else:
                                        print(
                                            f"[{completed_files}/{len(file_list)}] ✓ Client {chunk_id+1}: {filename}"
                                        )
                                        sys.stdout.flush()

                        except Exception as e:
                            chunk_failed_files.append((idx, str(e)))
                            with progress_lock:
                                completed_files += 1
                                if use_progress_bar and progress_bar:
                                    progress_bar.update(1)
                                    progress_bar.refresh()
                                else:
                                    print(
                                        f"[{completed_files}/{len(file_list)}] ✗ Client {chunk_id+1}: {filename} - {e}"
                                    )
                                    sys.stdout.flush()

            except Exception as e:
                print(f"Client {chunk_id+1} failed: {e}")
                # Mark all files in this chunk as failed
                for idx in chunk_files.index:
                    chunk_failed_files.append((idx, f"Client error: {e}"))
                    with progress_lock:
                        completed_files += 1
                        if use_progress_bar and progress_bar:
                            progress_bar.update(1)
                            progress_bar.refresh()

            return chunk_report_files, chunk_failed_files, chunk_skipped_files

        # Initialize progress bar for parallel processing
        progress_bar = None
        if use_progress_bar:
            force_terminal = (
                os.environ.get("FORCE_COLOR", "0") == "1" or sys.stdout.isatty()
            )
            progress_bar = tqdm(
                total=len(file_list),
                desc="Analyzing files",
                unit="file",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                disable=not force_terminal and not verbose,
                file=sys.stdout,
            )

        try:
            # Process chunks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=actual_clients) as executor:
                # Submit all chunk processing tasks
                future_to_chunk = {
                    executor.submit(process_chunk, i, chunk): i
                    for i, chunk in enumerate(file_chunks)
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_reports, chunk_failures, chunk_skipped = future.result()
                        all_report_files.extend(chunk_reports)
                        all_failed_files.extend(chunk_failures)
                        all_skipped_files.extend(chunk_skipped)
                        if verbose:
                            processed_count = (
                                len(chunk_reports) // 2 if len(chunk_reports) > 0 else 0
                            )
                            print(
                                f"Client {chunk_id+1} completed: {processed_count} processed, {len(chunk_skipped)} skipped"
                            )
                    except Exception as e:
                        print(f"Client {chunk_id+1} failed with exception: {e}")

        finally:
            if use_progress_bar and progress_bar:
                progress_bar.close()

        # Summary
        processed_files = (
            len(file_list) - len(all_failed_files) - len(all_skipped_files)
        )
        print("\nParallel Analysis Complete:")
        print(f"   Successfully processed: {processed_files} files")
        print(f"   Skipped (already done): {len(all_skipped_files)} files")
        print(f"   Failed: {len(all_failed_files)} files")
        print(f"   Reports generated/found: {len(all_report_files)} files")
        print(f"   Output directory: {output_dir}")
        print(f"   Parallel clients used: {actual_clients}")

        if all_failed_files and verbose:
            print("\nFailed files:")
            for idx, error in all_failed_files:
                print(f"   File {idx}: {error}")

        if all_skipped_files and verbose:
            print("\nSkipped files:")
            for idx in all_skipped_files:
                filename = (
                    file_list.iloc[idx].filename
                    if "filename" in file_list.columns
                    else f"file_{idx}"
                )
                print(f"   File {idx}: {filename}")

        return all_report_files


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Weather Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze files from January 2025 (sequential, skip already processed)
  python validate_input.py --start-date 20250101 --end-date 20250131 --output ./results

  # Force reprocess all files, overwriting existing outputs
  python validate_input.py --start-date 20250101 --end-date 20250105 --overwrite

  # Use parallel processing with 4 clients, 16 total workers (4 workers per client)
  python validate_input.py --start-date 20250101 --end-date 20250105 --workers 16 --parallel-clients 4

  # Resume interrupted job (skips already processed files automatically)
  python validate_input.py --start-date 20250101 --end-date 20250131 --workers 32 --parallel-clients 8

  # Force reprocess with parallel clients (overwrite existing files)
  python validate_input.py --start-date 20250101 --end-date 20250105 --workers 16 --parallel-clients 4 --overwrite

  # Test S3 access only
  python validate_input.py --test-access

  # Analyze single file (skip if already processed)
  python validate_input.py --single-file s3://opera-ecmwf/20250318/ECMWF_TROP_202503180600_202503180600_1.nc

  # Force reprocess single file
  python validate_input.py --single-file s3://opera-ecmwf/20250318/ECMWF_TROP_202503180600_202503180600_1.nc --overwrite

  # Optimal parallel setup for many files (balance parallelism vs resources)
  python validate_input.py --start-date 20250101 --end-date 20250131 --workers 32 --parallel-clients 8
        """,
    )

    # Date range options
    parser.add_argument("--start-date", type=str, help="Start date in YYYYMMDD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYYMMDD format")

    # Single file option
    parser.add_argument("--single-file", type=str, help="Analyze a single file by URL")

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./weather_stats",
        help="Output directory (default: ./weather_stats)",
    )

    # Processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Total number of Dask workers (default: 4)",
    )
    parser.add_argument(
        "--worker-mem",
        "-wm",
        type=int,
        default=2,
        help="Given emory per worker (default: 2GB)",
    )
    parser.add_argument(
        "--parallel-clients",
        "-p",
        type=int,
        default=1,
        help="Number of parallel Dask clients (default: 1 = sequential)",
    )
    parser.add_argument(
        "--s3-profile",
        type=str,
        default="saml-pub",
        help="AWS S3 profile name (default: saml-pub)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (use simple print statements)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip already processed files)",
    )

    # Utility options
    parser.add_argument(
        "--test-access", action="store_true", help="Test S3 access and exit"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--list-files", action="store_true", help="List available files and exit"
    )
    parser.add_argument(
        "--test-progress", action="store_true", help="Test progress bar functionality"
    )

    args = parser.parse_args()

    # Test progress bar if requested
    if args.test_progress:
        print("Testing progress bar...")
        import time

        with tqdm(range(10), desc="Test progress") as pbar:
            for i in pbar:
                time.sleep(1)
                pbar.set_description(f"Step {i+1}")
                pbar.refresh()
        print("Progress bar test completed!")
        sys.exit(0)

    # Setup workflow
    workflow = WeatherDataWorkflow(s3_profile=args.s3_profile)

    # Test access if requested
    if args.test_access:
        success = workflow.check_s3_access()
        sys.exit(0 if success else 1)

    # Validate arguments
    if not args.single_file and (not args.start_date or not args.end_date):
        parser.error(
            "Must specify either --single-file or both --start-date and --end-date"
        )

    if args.parallel_clients < 1:
        parser.error("parallel-clients must be at least 1")

    if args.workers < 1:
        parser.error("workers must be at least 1")

    try:
        if args.single_file:
            # Analyze single file
            print(f"Analyzing single file: {args.single_file}")
            analyzer = WeatherDataAnalyzer(
                output_dir=args.output,
                n_workers=args.workers,
                s3_profile=args.s3_profile,
                verbose=args.verbose,
            )
            summary_df, txt_file, csv_file = analyzer.analyze_file(
                args.single_file, overwrite=args.overwrite
            )
            print("Analysis complete. Reports saved to:")
            print(f"   {txt_file}")
            print(f"   {csv_file}")

        else:
            # Check S3 access first
            if not workflow.check_s3_access():
                print(
                    "Cannot proceed without S3 access. Please check your AWS credentials."
                )
                sys.exit(1)

            # Get file list
            file_list = workflow.get_file_list(args.start_date, args.end_date)

            if args.list_files:
                print("\nAvailable files:")
                print(file_list.to_string(index=False))
                sys.exit(0)

            if len(file_list) == 0:
                print("No files found for the specified date range.")
                sys.exit(1)

            # Analyze all files
            workflow.analyze_all_files(
                file_list=file_list,
                output_dir=args.output,
                n_workers=args.workers,
                worker_mem=args.worker_mem,
                verbose=args.verbose,
                use_progress_bar=not args.no_progress,
                parallel_clients=args.parallel_clients,
                overwrite=args.overwrite,
            )

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
