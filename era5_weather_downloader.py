"""
ERA5 Weather Downloader

Downloads, summarizes, and merges ERA5 weather data for a given client and time period.
Handles configuration, logging, and robust error handling.
"""

import argparse
import calendar
import glob
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cdsapi
import xarray as xr
import yaml

# -----------------------
# LOGGING CONFIGURATION
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------
# CONSTANTS
# -----------------------
DEFAULT_DELTA = 0.01  # ~1km bounding box for area
DEFAULT_DISK_GB = 20
MAX_RETRIES = 3
RETRY_WAIT_BASE = 10  # seconds

VARIABLES_TIME: Dict[str, List[str]] = {
    "surface_solar_radiation_downwards": [f"{h:02d}:00" for h in range(24)],
    "surface_solar_radiation_downward_clear_sky": [f"{h:02d}:00" for h in range(24)],
    "total_cloud_cover": [f"{h:02d}:00" for h in range(24)],
    "2m_temperature": [f"{h:02d}:00" for h in range(24)],
    "10m_u_component_of_wind": [f"{h:02d}:00" for h in range(24)],
    "10m_v_component_of_wind": [f"{h:02d}:00" for h in range(24)],
    "snow_depth": [f"{h:02d}:00" for h in range(24)],
    "total_precipitation": [f"{h:02d}:00" for h in range(24)],
}

DATASET = "reanalysis-era5-single-levels"

VAR_MAP: Dict[str, str] = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_solar_radiation_downward_clear_sky": "ssrdc",
    "total_cloud_cover": "tcc",
    "snow_depth": "sd",
    "total_precipitation": "tp",
}


# -----------------------
# CONFIGURATION
# -----------------------
def load_client_config_and_setup() -> (
    Tuple[Dict[str, Any], str, List[str], str, List[float]]
):
    """
    Loads client config and prepares output directory and area.

    Returns:
        config (dict): Client config dictionary.
        client_id (str): Client ID.
        years (list): List of years as strings.
        output_dir (str): Output directory path.
        area (list): Bounding box [N, W, S, E].
    """
    client_id = input("Enter client ID: ").strip()
    config_path = os.path.join("configs", "clients", client_id, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    installation_date = config.get("installation_date")
    if not installation_date:
        raise ValueError("installation_date missing from config.")
    try:
        install_year = datetime.strptime(installation_date, "%Y-%m-%d").year
    except ValueError:
        raise ValueError(f"Invalid installation_date format: {installation_date}")

    current_year = datetime.now().year
    years = [str(y) for y in range(install_year, current_year + 1)]

    output_dir = os.path.join("data", client_id, "Raw", "era5_data")
    os.makedirs(output_dir, exist_ok=True)

    lat = config.get("latitude")
    lon = config.get("longitude")
    if lat is None or lon is None:
        raise ValueError("latitude and/or longitude missing from config.")
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        raise ValueError(f"Invalid latitude or longitude: {lat}, {lon}")
    area = get_area_from_latlon(lat, lon, delta=DEFAULT_DELTA)

    return config, client_id, years, output_dir, area


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def get_days_for_month(year: str, month: int) -> List[str]:
    """Get valid days for a given year and month."""
    _, last_day = calendar.monthrange(int(year), int(month))
    return [f"{d:02d}" for d in range(1, last_day + 1)]


def get_area_from_latlon(
    lat: float, lon: float, delta: float = DEFAULT_DELTA
) -> List[float]:
    """
    Returns a bounding box [N, W, S, E] for ERA5 API given a center lat/lon.
    """
    north = lat + delta
    south = lat - delta
    west = lon - delta
    east = lon + delta
    return [north, west, south, east]


def check_disk_space(path: str, required_gb: int = DEFAULT_DISK_GB) -> bool:
    """Check if enough disk space is available."""
    try:
        free_bytes = shutil.disk_usage(path).free
        free_gb = free_bytes / (1024**3)
        return free_gb >= required_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


def get_var_from_filename(filename: str) -> str:
    """Extract variable name from NetCDF filename."""
    match = re.match(r"([a-zA-Z0-9_]+)_\d{4}_\d{2}\.nc", os.path.basename(filename))
    return match.group(1) if match else None


def download_era5_data(
    c: cdsapi.Client,
    output_dir: str,
    years: List[str],
    variables_time: Dict[str, List[str]],
    area: List[float],
    dataset: str,
) -> None:
    """Download ERA5 data month-by-month for each variable."""
    total_downloads = len(years) * 12 * len(variables_time)
    current_download = 0
    for year in years:
        for month_num in range(1, 13):
            month = f"{month_num:02d}"
            days = get_days_for_month(year, month_num)
            for var, times in variables_time.items():
                current_download += 1
                filename = f"{output_dir}/{var}_{year}_{month}.nc"
                logger.info(
                    f"[{current_download}/{total_downloads}] Processing {var} for {year}-{month}"
                )
                if os.path.exists(filename):
                    logger.info(f"Skipping {filename}, already exists.")
                    continue
                request = {
                    "product_type": "reanalysis",
                    "variable": [var],
                    "year": [year],
                    "month": [month],
                    "day": days,
                    "time": times,
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                }
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        logger.info(f"  Downloading attempt {attempt}...")
                        c.retrieve(dataset, request).download(filename)
                        logger.info(f"  ✓ Saved to {filename}")
                        break
                    except Exception as e:
                        logger.error(f"  ✗ Error: {e}")
                        if attempt < MAX_RETRIES:
                            wait_time = RETRY_WAIT_BASE * attempt
                            logger.info(f"  Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"  ✗ Failed after {MAX_RETRIES} attempts.")
                            break
                time.sleep(2)  # Be nice to the API


def summarise_netcdf_files(output_dir: str) -> None:
    """Print a summary of NetCDF files in the output directory for consistency check."""
    files = glob.glob(f"{output_dir}/*.nc")
    if not files:
        logger.info("No NetCDF files found for summary.")
        return
    logger.info("Summary of NetCDF files for consistency check:")
    summary = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            summary.append(
                {
                    "file": os.path.basename(f),
                    "vars": tuple(ds.data_vars.keys()),
                    "lat": float(ds.latitude.values[0]),
                    "lon": float(ds.longitude.values[0]),
                    "start": str(ds.valid_time.min().values)[:19],
                    "end": str(ds.valid_time.max().values)[:19],
                    "shape": ds[list(ds.data_vars.keys())[0]].shape,
                }
            )
            ds.close()
        except Exception as e:
            summary.append({"file": os.path.basename(f), "error": str(e)})
    grouped = defaultdict(list)
    for entry in summary:
        if "vars" in entry:
            grouped[entry["vars"]].append(entry)
    for var_set, entries in grouped.items():
        logger.info(f"\nVariable(s): {var_set}")
        for entry in entries[:5]:
            logger.info(entry)
        if len(entries) > 5:
            logger.info(f"... {len(entries)} files for {var_set}")
    logger.info(f"Total files checked: {len(summary)}")
    unique_lats = set(e["lat"] for e in summary if "lat" in e)
    unique_lons = set(e["lon"] for e in summary if "lon" in e)
    logger.info(f"Unique latitudes: {unique_lats}")
    logger.info(f"Unique longitudes: {unique_lons}")


def merge_netcdf_files(
    output_dir: str, var_map: Dict[str, str], years: List[str]
) -> None:
    """Merge NetCDF files in output_dir by variable using Dask chunks."""
    logger.info("=" * 50)
    logger.info("ROBUST VARIABLE-WISE MERGING")
    logger.info("=" * 50)
    files = glob.glob(f"{output_dir}/*.nc")
    if not files:
        logger.info("No NetCDF files found to merge.")
        return
    logger.info(f"Found {len(files)} NetCDF files to merge...")
    files.sort()
    var_files = {}
    for f in files:
        var = get_var_from_filename(f)
        if var:
            var_files.setdefault(var, []).append(f)
    merged_data = {}
    for var, flist in var_files.items():
        nc_var = var_map.get(var, var)
        logger.info(
            f"Merging {len(flist)} files for variable '{var}' (NetCDF var: '{nc_var}')..."
        )
        flist.sort()
        ds_var = xr.open_mfdataset(
            flist,
            combine="by_coords",
            coords="minimal",
            compat="override",
            chunks={"valid_time": 100},
        )
        da = ds_var[nc_var].load()
        if "expver" in da.coords:
            da = da.drop_vars("expver")
        merged_data[nc_var] = da
        ds_var.close()
    ds_merged = xr.Dataset(merged_data)
    merged_file = os.path.join(output_dir, f"era5_{years[0]}_{years[-1]}_merged.nc")
    logger.info(f"Saving robust merged dataset to {merged_file}...")
    encoding = {
        var: {"zlib": True, "complevel": 4, "chunksizes": (100, 1, 1)}
        for var in ds_merged.data_vars
    }
    ds_merged.to_netcdf(merged_file, encoding=encoding)
    logger.info(f"✓ Merged dataset saved to {merged_file}")
    logger.info(f"  Variables: {list(ds_merged.data_vars.keys())}")
    logger.info(
        f"  Time range: {ds_merged['valid_time'].min().values} to {ds_merged['valid_time'].max().values}"
    )


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    config, client_id, years, output_dir, area = load_client_config_and_setup()

    # Initialise CDS API client
    c = cdsapi.Client()
    # Check disk space
    if not check_disk_space(output_dir, DEFAULT_DISK_GB):
        logger.warning("Warning: Low disk space detected.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            exit()
    # Download data
    download_era5_data(c, output_dir, years, VARIABLES_TIME, area, DATASET)
    # Summarise files
    summarise_netcdf_files(output_dir)
    # Merge files with error handling
    try:
        merge_netcdf_files(output_dir, VAR_MAP, years)
    except Exception as e:
        logger.error(f"Error during merging NetCDF files: {e}")


if __name__ == "__main__":
    main()
