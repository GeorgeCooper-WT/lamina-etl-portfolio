"""
SCADA & ERA5 Data Combiner

Aligns, merges, and disaggregates SCADA (site) and ERA5 weather data for solar analytics.
- Robust timezone alignment and verification
- Disaggregation of hourly ERA5 data to high-frequency intervals using solar geometry
- Rainfall pattern analysis
- Detailed logging and validation

Usage:
    python scada_era5_combine.py --client_id myclient --scada_file /path/to/scada.csv --era5_file /path/to/era5.nc

Dependencies:
    pandas, xarray, numpy, scipy, pvlib, pyyaml
"""

import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import gamma
from pvlib import solarposition
import time
import logging
import os
import yaml
import argparse

# Configure logging
logger = logging.getLogger("scada_era5_combine")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_client_config(client_id: str) -> dict:
    config_path = os.path.join("configs", "clients", client_id, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config for client {client_id}: {config_path}")
    return config


def align_timezones(
    site_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    site_timezone: str = "Europe/London",
    era5_timezone: str = "UTC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align timezones between site data and ERA5 data.
    Returns site_df_utc, era5_df_utc.
    """
    logger.info(f"\n=== TIMEZONE ALIGNMENT ===")
    logger.info(f"Site data timezone: {site_timezone}")
    logger.info(f"ERA5 data timezone: {era5_timezone}")

    site_df_aligned = site_df.copy()
    era5_df_aligned = era5_df.copy()

    # 1. Handle site data timezone
    if site_df_aligned.index.tz is None:
        logger.info(
            "Site data has no timezone info - detecting from production patterns..."
        )
        summer_data = site_df_aligned[site_df_aligned.index.month == 6]
        peak_hour = None
        if len(summer_data) > 288 and "actual_yield_w" in site_df_aligned.columns:
            hourly_production = summer_data.groupby(summer_data.index.hour)[
                "actual_yield_w"
            ].mean()
            peak_hour = hourly_production.idxmax()
            logger.info(f"Production peaks at hour {peak_hour}")
        if peak_hour is not None:
            if 11 <= peak_hour <= 13:
                logger.info("✓ Data appears to already be in UTC - adding UTC timezone")
                site_df_aligned.index = site_df_aligned.index.tz_localize("UTC")
            elif 9 <= peak_hour <= 11:
                logger.warning(
                    f"Data appears to be in local time (peak at {peak_hour}) - localizing to {site_timezone}"
                )
                try:
                    site_df_aligned.index = site_df_aligned.index.tz_localize(
                        site_timezone, ambiguous="infer"
                    )
                    site_df_aligned.index = site_df_aligned.index.tz_convert("UTC")
                    logger.info(f"✓ Site data converted from {site_timezone} to UTC")
                except Exception as e:
                    logger.error(f"Failed to convert timezone: {e}")
                    site_df_aligned.index = site_df_aligned.index.tz_localize("UTC")
            else:
                logger.warning(
                    f"Unusual peak hour {peak_hour} - assuming local time and converting"
                )
                try:
                    site_df_aligned.index = site_df_aligned.index.tz_localize(
                        site_timezone, ambiguous="infer"
                    )
                    site_df_aligned.index = site_df_aligned.index.tz_convert("UTC")
                    logger.info(f"✓ Site data converted from {site_timezone} to UTC")
                except Exception as e:
                    logger.error(f"Failed to convert timezone: {e}")
                    site_df_aligned.index = site_df_aligned.index.tz_localize("UTC")
        else:
            logger.warning(
                "Could not determine peak hour for timezone detection. Assuming UTC."
            )
            site_df_aligned.index = site_df_aligned.index.tz_localize("UTC")

    # 2. Convert site data to UTC
    if site_df_aligned.index.tz != "UTC":
        logger.info("Converting site data to UTC...")
        site_df_aligned.index = site_df_aligned.index.tz_convert("UTC")
        logger.info("✓ Site data converted to UTC")

    # 3. Handle ERA5 data timezone
    if era5_df_aligned.index.tz is None:
        logger.info(f"ERA5 data has no timezone - adding {era5_timezone}")
        era5_df_aligned.index = era5_df_aligned.index.tz_localize(era5_timezone)
    if era5_df_aligned.index.tz != "UTC":
        era5_df_aligned.index = era5_df_aligned.index.tz_convert("UTC")

    # 5. Verification
    logger.info("\nAfter alignment:")
    logger.info(f"Site data timezone: {site_df_aligned.index.tz}")
    logger.info(f"ERA5 data timezone: {era5_df_aligned.index.tz}")
    logger.info(
        f"Site data range: {site_df_aligned.index.min()} to {site_df_aligned.index.max()}"
    )
    logger.info(
        f"ERA5 data range: {era5_df_aligned.index.min()} to {era5_df_aligned.index.max()}"
    )

    # 6. Final verification with production patterns
    if "actual_yield_w" in site_df_aligned.columns:
        summer_data = site_df_aligned[site_df_aligned.index.month == 6]
        if len(summer_data) > 288:
            hourly_production = summer_data.groupby(summer_data.index.hour)[
                "actual_yield_w"
            ].mean()
            peak_hour = hourly_production.idxmax()
            logger.info(f"\nVerification: Production now peaks at hour {peak_hour} UTC")
            if 11 <= peak_hour <= 14:
                logger.info("✓ Timezone alignment appears successful!")
            else:
                logger.warning(
                    "Production peak still unusual - may need further investigation"
                )

    return site_df_aligned, era5_df_aligned


def disaggregate_solar_radiation(
    hourly_data: pd.Series,
    target_index: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> pd.Series:
    """Disaggregate solar radiation data using solar geometry (1hour > 5minute)."""
    start_time = time.time()
    logger.info(f"Calculating solar positions for {len(target_index)} timestamps...")
    solar_pos = solarposition.get_solarposition(target_index, latitude, longitude)
    logger.info(
        f"Solar position calculation completed in {time.time() - start_time:.1f}s"
    )
    disagg = pd.Series(0.0, index=target_index)
    night_mask = solar_pos["elevation"] <= 0
    disagg[night_mask] = 0
    total_hours = len(hourly_data)
    logger.info(f"Processing {total_hours} hours of data...")
    for i, (hour_start, value) in enumerate(hourly_data.items(), 1):
        if value > 0:
            hour_end = hour_start + pd.Timedelta(hours=1)
            mask = (target_index >= hour_start) & (target_index < hour_end)
            intervals = target_index[mask]
            if len(intervals) > 0:
                hour_elevation = solar_pos.loc[intervals, "elevation"]
                weights = np.maximum(0, np.sin(np.radians(hour_elevation)))
                if weights.sum() > 0:
                    weights = weights / weights.sum() * value
                    disagg[intervals] = weights
    logger.info(
        f"Solar radiation disaggregation completed in {time.time() - start_time:.1f}s"
    )
    return disagg


def disaggregate_rainfall(
    hourly_data: pd.Series, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Disaggregate rainfall data into realistic noise patterns (1hour > 5min)."""
    hourly_vals = hourly_data * 1000  # m to mm
    disagg = pd.Series(0.0, index=target_index)
    for hour_start, value in hourly_vals.items():
        if value > 0:
            hour_end = hour_start + pd.Timedelta(hours=1)
            intervals = target_index[
                (target_index >= hour_start) & (target_index < hour_end)
            ]
            n_intervals = len(intervals)
            if n_intervals > 0:
                rain_intervals = max(1, int(n_intervals * 0.3))
                rain_indices = np.random.seed(42)(
                    n_intervals, size=rain_intervals, replace=False
                )
                pattern = np.zeros(n_intervals)
                pattern[rain_indices] = value / rain_intervals
                disagg[intervals] = pattern
    return disagg


def disaggregate_era5_data(
    era5_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Main function to handle all ERA5 data disaggregation."""
    logger.info("\nDisaggregating ERA5 variables...")
    era5_5min = pd.DataFrame(index=target_index)
    for col in era5_df.columns:
        logger.info(f"Processing {col}...")
        if col in ["latitude", "longitude"]:
            era5_5min[col] = era5_df[col].iloc[0]
        elif col in ["ssrd", "ssrdc"]:
            era5_5min[col] = disaggregate_solar_radiation(
                era5_df[col], target_index, latitude, longitude
            )
        elif col == "tp":
            era5_5min[col] = disaggregate_rainfall(era5_df[col], target_index)
        else:
            era5_5min[col] = (
                era5_df[col].reindex(target_index).interpolate(method="time")
            )
        logger.info(f"✓ {col} processed")
    logger.info("\nVerifying disaggregation:")
    for col in ["tp", "ssrd", "ssrdc"]:
        if col in era5_5min.columns:
            original_sum = era5_df[col].sum()
            disagg_sum = era5_5min[col].resample("1H").sum().sum()
            diff_pct = abs(100 * (disagg_sum - original_sum) / original_sum)
            logger.info(
                f"{col}: Original sum = {original_sum:.3f}, Disaggregated sum = {disagg_sum:.3f}, Difference = {diff_pct:.2f}%"
            )
    return era5_5min


def analyze_rain_patterns(
    combined_df: pd.DataFrame, thresholds=[0.1, 0.5, 1.0, 2.0]
) -> None:
    """Analyze rainfall patterns with different thresholds."""
    logger.info("\n=== RAINFALL ANALYSIS ===")
    daily_rain = combined_df["tp"].resample("D").sum()
    years = daily_rain.index.year.unique()
    logger.info("Rain days per year:")
    header = "Year\t" + "\t".join([f">{t}mm" for t in thresholds])
    logger.info(header)
    yearly_totals = {t: [] for t in thresholds}
    for year in years:
        year_data = daily_rain[daily_rain.index.year == year]
        row = f"{year}\t"
        for threshold in thresholds:
            rain_days = (year_data > threshold).sum()
            yearly_totals[threshold].append(rain_days)
            row += f"{rain_days}\t"
        logger.info(row)
    logger.info("\nAverage rain days per year:")
    for threshold in thresholds:
        avg_days = np.mean(yearly_totals[threshold])
        logger.info(f">{threshold}mm: {avg_days:.1f} days")
    annual_rainfall = daily_rain.groupby(daily_rain.index.year).sum()
    logger.info(f"\nAverage annual rainfall: {annual_rainfall.mean():.1f}mm")


def format_datetime_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime index to original UK format string keeping UTC times."""
    df_formatted = df.copy()
    datetime_strings = df_formatted.index.strftime("%d/%m/%Y %H:%M")
    df_formatted = df_formatted.reset_index(drop=True)
    df_formatted.insert(0, "datetime", datetime_strings)
    return df_formatted


def verify_saved_file(output_filename: str, combined_df: pd.DataFrame) -> None:
    logger.info("\n=== VERIFICATION OF SAVED FILE ===")
    verification_df = pd.read_csv(output_filename)
    verification_df["datetime"] = pd.to_datetime(
        verification_df["datetime"], format="%d/%m/%Y %H:%M", dayfirst=True
    )
    verification_df = verification_df.set_index("datetime").sort_index()
    if "actual_yield_w" in verification_df.columns:
        summer_verification = verification_df[verification_df.index.month == 6]
        if len(summer_verification) > 288:
            hourly_verification = summer_verification.groupby(
                summer_verification.index.hour
            )["actual_yield_w"].mean()
            peak_hour_verification = hourly_verification.idxmax()
            logger.info(f"Production peak in saved file: hour {peak_hour_verification}")
            if 11 <= peak_hour_verification <= 14:
                logger.info(
                    "✅ SUCCESS: Saved file will work correctly with analysis pipeline!"
                )
            else:
                logger.error(
                    f"❌ PROBLEM: Saved file still shows peak at hour {peak_hour_verification}"
                )
                logger.error("The timezone conversion is not working correctly.")
                logger.error(f"\nDEBUG INFO:")
                logger.error(
                    f"Combined DF (UTC) production peaks at: {combined_df[combined_df.index.month==6].groupby(combined_df.index.hour)['actual_yield_w'].mean().idxmax()}"
                )
                logger.error(
                    f"Saved file production peaks at: {peak_hour_verification}"
                )
                logger.error("These should be different by the timezone offset!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine SCADA and ERA5 data for solar analytics."
    )
    parser.add_argument("--client_id", required=True, help="Client ID")
    parser.add_argument(
        "--scada_file", required=True, help="Path to SCADA (site) CSV file"
    )
    parser.add_argument("--era5_file", required=True, help="Path to ERA5 NetCDF file")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: data/<client_id>/processed)",
    )
    args = parser.parse_args()

    client_id = args.client_id
    scada_file = args.scada_file
    era5_file = args.era5_file
    output_dir = args.output_dir or os.path.join("data", client_id, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{client_id}_master_data_set_combined.csv"
    output_path = os.path.join(output_dir, output_filename)

    config = load_client_config(client_id)

    client_data_dir = os.path.join("data", client_id)
    if not os.path.exists(client_data_dir):
        logger.error(f"Client data directory does not exist: {client_data_dir}")
        exit(1)

    # # --- Developer diagnostic: View contents of NetCDF file ---
    # # Uncomment for debugging NetCDF structure
    # era5_ds = xr.open_dataset(era5_file)
    # logger.info("ERA5 variables in merged file:")
    # logger.info(list(era5_ds.data_vars))
    # logger.info("\nERA5 coordinates:")
    # logger.info(list(era5_ds.coords))

    # --- Load site dataset ---
    site_df = pd.read_csv(scada_file)
    site_df["temp_datetime"] = pd.to_datetime(site_df["temp_datetime"])
    site_df = site_df.set_index("temp_datetime").sort_index()
    site_df = site_df[~site_df.index.duplicated(keep="first")]

    logger.info(f"\nOriginal site data sample (first 5 timestamps):")
    logger.info(site_df.index[:5])
    logger.info(f"Site data timezone: {site_df.index.tz}")

    # --- Load ERA5 dataset ---
    era5_ds = xr.open_dataset(era5_file)
    required_vars = [
        "sd",
        "ssrdc",
        "ssrd",
        "tcc",
        "tp",
        "u10",
        "v10",
        "t2m",
        "latitude",
        "longitude",
    ]
    available_vars = [var for var in required_vars if var in era5_ds.data_vars]
    logger.info(f"Variables selected for merging: {available_vars}")

    era5_df = era5_ds[available_vars].to_dataframe().reset_index()
    era5_df["valid_time"] = pd.to_datetime(era5_df["valid_time"])
    era5_df = era5_df.set_index("valid_time").sort_index()
    era5_df = era5_df[~era5_df.index.duplicated(keep="first")]

    logger.info(f"\nOriginal ERA5 data sample (first 5 timestamps):")
    logger.info(era5_df.index[:5])
    logger.info(f"ERA5 data timezone: {era5_df.index.tz}")

    # --- APPLY TIMEZONE ALIGNMENT ---
    site_df_aligned, era5_df_aligned = align_timezones(site_df, era5_df)

    # --- Add missing timestamps to site data ---
    full_index = pd.date_range(
        site_df_aligned.index.min(),
        site_df_aligned.index.max(),
        freq="2min",
        tz=site_df_aligned.index.tz,
    )
    site_df_aligned = site_df_aligned.reindex(full_index)

    # --- Interpolate ERA5 to 5-min intervals ---
    logger.info(f"\nInterpolating ERA5 data to site data frequency...")
    new_index = site_df_aligned.index
    era5_5min = era5_df_aligned.reindex(new_index)
    disagg_cols = ["tp", "ssrdc", "ssrd"]
    columns_to_interp = [col for col in era5_5min.columns if col not in disagg_cols]
    era5_5min[columns_to_interp] = era5_5min[columns_to_interp].interpolate(
        method="time"
    )
    era5_5min[columns_to_interp] = era5_5min[columns_to_interp].bfill()

    logger.info("\nStarting ERA5 disaggregation...")
    latitude = era5_df_aligned["latitude"].iloc[0]
    longitude = era5_df_aligned["longitude"].iloc[0]

    era5_5min = disaggregate_era5_data(
        era5_df_aligned, site_df_aligned.index, latitude, longitude
    )

    # --- Merge aligned datasets ---
    combined_df = site_df_aligned.join(era5_5min, how="left")

    # --- Rainfall analysis ---
    analyze_rain_patterns(combined_df)

    # --- Disaggregation checks ---
    for col in ["ssrd", "ssrdc", "tp"]:
        if col not in era5_5min.columns:
            logger.warning(f"Warning: Missing expected column: {col}")

    nan_check = era5_5min[["ssrd", "ssrdc", "tp"]].isna().sum()
    if nan_check.any():
        logger.warning("Warning: NaN values found:")
        logger.warning(nan_check[nan_check > 0])

    for hour, group in era5_5min["tp"].groupby(era5_5min.index.floor("H")):
        original = era5_df_aligned.loc[hour, "tp"] * 1000
        disagg_sum = group.sum()
        if not np.isclose(original, disagg_sum, atol=1e-3):
            logger.warning(f"Hour {hour}: original={original}, disagg_sum={disagg_sum}")

    # --- Final verification ---
    logger.info(f"\nFinal combined dataset:")
    logger.info(f"Shape: {combined_df.shape}")
    logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    logger.info(f"Timezone: {combined_df.index.tz}")

    logger.info("\nSample of combined dataset (first 5 rows):")
    logger.info(combined_df.head())

    # --- Format and save ---
    logger.info("\nFormatting datetime for compatibility with existing pipeline...")
    formatted_df = format_datetime_for_output(combined_df)
    formatted_df.to_csv(output_path, index=False)
    logger.info(
        f"✓ Timezone-aligned dataset saved with original date format: {output_path}"
    )

    logger.info("\nSample of formatted output:")
    logger.info(formatted_df[["datetime"]].head())
    logger.info("Format: DD/MM/YYYY HH:MM (UTC times in string format)")

    # --- Verification of saved file ---
    verify_saved_file(output_path, combined_df)

    # --- Final production peak verification from the in-memory data ---
    if "actual_yield_w" in combined_df.columns:
        summer_final = combined_df[combined_df.index.month == 6]
        if len(summer_final) > 288:
            final_hourly = summer_final.groupby(summer_final.index.hour)[
                "actual_yield_w"
            ].mean()
            final_peak = final_hourly.idxmax()
            logger.info(
                f"\nIn-memory UTC data: Production peaks at hour {final_peak} UTC"
            )
            if 11 <= final_peak <= 14:
                logger.info("✅ UTC alignment in memory is correct!")
            else:
                logger.warning("⚠️  Warning: Even UTC data peak is unusual")


if __name__ == "__main__":
    main()
