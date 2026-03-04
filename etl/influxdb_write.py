"""
Utilities for writing scored solar data to InfluxDB, including batching, validation, and diagnostics.

Usage Example:
    from influxdb_write import orchestrate_influxdb_write

    scored = pd.read_csv("scored_data.csv")
    df = pd.read_csv("extra_data.csv")

    orchestrate_influxdb_write(
        scored,
        df,
        influx_url,
        token,
        org,
        bucket,
        client_id,
        confirm=True
    )
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("lamina.influxdb_write")

COLUMNS_TO_WRITE: List[str] = [
    "actual_yield_kw_preclip",
    "actual_yield_kw",
    "physics_predicted_yield",
    "expected_kw",
    "expected_kw_preclip",
    "expected_kw_golden",
    "ml_is_clipping",
    "ml_clipping_loss_kw",
    "ghi",
    "residual_kw",
    "residual_pct_cap",
    "residual_z",
    "underperf_flag",
    "severe_underperf_flag",
    "solar_elevation_deg",
    "_poa_wm2",
    "temp_derate",
    "month",
    "performance_ratio",
    "rolling_pr",
    "air_temp",
    "wind_speed",
    "wind_direction",
    "sd",
    "ssrdc",
    "ssrd",
    "tcc",
    "tp",
    "aoi",
    "iam",
    "airmass",
    "clearsky_index_poa",
    "pr_string1",
    "pr_string2",
    "string1_kw",
    "string2_kw",
    "s1_physics_predicted_yield",
    "s2_physics_predicted_yield",
    "degradation_factor",
    "rolling_pr_daily_mean",
    "module_temp",
    "solar_azimuth",
    "physics_predicted_yield_unclipped",
    "s1_physics_predicted_yield_unclipped",
    "s2_physics_predicted_yield_unclipped",
    "is_clipping",
    "clipping_loss_kw",
    "r2_score",
    "nrmse_stc",
]


def prepare_scored_for_influx(
    scored: pd.DataFrame, df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add/compute columns needed for InfluxDB write.

    Args:
        scored (pd.DataFrame): The scored DataFrame to process.
        df (pd.DataFrame, optional): Optional DataFrame for additional columns.

    Returns:
        pd.DataFrame: The processed DataFrame ready for InfluxDB.
    """
    scored = scored.copy()
    scored["temp_derate"] = (
        df["temp_derate"] if df is not None and "temp_derate" in df.columns else np.nan
    )
    scored["month"] = scored.index.month
    scored["is_clipping"] = (
        scored["physics_predicted_yield_unclipped"] > scored["physics_predicted_yield"]
    )
    scored["clipping_loss_kw"] = np.where(
        scored["is_clipping"],
        scored["physics_predicted_yield_unclipped"] - scored["physics_predicted_yield"],
        0,
    )
    return scored


def filter_valid_solar(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out nighttime/low solar elevation periods.

    Args:
        scored (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with only valid solar periods.
    """
    valid_solar_mask = scored["solar_elevation_deg"] >= 5
    return scored[valid_solar_mask].copy()


def get_existing_columns(scored: pd.DataFrame, columns_to_write: List[str]) -> List[str]:
    """
    Return only columns that exist in the DataFrame.

    Args:
        scored (pd.DataFrame): The DataFrame to check.
        columns_to_write (list): List of columns to check for.

    Returns:
        list: List of columns that exist in the DataFrame.
    """
    return [col for col in columns_to_write if col in scored.columns]


def log_influxdb_test_output(
    scored_filtered: pd.DataFrame, existing_columns: List[str]
) -> None:
    """
    Log diagnostics and sample output before writing to InfluxDB.

    Args:
        scored_filtered (pd.DataFrame): The filtered DataFrame to be written.
        existing_columns (list): List of columns to write.
    """
    logger.info("=" * 80)
    logger.info("INFLUXDB OUTPUT TEST (NOT WRITING TO DATABASE)")
    logger.info("=" * 80)
    test_sample = scored_filtered[existing_columns].sample(
        n=min(20, len(scored_filtered)), random_state=42
    )
    logger.info(f"Columns to write: {len(existing_columns)}")
    logger.info(f"Column names: {existing_columns[:10]}... (showing first 10)")
    logger.info("Sample of data being sent to InfluxDB:\n%s", test_sample.to_string())
    nan_counts = scored_filtered[existing_columns].isna().sum()
    logger.info("NaN counts per column:\n%s", nan_counts[nan_counts > 0])
    logger.info(
        "Value ranges:\n%s",
        scored_filtered[existing_columns].describe().T[["min", "max", "mean"]],
    )
    logger.info(
        "Nighttime Filtering Verification: min=%.2f°, max=%.2f°",
        scored_filtered["solar_elevation_deg"].min(),
        scored_filtered["solar_elevation_deg"].max(),
    )
    logger.info(
        "Min physics_predicted_yield: %.4f kW",
        scored_filtered["physics_predicted_yield"].min(),
    )
    hourly_dist = scored_filtered.index.hour.value_counts().sort_index()
    logger.info("Hourly distribution of records:\n%s", hourly_dist)


def write_scored_data_in_batches(
    scored_df: pd.DataFrame,
    measurement_name: str,
    influx_url: str,
    token: str,
    org: str,
    bucket: str,
    client_id: str,
    batch_size: int = 1000,
) -> None:
    """
    Write scored data to InfluxDB in batches.

    Args:
        scored_df (pd.DataFrame): DataFrame of scored data.
        measurement_name (str): InfluxDB measurement name.
        influx_url (str): InfluxDB URL.
        token (str): InfluxDB token.
        org (str): InfluxDB organization.
        bucket (str): InfluxDB bucket.
        client_id (str): Client/system identifier.
        batch_size (int, optional): Batch size for writing. Defaults to 1000.
    """

    total_records = len(scored_df)
    batches = range(0, total_records, batch_size)
    logger.info(
        f"Writing {total_records} scored records in {len(list(batches))} batches of {batch_size} to InfluxDB ({measurement_name})..."
    )

    client = influxdb_client.InfluxDBClient(
        url=influx_url, token=token, org=org, timeout="30s"
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for batch_start in batches:
        batch_end = min(batch_start + batch_size, total_records)
        batch_df = scored_df.iloc[batch_start:batch_end]
        points = []
        for timestamp, row in batch_df.iterrows():
            point = (
                influxdb_client.Point(measurement_name)
                .tag("system_id", client_id)
                .time(timestamp)
            )
            for col, value in row.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    point = point.field(col, float(value))
            points.append(point)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                write_api.write(bucket=bucket, org=org, record=points)
                logger.info(
                    f"✓ Batch {batch_start//batch_size + 1}: Records {batch_start+1}-{batch_end} written"
                )
                break
            except Exception as e:
                if "Read timed out" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"⚠ Timeout on attempt {attempt+1}, retrying in 5s..."
                    )
                    time.sleep(5)
                else:
                    logger.error(
                        f"✗ Error writing batch {batch_start//batch_size + 1}: {e}"
                    )
                    break

    client.close()
    logger.info(
        f"✓ Successfully ingested all {total_records} scored records into InfluxDB ({measurement_name})."
    )


def orchestrate_influxdb_write(
    scored: pd.DataFrame,
    df: pd.DataFrame,
    influx_url: str,
    token: str,
    org: str,
    bucket: str,
    client_id: str,
    confirm: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Orchestrate the full process of preparing, validating, and writing scored data to InfluxDB.

    Args:
        scored (pd.DataFrame): The scored DataFrame.
        df (pd.DataFrame): Additional DataFrame for extra columns.
        influx_url (str): InfluxDB URL.
        token (str): InfluxDB token.
        org (str): InfluxDB organization.
        bucket (str): InfluxDB bucket.
        client_id (str): Client/system identifier.
        confirm (bool, optional): Whether to prompt for confirmation before writing. Defaults to True.

    Returns:
        dict: Status and metadata about the write operation.
    """

    # --- Parameter validation ---
    missing_params = []
    for name, value in [
        ("influx_url", influx_url),
        ("token", token),
        ("org", org),
        ("bucket", bucket),
        ("client_id", client_id),
    ]:
        if not value:
            missing_params.append(name)
    if missing_params:
        logger.error(
            f"Missing required parameters: {', '.join(missing_params)}. Aborting write."
        )
        return {
            "status": "error",
            "reason": f"Missing parameters: {', '.join(missing_params)}",
        }

    scored = prepare_scored_for_influx(scored, df)
    scored_filtered = filter_valid_solar(scored)

    existing_columns = get_existing_columns(scored_filtered, COLUMNS_TO_WRITE)
    log_influxdb_test_output(scored_filtered, existing_columns)

    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    measurement_name = f"expected_power_{client_id}_v{version_timestamp}"

    logger.info(f"Measurement: {measurement_name}")
    logger.info(f"Data range: {scored.index.min()} to {scored.index.max()}")
    logger.info(f"Total records: {len(scored):,}")
    logger.info(f"Valid solar records (elevation >= 5°): {len(scored_filtered):,}")
    logger.info(
        f"Filtered out: {len(scored) - len(scored_filtered):,} nighttime records"
    )

    if confirm:
        response = input("Do you want to proceed with writing to InfluxDB? (yes/no): ")
        if response.lower() != "yes":
            logger.info("❌ Skipping InfluxDB write")
            return {
                "status": "skipped",
                "measurement": measurement_name,
                "records_total": len(scored),
                "records_written": 0,
            }

    try:
        write_scored_data_in_batches(
            scored_filtered[existing_columns],
            measurement_name,
            influx_url,
            token,
            org,
            bucket,
            client_id=client_id,
            batch_size=5000,
        )
        return {
            "status": "success",
            "measurement": measurement_name,
            "records_total": len(scored),
            "records_written": len(scored_filtered),
        }
    except Exception as e:
        logger.error(f"Failed to write to InfluxDB: {e}")
        return {
            "status": "error",
            "measurement": measurement_name,
            "records_total": len(scored),
            "records_written": 0,
            "reason": str(e),
        }
