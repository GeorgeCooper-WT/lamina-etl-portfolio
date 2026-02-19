"""
Client Configuration and Data Utilities

This module handles loading, validating, and cleaning client configuration and solar data.
Includes robust logging, error handling, and timezone verification.

NOTE:
- String configuration (panel string parameters) is currently hardcoded for demonstration.
- For production, implement a separate script or function to load/generate string configuration
  (e.g., from a YAML, JSON, or database source) and pass it to the relevant functions.
"""

import logging
import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
import os
from typing import Optional, Tuple, Dict, Any

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("lamina.data")


# --- Configuration Loading ---
def load_client_config(
    client_id: str, configs_dir: Path = Path("configs/clients")
) -> Dict[str, Any]:
    """
    Load client YAML configuration.

    Args:
        client_id: Client identifier.
        configs_dir: Directory containing client configs.

    Returns:
        Client configuration dictionary.
    """
    config_path = configs_dir / client_id / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(client_config: Dict[str, Any], config_filename: str) -> bool:
    """
    Validate that the YAML config dict has all required parameters.

    Args:
        client_config: Client configuration dictionary.
        config_filename: Filename for logging.

    Returns:
        True if validation passes, raises KeyError otherwise.
    """
    required_attrs = ["client_id", "latitude", "longitude", "system_parameters"]
    required_system_params = ["surface_tilt", "surface_azimuth", "dc_capacity_kw"]

    logger.info(f"=== Validating {config_filename} ===")

    for attr in required_attrs:
        if attr not in client_config:
            logger.error(f"Missing required attribute: {attr}")
            raise KeyError(f"Missing required attribute: {attr}")
        logger.info(f"✓ {attr}: {client_config[attr]}")

    system_params = client_config["system_parameters"]
    for param in required_system_params:
        if param not in system_params:
            logger.error(f"Missing required system parameter: {param}")
            raise KeyError(f"Missing required system parameter: {param}")
        logger.info(f"✓ system_parameters.{param}: {system_params[param]}")

    # Optional parameters with defaults
    installation_date = client_config.get("installation_date", "2022-01-01")
    altitude = client_config.get("altitude", 0)
    annual_degradation = system_params.get("annual_degradation", 0.005)

    logger.info(f"✓ installation_date: {installation_date} (default if not specified)")
    logger.info(f"✓ altitude: {altitude} (default if not specified)")
    logger.info(
        f"✓ annual_degradation: {annual_degradation} (default if not specified)"
    )
    logger.info("✓ Configuration validation passed!")

    return True


def load_panel_config(panel_config_path: Path) -> list[dict]:
    """
    Load panel (string) configuration from a YAML file.
    """
    with open(panel_config_path, "r") as f:
        panel_config = yaml.safe_load(f)
    return panel_config


# --- Model Paths ---
def get_model_paths(project_root: Path, client_id: str) -> Dict[str, Path]:
    """
    Get paths to model, features, and scaler files.

    Args:
        project_root: Root directory of the project.
        client_id: Client identifier.

    Returns:
        Dictionary of model file paths.
    """
    model_dir = project_root / "models" / client_id
    return {
        "model": model_dir / f"residual_model_{client_id}.pkl",
        "features": model_dir / f"residual_features_{client_id}.pkl",
        "scaler": model_dir / f"scaler_{client_id}.pkl",
    }


# --- Data Loading & Cleaning ---
def load_and_clean_dataset(
    path: Path, inverter_ac_capacity_kw: float, strict: bool = True
) -> pd.DataFrame:
    """
    Load and clean the master dataset with proper timezone handling.

    Args:
        path: Path to the CSV file.
        inverter_ac_capacity_kw: Inverter AC capacity for clipping.
        strict: If True, raise on missing required columns; else, log and return empty DataFrame.

    Returns:
        Cleaned DataFrame indexed by UTC datetime.
    """
    logger.info(f"Loading dataset: {path}")
    try:
        df = pd.read_csv(path)
        # Convert datetime column to pandas datetime with explicit UTC timezone
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"], format="%d/%m/%Y %H:%M", dayfirst=True
            )
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
            df = df.set_index("datetime").sort_index()
        elif "temp_datetime" in df.columns:
            df["temp_datetime"] = pd.to_datetime(df["temp_datetime"], dayfirst=True)
            df["temp_datetime"] = df["temp_datetime"].dt.tz_localize("UTC")
            df = df.set_index("temp_datetime").sort_index()
        logger.info("✓ Dataset loaded with proper UTC timezone")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Accept legacy truncated column name fallback
    if "actual_yield_w" not in df.columns and "actual_yie" in df.columns:
        df.rename(columns={"actual_yie": "actual_yield_w"}, inplace=True)

    required_cols = ["actual_yield_w", "ghi", "t2m"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        if strict:
            raise ValueError(f"Missing required columns: {missing}")
        else:
            return pd.DataFrame()

    optional_cols = ["surface_solar_radiation_W_m2", "direct_solar_radiation_W_m2"]
    present_optional = [c for c in optional_cols if c in df.columns]
    if present_optional:
        logger.info(f"Optional irradiance columns found: {present_optional}")

    # Remove duplicate timestamps
    dup = df.index.duplicated().sum()
    if dup:
        logger.warning(f"Removing {dup} duplicate timestamps")
        df = df[~df.index.duplicated(keep="first")]

    # t2m may be ERA5 (Kelvin) — detect and convert to Celsius if needed
    if "t2m" in df.columns:
        median_t2m = df["t2m"].median()
        df["air_temp"] = df["t2m"] - 273.15 if median_t2m > 80 else df["t2m"]
    else:
        df["air_temp"] = np.nan

    # Convert to kW and create pre-clip versions
    df["actual_yield_kw"] = df["actual_yield_w"] / 1000.0
    df["string1_kw"] = df["string1_w"] / 1000.0 if "string1_w" in df.columns else np.nan
    df["string2_kw"] = df["string2_w"] / 1000.0 if "string2_w" in df.columns else np.nan
    df["actual_yield_kw_preclip"] = df["actual_yield_kw"]
    df["string1_kw_preclip"] = (
        df["string1_kw"] if "string1_kw" in df.columns else np.nan
    )
    df["string2_kw_preclip"] = (
        df["string2_kw"] if "string2_kw" in df.columns else np.nan
    )

    # Apply inverter clipping
    df["actual_yield_kw"] = df["actual_yield_kw"].clip(0, inverter_ac_capacity_kw)
    if "string1_kw" in df.columns:
        df["string1_kw"] = df["string1_kw"].clip(0, inverter_ac_capacity_kw)
    if "string2_kw" in df.columns:
        df["string2_kw"] = df["string2_kw"].clip(0, inverter_ac_capacity_kw)

    # Timezone verification
    logger.info("=== TIMEZONE VERIFICATION ===")
    summer_check = df[df.index.month == 6]
    if len(summer_check) > 288 and "actual_yield_w" in df.columns:
        peak_hour = (
            summer_check.groupby(summer_check.index.hour)["actual_yield_w"]
            .mean()
            .idxmax()
        )
        logger.info(f"✓ Production peaks at hour {peak_hour} UTC")
        if 11 <= peak_hour <= 14:
            logger.info("✅ Timezone appears correct for UK solar data")
        else:
            logger.warning(
                f"⚠️  WARNING: Peak at hour {peak_hour} suggests timezone issue!"
            )

    return df


# --- Data Validation ---
def validate_solar_data(df: pd.DataFrame, asset_dc_capacity_kw: float) -> pd.DataFrame:
    """
    Solar-specific data validation.

    Args:
        df: DataFrame with solar data.
        asset_dc_capacity_kw: DC capacity of the asset (for ramp checks).

    Returns:
        DataFrame of boolean flags for each validation rule.
    """
    flags = pd.DataFrame(index=df.index)

    # 1. Ramp rate checks
    if "actual_yield_kw" in df.columns:
        max_ramp_up = 0.3 * asset_dc_capacity_kw
        max_ramp_down = -0.4 * asset_dc_capacity_kw
        ramps = df["actual_yield_kw"].diff()
        flags["ramp_violation"] = (ramps > max_ramp_up) | (ramps < max_ramp_down)
    else:
        logger.warning("'actual_yield_kw' column missing: skipping ramp rate check.")
        flags["ramp_violation"] = False

    # 2. Clear-sky envelope validation (if available)
    if "clearsky_index_poa" in df.columns:
        flags["clearsky_violation"] = df["clearsky_index_poa"] > 1.2
    else:
        flags["clearsky_violation"] = False

    # 3. Physical limits check (if available)
    if "_poa_wm2" in df.columns:
        flags["poa_violation"] = df["_poa_wm2"] > 1200
    else:
        flags["poa_violation"] = False

    # 4. Basic GHI validation
    if "ghi" in df.columns:
        flags["ghi_violation"] = df["ghi"] > 1400
    else:
        flags["ghi_violation"] = False

    return flags


def assign_panel_config(df, panel_config):
    """
    Assigns panel configuration parameters to each timestamp in the DataFrame index,
    using an external panel config data structure (list of dicts).

    Args:
        df: DataFrame with a DatetimeIndex.
        panel_config: List of dicts with 'start', 'end', and config parameters.

    Returns:
        DataFrame with panel configuration columns aligned to df.index.
    """
    logger = logging.getLogger("lamina.data")
    config = pd.DataFrame(index=df.index)
    for period in panel_config:
        # Determine start and end timestamps for the period
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"]) if period["end"] else None

        # Handle timezone localization/conversion
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end is not None:
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            else:
                end = end.tz_convert("UTC")
        else:
            end = df.index.max() + pd.Timedelta("1ns")

        mask = (df.index >= start) & (df.index < end)
        for key, value in period.items():
            if key not in ("start", "end"):
                config.loc[mask, key] = value

    # Optional: Log config for summer months, both strings (for diagnostics)
    summer_mask = df.index.month.isin([6, 7, 8])
    logger.info(
        "\nString 1 config (summer):\n%s",
        config.loc[
            summer_mask, ["num_panels_1", "eff_1", "panel_area_1", "temp_coeff_1"]
        ].drop_duplicates(),
    )
    logger.info(
        "\nString 2 config (summer):\n%s",
        config.loc[
            summer_mask, ["num_panels_2", "eff_2", "panel_area_2", "temp_coeff_2"]
        ].drop_duplicates(),
    )

    # After last change for String 2
    last_change = max(
        pd.Timestamp(period["start"]).tz_localize("UTC") if pd.Timestamp(period["start"]).tzinfo is None
        else pd.Timestamp(period["start"]).tz_convert("UTC")
        for period in panel_config
        if period["start"]
    )
    post_change_mask = (df.index >= last_change) & summer_mask
    logger.info(
        "\nString 2 config (summer, post-change):\n%s",
        config.loc[
            post_change_mask, ["num_panels_2", "eff_2", "panel_area_2", "temp_coeff_2"]
        ].drop_duplicates(),
    )

    return config


def aggregate_to_5min(
    df: pd.DataFrame, freq: str = "5min", extra_agg: dict = None
) -> pd.DataFrame:
    """
    Aggregate DataFrame to specified frequency (default 5min) with timezone preservation.

    Args:
        df: Input DataFrame with datetime index.
        freq: Resampling frequency string (default "5min").
        extra_agg: Optional dict of additional columns/aggregation rules.

    Returns:
        Aggregated DataFrame with preserved timezone.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.error("DataFrame index must be datetime type for resampling.")
        raise ValueError("DataFrame index must be datetime type for resampling.")

    agg_dict = {
        "actual_yield_kw": "mean",
        "actual_yield_kw_preclip": "mean",
        "ghi": "mean",
        "air_temp": "mean",
        "t2m": "mean",
        "u10": "mean",
        "v10": "mean",
        "sd": "mean",
        "ssrdc": "sum",
        "ssrd": "sum",
        "tcc": "mean",
        "tp": "sum",
        "string1_kw": "mean",
        "string2_kw": "mean",
    }

    # Only add optional columns if present
    if "surface_solar_radiation_W_m2" in df.columns:
        agg_dict["surface_solar_radiation_W_m2"] = "mean"
    if "direct_solar_radiation_W_m2" in df.columns:
        agg_dict["direct_solar_radiation_W_m2"] = "mean"

    # Add any extra aggregation rules
    if extra_agg:
        agg_dict.update(extra_agg)

    original_tz = df.index.tz
    try:
        df_agg = df.resample(freq).agg(
            {k: v for k, v in agg_dict.items() if k in df.columns}
        )
        df_agg.index.name = "datetime"
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        raise

    # Restore timezone if lost during resampling
    if df_agg.index.tz is None and original_tz is not None:
        df_agg.index = df_agg.index.tz_localize(original_tz)
        logger.info(f"✓ Restored {original_tz} timezone after aggregation")

    return df_agg


def check_post_aggregation_timezone(
    df: pd.DataFrame,
    month: int = 6,
    min_points: int = 288,
    expected_peak_range: tuple = (11, 14),
) -> dict:
    """
    Quick post-aggregation timezone check.
    Logs diagnostics and returns dict: {'peak_hour': int or None, 'ok': bool}

    Args:
        df: Aggregated DataFrame with datetime index.
        month: Month to check (default: June).
        min_points: Minimum number of points to consider check valid.
        expected_peak_range: Tuple of expected peak hours (inclusive).

    Returns:
        dict: {'peak_hour': int or None, 'ok': bool}
    """
    logger = logging.getLogger("lamina.data")
    logger.info("=== POST-AGGREGATION TIMEZONE CHECK ===")
    logger.info(f"Timezone after aggregation: {df.index.tz}")
    summer_check = df[df.index.month == month]
    result = {"peak_hour": None, "ok": False}

    if len(summer_check) > min_points and "actual_yield_kw" in df.columns:
        peak_hour = (
            summer_check.groupby(summer_check.index.hour)["actual_yield_kw"]
            .mean()
            .idxmax()
        )
        result["peak_hour"] = int(peak_hour)
        logger.info(f"Production peaks at hour {peak_hour} UTC after aggregation")
        if expected_peak_range[0] <= peak_hour <= expected_peak_range[1]:
            logger.info("✅ Timezone still correct after aggregation")
            result["ok"] = True
        else:
            logger.warning(
                f"⚠️  WARNING: Peak shifted to hour {peak_hour} after aggregation!"
            )
    else:
        logger.warning(
            "Not enough data or missing 'actual_yield_kw' for timezone check."
        )

    logger.debug(f"DEBUG: df timezone before quick checks: {df.index.tz}")
    logger.debug(f"DEBUG: df sample timestamps: {df.index[:5]}")
    return result


def final_timezone_verification(
    df: pd.DataFrame,
    month: int = 6,
    sample_days: int = 3,
    expected_peak_range: tuple = (11, 14),
) -> dict:
    """
    More robust final timezone verification using sample_days * 288 rows (5min data).
    Logs diagnostics and returns dict: {'peak_hour': int or None, 'verification_peak': int or None, 'ok': bool}

    Args:
        df: Aggregated DataFrame with datetime index.
        month: Month to check (default: June).
        sample_days: Number of days to sample (default: 3).
        expected_peak_range: Tuple of expected peak hours (inclusive).

    Returns:
        dict: {'peak_hour': int or None, 'verification_peak': int or None, 'ok': bool}
    """
    logger = logging.getLogger("lamina.data")
    logger.info("=== FINAL TIMEZONE VERIFICATION ===")
    nrows = 288 * sample_days
    summer_day = df[df.index.month == month].head(nrows)
    out = {"peak_hour": None, "verification_peak": None, "ok": False}

    if len(summer_day) > 0 and "actual_yield_kw" in df.columns:
        hourly_production = summer_day.groupby(summer_day.index.hour)[
            "actual_yield_kw"
        ].mean()
        peak_hour = int(hourly_production.idxmax())
        out["peak_hour"] = peak_hour
        logger.info(f"Final check: Summer production peaks at hour {peak_hour}")
        logger.info(
            f"Expected: {expected_peak_range[0]}-{expected_peak_range[1]} for UK solar data"
        )
        if expected_peak_range[0] <= peak_hour <= expected_peak_range[1]:
            logger.info("✅ Final timezone verification: PASSED")
            out["ok"] = True
        else:
            logger.warning(
                f"⚠️  WARNING: Final timezone verification shows peak at hour {peak_hour}"
            )

        # Redundant verification using same method as post-aggregation
        verification_peak = int(hourly_production.idxmax())
        out["verification_peak"] = verification_peak
        logger.info(
            f"Verification using same method as post-aggregation: hour {verification_peak}"
        )
    else:
        logger.warning("Not enough data for final timezone verification.")

    return out


def despike_and_clip_ghi(
    df: pd.DataFrame,
    clip_max: float = 1400,
    spike_threshold: float = 1200,
    interp_limit: int = 3,
    fill_value: float = 0,
) -> pd.DataFrame:
    """
    Despike and clip GHI to physically plausible range.
    Modifies df in-place and returns df.

    Args:
        df: DataFrame containing 'ghi' column.
        clip_max: Maximum physically plausible GHI value.
        spike_threshold: Threshold above which values are considered spikes.
        interp_limit: Max consecutive NaNs to fill by interpolation.
        fill_value: Value to fill remaining NaNs after interpolation.

    Returns:
        DataFrame with despiked and clipped 'ghi' column.
    """
    logger = logging.getLogger("lamina.data")
    if "ghi" not in df.columns:
        logger.warning("'ghi' column not found. No despiking or clipping applied.")
        return df

    # Clip to max physically plausible
    df["ghi"] = df["ghi"].clip(lower=0, upper=clip_max)

    # Mask and interpolate very large isolated spikes
    spike_mask = df["ghi"] > spike_threshold
    n_spikes = int(spike_mask.sum())
    if n_spikes > 0:
        df.loc[spike_mask, "ghi"] = np.nan
        df["ghi"] = df["ghi"].interpolate(limit=interp_limit).fillna(fill_value)
        logger.info(f"Despiked GHI: {n_spikes} values were replaced and interpolated")
    else:
        logger.info("No GHI spikes detected above threshold.")

    return df


def get_influx_credentials(
    config_module=None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Read Influx credentials from config module (if provided) or environment variables.
    Returns tuple: (influx_url, token, org, bucket)
    Logs a warning if any credential is missing.
    """
    logger = logging.getLogger("lamina.data")
    influx_url = None
    token = None
    org = None
    bucket = None

    if config_module is not None:
        influx_url = getattr(config_module, "INFLUX_URL", None)
        token = getattr(config_module, "INFLUX_TOKEN", None)
        org = getattr(config_module, "INFLUX_ORG", None)
        bucket = getattr(config_module, "INFLUX_BUCKET", None)

    # Fall back to environment variables if any are missing
    influx_url = influx_url or os.getenv("INFLUX_URL")
    token = token or os.getenv("INFLUX_TOKEN")
    org = org or os.getenv("INFLUX_ORG")
    bucket = bucket or os.getenv("INFLUX_BUCKET")

    # Log warnings for missing credentials
    if not influx_url:
        logger.warning("INFLUX_URL is missing (not set in config or environment).")
    if not token:
        logger.warning("INFLUX_TOKEN is missing (not set in config or environment).")
    if not org:
        logger.warning("INFLUX_ORG is missing (not set in config or environment).")
    if not bucket:
        logger.warning("INFLUX_BUCKET is missing (not set in config or environment).")

    return influx_url, token, org, bucket


def run_validation_and_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run validate_solar_data() and log a concise report of violations.
    Returns the validation_flags DataFrame.

    Args:
        df: DataFrame to validate.

    Returns:
        DataFrame of validation flags.
    """
    logger = logging.getLogger("lamina.data")
    validation_flags = validate_solar_data(df)
    logger.info("Validation Results:")
    if validation_flags.empty:
        logger.info("No data to validate.")
        return validation_flags
    for col in validation_flags.columns:
        violations = int(validation_flags[col].sum())
        if violations > 0:
            logger.warning(f"Found {violations} {col}")
        else:
            logger.info(f"No {col} violations")
    return validation_flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", required=True, help="Client identifier")
    args = parser.parse_args()
    client_id = args.client_id

    try:
        config_filename = f"{client_id}/config.yaml"
        BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
        BASE_CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "configs/clients"))

        client_config = load_client_config(client_id, BASE_CONFIG_DIR)
        validate_config(client_config, config_filename)

        # Extract configuration parameters from dict
        CLIENT_ID = client_config["client_id"]
        INSTALLATION_DATE = client_config.get("installation_date", "2022-01-01")
        LATITUDE = client_config["latitude"]
        LONGITUDE = client_config["longitude"]
        ALTITUDE = client_config.get("altitude", 0)
        SURFACE_TILT = client_config["system_parameters"]["surface_tilt"]
        SURFACE_AZIMUTH = client_config["system_parameters"]["surface_azimuth"]
        ASSET_DC_CAPACITY_KW = client_config["system_parameters"]["dc_capacity_kw"]
        ANNUAL_DEGRADATION = client_config["system_parameters"].get(
            "annual_degradation", 0.005
        )
        INVERTER_AC_CAPACITY_KW = client_config["system_parameters"][
            "inverter_ac_capacity_kw"
        ]
        ARRAY_HEIGHT = client_config["system_parameters"]["array_height"]

        MASTER_DATASET_PATH = (
            BASE_DATA_DIR
            / client_id
            / "processed"
            / f"{client_id}_master_data_set_combined.csv"
        )
        PANEL_CONFIG_PATH = BASE_CONFIG_DIR / client_id / "panel_config.yaml"

        # Load and clean main dataset
        df = load_and_clean_dataset(MASTER_DATASET_PATH, INVERTER_AC_CAPACITY_KW)

        # Load panel config from YAML
        panel_config = load_panel_config(PANEL_CONFIG_PATH)
        config_df = assign_panel_config(df, panel_config)

        # (Optional) Print or log a sample of the config_df
        logger.info(config_df.head())

    except Exception as e:
        logger = logging.getLogger("lamina.data")
        logger.error(f"Error loading client configuration: {e}")
        logger.error("Expected config file format (YAML):")
        logger.error("  client_id:")
        logger.error("  latitude:")
        logger.error("  longitude:")
        logger.error("  system_parameters:")
        logger.error("    surface_tilt:")
        logger.error("    surface_azimuth:")
        logger.error("    dc_capacity_kw:")
        logger.error("    inverter_ac_capacity_kw:")
        logger.error("    array_height:")
        logger.error("  master_dataset_path:")
