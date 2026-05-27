# Lamina ETL Pipeline

Production-ready, containerised ETL pipeline for solar energy performance analytics, covering weather API ingestion to time-series database storage.

Downloads ERA5 reanalysis weather data, merges with SCADA production data, performs timezone alignment and temporal disaggregation to 5-minute resolution, and writes analytics-ready datasets to InfluxDB for ML and performance monitoring.

Docker containerisation ensures reproducible deployments across environments.


## Scripts

**run_etl.py**
- Orchestrates the full ETL pipeline with logging and error handling.
- Supports flexible data directory configuration via `--data_dir` argument.

**config_and_data.py**
- Loads and validates solar site configuration and SCADA data.
- Enforces physical limits and checks for timezone consistency.

**era5_weather_downloader.py**
- Downloads ERA5 weather data using `cdsapi` and `xarray`.
- Handles chunked downloads, disk space checks, and API retries.

**scada_era5_combine.py**
- Aligns and merges SCADA and ERA5 data.
- Disaggregates hourly weather data to 5-minute intervals using solar geometry.

**influxdb_write.py**
- Prepares and writes analytics data to InfluxDB in batches.
- Includes diagnostics, validation, and retry logic.

---

**Note:** No proprietary data or ML modelling code included. Presented for code review and portfolio demonstration.
