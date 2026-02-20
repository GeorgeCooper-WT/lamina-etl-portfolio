# Lamina ETL Portfolio

This repository contains four standalone Python scripts demonstrating production-ready ETL pipelines for solar energy data.

## Scripts

- **config_and_data.py**
  - Loads and validates solar site configuration and SCADA data.
  - Enforces physical limits and checks for timezone consistency.

- **era5_weather_downloader.py**
  - Downloads ERA5 weather data using `cdsapi` and `xarray`.
  - Handles chunked downloads, disk space checks, and API retries.

- **scada_era5_combine.py**
  - Aligns and merges SCADA and ERA5 data.
  - Disaggregates hourly weather data to 5-minute intervals using solar geometry.

- **influxdb_write.py**
  - Prepares and writes analytics data to InfluxDB in batches.
  - Includes diagnostics, validation, and retry logic.

## Engineering Practices Demonstrated

- Robust error handling and logging
- Defensive programming for edge cases (e.g., timezone drift, disk space)
- Modular, reusable code with clear docstrings
- Batch processing and API/database interaction

## Notes

- No proprietary data or ML code is included.
- Scripts are presented for code review and portfolio demonstration only.
