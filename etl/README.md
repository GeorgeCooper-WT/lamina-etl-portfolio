# Lamina ETL Portfolio

Production-ready ETL pipeline for solar energy performance analytics, demonstrating end-to-end data engineering from weather API integration to time-series database storage.

## Overview

This pipeline downloads ERA5 weather data, combines it with SCADA (site) production data, performs timezone alignment and temporal disaggregation, and prepares analytics-ready datasets for machine learning and performance monitoring. Docker is used for containerisation.

## Scripts

- **run_etl.py**
  - Orchestrates the full ETL pipeline with logging and error handling.
  - Supports flexible data directory configuration via `--data_dir` argument.

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

## Docker Support

The pipeline is fully containerized for portability and reproducibility:

```bash
docker build -t lamina-etl .
docker run --rm -it \
  -v ~/.cdsapirc:/root/.cdsapirc \
  -v /path/to/data:/data \
  lamina-etl --client_id <CLIENT_ID> --data_dir /data
```

## Engineering Practices Demonstrated

- **Containerisation** with Docker for reproducible deployments
- **Flexible configuration** via command-line arguments and YAML files
- Robust error handling and logging
- Defensive programming for edge cases (e.g., timezone drift, disk space)
- Modular, reusable code with clear docstrings
- Batch processing and API/database interaction

## Notes

- No proprietary data or ML code is included.
- Scripts are presented for code review and portfolio demonstration only.