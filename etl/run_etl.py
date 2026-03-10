import argparse
import subprocess
import sys
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_etl")

def run_script(script, args):
    cmd = [sys.executable, script] + args
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Script {script} failed with exit code {result.returncode}")
        logger.error(result.stderr)
        sys.exit(result.returncode)
    logger.info(result.stdout)

def main():
    parser = argparse.ArgumentParser(description="Run full ETL pipeline for ML prep.")
    parser.add_argument("--client_id", required=True, help="Client ID")
    parser.add_argument("--data_dir", default="data", help="Base directory for all data files")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--ignore_disk_space_warning", action="store_true", help="Ignore low disk space warning in weather downloader")
    args = parser.parse_args()

    # 1. Download and merge ERA5 weather data
    weather_args = ["--client_id", args.client_id, "--data_dir", args.data_dir]
    if args.ignore_disk_space_warning:
        weather_args.append("--ignore_disk_space_warning")
    run_script("era5_weather_downloader.py", weather_args)

    # 2. Combine SCADA and ERA5 data
    combine_args = [
        "--client_id", args.client_id,
        "--data_dir", args.data_dir,
    ]
    if args.output_dir:
        combine_args += ["--output_dir", args.output_dir]
    run_script("scada_era5_combine.py", combine_args)

    # 3. load config and prepare data for ML
    run_script("config_and_data.py", ["--client_id", args.client_id, "--data_dir", args.data_dir])

    logger.info("ETL pipeline completed successfully.")

if __name__ == "__main__":
    main()