# Lamina Portfolio

This repository showcases an approach to building secure, scalable, and production-ready data solutions for the Lamina Energy solar analytics platform.

## Contents

### 1. etl
Standalone Python scripts demonstrating robust ETL (Extract, Transform, Load) pipelines for solar energy data analytics. Highlights include:
- Data validation and configuration
- Weather data ingestion and alignment
- Data merging and transformation
- Batch writing to InfluxDB

See [etl/README.md](https://github.com/GeorgeCooper-WT/lamina-etl-portfolio/blob/main/etl/README.md) for details.

### 2. terraform
Terraform infrastructure-as-code for a scalable AWS data lake architecture. Features include:
- Multi-bucket S3 structure for raw, processed, and report data
- Lifecycle and cost optimisation policies
- Secure IAM, encryption, and secrets management

See [terraform/README.md](https://github.com/GeorgeCooper-WT/lamina-etl-portfolio/blob/main/terraform/README.md) for details.

---

**Note:** This portfolio is intended for demonstration and code review purposes. No proprietary data or credentials are included.
