# Lamina Intelligence: Forensic Solar Analytics


This repository showcases the MLOps framework used for building a secure and scalable data solution within the Lamina Energy analytics platform.

**Validation:** <br>
Validated over 2,742 days of 5-minute SCADA telemetry, the Lamina Hybrid Engine identified rising asset degradation trends up to 3 years prior to failure, isolated recoverable revenue losses and OpEx optimisation opportunities.

**Model Validation Report Abstract:** <br>
See [lamina-ml-validation-abstract.pdf](reports/lamina-ml-validation-abstract.pdf) for the executive summary and contents of the Lamina ML model validation report. The full report is available on request.

<br>

<p align="center">
  <img src="assets/lamina-streamlit-app-demo.png" width="90%" alt="Lamina Forensic Portal Demo">
  <br>
  <em>UI Demo: Integrated Forensic Portal for Portfolio Overview & Site Performance Analysis.</em>
</p>

<br>

## System Architecture

### 1. Data Engineering & ETL ```/etl```
Production-ready, containerised Python ETL pipeline for solar energy data analytics. Highlights include:
- Docker containerisation for reproducible deployments
- Orchestrated pipeline with flexible configuration
- Weather data ingestion and temporal alignment
- Data validation, transformation, and merging
- Batch writing to InfluxDB with retry logic

See [etl/README.md](https://github.com/GeorgeCooper-WT/lamina-etl-portfolio/blob/main/etl/README.md) for details.

<br>

### 2. Infrastructure as Code ```/terraform```
Terraform infrastructure-as-code for a scalable AWS data lake architecture. Features include:
- Multi-bucket S3 structure for raw, processed, and report data
- Lifecycle and cost optimisation policies
- Secure IAM, encryption, and secrets management

```
terraform/
├── main.tf           # Root orchestration
├── variables.tf      # Global configuration
├── outputs.tf        # Global outputs
├── lambda/           # Source code for AWS Lambda functions
└── modules/          # Encapsulated, reusable infrastructure
    ├── data-lake/    # S3 Tiering & Lifecycle policies
    ├── iam/          # Least-privilege role definitions
    ├── secrets/      # AWS Secrets Manager integration
    └── lambda/       # Infrastructure definitions for compute
```

See [terraform/README.md](https://github.com/GeorgeCooper-WT/lamina-etl-portfolio/blob/main/terraform/README.md) for details.

<br>

### 3. ML Model & Analytics
*Note: The core ML modeling codebase is proprietary and has been omitted from this public repository. This section outlines the technical framework and validation methodology.*

<br>

**Tech Stack:** 
<br>Python (XGBoost, Scikit-learn, NumPy, Pandas, SHAP)

**Physics Modelling:** 
<br>Baseline performance modelled using PVLib to integrate theoretical physical constraints e.g. clear-sky and diffuse irradiance curves.

**Decomposition Logic:** 
<br>Utilises a hybrid approach to separate reversible losses (soiling/shading) correlations from irreversible degradation trend-lines.


**Technical Highlights:**
- Benchmarked against 8 years of high-fidelity meteorological data with a baseline fidelity of 98.7% R<sup>2</sup>
- Identified string-level signatures of degradation up to 3 years prior to failure utilising rolling volatility analysis of the Performance Index (PI)
- Leverages the RMSE delta (0.11 vs 0.25) between filtered and unfiltered datasets as a mathematical proxy for recoverable yield loss
- Validated via 5-fold rolling-origin cross-validation to maintain temporal integrity and prevent data leakage.

<br>

*Detailed validation report available upon request. See [lamina-ml-validation-abstract.pdf](reports/lamina-ml-validation-abstract.pdf) for the executive summary.*

<br>

---

**Note:** This portfolio is intended for demonstration and code review purposes. No proprietary data or credentials are included.
