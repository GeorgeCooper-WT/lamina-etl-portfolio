# Lamina Intelligence: Solar Analytics Portfolio

<br>

> **Repository Scope & IP Notice**
>
> **Please note:** This public repository is a representative snapshot intended purely for code review and architectural demonstration. It showcases coding standards, IaC structure (Terraform), containerisation (Docker), and general data engineering capabilities, not the full working codebase.
> 
> To protect commercial IP, the core proprietary physics-ML engine, the dashboard source code, and the scaled OpenCV/OCR + VLM parsing architecture (which now handles dense 100MWp+ SLD diagrams) have been withheld.
>
> The images below demonstrate the actual outcomes of the current, proprietary production platform, whilst the code highlights general coding quality.

<br>

## 1. Document Intelligence: VLM & OCR Parser Overview
*Transforming unstructured engineering diagrams into queryable databases.*

<br>

<p align="center">
  <img src="assets/test_sld_100mwp.png" width="90%" alt="Example Input SLD">
  <br>
  <em><strong>The Input:</strong> Unstructured, high-density 100MWp AutoCAD Single Line Diagram (SLD).</em>
</p>

<br>
<br>

<p align="center">
  <img src="assets/test_sql_app_100mwp.png" width="90%" alt="Lamina SQL Visual">
  <br>
  <em><strong>The Output:</strong> Interactive relational digital twin. Site → Inverter → DC Combiner → String hierarchy with full spec sheet drill-down.</em>
</p>

<br>

## 2. Physics-ML Engine: Performance Analysis Overview
*Isolating reversible losses from irreversible degradation.*

<br>

**Validation:** <br>
Validated over **2,742 days** of 5-minute SCADA telemetry, the Lamina Hybrid Engine identified rising asset degradation trends up to **3 years** prior to failure, isolated recoverable revenue losses and OpEx optimisation opportunities.

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
- Validated via 5-fold rolling-origin cross-validation to maintain temporal integrity and prevent data leakage

<br>

*Detailed validation report available upon request. See [lamina-ml-validation-abstract.pdf](reports/lamina-ml-validation-abstract.pdf) for the executive summary.*

<br>

### 4. Document Intelligence & Site Configuration ```/vlm_parser```

*Note: Prompts and proprietary code are omitted from this public repository.*

<br>

VLM/LLM pipeline for automated solar site configuration. Ingests raw Single Line Diagrams and Bills of Materials, extracts structured component data via modular per-component vision extractors, and populates a queryable SQLite database enriched with manufacturer spec sheet data.

<br>

**The MVP (Code included in /vlm_parser)**
The codebase in this folder represents snippets from the initial MVP. It highlights the initial functional method for residential and commercial-scale SLDs (~1MWp).

- Evaluation: Field-level accuracy measured against manually verified ground truth JSON using a recursive leaf-node diff framework.
- Constraints: Error rate ~1% but is constrained to small-scale SLDs and overfit to initial annotation styles.

<br>

**Product Evolution (Proprietary IP)**
To solve for VLM context window size and handle the massive information density of utility-scale CAD drawings (100MWp+), the architecture was evolved far beyond this MVP:

- **OCR Text Extraction:** Moved away from VLM-exclusive inference by introducing an OCR pre-processing layer to ground the LLM with high-accuracy text.
- **Image Segmentation:** Implemented automated OpenCV image segmentation to break down complex SLDs into classified chunks, enabling tighter context focus and drastically reduced token usage.
- **Functional Application:** A functional frontend application to upload files, and interact with the SQLDB has been developed.

<br>

See [vlm_parser/README.md](vlm_parser/README.md) for MVP methodology, code snippets, before/after visuals, and folder structure.

<br>

---

**Note:** This portfolio is intended for demonstration and code review purposes. No proprietary data or credentials are included.
