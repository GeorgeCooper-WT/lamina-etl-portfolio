# Lamina Data Lake - Terraform Infrastructure

Terraform infrastructure-as-code for a secure, scalable AWS data lake architecture for the Lamina Intelligence Platform. 

Provided as a portfolio piece demonstrating cloud infrastructure design and IaC practices.

## Architecture

### Multi-Bucket Structure
Separate buckets per data lifecycle stage enforce clean boundaries and allow independent access policies and lifecycle rules per data type.
- **lamina-raw-data-{env}**: Raw SCADA ingestion
- **lamina-processed-data-{env}**: Transformed and cleaned data
- **lamina-reports-{env}**: Client-facing reports

### Features
- Separate buckets for different data lifecycle stages  
- Cost-optimised lifecycle policies  
- S3 versioning enabled (data protection)  
- Server-side encryption (AES-256)  
- Public access blocked (security)  
- IAM roles for Lambda, Glue, EC2 
- Lambda scripts (client-setup) 
- AWS Secrets Manager for credentials (InfluxDB, Supabase)  
- KMS encryption for secrets  


## Project Structure

```
terraform/
├── main.tf           # Root orchestration
├── variables.tf      # Global configuration
├── outputs.tf        # Global outputs
├── lambda/           # AWS Lambda function source
└── modules/
    ├── data-lake/    # S3 buckets and lifecycle policies
    ├── secrets/      # AWS Secrets Manager and KMS
    ├── iam/          # Least-privilege role definitions
    └── lambda/       # Lambda infrastructure
```

### Data Lifecycle Policies

**Raw Data:** tiered quickly, rarely re-accessed after processing.
- 0-30 days: Standard
- 30-90 days: Intelligent-Tiering
- 90-180 days: Glacier Instant Retrieval
- 180+ days: Deep Archive

**Processed Data:** slower archival transition, retained longer for reprocessing and model retraining.
- 0-60 days: Standard
- 60-180 days: Intelligent-Tiering
- 180-365 days: Glacier IR
- 365+ days: Deep Archive

**Reports:** Standard storage, frequent client access requires consistent low-latency retrieval.
