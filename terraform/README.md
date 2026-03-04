# Lamina Data Lake - Terraform Infrastructure

This Terraform project creates a scalable, production-ready data lake architecture on AWS for the Lamina platform.

This infrastructure is provided as a portfolio piece, demonstrating a secure and scalable approach to cloud architecture and Infrastructure as Code.

## Architecture

### Multi-Bucket Structure
- **lamina-raw-data-{env}**: Raw SCADA data ingestion
- **lamina-processed-data-{env}**: Transformed and cleaned data
- **lamina-reports-{env}**: Client-facing reports

### Features
Separate buckets for different data lifecycle stages  
Cost-optimised lifecycle policies  
S3 versioning enabled (data protection)  
Server-side encryption (AES-256)  
Public access blocked (security)  
IAM roles for Lambda, Glue, EC2 
Lambda scripts (client-setup) 
AWS Secrets Manager for credentials (InfluxDB, Supabase)  
KMS encryption for secrets  

## Prerequisites

1. **AWS CLI** installed and configured
   ```powershell
   aws configure
   ```

2. **Terraform** installed (v1.0+)
   ```powershell
   terraform version
   ```

3. **AWS Credentials** with AdministratorAccess (or specific permissions)

## Project Structure

```
terraform/
├── main.tf                 # Root configuration
├── variables.tf            # Input variable definitions
├── outputs.tf              # Output definitions
├── terraform.tfvars        # Your secret values (DO NOT COMMIT)
├── terraform.tfvars.example # Template for variables
├── .gitignore              # Prevents committing secrets
│
├── modules/
│   ├── data-lake/          # S3 buckets with lifecycle policies
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── secrets/            # AWS Secrets Manager & KMS
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   └── iam/                # IAM roles for Lambda, Glue, EC2
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
```

## Security Best Practices

### DO
- Keep `terraform.tfvars` out of version control (already in .gitignore)
- Use AWS Secrets Manager to store credentials
- Rotate access keys regularly
- Enable MFA on your AWS account
- Use least privilege IAM policies

### DON'T
- Hardcode secrets in .tf files
- Commit `terraform.tfvars` to Git
- Share your `terraform.tfstate` file (contains secrets)
- Use root AWS account for Terraform

## Cost Optimization

### Lifecycle Policies Configured

**Raw Data:**
- 0-30 days: Standard ($0.023/GB/month)
- 30-90 days: Intelligent-Tiering ($0.023-$0.0025/GB)
- 90-180 days: Glacier Instant Retrieval ($0.004/GB)
- 180+ days: Deep Archive ($0.00099/GB)

**Processed Data:**
- 0-60 days: Standard
- 60-180 days: Intelligent-Tiering
- 180-365 days: Glacier IR
- 365+ days: Deep Archive

**Reports:** Always Standard (frequently accessed)

### Folder Structure (Recommended)
```
lamina-raw-data-dev/
├── {client-uuid}/
│   └── year=2026/
│       └── month=03/
│           └── day=02/
│               └── scada-data.parquet
```
