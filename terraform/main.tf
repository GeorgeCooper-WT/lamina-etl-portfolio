terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}


provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Lamina"
      ManagedBy   = "Terraform"
      Environment = var.environment
    }
  }
}

# Data Lake Module - Multi-bucket architecture
module "data_lake" {
  source = "./modules/data-lake"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region
}

# Secrets Management Module
module "secrets" {
  source = "./modules/secrets"

  project_name         = var.project_name
  environment          = var.environment
  supabase_url         = var.supabase_url
  supabase_anon_key    = var.supabase_anon_key
  supabase_service_key = var.supabase_service_key
  influxdb_url         = var.influxdb_url
  influxdb_token       = var.influxdb_token
  influxdb_org         = var.influxdb_org

  # New secrets for buckets and Lambda endpoint
  raw_data_bucket_name       = module.data_lake.raw_data_bucket_name
  processed_data_bucket_name = module.data_lake.processed_data_bucket_name
  reports_bucket_name        = module.data_lake.reports_bucket_name
  lambda_folder_creation_url = module.client_setup_lambda.api_gateway_url # Update this output name if needed
}

# IAM Roles Module
module "iam" {
  source = "./modules/iam"

  project_name              = var.project_name
  environment               = var.environment
  raw_data_bucket_arn       = module.data_lake.raw_data_bucket_arn
  processed_data_bucket_arn = module.data_lake.processed_data_bucket_arn
  reports_bucket_arn        = module.data_lake.reports_bucket_arn
  secrets_arns              = module.secrets.secrets_arns
}

# Lambda Functions Module
module "client_setup_lambda" {
  source                = "./modules/lambda"
  function_name         = "lamina-client-setup"
  handler               = "main.lambda_handler"
  runtime               = "python3.11"
  role_arn              = module.iam.lambda_execution_role_arn
  source_path           = "${path.root}/lambda/client-setup"
  environment_variables = {}
  aws_region            = var.aws_region
}
