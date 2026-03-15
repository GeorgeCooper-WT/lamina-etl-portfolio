variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "supabase_url" {
  description = "Supabase project URL"
  type        = string
  sensitive   = true
}

variable "supabase_anon_key" {
  description = "Supabase anonymous key"
  type        = string
  sensitive   = true
}

variable "supabase_service_key" {
  description = "Supabase service role key"
  type        = string
  sensitive   = true
}

variable "influxdb_url" {
  description = "InfluxDB URL"
  type        = string
  sensitive   = true
}

variable "influxdb_token" {
  description = "InfluxDB authentication token"
  type        = string
  sensitive   = true
}

variable "influxdb_org" {
  description = "InfluxDB organization"
  type        = string
  sensitive   = true
}

# S3 bucket names
variable "raw_data_bucket_name" {
  description = "Raw data S3 bucket name"
  type        = string
}

variable "processed_data_bucket_name" {
  description = "Processed data S3 bucket name"
  type        = string
}

variable "reports_bucket_name" {
  description = "Reports S3 bucket name"
  type        = string
}

# Lambda endpoint
variable "lambda_folder_creation_url" {
  description = "API Gateway endpoint for client folder creation Lambda"
  type        = string
}

# API Key for Lambda endpoint
variable "lambda_api_key" {
  description = "API Gateway API Key for client folder creation Lambda"
  type        = string
  sensitive   = true
}
