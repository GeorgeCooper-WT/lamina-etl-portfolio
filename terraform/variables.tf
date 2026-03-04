variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "lamina"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# Supabase Variables
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

# InfluxDB Variables
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
