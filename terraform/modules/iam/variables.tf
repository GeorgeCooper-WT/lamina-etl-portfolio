variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "raw_data_bucket_arn" {
  description = "ARN of raw data bucket"
  type        = string
}

variable "processed_data_bucket_arn" {
  description = "ARN of processed data bucket"
  type        = string
}

variable "reports_bucket_arn" {
  description = "ARN of reports bucket"
  type        = string
}

variable "secrets_arns" {
  description = "Map of secret ARNs"
  type        = map(string)
}
