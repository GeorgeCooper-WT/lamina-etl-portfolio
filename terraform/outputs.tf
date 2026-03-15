output "raw_data_bucket_name" {
  description = "Name of the raw data S3 bucket"
  value       = module.data_lake.raw_data_bucket_name
}

output "processed_data_bucket_name" {
  description = "Name of the processed data S3 bucket"
  value       = module.data_lake.processed_data_bucket_name
}

output "reports_bucket_name" {
  description = "Name of the reports S3 bucket"
  value       = module.data_lake.reports_bucket_name
}

output "lambda_execution_role_arn" {
  description = "ARN of Lambda execution role"
  value       = module.iam.lambda_execution_role_arn
}

output "secrets_manager_arns" {
  description = "ARNs of secrets in AWS Secrets Manager"
  value       = module.secrets.secrets_arns
  sensitive   = true
}

output "client_setup_api_url" {
  description = "API Gateway URL for client setup Lambda"
  value       = module.client_setup_lambda.api_gateway_url
}

output "client_setup_api_key_secret_name" {
  description = "AWS Secrets Manager secret name containing the API key for client setup Lambda"
  value       = module.secrets.lambda_api_key_secret_name
}
