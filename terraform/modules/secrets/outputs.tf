output "kms_key_id" {
  description = "KMS key ID for secrets encryption"
  value       = aws_kms_key.secrets.id
}

output "kms_key_arn" {
  description = "KMS key ARN for secrets encryption"
  value       = aws_kms_key.secrets.arn
}

output "secrets_arns" {
  description = "Map of all secret ARNs"
  value = {
    supabase_url               = aws_secretsmanager_secret.supabase_url.arn
    supabase_anon_key          = aws_secretsmanager_secret.supabase_anon_key.arn
    supabase_service_key       = aws_secretsmanager_secret.supabase_service_key.arn
    influxdb_url               = aws_secretsmanager_secret.influxdb_url.arn
    influxdb_token             = aws_secretsmanager_secret.influxdb_token.arn
    influxdb_org               = aws_secretsmanager_secret.influxdb_org.arn
    raw_data_bucket_name       = aws_secretsmanager_secret.raw_data_bucket_name.arn
    processed_data_bucket_name = aws_secretsmanager_secret.processed_data_bucket_name.arn
    reports_bucket_name        = aws_secretsmanager_secret.reports_bucket_name.arn
    lambda_folder_creation_url = aws_secretsmanager_secret.lambda_folder_creation_url.arn
    lambda_api_key             = aws_secretsmanager_secret.lambda_api_key.arn
  }
}

output "supabase_url_secret_name" {
  description = "Name of Supabase URL secret"
  value       = aws_secretsmanager_secret.supabase_url.name
}

output "influxdb_token_secret_name" {
  description = "Name of InfluxDB token secret"
  value       = aws_secretsmanager_secret.influxdb_token.name
}

output "lambda_api_key_secret_name" {
  description = "Name of Lambda API Key secret"
  value       = aws_secretsmanager_secret.lambda_api_key.name
}
