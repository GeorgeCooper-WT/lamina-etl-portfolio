# KMS Key for encrypting secrets
resource "aws_kms_key" "secrets" {
  description             = "${var.project_name}-${var.environment}-secrets-key"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "${var.project_name}-secrets-key"
  }
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.project_name}-${var.environment}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# Supabase URL
resource "aws_secretsmanager_secret" "supabase_url" {
  name        = "${var.project_name}/${var.environment}/supabase/url"
  description = "Supabase project URL"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "Supabase"
  }
}

resource "aws_secretsmanager_secret_version" "supabase_url" {
  secret_id     = aws_secretsmanager_secret.supabase_url.id
  secret_string = var.supabase_url
}

# Supabase Anonymous Key
resource "aws_secretsmanager_secret" "supabase_anon_key" {
  name        = "${var.project_name}/${var.environment}/supabase/anon-key"
  description = "Supabase anonymous key"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "Supabase"
  }
}

resource "aws_secretsmanager_secret_version" "supabase_anon_key" {
  secret_id     = aws_secretsmanager_secret.supabase_anon_key.id
  secret_string = var.supabase_anon_key
}

# Supabase Service Key
resource "aws_secretsmanager_secret" "supabase_service_key" {
  name        = "${var.project_name}/${var.environment}/supabase/service-key"
  description = "Supabase service role key"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "Supabase"
  }
}

resource "aws_secretsmanager_secret_version" "supabase_service_key" {
  secret_id     = aws_secretsmanager_secret.supabase_service_key.id
  secret_string = var.supabase_service_key
}

# InfluxDB URL
resource "aws_secretsmanager_secret" "influxdb_url" {
  name        = "${var.project_name}/${var.environment}/influxdb/url"
  description = "InfluxDB server URL"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "InfluxDB"
  }
}

resource "aws_secretsmanager_secret_version" "influxdb_url" {
  secret_id     = aws_secretsmanager_secret.influxdb_url.id
  secret_string = var.influxdb_url
}

# InfluxDB Token
resource "aws_secretsmanager_secret" "influxdb_token" {
  name        = "${var.project_name}/${var.environment}/influxdb/token"
  description = "InfluxDB authentication token"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "InfluxDB"
  }
}

resource "aws_secretsmanager_secret_version" "influxdb_token" {
  secret_id     = aws_secretsmanager_secret.influxdb_token.id
  secret_string = var.influxdb_token
}

# InfluxDB Organization
resource "aws_secretsmanager_secret" "influxdb_org" {
  name        = "${var.project_name}/${var.environment}/influxdb/org"
  description = "InfluxDB organization name"
  kms_key_id  = aws_kms_key.secrets.id

  tags = {
    Service = "InfluxDB"
  }
}

resource "aws_secretsmanager_secret_version" "influxdb_org" {
  secret_id     = aws_secretsmanager_secret.influxdb_org.id
  secret_string = var.influxdb_org
}

# Secret rotation configuration (optional - for future)
# You can add Lambda-based secret rotation here

# S3 bucket names as secrets
resource "aws_secretsmanager_secret" "raw_data_bucket_name" {
  name        = "${var.project_name}/${var.environment}/raw_data_bucket_name"
  description = "Raw data S3 bucket name"
  kms_key_id  = aws_kms_key.secrets.id
  tags = {
    Service = "DataLake"
  }
}

resource "aws_secretsmanager_secret_version" "raw_data_bucket_name" {
  secret_id     = aws_secretsmanager_secret.raw_data_bucket_name.id
  secret_string = var.raw_data_bucket_name
}

resource "aws_secretsmanager_secret" "processed_data_bucket_name" {
  name        = "${var.project_name}/${var.environment}/processed_data_bucket_name"
  description = "Processed data S3 bucket name"
  kms_key_id  = aws_kms_key.secrets.id
  tags = {
    Service = "DataLake"
  }
}

resource "aws_secretsmanager_secret_version" "processed_data_bucket_name" {
  secret_id     = aws_secretsmanager_secret.processed_data_bucket_name.id
  secret_string = var.processed_data_bucket_name
}

resource "aws_secretsmanager_secret" "reports_bucket_name" {
  name        = "${var.project_name}/${var.environment}/reports_bucket_name"
  description = "Reports S3 bucket name"
  kms_key_id  = aws_kms_key.secrets.id
  tags = {
    Service = "DataLake"
  }
}

resource "aws_secretsmanager_secret_version" "reports_bucket_name" {
  secret_id     = aws_secretsmanager_secret.reports_bucket_name.id
  secret_string = var.reports_bucket_name
}

# Lambda endpoint as secret
resource "aws_secretsmanager_secret" "lambda_folder_creation_url" {
  name        = "${var.project_name}/${var.environment}/lambda_folder_creation_url"
  description = "API Gateway endpoint for client folder creation Lambda"
  kms_key_id  = aws_kms_key.secrets.id
  tags = {
    Service = "Lambda"
  }
}

resource "aws_secretsmanager_secret_version" "lambda_folder_creation_url" {
  secret_id     = aws_secretsmanager_secret.lambda_folder_creation_url.id
  secret_string = var.lambda_folder_creation_url
}

# Lambda API Key as secret
resource "aws_secretsmanager_secret" "lambda_api_key" {
  name        = "${var.project_name}/${var.environment}/lambda_api_key"
  description = "API Gateway API Key for client folder creation Lambda"
  kms_key_id  = aws_kms_key.secrets.id
  tags = {
    Service = "Lambda"
  }
}

resource "aws_secretsmanager_secret_version" "lambda_api_key" {
  secret_id     = aws_secretsmanager_secret.lambda_api_key.id
  secret_string = var.lambda_api_key
}
