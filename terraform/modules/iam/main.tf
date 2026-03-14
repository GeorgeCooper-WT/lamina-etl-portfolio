# IAM Role for Lambda Functions (ETL, Data Processing)
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-lambda-role"
  }
}

# Policy for Lambda to access S3 buckets
resource "aws_iam_policy" "lambda_s3_access" {
  name        = "${var.project_name}-lambda-s3-access-${var.environment}"
  description = "Allow Lambda to read/write data lake buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${var.raw_data_bucket_arn}/*",
          "${var.processed_data_bucket_arn}/*",
          "${var.reports_bucket_arn}/*",
          var.raw_data_bucket_arn,
          var.processed_data_bucket_arn,
          var.reports_bucket_arn
        ]
      }
    ]
  })
}

# Policy for Lambda to access Secrets Manager
resource "aws_iam_policy" "lambda_secrets_access" {
  name        = "${var.project_name}-lambda-secrets-access-${var.environment}"
  description = "Allow Lambda to read secrets from Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          for arn in values(var.secrets_arns) : arn
        ]
      }
    ]
  })
}

# Attach policies to Lambda role
resource "aws_iam_role_policy_attachment" "lambda_s3" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = aws_iam_policy.lambda_s3_access.arn
}

resource "aws_iam_role_policy_attachment" "lambda_secrets" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = aws_iam_policy.lambda_secrets_access.arn
}

# Attach AWS managed policy for basic Lambda execution
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# IAM Role for Glue (ETL Jobs)
resource "aws_iam_role" "glue_execution" {
  name = "${var.project_name}-glue-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "glue.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-glue-role"
  }
}

# Attach Glue service role and S3 access
resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy_attachment" "glue_s3" {
  role       = aws_iam_role.glue_execution.name
  policy_arn = aws_iam_policy.lambda_s3_access.arn
}

# IAM Role for EC2 (if self-hosting InfluxDB)
resource "aws_iam_role" "ec2_data_access" {
  name = "${var.project_name}-ec2-data-access-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-ec2-role"
  }
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_data_access.name
}

# EC2 can read processed data and secrets
resource "aws_iam_role_policy_attachment" "ec2_secrets" {
  role       = aws_iam_role.ec2_data_access.name
  policy_arn = aws_iam_policy.lambda_secrets_access.arn
}

# Policy for EC2 to read processed data (read-only)
resource "aws_iam_policy" "ec2_s3_read" {
  name        = "${var.project_name}-ec2-s3-read-${var.environment}"
  description = "Allow EC2 to read processed data"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${var.processed_data_bucket_arn}/*",
          var.processed_data_bucket_arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_read" {
  role       = aws_iam_role.ec2_data_access.name
  policy_arn = aws_iam_policy.ec2_s3_read.arn
}
