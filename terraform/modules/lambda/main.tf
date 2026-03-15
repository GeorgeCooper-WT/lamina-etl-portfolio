data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = var.source_path
  output_path = "${path.module}/lambda.zip"
}

#tfsec:ignore:aws-lambda-enable-tracing Reason: X-Ray tracing not required for portfolio project to minimize costs
resource "aws_lambda_function" "client_setup" {
  function_name = var.function_name
  handler       = var.handler
  runtime       = var.runtime
  role          = var.role_arn

  filename = data.archive_file.lambda_zip.output_path

  environment {
    variables = var.environment_variables
  }

  timeout     = 30
  memory_size = 128
}

# API Gateway REST API
resource "aws_api_gateway_rest_api" "client_setup" {
  name        = "${var.function_name}-api"
  description = "API Gateway for Lambda client setup function"
}

# API Gateway Resource (root resource)
resource "aws_api_gateway_resource" "client_setup" {
  rest_api_id = aws_api_gateway_rest_api.client_setup.id
  parent_id   = aws_api_gateway_rest_api.client_setup.root_resource_id
  path_part   = "client-setup"
}

# API Gateway Method (POST)
#tfsec:ignore:aws-api-gateway-no-public-access Reason: API key authentication is enforced via api_key_required=true; authorization="NONE" is the correct pattern for API key auth
resource "aws_api_gateway_method" "client_setup" {
  rest_api_id      = aws_api_gateway_rest_api.client_setup.id
  resource_id      = aws_api_gateway_resource.client_setup.id
  http_method      = "POST"
  authorization    = "NONE"
  api_key_required = true
}

# API Gateway Integration (Lambda)
resource "aws_api_gateway_integration" "client_setup" {
  rest_api_id             = aws_api_gateway_rest_api.client_setup.id
  resource_id             = aws_api_gateway_resource.client_setup.id
  http_method             = aws_api_gateway_method.client_setup.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.client_setup.invoke_arn
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.client_setup.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.client_setup.execution_arn}/*/*"
}

# API Gateway Deployment
resource "aws_api_gateway_deployment" "client_setup" {
  depends_on  = [aws_api_gateway_integration.client_setup]
  rest_api_id = aws_api_gateway_rest_api.client_setup.id
}

# CloudWatch Log Group for API Gateway
#tfsec:ignore:aws-cloudwatch-log-group-customer-key Reason: Default CloudWatch encryption is sufficient for portfolio project; customer-managed keys add unnecessary complexity and cost
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.function_name}-api"
  retention_in_days = 7

  tags = {
    Name = "${var.function_name}-api-logs"
  }
}

# IAM Role for API Gateway to write to CloudWatch
resource "aws_iam_role" "api_gateway_cloudwatch" {
  name = "${var.function_name}-api-gateway-cloudwatch"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for API Gateway to write logs
#tfsec:ignore:aws-iam-no-policy-wildcards Reason: Wildcard suffix (:*) is required to allow API Gateway to create and write to log streams within the specific log group
resource "aws_iam_role_policy" "api_gateway_cloudwatch" {
  name = "${var.function_name}-api-gateway-cloudwatch-policy"
  role = aws_iam_role.api_gateway_cloudwatch.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ]
        Resource = [
          aws_cloudwatch_log_group.api_gateway.arn,
          "${aws_cloudwatch_log_group.api_gateway.arn}:*"
        ]
      }
    ]
  })
}

# API Gateway Account (for CloudWatch logging)
resource "aws_api_gateway_account" "main" {
  cloudwatch_role_arn = aws_iam_role.api_gateway_cloudwatch.arn
}

# API Gateway Stage with access logging
#tfsec:ignore:aws-api-gateway-enable-tracing Reason: X-Ray tracing not required for portfolio project to minimize costs
resource "aws_api_gateway_stage" "client_setup" {
  rest_api_id   = aws_api_gateway_rest_api.client_setup.id
  deployment_id = aws_api_gateway_deployment.client_setup.id
  stage_name    = "prod"

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      caller         = "$context.identity.caller"
      user           = "$context.identity.user"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  depends_on = [aws_api_gateway_account.main]
}

# API Key for authentication
resource "aws_api_gateway_api_key" "client_setup" {
  name    = "${var.function_name}-api-key"
  enabled = true
}

# Usage Plan to associate API key with stage
resource "aws_api_gateway_usage_plan" "client_setup" {
  name = "${var.function_name}-usage-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.client_setup.id
    stage  = aws_api_gateway_stage.client_setup.stage_name
  }

  quota_settings {
    limit  = 1000
    period = "DAY"
  }

  throttle_settings {
    burst_limit = 10
    rate_limit  = 5
  }
}

# Associate API Key with Usage Plan
resource "aws_api_gateway_usage_plan_key" "client_setup" {
  key_id        = aws_api_gateway_api_key.client_setup.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.client_setup.id
}