data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = var.source_path
  output_path = "${path.module}/lambda.zip"
}

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
resource "aws_api_gateway_method" "client_setup" {
  rest_api_id   = aws_api_gateway_rest_api.client_setup.id
  resource_id   = aws_api_gateway_resource.client_setup.id
  http_method   = "POST"
  authorization = "NONE"
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

# API Gateway Stage
resource "aws_api_gateway_stage" "client_setup" {
  rest_api_id   = aws_api_gateway_rest_api.client_setup.id
  deployment_id = aws_api_gateway_deployment.client_setup.id
  stage_name    = "prod"
}