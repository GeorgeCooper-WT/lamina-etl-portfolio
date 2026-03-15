output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.client_setup.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.client_setup.arn
}

output "api_gateway_url" {
  description = "API Gateway endpoint for client folder creation Lambda"
  value       = "https://${aws_api_gateway_rest_api.client_setup.id}.execute-api.${var.aws_region}.amazonaws.com/${aws_api_gateway_stage.client_setup.stage_name}/client-setup"
}

output "api_key_id" {
  description = "ID of the API Gateway API Key"
  value       = aws_api_gateway_api_key.client_setup.id
}

output "api_key_value" {
  description = "Value of the API Gateway API Key (sensitive)"
  value       = aws_api_gateway_api_key.client_setup.value
  sensitive   = true
}