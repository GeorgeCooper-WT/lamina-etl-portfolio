variable "function_name" {
  description = "Name of the Lambda function"
  type        = string
}

variable "role_arn" {
  description = "IAM role ARN for Lambda execution"
  type        = string
}

variable "source_path" {
  description = "Path to Lambda source code directory"
  type        = string
}

variable "environment_variables" {
  description = "Map of environment variables for Lambda"
  type        = map(string)
  default     = {}
}

variable "handler" {
  description = "Lambda handler (e.g., main.lambda_handler)"
  type        = string
}

variable "runtime" {
  description = "Lambda runtime"
  type        = string
}

variable "aws_region" {
  description = "AWS region for API Gateway invoke URL"
  type        = string
}