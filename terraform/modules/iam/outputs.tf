output "lambda_execution_role_arn" {
  description = "ARN of Lambda execution role"
  value       = aws_iam_role.lambda_execution.arn
}

output "lambda_execution_role_name" {
  description = "Name of Lambda execution role"
  value       = aws_iam_role.lambda_execution.name
}

output "glue_execution_role_arn" {
  description = "ARN of Glue execution role"
  value       = aws_iam_role.glue_execution.arn
}

output "glue_execution_role_name" {
  description = "Name of Glue execution role"
  value       = aws_iam_role.glue_execution.name
}

output "ec2_instance_profile_name" {
  description = "Name of EC2 instance profile"
  value       = aws_iam_instance_profile.ec2_profile.name
}

output "ec2_role_arn" {
  description = "ARN of EC2 IAM role"
  value       = aws_iam_role.ec2_data_access.arn
}
