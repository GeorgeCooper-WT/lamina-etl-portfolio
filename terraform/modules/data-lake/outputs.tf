output "raw_data_bucket_name" {
  description = "Name of the raw data bucket"
  value       = aws_s3_bucket.raw_data.id
}

output "raw_data_bucket_arn" {
  description = "ARN of the raw data bucket"
  value       = aws_s3_bucket.raw_data.arn
}

output "processed_data_bucket_name" {
  description = "Name of the processed data bucket"
  value       = aws_s3_bucket.processed_data.id
}

output "processed_data_bucket_arn" {
  description = "ARN of the processed data bucket"
  value       = aws_s3_bucket.processed_data.arn
}

output "reports_bucket_name" {
  description = "Name of the reports bucket"
  value       = aws_s3_bucket.reports.id
}

output "reports_bucket_arn" {
  description = "ARN of the reports bucket"
  value       = aws_s3_bucket.reports.arn
}
