# S3 Bucket for Raw Data
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-raw-data-${var.environment}"
  
  tags = {
    Name        = "Lamina Raw Data"
    Layer       = "Raw"
    Description = "Raw SCADA and Satellite data ingestion"
  }
}

# S3 Bucket for Processed Data
resource "aws_s3_bucket" "processed_data" {
  bucket = "${var.project_name}-processed-data-${var.environment}"
  
  tags = {
    Name        = "Lamina Processed Data"
    Layer       = "Processed"
    Description = "Transformed and cleaned data"
  }
}

# S3 Bucket for Reports
resource "aws_s3_bucket" "reports" {
  bucket = "${var.project_name}-reports-${var.environment}"
  
  tags = {
    Name        = "Lamina Reports"
    Layer       = "Reports"
    Description = "Client-facing reports and visualizations"
  }
}

# Enable versioning for all buckets (data protection)
resource "aws_s3_bucket_versioning" "raw_versioning" {
  bucket = aws_s3_bucket.raw_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "processed_versioning" {
  bucket = aws_s3_bucket.processed_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "reports_versioning" {
  bucket = aws_s3_bucket.reports.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption (AES-256)
resource "aws_s3_bucket_server_side_encryption_configuration" "raw_encryption" {
  bucket = aws_s3_bucket.raw_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed_encryption" {
  bucket = aws_s3_bucket.processed_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "reports_encryption" {
  bucket = aws_s3_bucket.reports.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access (security best practice)
resource "aws_s3_bucket_public_access_block" "raw_public_access_block" {
  bucket = aws_s3_bucket.raw_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed_public_access_block" {
  bucket = aws_s3_bucket.processed_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "reports_public_access_block" {
  bucket = aws_s3_bucket.reports.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for Raw Data (cost optimization)
resource "aws_s3_bucket_lifecycle_configuration" "raw_lifecycle" {
  bucket = aws_s3_bucket.raw_data.id
  
  rule {
    id     = "archive-old-raw-data"
    status = "Enabled"
    filter {}
    
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
    
    transition {
      days          = 180
      storage_class = "DEEP_ARCHIVE"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER_IR"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Lifecycle policy for Processed Data
resource "aws_s3_bucket_lifecycle_configuration" "processed_lifecycle" {
  bucket = aws_s3_bucket.processed_data.id
  
  rule {
    id     = "archive-old-processed-data"
    status = "Enabled"
    filter {}
    
    transition {
      days          = 60
      storage_class = "INTELLIGENT_TIERING"
    }
    
    transition {
      days          = 180
      storage_class = "GLACIER_IR"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}

# Reports kept in Standard storage (frequently accessed)
# No lifecycle policy for reports bucket
