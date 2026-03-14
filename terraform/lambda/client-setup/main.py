import json
import boto3
from botocore.exceptions import ClientError
import os
import logging

logging.basicConfig(level=logging.INFO)

region = os.environ.get("AWS_REGION", "eu-west-2")
s3 = boto3.client("s3", region_name=region)


def lambda_handler(event, context):
    """
    Lambda function to create S3 folders for new clients in a multi-bucket data lake architecture.

    Triggered by: API Gateway POST request
    Expected payload:
        {
            "client_id": "uuid",
            "raw_bucket": "lamina-raw-data-dev",
            "processed_bucket": "lamina-processed-data-dev",
            "reports_bucket": "lamina-reports-dev"
        }

    IAM Role Required Permissions:
        - s3:PutObject on arn:aws:s3:::lamina-raw-data-dev/*
        - s3:PutObject on arn:aws:s3:::lamina-processed-data-dev/*
        - s3:PutObject on arn:aws:s3:::lamina-reports-dev/*

    Returns:
        - 200: Success with folder details
        - 400: Invalid input (missing fields)
        - 500: S3 error
    """
    logging.info("Lambda invoked with event: %s", json.dumps(event))

    # Parse request body
    try:
        if "body" in event:
            body = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            body = event

        client_id = body.get("client_id")
        raw_bucket = body.get("raw_bucket")
        processed_bucket = body.get("processed_bucket")
        reports_bucket = body.get("reports_bucket")

        missing = []
        if not client_id:
            missing.append("client_id")
        if not raw_bucket:
            missing.append("raw_bucket")
        if not processed_bucket:
            missing.append("processed_bucket")
        if not reports_bucket:
            missing.append("reports_bucket")

        if missing:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {"error": f"Missing required fields: {', '.join(missing)}"}
                ),
            }

    except Exception as e:
        logging.error("Error parsing request: %s", e)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Invalid request: {str(e)}"}),
        }

    # Create S3 client (uses Lambda execution role automatically - no credentials needed!)
    s3 = boto3.client("s3", region_name=region)
    logging.info("S3 client created using Lambda execution role")

    created_folders = []

    for bucket in [raw_bucket, processed_bucket, reports_bucket]:
        key = f"{client_id}/.init"
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=b"init")
            created_folders.append(f"{bucket}/{client_id}/")
            logging.info("Created: %s", key)
        except ClientError as e:
            logging.error("S3 error for %s: %s", key, e)
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {"error": f"S3 error: {str(e)}", "created_folders": created_folders}
                ),
            }

    # Success response
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "message": "Folders created successfully",
                "client_id": client_id,
                "folders": created_folders,
            }
        ),
    }
