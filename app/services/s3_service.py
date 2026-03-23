import boto3
from app.core.config import settings

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY
)

def upload_file_to_s3(file_content, file_name):
    s3_client.put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=file_name,
        Body=file_content
    )
    return f"s3://{settings.S3_BUCKET_NAME}/{file_name}"