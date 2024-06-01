import boto3
import os
#from bing_image_downloader import downloader

# Set up S3 client
s3 = boto3.client('s3')

# Function to upload file to S3
def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    s3.upload_file(file_name, bucket, object_name)

# Download images locally
#downloader.download("your search query", limit=5, output_dir='images', adult_filter_off=True, force_replace=False, timeout=60)

# Upload images to S3 bucket
bucket_name = 'your-s3-bucket-name'
for root, dirs, files in os.walk('images'):
    for file in files:
        file_path = os.path.join(root, file)
        upload_to_s3(file_path, bucket_name, object_name=file)

# Clean up local files if needed
import shutil
shutil.rmtree('images')

