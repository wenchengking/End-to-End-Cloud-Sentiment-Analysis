from pathlib import Path

import boto3


def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    s3 = boto3.client("s3")
    print(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        print(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def list_paths_s3(bucket_name: str, prefix: str):
    # Initialize the S3 resource
    s3 = boto3.resource('s3')
    print(f"Fetching path list")
    # Retrieve the bucket object
    bucket = s3.Bucket(bucket_name)
    # Get all objects in the bucket
    objects = list(bucket.objects.all())
    # Extract the folder names from the objects
    folders = [obj.key.split('/')[1] for obj in objects if obj.key.startswith(str(prefix)) and obj.key.endswith('/')]
    print("%d versions of model exist" % len(folders))
    return folders

if __name__ == "__main__":
    list_paths_s3("msia423-group8-artifact", "") 