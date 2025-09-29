#tar -czvf falcon.model.tar.gz ./fine_tuned_model_v1/
#tar -czvf falcon.model.tar.gz ./inference.py ./requirements.txt ./fine_tuned_model_v1/

import os
import sagemaker
from sagemaker.s3 import S3Uploader

file_path = 'falcon.model.tar.gz' # Or the absolute path

if os.path.exists(file_path):
    print(f"'{file_path}' exists.")
    # Proceed with opening/using the file
else:
    print(f"Error: '{file_path}' not found. Please check the path and file name.")

sagemaker_session = sagemaker.Session()
bucket = 'ml-finetune'
prefix = "fine-tuned-model" # A path within the bucket

local_model_path = "falcon.model.tar.gz" # Or the path to your saved model file/directory
s3_uri = S3Uploader.upload(local_model_path, f"s3://{bucket}/{prefix}")

print(f"Model uploaded to: {s3_uri}")
