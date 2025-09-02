import boto3
import tarfile

s3 = boto3.client('s3')
bucket_name = 'ml-finetune'
s3_key = 'fine-tuned-model/model.tar.gz'
local_path = 'model.tar.gz'

s3.download_file(bucket_name, s3_key, local_path)

local_model_directory = "./"

with tarfile.open(local_path, 'r:gz') as tar:
    tar.extractall(local_model_directory)
tar.close()
