import boto3
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer

s3 = boto3.client('s3')
bucket_name = 'ml-finetune'
s3_key = 'model.tar.gz'
local_path = 'model.tar'

s3.download_file(bucket_name, s3_key, local_path)

local_model_directory = "./local_model_directory"

with tarfile.open(local_path, 'r') as tar:
    tar.extractall('./local_model_directory')

model_dir = './local_model_directory'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
