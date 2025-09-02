import os
import sagemaker
from sagemaker.s3 import S3Uploader

file_path = 'model.tar.gz' # Or the absolute path

if os.path.exists(file_path):
    print(f"'{file_path}' exists.")
    # Proceed with opening/using the file
else:
    print(f"Error: '{file_path}' not found. Please check the path and file name.")

sagemaker_session = sagemaker.Session()
bucket = 'ml-finetune'
prefix = "fine-tuned-model" # A path within the bucket

local_model_path = "model.tar.gz" # Or the path to your saved model file/directory
s3_uri = S3Uploader.upload(local_model_path, f"s3://{bucket}/{prefix}")

print(f"Model uploaded to: {s3_uri}")

data = [
    {"instruction": "version", "response": "The VPC Flow Logs version. If you use the default format, the version is , If you use a custom format, the version is the highest version among the specified fields.  Parquet data type: INT"},
    {"instruction": "account-id", "response": "The AWS account ID of the owner of the source network interface for which traffic is recorded. If the network interface is created by an AWS service, for example when creating a VPC endpoint or Network Load Balancer, the record might display unknown for this field. Parquet data type: STRING"},
    {"instruction": "interface-id", "response": "The ID of the network interface for which the traffic is recorded. Parquet data type: STRING"},
    {"instruction": "srcaddr", "response": "For incoming traffic, this is the IP address of the source of traffic. For outgoing traffic, this is the private IPv, address or the IPv, address of the network interface sending the traffic. See also pkt-srcaddr. Parquet data type: STRING"},
    {"instruction": "dstaddr", "response": "The destination address for outgoing traffic, or the IPv, or IPv, address of the network interface for incoming traffic on the network interface. The IPv, address of the network interface is always its private IPv, address. See also pkt-dstaddr. Parquet data type: STRING"},
    {"instruction": "srcport", "response": "The source port of the traffic. Parquet data type: INT"},
    {"instruction": "dstport", "response": "The destination port of the traffic. Parquet data type: INT"},
    {"instruction": "protocol", "response": "The IANA protocol number of the traffic. For more information, see Assigned Internet Protocol Numbers. Parquet data type: INT"},
    {"instruction": "packets", "response": "The number of packets transferred during the flow. Parquet data type: INT"},
    {"instruction": "bytes", "response": "The number of bytes transferred during the flow. Parquet data type: INT"},
]
