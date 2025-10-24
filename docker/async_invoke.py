import sagemaker
import boto3

# Initialize SageMaker session and S3 client
sagemaker_session = sagemaker.Session()
boto_session = boto3.Session()
sm_runtime = boto_session.client("sagemaker-runtime")

# Define your SageMaker endpoint name
endpoint_name = "falcon3-10b"
s3_input_location = "s3://ml-finetune/fine-tuned-model/input.json"
response = sm_runtime.invoke_endpoint_async(
    EndpointName=endpoint_name,
    InputLocation=s3_input_location,
    # Optionally, specify an S3 location for output if not already configured in endpoint
    # ExpectedOutputLocation=f"s3://{default_bucket}/{output_prefix}/"
)
# output_location = response["OutputLocation"]
print(f"Asynchronous invocation successful")
