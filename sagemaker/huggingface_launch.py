import sagemaker
from sagemaker.huggingface import HuggingFace

# SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Upload your JSONL data to S3
s3_data_path = sagemaker_session.upload_data(path="path/to/your_data.jsonl", key_prefix="llm_training_data")

# sess = sagemaker.Session()
# bucket = sess.default_bucket()
# s3_training_data_path = f's3://{bucket}/your-data/train.jsonl'

# Configure the HuggingFace Estimator
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./",  # Directory containing train.py
    instance_type="ml.g4dn.xlarge",  # Choose appropriate instance type
    instance_count=1,
    role=role,
    transformers_version="4.37.2",  # Specify Hugging Face Transformers version
    pytorch_version="2.1.0",      # Specify PyTorch version
    py_version="py310",           # Specify Python version
    hyperparameters={
        "model_name": "gpt2",
        "epochs": 3,
    },
    # hyperparameters={'model_id': 'meta-llama/Llama-2-7b-hf', 'epochs': 5},
    input_mode="File",
    distribution={"smdistributed": {"dataparallel": {"enabled": False}}}, # Adjust for distributed training if needed
    
)
huggingface_estimator.fit({"train": s3_data_path})
# huggingface_estimator.fit({'training': s3_training_data_path})

# Start the training job
huggingface_estimator.fit({"train": s3_data_path})
