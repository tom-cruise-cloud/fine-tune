import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define your input data path in S3
input_data_path = "s3://your-s3-bucket/your-data-folder/"

# Configure the PyTorch estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir="./",  # Directory containing train.py and any other necessary files
    role=role,
    framework_version="1.13.1",  # Specify PyTorch version
    py_version="py39",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",  # Choose an appropriate GPU instance type
    hyperparameters={
        "model_name_or_path": "gpt2",
        "epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
    },
    input_mode="File",
)

# Start the training job
estimator.fit({"training": input_data_path})
