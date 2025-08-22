import sagemaker
from sagemaker.pytorch import PyTorchModel
import torch

# Assuming your model artifacts are in a 'model.tar.gz' file locally
model_data_path = 'file://./model.tar.gz'

# Define the SageMaker PyTorchModel
# Replace 'your_entry_point_script.py' with your inference script
# Replace 'your_model_dir' with the directory containing your model artifacts
pytorch_model = PyTorchModel(
    model_data=model_data_path,
    role=sagemaker.get_execution_role(), # Or a specific IAM role if needed
    framework_version='1.13.1', # Specify your PyTorch version
    py_version='py39', # Specify your Python version
    entry_point='your_entry_point_script.py',
    source_dir='your_model_dir' # Directory containing entry_point_script and other necessary files
)

# Deploy the model locally for testing
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='local'
)

# Example inference
test_data = {"inputs": "Your test input text here."}
response = predictor.predict(test_data)
print(response)

# Clean up the local endpoint
predictor.delete_endpoint()
