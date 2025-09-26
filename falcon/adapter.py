from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
ADAPTER_PATH = 'path/to/adapter_model.safetensors'

# Load base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load adapter configuration and model
adapter_config = PeftConfig.from_pretrained(ADAPTER_PATH)
model = PeftModel.from_pretrained(model, adapter_config, adapter_path=ADAPTER_PATH)

# Now you can use the model with the fine-tuned adapter
input_ids = tokenizer("Your input text here", return_tensors="pt").input_ids
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
