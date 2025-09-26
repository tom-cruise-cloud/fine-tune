from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_NAME = 'tiiuae/Falcon3-10B-Instruct'
ADAPTER_PATH = './fine_tuned_model_v1/adapter_model.safetensors'

# Load base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load adapter configuration and model
adapter_config = PeftConfig.from_pretrained(ADAPTER_PATH)
model = PeftModel.from_pretrained(model, adapter_config, adapter_path=ADAPTER_PATH)

# Now you can use the model with the fine-tuned adapter
input_ids = tokenizer("2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK", return_tensors="pt").input_ids
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
