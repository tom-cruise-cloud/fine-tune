from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
