from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Use 'auto' to let Hugging Face determine the best dtype
    device_map="auto"    # Automatically map model to available devices (e.g., GPU)
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
