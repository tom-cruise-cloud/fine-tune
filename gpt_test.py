from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./results/checkpoint-10/" # Replace with your model's local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

text_prompt = "A bustling city street"
inputs = tokenizer(text_prompt, return_tensors="pt")
generated_ids = model.generate(inputs["input_ids"], max_new_tokens=100, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id,attention_mask=inputs['attention_mask'],)
generated_description = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_description)
