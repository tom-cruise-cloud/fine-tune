from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./results/checkpoint-21" # Path to your locally saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "explain '2 123456789010 eni-1235b8ca123456789 - - - - - - - 1431280876 1431280934 - NODATA'"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=200, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
