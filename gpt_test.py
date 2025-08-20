from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./results/checkpoint-40/" # Replace with your model's local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

text_prompt = "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"
inputs = tokenizer(text_prompt, return_tensors="pt")
generated_ids = model.generate(
    inputs["input_ids"], 
    max_new_tokens=150, 
    num_beams=3, 
    pad_token_id=tokenizer.eos_token_id, 
    attention_mask=inputs['attention_mask'],
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
generated_description = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_description)
