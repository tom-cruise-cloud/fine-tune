from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./results/checkpoint-5/" # Replace with your model's local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK"

inputs = tokenizer(prompt, return_tensors="pt")
# print(inputs)

output_sequences = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,  # Maximum number of new tokens to generate
    num_beams=5,        # For beam search decoding
    do_sample=True,     # For sampling-based generation
    temperature=0.7,    # Controls randomness in sampling
    top_k=50,           # Top-k sampling
    top_p=0.95          # Nucleus sampling
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)

