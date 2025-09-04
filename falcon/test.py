from transformers import AutoTokenizer, AutoModelForCausalLM

model_name='./fine_tuned_model_v1/'
# model_name='./results/checkpoint-5'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")
# print(model_inputs.input_ids)

# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs['attention_mask'],
    pad_token_id=tokenizer.eos_token_id, 
    max_new_tokens=1024,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.85,
)
# print(generated_ids)

# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# print(generated_ids)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response: " + response)
