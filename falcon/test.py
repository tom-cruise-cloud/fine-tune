from transformers import AutoTokenizer, AutoModelForCausalLM
# import evaluate

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
reference = "In this example, RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected."
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

# predictions = []
references = []

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response: " + response)
# predictions.append(response)
# references.append(reference)

# The 'evaluate' library expects predictions and references as lists
# Example using ROUGE for text generation evaluation
# rouge_metric = evaluate.load("rouge")
# rouge_result = rouge_metric.compute(predictions=predictions, references=references)
# print("Rouge Result: ", rouge_result)

# Example using accuracy for a hypothetical classification task (requires adjustments to generate_response)
# accuracy_metric = evaluate.load("accuracy")
# accuracy_result = accuracy_metric.compute(predictions=predictions, references=references)
# print("Accuracy Results: ", accuracy_result)
