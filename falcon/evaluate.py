from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
import torch
import evaluate

model_path = "your_local_model_path"
base_model_id = "tiiuae/falcon3-10b-instruct" # The base model you used for fine-tuning

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load the PEFT adapter
model = PeftModel.from_pretrained(base_model, model_path)

# Merge the adapter with the base model for inference
model = model.merge_and_unload()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

test_data = [
    {
        "instruction": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK", 
        "response": "In this example, RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected."
    },
]
# 3. Generate predictions and evaluate
predictions = []
references = []

for item in test_data:
    prompt = item["instruction"]
    reference = item["response"]
    references.append(reference)

    sequences = pipeline(
    prompt,
    max_length=150,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = sequences[0]['generated_text'].split(prompt)[-1].strip()
    predictions.append(generated_text)
    
    print(f"Prompt: {prompt}\n")
    print(f"Generated: {generated_text}\n")
    print("-" * 50)

rouge = evaluate.load("rouge")
results = rouge.compute(predictions=predictions, references=references)

# eval_dataset = Dataset.from_dict({"prompt": prompts, "reference": references})

# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     # Adjust max_new_tokens as needed for your task
#     outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Remove the prompt from the generated response for cleaner evaluation
#     return response[len(prompt):].strip()

# def predict_and_evaluate(dataset, metric):
#     predictions = []
#     for example in dataset:
#         generated_text = generate_response(example["prompt"])
#         predictions.append(generated_text)

#     # The 'evaluate' library expects predictions and references as lists
#     results = metric.compute(predictions=predictions, references=dataset["reference"])
#     return results

# # Example using ROUGE for text generation evaluation
# rouge = evaluate.load("rouge")
# rouge_results = predict_and_evaluate(eval_dataset, rouge)
# print("ROUGE Results:", rouge_results)

# # Example using accuracy for a hypothetical classification task (requires adjustments to generate_response)
# # accuracy = evaluate.load("accuracy")
# # accuracy_results = predict_and_evaluate(eval_dataset, accuracy)
# # print("Accuracy Results:", accuracy_results)
