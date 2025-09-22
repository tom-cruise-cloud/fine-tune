from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import evaluate

model_path = "./fine_tuned_model_v1/"
base_model_id = "tiiuae/falcon3-10B-instruct" # The base model you used for fine-tuning

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
        "response": "RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected."
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
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = sequences[0]['generated_text'].split(prompt)[-1].strip()
    predictions.append(generated_text)
    
    print(f"Prompt: {prompt}\n")
    print(f"Generated: {generated_text}\n")

rouge = evaluate.load("rouge")
results = rouge.compute(predictions=predictions, references=references)

print("\n--- Automated Evaluation Results (ROUGE) ---")
print(results)

# 5. Conduct human evaluation (optional but recommended)
print("\n--- Human Evaluation ---")
print("Examine the generated text above and answer the following questions for each response:")
print("- **Relevance:** Does the response directly answer the prompt?")
print("- **Correctness:** Is the information factually accurate?")
print("- **Fluency:** Is the response well-written and easy to read?")
print("- **Instruction-Following:** Does it follow any specific instructions given in the prompt?")


