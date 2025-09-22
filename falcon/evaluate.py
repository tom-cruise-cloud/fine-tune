from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
import torch
import evaluate

test_data = [
    {
        "prompt": "What are the main benefits of a balanced diet?",
        "reference": "A balanced diet provides essential nutrients, improves energy levels, and supports a healthy immune system."
    },
]

# 3. Generate predictions and evaluate
predictions = []
references = []

prompts = [
    "What is the capital of France?",
]
references = [
    "Paris",
]

eval_dataset = Dataset.from_dict({"prompt": prompts, "reference": references})

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Adjust max_new_tokens as needed for your task
    outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the generated response for cleaner evaluation
    return response[len(prompt):].strip()

def predict_and_evaluate(dataset, metric):
    predictions = []
    for example in dataset:
        generated_text = generate_response(example["prompt"])
        predictions.append(generated_text)

    # The 'evaluate' library expects predictions and references as lists
    results = metric.compute(predictions=predictions, references=dataset["reference"])
    return results

# Example using ROUGE for text generation evaluation
rouge = evaluate.load("rouge")
rouge_results = predict_and_evaluate(eval_dataset, rouge)
print("ROUGE Results:", rouge_results)

# Example using accuracy for a hypothetical classification task (requires adjustments to generate_response)
# accuracy = evaluate.load("accuracy")
# accuracy_results = predict_and_evaluate(eval_dataset, accuracy)
# print("Accuracy Results:", accuracy_results)
