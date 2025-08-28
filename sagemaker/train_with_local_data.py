import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

data = [
    {"instruction": "2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK", "response": "Accepted and rejected trafficthis is example of default flow log records.In this example, SSH traffic (destination port 22, TCP protocol) from IP address 172.31.16.139 to network interface with private IP address is 172.31.16.21 and ID eni-1235b8ca123456789 in account 123456789010 was allowed."},
    {"instruction": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK", "response": "In this example, RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected."},
    {"instruction": "2 123456789010 eni-1235b8ca123456789 - - - - - - - 1431280876 1431280934 - NODATA", "response": "No data and skipped recordsthis is example of default flow log records.In this example, no data was recorded during the aggregation interval."},
    {"instruction": "2 123456789010 eni-11111111aaaaaaaaa - - - - - - - 1431280876 1431280934 - SKIPDATA", "response": "VPC Flow Logs skips records when it can't capture flow log data during an aggregation interval because it exceeds internal capacity. A single skipped record can represent multiple flows that were not captured for the network interface during the aggregation interval."}
]
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in data], "response": [d["response"] for d in data]})
# print(dataset[0])
# Convert to Hugging Face Dataset

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["instruction"], text_target=examples["response"], truncation=True, padding='max_length', max_length=256)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
