from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

data = [
    {
        "sentence1": "2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK",
        "sentence2": "In this example, SSH traffic (destination port 22, TCP protocol) from IP address 172.31.16.139 to network interface with private IP address is 172.31.16.21 and ID eni-1235b8ca123456789 in account 123456789010 was allowed.",
        "label": 1
    },
    {
        "sentence1": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK",
        "sentence2": "In this example, RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected.",
        "label": 1
    },
    {
        "sentence1": "2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK",
        "sentence2": "In this example, RDP traffic (destination port 3389, TCP protocol) to network interface eni-1235b8ca123456789 in account 123456789010 was rejected.",
        "label": 0
    },
]

dataset = Dataset.from_list(data)

model_name = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length", # or "longest"
            max_length=256 # specify a max length if using "max_length"
)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.add_column("labels", dataset["label"])
# tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized_dataset[0])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Use 'auto' to let Hugging Face determine the best dtype
    device_map="auto"    # Automatically map model to available devices (e.g., GPU)
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    save_steps=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
