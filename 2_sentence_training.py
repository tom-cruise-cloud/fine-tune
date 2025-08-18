from datasets import Dataset
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

data = {
    "sentence1": ["2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK"],
    "sentence2": ["In this example, SSH traffic (destination port 22, TCP protocol) from IP address 172.31.16.139 to network interface with private IP address is 172.31.16.21 and ID eni-1235b8ca123456789 in account 123456789010 was allowed."],
    "label": [1]
}
dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized_dataset["labels"])
# print(tokenized_dataset['input_ids'])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) # Adjust num_labels

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
