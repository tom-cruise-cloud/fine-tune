from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "trainingdata2.txt"})
# print(dataset)

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# print(tokenized_datasets)

training_args = TrainingArguments(
    output_dir="./mlm_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
