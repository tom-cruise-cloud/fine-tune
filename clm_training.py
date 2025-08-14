from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

dataset = load_dataset("text", data_files={"train": "trainingdata2.txt"})

# model_name = "gpt2"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name) # Or your chosen CLM
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name) # Or your chosen CLM

training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        # per_device_eval_batch_size=8,
        # logging_dir="./logs",
        # logging_steps=500,
        save_steps=10_000,
        save_total_limit=2,
    )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    data_collator=data_collator
)

trainer.train()
