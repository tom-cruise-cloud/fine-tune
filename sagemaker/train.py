# train.py
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_data_path", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load data
    train_data = []
    with open(os.path.join(args.train_data_path, "vpc-flowlog.jsonl"), "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["instruction"], text_target=examples["response"], truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        logging_dir=f"{args.model_dir}/logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

if __name__ == "__main__":
    main()
