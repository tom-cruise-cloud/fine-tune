# train.py
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--train_data_path", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load your custom dataset
    # Assuming your data is in JSON Lines format, with a 'text' field
    dataset = load_dataset("json", data_files=os.path.join(args.train_data_path, "your_data.jsonl"))

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
