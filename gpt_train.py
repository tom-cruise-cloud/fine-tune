from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load your dataset

data = [
  {"prompt": "A sunny day at the beach", "description": "The golden sand stretched endlessly under the warm sun, meeting the gentle lapping waves of the turquoise ocean."},
  {"prompt": "A bustling city street", "description": "Skyscrapers towered over the crowded sidewalks, filled with the sounds of traffic and chatter, as neon signs glowed in the evening light."}
]

dataset = Dataset.from_list(data)
print(dataset)
# Load a pre-trained tokenizer (e.g., for GPT-2)
model_name='gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Add a pad token if the tokenizer doesn't have one (common for GPT-2)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def preprocess_function(examples):
    inputs = [f"Prompt: {p}" for p in examples["prompt"]]
    targets = [d for d in examples["description"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
# print(tokenized_dataset[0])

model.resize_token_embeddings(len(tokenizer)) # Resize embeddings if pad token was added

 # Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
