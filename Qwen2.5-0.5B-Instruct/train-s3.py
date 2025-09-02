import boto3
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer

s3 = boto3.client('s3')
bucket_name = 'ml-finetune'
s3_key = 'model.tar.gz'
local_path = 'model.tar'

s3.download_file(bucket_name, s3_key, local_path)

local_model_directory = "./local_model_directory"

with tarfile.open(local_path, 'r') as tar:
    tar.extractall('./local_model_directory')

model_dir = './local_model_directory'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Set the device to CUDA if available, otherwise CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

train_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in data], "response": [d["response"] for d in data]})
# print(dataset[0])
# Convert to Hugging Face Dataset

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["instruction"], text_target=examples["response"], truncation=True, padding='max_length', max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
print(tokenized_train_dataset)

training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=tokenized_train_dataset,
    args=training_args,
    # formatting_func=lambda x: f"### Instruction:\n{x['instruction']}\n### Output:\n{x['response']}",
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
