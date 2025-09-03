import boto3
import tarfile
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

data = [
    {"instruction": "version", "response": "The VPC Flow Logs version. If you use the default format, the version is , If you use a custom format, the version is the highest version among the specified fields.  Parquet data type: INT"},
    {"instruction": "account-id", "response": "The AWS account ID of the owner of the source network interface for which traffic is recorded. If the network interface is created by an AWS service, for example when creating a VPC endpoint or Network Load Balancer, the record might display unknown for this field. Parquet data type: STRING"},
    {"instruction": "interface-id", "response": "The ID of the network interface for which the traffic is recorded. Parquet data type: STRING"},
    {"instruction": "srcaddr", "response": "For incoming traffic, this is the IP address of the source of traffic. For outgoing traffic, this is the private IPv, address or the IPv, address of the network interface sending the traffic. See also pkt-srcaddr. Parquet data type: STRING"},
    {"instruction": "dstaddr", "response": "The destination address for outgoing traffic, or the IPv, or IPv, address of the network interface for incoming traffic on the network interface. The IPv, address of the network interface is always its private IPv, address. See also pkt-dstaddr. Parquet data type: STRING"},
    {"instruction": "srcport", "response": "The source port of the traffic. Parquet data type: INT"},
    {"instruction": "dstport", "response": "The destination port of the traffic. Parquet data type: INT"},
    {"instruction": "protocol", "response": "The IANA protocol number of the traffic. For more information, see Assigned Internet Protocol Numbers. Parquet data type: INT"},
    {"instruction": "packets", "response": "The number of packets transferred during the flow. Parquet data type: INT"},
    {"instruction": "bytes", "response": "The number of bytes transferred during the flow. Parquet data type: INT"},
]

model_name = "fine_tuned_model"
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
