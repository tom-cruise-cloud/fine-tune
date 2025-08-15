from PyPDF2 import PdfReader
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf("training1.pdf")

# Example for a simple text dataset
data = {"text": [pdf_text]} # Or break it into smaller chunks if it's very long
dataset = Dataset.from_dict(data)
# print(dataset['text'])

# model_checkpoint = "bert-base-uncased" # Choose a suitable model
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# print(tokenized_dataset.column_names)
# # print(tokenized_dataset['text'][0])
# # print(tokenized_dataset['input_ids'])
# # print(tokenized_dataset['attention_mask'])
# # print(tokenized_dataset['token_type_ids'])

# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3) # Adjust num_labels

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='longest', truncation=True, return_tensors="pt")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    overwrite_output_dir=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
