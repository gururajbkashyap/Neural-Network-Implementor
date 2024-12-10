import json
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling

# Load your dataset from the 'data.json' file
with open('data.json', 'r') as f:
    raw_data = json.load(f)

# Preprocess data: combine questions and answers
data = [
    {"text": f"Q: {item['question']}\nA: {item['answer']}"}
    for item in raw_data
]
#  trainer.train()
#  model.save_pretrained("./finetuned_model")
#     tokenizer.save_pretrained("./finetuned_model")
#     print("Fine-tuning completed and model saved.")
# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"text": [item["text"] for item in data]})

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad_token to eos_token (since GPT-2 doesn't have a pad token)
tokenizer.pad_token = tokenizer.eos_token  # You can also choose to define a new pad token

# Load the model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=5e-5  # You can adjust this learning rate as necessary
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
