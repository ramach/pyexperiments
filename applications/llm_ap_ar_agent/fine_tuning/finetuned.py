from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
dataset_path = "ap_ar_data.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

# Prepare prompts
def tokenize(example):
    input_ids = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)
    label_ids = tokenizer(example["completion"], truncation=True, padding="max_length", max_length=128)
    input_ids["labels"] = label_ids["input_ids"]
    return input_ids

tokenized_dataset = dataset.map(tokenize)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Training
training_args = TrainingArguments(
    output_dir="./ap_ar_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
