import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_path = "train_invoice_sow_sample.jsonl"
output_dir = "./output_lora_mistralai"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# ✅ Load and tokenize dataset manually with labels
with open(dataset_path, "r") as f:
    raw_data = [json.loads(line) for line in f]

formatted_data = [{"instruction": f"{item['input']}\n{item['output']}"} for item in raw_data]

def tokenize(example):
    enc = tokenizer(example["instruction"], truncation=True, padding="max_length", max_length=512)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_data = list(map(tokenize, formatted_data))

split_idx = int(0.95 * len(tokenized_data))
train_data = tokenized_data[:split_idx]
test_data = tokenized_data[split_idx:]

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=5,
    fp16=True,
    save_strategy="epoch",
    report_to="none",
    logging_dir="./logs",
    disable_tqdm=False
)

print("✅ Initialized training arguments.")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

print("🚀 Starting training...")
try:
    trainer.train(resume_from_checkpoint=False)
    print("✅ Training completed successfully.")
except Exception as e:
    print("❌ Training failed:", str(e))

print("💾 Saving model...")
model.save_pretrained(output_dir)
