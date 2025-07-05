from datasets import load_dataset

dataset = load_dataset("json", data_files="ap_ar_train.jsonl", split="train")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./fine_tuned_ap_ar_model",
    save_strategy="no"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="instruction",  # if both `instruction` and `output` exist, concatenate
)

trainer.train()
model.save_pretrained("./fine_tuned_ap_ar_model")
