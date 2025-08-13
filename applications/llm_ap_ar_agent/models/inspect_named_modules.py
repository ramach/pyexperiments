from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Replace these with your actual local paths
base_model_path = "./mistralai_base_model"
adapter_path = "./mistralai_trained_adapter"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    local_files_only=True
)

# Inspect the full module structure
print("\n🔍 Model.named_modules()")
for name, _ in model.named_modules():
    print(name)
