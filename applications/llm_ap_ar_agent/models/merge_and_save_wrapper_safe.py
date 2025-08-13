
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class TripleModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self._base_model = base_model
        self.config = base_model.config

        # Register actual model under correct path
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.model = base_model.model

        # Register children properly for state dict
        self.add_module("model", self.model)

    def forward(self, *args, **kwargs):
        return self.model.model.model(*args, **kwargs)

    def __getattr__(self, name):
        if name in ["_base_model", "config", "model"]:
            return super().__getattribute__(name)
        return getattr(self._base_model, name)

def merge_and_save_lora(base_model_path, adapter_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    wrapped_model = TripleModelWrapper(base_model)

    model = PeftModel.from_pretrained(
        wrapped_model,
        adapter_path,
        local_files_only=True
    )

    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Merged model saved at: {output_path}")

if __name__ == "__main__":
    merge_and_save_lora(
        base_model_path="./mistralai_base_model",
        adapter_path="./mistralai_trained_adapter",
        output_path="./merged_mistral_model"
    )
