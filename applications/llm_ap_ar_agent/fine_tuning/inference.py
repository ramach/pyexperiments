from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("./ap_ar_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

prompt = "Invoice: Contractor billed 44 hours at $141/hr. Is it valid?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#with langchain

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

llm("Invoice: Contractor billed 50 hours this week without approval.")


