from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_ap_ar_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = pipe("Verify if invoice INV-1001 is valid", max_new_tokens=150)
print(response[0]['generated_text'])
