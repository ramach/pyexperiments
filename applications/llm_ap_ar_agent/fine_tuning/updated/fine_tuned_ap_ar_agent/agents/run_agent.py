from langchain.agents import initialize_agent, AgentType
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

def load_model():
    config = PeftConfig.from_pretrained("model/")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, "model/")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def get_agent():
    from invoice_tools import validate_invoice_against_rules
    llm = load_model()
    tools = [validate_invoice_against_rules]
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
