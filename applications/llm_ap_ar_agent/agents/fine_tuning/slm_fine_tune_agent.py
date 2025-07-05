from langchain.tools import Tool
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType

MODEL_DIR = "./flan_apar_model"  # Your fine-tuned model dir

# Assume llm (HuggingFacePipeline) already created
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    hf_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=hf_pipe)

llm = load_flan_model()
def validation_tool_fn(invoice_text: str) -> str:
    prompt = f"Validate this invoice: {invoice_text}"
    return llm(prompt)

def business_rule_tool_fn(invoice_text: str) -> str:
    import re
    match = re.search(r"\$?(\d+(?:,\d{3})*(?:\.\d{2})?)", invoice_text)
    if match:
        amount = float(match.group(1).replace(",", ""))
        if amount > 10000:
            return f"❌ Amount ${amount} exceeds allowed max."
        else:
            return f"✅ Amount ${amount} is within allowed max."
    return "⚠ Could not extract amount."

def po_match_tool_fn(invoice_text: str) -> str:
    return "✅ PO matched successfully."

def approval_tool_fn(invoice_text: str) -> str:
    return "✅ Approved by manager."

def payment_tool_fn(invoice_text: str) -> str:
    return "✅ Payment processed."

# LangChain Tools
validation_tool = Tool.from_function(validation_tool_fn, name="Validation", description="Validate the invoice")
business_rule_tool = Tool.from_function(business_rule_tool_fn, name="BusinessRule", description="Check business rules")
po_match_tool = Tool.from_function(po_match_tool_fn, name="POMatch", description="Match to purchase order")
approval_tool = Tool.from_function(approval_tool_fn, name="Approval", description="Simulate approval process")
payment_tool = Tool.from_function(payment_tool_fn, name="Payment", description="Simulate payment processing")

def run_slm_agent():
    tools = [validation_tool, business_rule_tool, po_match_tool, approval_tool, payment_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

