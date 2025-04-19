from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class ARChain:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model_name="gpt-4", temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["customer", "invoice_id", "amount", "due_date"],
            template=(
                "You are an AR agent. Review this receivable invoice:\n"
                "- Customer: {customer}\n"
                "- Invoice ID: {invoice_id}\n"
                "- Amount: ${amount}\n"
                "- Due Date: {due_date}\n\n"
                "Respond with:\n"
                "1. Payment expected or delayed\n"
                "2. Follow-up recommendation"
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, invoice_data: dict):
        return self.chain.run(**invoice_data)