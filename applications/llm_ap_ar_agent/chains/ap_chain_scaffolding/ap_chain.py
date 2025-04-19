from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class APChain:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model_name="gpt-4", temperature=0)

        self.prompt = PromptTemplate(
            input_variables=["vendor", "invoice_id", "amount", "due_date"],
            template=(
                "You are an AP agent. Analyze the following invoice:\n"
                "- Vendor: {vendor}\n"
                "- Invoice ID: {invoice_id}\n"
                "- Amount: ${amount}\n"
                "- Due Date: {due_date}\n"
                "\nRespond with:\n"
                "1. Whether the invoice is valid\n"
                "2. Whether it should be paid now or flagged\n"
                "3. Any follow-up actions needed"
            )
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, invoice_data: dict):
        return self.chain.run(**invoice_data)