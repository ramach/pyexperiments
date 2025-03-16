import getpass
import os

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# Initialize LLM and Embeddings
llm = OpenAI(temperature=0)  # Or your preferred LLM
embeddings = OpenAIEmbeddings()

# --- Synthetic Data for Healthcare Insurance Claims Processing ---
# (Replace with your actual data loading and formatting)

# Sample patient records
patient_records = """
Patient ID | Name | Age | Gender | Diagnosis | Treatment | Insurance Plan
------- | -------- | -------- | -------- | -------- | -------- | --------
12345 | John Doe | 45 | Male | Diabetes | Medication, Insulin Pump | Gold Plan
67890 | Jane Smith | 32 | Female | Asthma | Inhaler, Nebulizer | Silver Plan
"""

# Sample insurance policies
insurance_policies = """
Plan | Coverage | Deductible | Out-of-Pocket Max
------- | -------- | -------- | --------
Gold Plan | Comprehensive | 1000 | 5000
Silver Plan | Standard | 2000 | 7000
"""

# Sample medical billing codes
billing_codes = """
Code | Description | Cost
------- | -------- | --------
A1234 | Insulin Pump | 5000
B5678 | Inhaler | 100
"""

# Sample claim processing guidelines
claim_guidelines = """
- Verify patient eligibility and coverage.
- Check for pre-authorization requirements.
- Validate billing codes and costs.
- Apply deductibles and co-pays.
- Calculate reimbursement amount.
- Generate claim status report.
"""

# Combine data into a list of documents
documents = [patient_records, insurance_policies, billing_codes, claim_guidelines]

# --- End of Synthetic Data ---

# Load data into Chroma vector database
db = Chroma.from_texts(documents, embeddings, collection_name="healthcare_data")

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# --- Main execution block ---
if __name__ == "__main__":
    # Chatbot interaction loop
    while True:
        query = input("Insurance Agent: ")

        # Get response from RetrievalQA chain
        response = qa_chain.run(query)

        # Print response in chat format
        print("Chatbot:", response)

        # Exit condition
        if query.lower() in ["exit", "quit"]:
            break