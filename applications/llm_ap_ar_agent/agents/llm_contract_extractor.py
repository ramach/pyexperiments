import json

import openai
from docx import Document
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_contract_fields_with_llm_experimental(docx_file) -> dict:
    contract_text = extract_text_from_docx(docx_file)

    system_prompt = "Extract key fields from this consulting contract document in structured JSON format. Fields include:\n" \
                    "- Client Name and Address\n- Consulting Firm Name and Address\n- Scope of Work\n" \
                    "- Fees and Payment Terms\n- Terms and Termination\n- Effective Dates (if available)"

    user_prompt = f"Document:\n{contract_text[:80000]}"  # truncate for context length

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    content = response["choices"][0]["message"]["content"]
    return content

def extract_contract_fields_with_llm(contract_text: str) -> dict:
    from langchain.prompts import PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["contract_text"],
        template = f""" you are an AI assistant designed to extract structured contract from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {contract_text}
         
         The output should include:
         - Client Name
         - Client Address
         - Consulting Firm Name
         - Consulting Firm Address
         - Scope of Work
         - Fees and Payment Terms
         - Terms and Termination
         if the vendor name is missing use business_id or client id.
         If a field is not present, say "MISSING". Return a JSON object.
         """)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"contract_text": contract_text})
    data = json.loads(response)
    try:
        return data  # Replace with `json.loads()` if result is a valid JSON string
    except Exception as e:
        return {"error": f"Failed to parse result: {e}", "raw_result": response}
