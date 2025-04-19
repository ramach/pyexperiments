# ui/streamlit_ui.py
import sys
import os
#Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import json
import os
from agents.openai_function_handler_advanced import get_openai_tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import sqlite3
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from retrievers.serpapi_scraper import run_serpapi_scraper

# Load environment variables
load_dotenv()

# Load OpenAI API key safely
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# Initialize LLM and tools
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4", temperature=0)
tools = get_openai_tools()
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Load mock data
def load_mock_data(filename):
    filepath = os.path.join("mockdata", filename)
    with open(filepath, "r") as f:
        return json.load(f)

def run_agent(query):
    try:
        return agent.run(query)
    except Exception as e:
        return f"[Error] {str(e)}"

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"[PDF Error] {str(e)}"

def extract_text_from_image(uploaded_image):
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"[Image OCR Error] {str(e)}"

def save_chat_history_as_json(history, filename="chat_history.json"):
    try:
        filepath = os.path.join("chatlogs", filename)
        os.makedirs("chatlogs", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)
        return filepath
    except Exception as e:
        return f"[Save Error] {str(e)}"

def summarize_text(text):
    prompt = f"Summarize the following document:\n{text}"
    return run_agent(prompt)

def save_to_db(data, table_name="invoices"):
    try:
        conn = sqlite3.connect("data/invoices.db")
        cursor = conn.cursor()
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')
        cursor.execute(f'''INSERT INTO {table_name} (content) VALUES (?)''', (json.dumps(data),))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return f"[DB Error] {str(e)}"

def embed_and_query(document_text, user_query, vendor_context=""):
    docs = text_splitter.create_documents([document_text])
    vector_store = FAISS.from_documents(docs, embedding_model)
    top_docs = vector_store.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in top_docs])
    full_context = f"{vendor_context}\n\n{context}" if vendor_context else context
    return run_agent(f"{user_query}\n\nContext:\n{full_context}")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.set_page_config(page_title="AP/AR Agent", layout="centered")
st.title("\U0001F4BC Accounts Payable & Receivable Chat Agent")

option = st.radio("Choose Mode:", ["Accounts Payable", "Accounts Receivable"])

upload_tab, mock_tab, image_tab, vendor_tab = st.tabs(["Upload PDF", "Use Mock Data", "Image OCR", "Vendor Info"])

query = ""
document_data = ""
vendor_context = ""

with upload_tab:
    uploaded_file = st.file_uploader("Upload invoice PDF", type="pdf")
    summarize = st.checkbox("Summarize PDF before asking question")
    if uploaded_file is not None:
        document_data = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted text from PDF:", document_data, height=200)
        if summarize:
            summary = summarize_text(document_data)
            st.text_area("\U0001F4C4 Summary:", summary, height=150)
            document_data = summary
        query = st.text_input("Ask a question about the uploaded invoice:", "Analyze this invoice")

with mock_tab:
    mock_file = "ap_mock_invoices.json" if option == "Accounts Payable" else "ar_mock_invoices.json"
    data = load_mock_data(mock_file)
    mock_choice = st.selectbox("Select mock invoice:", options=[f"{i+1}: {d['invoice_id']}" for i, d in enumerate(data)])
    selected_data = data[int(mock_choice.split(":")[0]) - 1]
    st.json(selected_data)
    query = st.text_input("Ask a question or let the agent analyze this invoice:", "Analyze this invoice")
    document_data = json.dumps(selected_data)

with image_tab:
    uploaded_image = st.file_uploader("Upload scanned invoice image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image_text = extract_text_from_image(uploaded_image)
        st.text_area("Extracted text from image:", image_text, height=200)
        document_data = image_text
        query = st.text_input("Ask a question about the image-based invoice:", "Analyze this invoice")

with vendor_tab:
    vendor_name = st.text_input("Enter vendor name to fetch public information:")
    if vendor_name:
        serp_results = run_serpapi_scraper(vendor_name)
        st.markdown("#### \U0001F4F0 Vendor Public Info")
        for result in serp_results:
            st.write(f"- {result['title']}: {result['link']}")
        vendor_context = "\n".join([result['snippet'] for result in serp_results if 'snippet' in result])
        query = st.text_input("Ask a question using the vendor info:", f"What can you tell me about {vendor_name}?")
        document_data = vendor_context if not document_data else document_data

if st.button("Run Agent") and query:
    response = embed_and_query(document_data, query, vendor_context=vendor_context)
    st.session_state.history.append({"query": query, "response": response})
    save_to_db({"query": query, "response": response})
    st.markdown("#### \U0001F916 Agent Response")
    st.write(response)

if st.session_state.history:
    st.markdown("---")
    st.markdown("### \U0001F4AC Chat History")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. You:** {item['query']}")
        st.markdown(f"**\U0001F916 Agent:** {item['response']}")

    if st.button("\U0001F4BE Export Chat History as JSON"):
        path = save_chat_history_as_json(st.session_state.history)
        st.success(f"Chat history saved to: {path}")
