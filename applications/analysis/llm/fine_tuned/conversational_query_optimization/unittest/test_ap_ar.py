import sys
import os

# --- START DIAGNOSTIC ---
# Calculate the absolute path to the directory containing apar_agent.py
# Assuming test_apar_agent.py is in the SAME directory:
source_dir = os.path.abspath(os.path.dirname(__file__))
# If test_apar_agent.py is in tests/ and apar_agent.py is in src/:
# source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# If test_apar_agent.py is in tests/ and apar_agent.py is in the root:
# source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(f"Attempting to add source directory to path: {source_dir}") # Debug print
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)
print("Updated sys.path:", sys.path) # Debug print

import pytest
from unittest.mock import patch, MagicMock, ANY # ANY helps match arguments flexibly
import sqlite3
from langchain_community.utilities import SerpAPIWrapper # Using SerpApi for search
from datetime import datetime, timedelta
import pandas as pd # Required if tools return DataFrames

# Import functions and classes to be tested from your main file
from .ap_ar_utils import (
    _run_query,
    get_internal_ap_data,
    get_internal_ar_data,
    search_public_data,
    agent_node,
    APARState,
    APARErrorState,
    # Make sure vector_store_retriever is accessible for mocking,
    # or refactor setup_vector_store to return it if needed for tests
)

from ..ap_ar_with_tokenization import (
    setup_vector_store,
    optimize_query_node,
    check_for_direct_answer,
    vector_store_retriever,
    SAMPLE_DOCS,
)
# Import LangChain classes used for mocking return values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.utilities import SerpAPIWrapper


# --- Fixtures for Mocking ---

@pytest.fixture(autouse=True) # Automatically use this fixture for all tests
def mock_env_vars(monkeypatch):
    """Mock environment variables to avoid real API calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
    monkeypatch.setenv("SERPAPI_API_KEY", "fake_serpapi_key")

@pytest.fixture
def mock_llm():
    """Fixture for a mock ChatOpenAI LLM."""
    mock = MagicMock(spec=ChatOpenAI)
    # Configure default behavior or specific responses later in tests
    return mock

@pytest.fixture
def mock_retriever():
    """Fixture for a mock vector store retriever."""
    mock = MagicMock()
    # Configure invoke method later in tests
    return mock

@pytest.fixture
def mock_serpapi():
    """Fixture for a mock SerpAPIWrapper."""
    mock = MagicMock(spec=SerpAPIWrapper)
    # Configure run method later in tests
    return mock

# --- Test Database Helper ---

@patch('ap_ar_utils.sqlite3.connect') # Patch where connect is looked up
def test_run_query_success(mock_connect):
    """Test _run_query successfully fetching data."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Simulate fetchall returning rows that can be dict-like
    mock_row1 = {'col1': 'val1', 'col2': 10}
    mock_row2 = {'col1': 'val2', 'col2': 20}
    # Mock row_factory behavior (returning dict-like objects)
    mock_cursor.fetchall.return_value = [mock_row1, mock_row2]
    # Make sure the Row objects can be converted to dict
    mock_cursor.description = [('col1',), ('col2',)] # Needed for dict conversion if row_factory mocked differently


    query = "SELECT col1, col2 FROM some_table WHERE id = ?"
    params = (1,)
    result = _run_query(query, params)

    mock_connect.assert_called_once_with(ANY) # Check connect called
    mock_conn.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once_with(query, params)
    mock_cursor.fetchall.assert_called_once()
    mock_conn.close.assert_called_once()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {'col1': 'val1', 'col2': 10}
    assert result[1] == {'col1': 'val2', 'col2': 20}

@patch('sqlite3.connect')
def test_run_query_no_results(mock_connect):
    """Test _run_query when no data is found."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [] # No rows found

    result = _run_query("SELECT * FROM empty_table")

    assert result == []
    mock_cursor.execute.assert_called_once_with("SELECT * FROM empty_table", ())

@patch('apar_agent.sqlite3.connect')
def test_run_query_db_error(mock_connect):
    """Test _run_query handling a database error."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = sqlite3.Error("Simulated DB Error") # Raise error

    result = _run_query("SELECT * FROM bad_query")

    assert isinstance(result, APARErrorState)
    assert "Database error" in result.message
    assert "Simulated DB Error" in result.message

# --- Test Internal Data Tools ---

@patch('..util.ap_ar_utils._run_query') # Patch the helper function used by the tool
def test_get_internal_ap_data_success(mock_run_query):
    """Test get_internal_ap_data finds vendor and invoices."""
    vendor_name = "Tech Solutions Inc."
    mock_vendor_info = [{'vendor_id': 'V1001', 'contact_email': 'ap@tech.com', 'payment_terms': 'NET 30'}]
    mock_invoice_info = [
        {'invoice_id': 'AP-001', 'invoice_date': '2025-03-21', 'due_date': '2025-04-20', 'amount': 1500.00, 'status': 'Unpaid'},
        {'invoice_id': 'AP-004', 'invoice_date': '2025-02-10', 'due_date': '2025-03-27', 'amount': 5000.00, 'status': 'Overdue'},
    ]

    # Configure mock_run_query to return different results based on call order or query
    mock_run_query.side_effect = [
        mock_vendor_info,  # First call (vendor lookup)
        mock_invoice_info  # Second call (invoice lookup)
    ]

    result = get_internal_ap_data.invoke({"vendor_name": vendor_name}) # Use invoke for tools

    assert mock_run_query.call_count == 2
    assert "Vendor Info:" in result
    assert "V1001" in result
    assert "Unpaid/Overdue AP Invoices:" in result
    assert "AP-001" in result
    assert "AP-004" in result
    assert "1500.0" in result # Check amounts are present

@patch('ap_ar_utils._run_query')
def test_get_internal_ap_data_no_invoices(mock_run_query):
    """Test get_internal_ap_data finds vendor but no invoices."""
    vendor_name = "Office Supplies Co."
    mock_vendor_info = [{'vendor_id': 'V1002', 'contact_email': 'billing@office.com', 'payment_terms': 'NET 15'}]
    mock_invoice_info = [] # No invoices found

    mock_run_query.side_effect = [mock_vendor_info, mock_invoice_info]

    result = get_internal_ap_data.invoke({"vendor_name": vendor_name})

    assert mock_run_query.call_count == 2
    assert "Vendor Info:" in result
    assert "V1002" in result
    assert "No unpaid or overdue AP invoices found" in result

@patch('apar_agent._run_query')
def test_get_internal_ap_data_vendor_not_found(mock_run_query):
    """Test get_internal_ap_data when vendor is not found."""
    vendor_name = "NonExistent Vendor"
    mock_vendor_info = [] # Vendor not found

    mock_run_query.return_value = mock_vendor_info # Only called once

    result = get_internal_ap_data.invoke({"vendor_name": vendor_name})

    mock_run_query.assert_called_once() # Should only be called for vendor lookup
    assert f"Vendor '{vendor_name}' not found" in result

# (Add similar tests for get_internal_ar_data)

# --- Test Public Search Tool ---

@patch('.SerpAPIWrapper') # Patch the external library
def test_search_public_data_success(MockSerpAPIWrapper, mock_serpapi):
    """Test search_public_data successful search."""
    MockSerpAPIWrapper.return_value = mock_serpapi # Ensure constructor returns our mock
    company_name = "Cloud Services LLC"
    search_result_text = "Cloud Services LLC announced new funding round..."
    mock_serpapi.run.return_value = search_result_text

    result = search_public_data.invoke({"company_name": company_name})

    mock_serpapi.run.assert_called_once_with(ANY) # Check run was called
    assert f"Web Search Results for '{company_name}'" in result
    assert search_result_text in result

@patch('apar_agent.SerpAPIWrapper')
def test_search_public_data_no_results(MockSerpAPIWrapper, mock_serpapi):
    """Test search_public_data when API returns no good results (and fallback)."""
    MockSerpAPIWrapper.return_value = mock_serpapi
    company_name = "Obscure Ltd."
    no_result_text = "No good search result found"
    fallback_result_text = "Official Website: obscureltd.com Contact: info@obscureltd.com"

    # Simulate first call returns no results, second call (fallback) returns contact
    mock_serpapi.run.side_effect = [no_result_text, fallback_result_text]

    result = search_public_data.invoke({"company_name": company_name})

    assert mock_serpapi.run.call_count == 2 # Original and fallback query
    assert f"Web Search Results for '{company_name}'" in result
    assert fallback_result_text in result

@patch('apar_agent.SerpAPIWrapper')
def test_search_public_data_api_error(MockSerpAPIWrapper, mock_serpapi):
    """Test search_public_data handling an API error."""
    MockSerpAPIWrapper.return_value = mock_serpapi
    company_name = "Error Prone Inc."
    mock_serpapi.run.side_effect = Exception("Simulated API Failure") # Raise error

    result = search_public_data.invoke({"company_name": company_name})

    assert isinstance(result, APARErrorState)
    assert "Error searching public data" in result.message
    assert "Simulated API Failure" in result.message


# --- Test Vector Store Setup ---

@patch('apar_agent.FAISS')
@patch('apar_agent.OpenAIEmbeddings')
@patch('apar_agent.RecursiveCharacterTextSplitter')
def test_setup_vector_store(mock_splitter_cls, mock_embeddings_cls, mock_faiss_cls, mocker):
    """Test vector store setup mocks dependencies."""
    # Mock instances and their methods
    mock_splitter_instance = MagicMock()
    mock_embeddings_instance = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_retriever_instance = MagicMock()

    mock_splitter_cls.return_value = mock_splitter_instance
    mock_embeddings_cls.return_value = mock_embeddings_instance
    mock_faiss_cls.from_documents.return_value = mock_faiss_instance
    mock_faiss_instance.as_retriever.return_value = mock_retriever_instance

    # Mock the global variable using mocker if it's global in apar_agent
    mocker.patch('apar_agent.vector_store_retriever', None)

    # Run the setup
    setup_vector_store()

    # Assertions
    mock_splitter_cls.assert_called_once()
    mock_embeddings_cls.assert_called_once()
    mock_splitter_instance.split_documents.assert_called_once_with(SAMPLE_DOCS)
    mock_faiss_cls.from_documents.assert_called_once()
    mock_faiss_instance.as_retriever.assert_called_once()

    # Check if the global (or returned) retriever is set
    # This depends on how vector_store_retriever is managed in apar_agent.py
    # If it's global and directly modified:
    from ..ap_ar_with_tokenization import vector_store_retriever as global_retriever
    assert global_retriever is mock_retriever_instance

# --- Test Optimizer Node ---

@patch('apar_agent.ChatOpenAI') # Patch the LLM
@patch('apar_agent.vector_store_retriever') # Patch the global retriever
def test_optimize_query_node_direct_answer(mock_retriever, MockChatOpenAI, mock_llm):
    """Test optimizer finds direct answer from docs."""
    MockChatOpenAI.return_value = mock_llm # Constructor returns mock

    # Mock retriever
    retrieved_docs = [Document(page_content="Payment Terms: NET 30.", metadata={"source": "contracts"})]
    mock_retriever.invoke.return_value = retrieved_docs

    # Mock LLM response (JSON format as expected by the node)
    direct_answer_json = '{"optimized_query_or_answer": "[Answer from Documents] The payment terms are NET 30.", "direct_answer_found": true}'
    mock_llm.invoke.return_value = AIMessage(content=direct_answer_json)

    # Initial state
    initial_state: APARState = {
        "user_prompt": "What are the payment terms?",
        "messages": [HumanMessage(content="What are the payment terms?")],
        # Fill other keys with None or defaults
        "optimized_prompt": None, "retrieved_docs_content": None, "direct_answer_found": False,
        "entity_name": None, "entity_type": None, "internal_data": None, "public_data": None,
        "analysis_result": None, "error_message": None
    }

    final_state = optimize_query_node(initial_state)

    mock_retriever.invoke.assert_called_once_with("What are the payment terms?")
    mock_llm.invoke.assert_called_once() # Check LLM was called
    assert final_state["direct_answer_found"] is True
    assert final_state["optimized_prompt"] == "[Answer from Documents] The payment terms are NET 30."
    assert final_state["analysis_result"] == "[Answer from Documents] The payment terms are NET 30."
    # Check if AIMessage was added
    assert isinstance(final_state["messages"][-1], AIMessage)
    assert final_state["messages"][-1].content == "[Answer from Documents] The payment terms are NET 30."


@patch('apar_agent.ChatOpenAI')
@patch('apar_agent.vector_store_retriever')
def test_optimize_query_node_optimizes_query(mock_retriever, MockChatOpenAI, mock_llm):
    """Test optimizer refines the query for the agent."""
    MockChatOpenAI.return_value = mock_llm

    # Mock retriever (maybe returns irrelevant docs or none)
    mock_retriever.invoke.return_value = []

    # Mock LLM response
    optimized_query_json = '{"optimized_query_or_answer": "Retrieve internal AP data for vendor Tech Solutions Inc.", "direct_answer_found": false}'
    mock_llm.invoke.return_value = AIMessage(content=optimized_query_json)

    initial_state: APARState = {
        "user_prompt": "Status Tech Solutions?",
        "messages": [HumanMessage(content="Status Tech Solutions?")],
        "optimized_prompt": None, "retrieved_docs_content": None, "direct_answer_found": False,
        "entity_name": None, "entity_type": None, "internal_data": None, "public_data": None,
        "analysis_result": None, "error_message": None
    }

    final_state = optimize_query_node(initial_state)

    mock_retriever.invoke.assert_called_once_with("Status Tech Solutions?")
    mock_llm.invoke.assert_called_once()
    assert final_state["direct_answer_found"] is False
    assert final_state["optimized_prompt"] == "Retrieve internal AP data for vendor Tech Solutions Inc."
    assert final_state["analysis_result"] is None # Should not be set yet

# (Add tests for optimizer LLM errors, JSON parsing errors, retrieval errors)

# --- Test Conditional Logic ---

def test_check_for_direct_answer_true():
    """Test conditional check when direct answer IS found."""
    state = {"direct_answer_found": True}
    assert check_for_direct_answer(state) == "end_workflow"

def test_check_for_direct_answer_false():
    """Test conditional check when direct answer IS NOT found."""
    state = {"direct_answer_found": False, "optimized_prompt": "Some query"}
    assert check_for_direct_answer(state) == "continue_to_agent"

def test_check_for_direct_answer_false_no_prompt():
    """Test conditional check handles missing prompt after failed optimization."""
    state = {"direct_answer_found": False, "optimized_prompt": None, "user_prompt": None}
    assert check_for_direct_answer(state) == "end_workflow" # Or error state

# --- Test Agent Node (More Complex) ---

# Mocking the AgentExecutor and its invoke method is often easier for unit tests
@patch('apar_agent.AgentExecutor')
@patch('apar_agent.ChatOpenAI') # Need to mock the LLM constructor potentially
@patch('apar_agent.create_openai_functions_agent') # Mock agent creation
def test_agent_node_success(mock_create_agent, MockChatOpenAI, MockAgentExecutor, mock_llm):
    """Test agent node successfully gets answer via tool."""
    # Mock LLM used inside agent_node
    MockChatOpenAI.return_value = mock_llm

    # Mock Agent creation (optional, depends if you test agent internals)
    mock_agent_instance = MagicMock()
    mock_create_agent.return_value = mock_agent_instance

    # Mock AgentExecutor instance and its invoke method
    mock_executor_instance = MagicMock()
    MockAgentExecutor.return_value = mock_executor_instance # Constructor return
    agent_final_output = "Here is the status for Tech Solutions Inc: Invoice AP-001 is Unpaid, due 2025-04-20."
    mock_executor_instance.invoke.return_value = {"output": agent_final_output}

    initial_state: APARState = {
        "user_prompt": "Status Tech Solutions?", # Original
        "optimized_prompt": "Retrieve internal AP data for vendor Tech Solutions Inc.", # Optimized
        "messages": [
            HumanMessage(content="Status Tech Solutions?")
        ],
        "retrieved_docs_content": None, "direct_answer_found": False,
        "entity_name": None, "entity_type": None, "internal_data": None, "public_data": None,
        "analysis_result": None, "error_message": None
    }

    final_state = agent_node(initial_state)

    # Check that AgentExecutor was invoked correctly
    # It should receive the full message history
    mock_executor_instance.invoke.assert_called_once()
    invoke_args = mock_executor_instance.invoke.call_args[0][0]
    assert "messages" in invoke_args
    assert len(invoke_args["messages"]) == 1 # Should have history
    assert invoke_args["messages"][0].content == "Status Tech Solutions?"


    # Check final state
    assert final_state["analysis_result"] == agent_final_output
    assert isinstance(final_state["messages"][-1], AIMessage)
    assert final_state["messages"][-1].content == agent_final_output


@patch('..util.ap_ar_utils.AgentExecutor')
@patch('..util.ap_ar_utils.ChatOpenAI')
@patch('..util.ap_ar_utils..create_openai_functions_agent')
def test_agent_node_error(mock_create_agent, MockChatOpenAI, MockAgentExecutor, mock_llm):
    """Test agent node handling an execution error."""
    MockChatOpenAI.return_value = mock_llm
    mock_agent_instance = MagicMock()
    mock_create_agent.return_value = mock_agent_instance
    mock_executor_instance = MagicMock()
    MockAgentExecutor.return_value = mock_executor_instance

    # Simulate agent execution raising an error
    error_message = "Simulated agent execution failure"
    mock_executor_instance.invoke.side_effect = Exception(error_message)

    initial_state: APARState = {
        "user_prompt": "Some query",
        "optimized_prompt": "Optimized query",
        "messages": [HumanMessage(content="Some query")],
        # ... other fields ...
        "retrieved_docs_content": None, "direct_answer_found": False,
        "entity_name": None, "entity_type": None, "internal_data": None, "public_data": None,
        "analysis_result": None, "error_message": None
    }

    final_state = agent_node(initial_state)

    mock_executor_instance.invoke.assert_called_once()
    assert final_state["error_message"] is not None
    assert error_message in final_state["error_message"]
    # Check error message added to history
    assert isinstance(final_state["messages"][-1], AIMessage)
    assert "Sorry, I encountered an error" in final_state["messages"][-1].content
    assert error_message in final_state["messages"][-1].content
    assert final_state["analysis_result"] == final_state["messages"][-1].content
