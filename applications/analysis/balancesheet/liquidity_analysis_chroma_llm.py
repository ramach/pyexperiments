import getpass

import chromadb
import openai
import pandas as pd
import os

def main():
    """Liquidity analysis using ChromaDB and OpenAI."""

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

    client = chromadb.Client()
    collection_name = "liquidity_analysis_collection"

    # Sample enterprise data (replace with your actual data)
    data = {
        'date': ['2023-10-26', '2023-10-27', '2023-10-28', '2023-10-29'],
        'cash_flow': [10000, -5000, 15000, -2000],
        'accounts_receivable': [50000, 52000, 48000, 55000],
        'accounts_payable': [30000, 31000, 29000, 32000],
        'report_summary': [
            "Increased sales led to positive cash flow.",
            "Large payment to supplier impacted cash flow.",
            "New client payment boosted cash flow.",
            "Unexpected expense decreased cash flow.",
        ],
    }
    df = pd.DataFrame(data)

    # Create ChromaDB collection or get existing
    try:
        collection = client.create_collection(name=collection_name)
    except ValueError:
        collection = client.get_collection(name=collection_name)

    # OpenAI embedding function
    def get_openai_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        try:
            response = openai.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    # LLM response function
    def get_llm_response(prompt):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "Sorry, I could not generate a response."

    # Ingest data into ChromaDB
    for index, row in df.iterrows():
        text_chunk = f"Date: {row['date']}, Cash Flow: {row['cash_flow']}, AR: {row['accounts_receivable']}, AP: {row['accounts_payable']}, Summary: {row['report_summary']}"
        embedding = get_openai_embedding(text_chunk)
        if embedding:
            collection.add(
                embeddings=[embedding],
                ids=[f"data_{index}"],
                metadatas=row.to_dict(),
                documents=[text_chunk],
            )

    # Liquidity analysis query
    def analyze_liquidity(query):
        query_embedding = get_openai_embedding(query)
        if query_embedding is None:
            return "Could not generate embedding."

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        context = "\n".join(results["documents"][0])
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnalyze the liquidity based on the context."
        return get_llm_response(prompt)

    # Example queries
    print("Query 1: Analyze recent cash flow trends.")
    print(analyze_liquidity("Analyze recent cash flow trends."))

    print("\nQuery 2: What factors impacted liquidity?")
    print(analyze_liquidity("What factors impacted liquidity?"))

    # Cleanup (optional)
    client.delete_collection(collection_name)
    print("\nCollection deleted.")

if __name__ == "__main__":
    main()