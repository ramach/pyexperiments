import getpass

import chromadb
import openai
import pandas as pd
import os

def main():
    """Runnable testable main function."""

    # Set your OpenAI API key from environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

    # Sample tabular data
    data = {
        'product_id': [1, 2, 3, 4],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'description': [
            'High-performance laptop for professionals',
            'Ergonomic mouse for comfortable use',
            'Mechanical keyboard with RGB lighting',
            '4K monitor with HDR support',
        ],
        'price': [1200, 25, 80, 300],
    }

    df = pd.DataFrame(data)

    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create a collection
    collection = client.create_collection(name="product_collection_openai_test")

    # Function to generate OpenAI embeddings
    def get_openai_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        try:
            response = openai.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None  # Return None in case of error

    # Chunk and add data to the collection
    for index, row in df.iterrows():
        text_chunk = f"Product: {row['product_name']}, Description: {row['description']}, Price: ${row['price']}"
        embedding = get_openai_embedding(text_chunk)

        if embedding is not None:  # Check if embedding was generated successfully
            collection.add(
                embeddings=[embedding],
                ids=[f"product_{row['product_id']}"],
                metadatas=[
                    {
                        "product_id": row["product_id"],
                        "product_name": row["product_name"],
                        "price": row["price"],
                    }
                ],
                documents=[text_chunk],
            )

    # Example query
    query_text = "Find a high-quality monitor"
    query_embedding = get_openai_embedding(query_text)

    if query_embedding is not None:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
        )
        print("Query Results:")
        print(results)
    else:
        print("Query embedding generation failed, cannot run query.")

    # Cleanup
    client.delete_collection("product_collection_openai_test")
    print("Collection deleted.")

if __name__ == "__main__":
    main()