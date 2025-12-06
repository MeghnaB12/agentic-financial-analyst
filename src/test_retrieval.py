import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env variables
load_dotenv()

DB_PATH = "./chroma_db"

def test_query(query_text):
    print(f"\n==================================================")
    print(f"Testing Query: '{query_text}'")
    print(f"==================================================")
    
    # Initialize the same embedding model we used for ingestion
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the DB
    if not os.path.exists(DB_PATH):
        print("Error: DB not found. Run ingest.py first.")
        return

    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # Perform a similarity search (Get top 3 matches)
    results = db.similarity_search_with_score(query_text, k=3)
    
    if not results:
        print("[FAIL] No results found.")
        return

    # Print the results
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} [Score: {score:.4f}]")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...") # Show first 200 chars
        print("-" * 50)

if __name__ == "__main__":
    # Test specific questions from the assignment to see if data exists
    test_query("common practice analysis market penetration")
    test_query("investment analysis financial viability")
    test_query("rice husk combustion technology")