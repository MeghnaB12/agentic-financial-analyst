import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# Load environment variables
load_dotenv()

DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def load_documents():
    documents = []
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data directory '{DATA_PATH}' not found.")
        return []

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    print(f"[INFO] Found {len(pdf_files)} PDF files in '{DATA_PATH}'")

    for file in pdf_files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"[INFO] Loading: {file}")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file 
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Could not load {file}: {e}")
    return documents

def split_documents(documents):
    print(f"[INFO] Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"[INFO] Cleared existing database at {DB_PATH}")

    print("[INFO] Saving to ChromaDB...")
    start_time = time.time()
    
    # Using the stable HuggingFace embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    end_time = time.time()
    print(f"[SUCCESS] Saved to {DB_PATH}. Time taken: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        save_to_chroma(chunks)