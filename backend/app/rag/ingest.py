import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = PROJECT_ROOT / "data" / "documents"
DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"

def ingest_pdfs():
    print(f"Reading PDFs from: {DOCS_DIR}")
    loader = PyPDFDirectoryLoader(str(DOCS_DIR))
    raw_docs = loader.load()

    if not raw_docs:
        print("No PDFs found! Please verify documents inside data/documents/.")
        return

    print(f"Loaded {len(raw_docs)} total pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} searchable chunks.")

    print("Generating local embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    print("Ingestion complete. ChromaDB ready at:", DB_DIR)

if __name__ == "__main__":
    ingest_pdfs()