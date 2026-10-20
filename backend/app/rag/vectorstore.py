import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

import chromadb
from langchain_core.tools import tool

DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"

@tool
def search_local_handbooks(query: str) -> str:
    """
    PRIMARY SEARCH TOOL.
    Searches official agricultural package of practices and ICAR crop disease manuals
    directly from the Chroma SQLite store without loading heavy PyTorch models.
    """
    if not DB_DIR.exists():
        return "NO_LOCAL_DATA_FOUND"

    try:
        # Connect directly to the SQLite Chroma store without loading PyTorch or HuggingFace
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collections = client.list_collections()
        if not collections:
            return "NO_LOCAL_DATA_FOUND"

        # Use the existing collection created by ingest.py
        collection = collections[0]
        
        # Tokenize query keywords (e.g. potato, blight, variety, spray)
        clean_words = [w.strip() for w in query.lower().split() if len(w.strip()) > 3]
        if not clean_words:
            clean_words = [w.strip() for w in query.lower().split() if len(w.strip()) > 1]

        results = []
        
        # Search the document chunks directly using keyword matching
        for word in clean_words[:4]:
            matched = collection.get(
                where_document={"$contains": word},
                limit=2,
                include=["documents", "metadatas"]
            )
            docs = matched.get("documents", [])
            metas = matched.get("metadatas", [])
            for doc, meta in zip(docs, metas):
                src = Path(meta.get("source", "PDF Manual")).name if meta else "PDF Manual"
                entry = f"[{src}]\n{doc.strip()}"
                if entry not in results:
                    results.append(entry)
            if len(results) >= 3:
                break

        if results:
            return "\n\n---\n\n".join(results[:3])

        # Fallback: Retrieve representative handbook excerpts if no exact keyword match
        sample = collection.get(limit=2, include=["documents", "metadatas"])
        if sample.get("documents"):
            for doc, meta in zip(sample["documents"], sample.get("metadatas", [])):
                src = Path(meta.get("source", "PDF Manual")).name if meta else "PDF Manual"
                results.append(f"[{src}]\n{doc.strip()}")
            return "\n\n---\n\n".join(results)

        return "NO_LOCAL_DATA_FOUND"

    except Exception as e:
        print(f"[CHROMA RETRIEVAL ERROR]: {e}")
        return "NO_LOCAL_DATA_FOUND"
        