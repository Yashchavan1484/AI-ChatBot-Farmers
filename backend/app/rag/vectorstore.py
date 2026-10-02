import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )

@tool
def search_local_handbooks(query: str) -> str:
    """
    PRIMARY SEARCH TOOL.
    Searches official agricultural package of practices, ICAR crop disease manuals, 
    and CIBRC registered pesticide dosage tables stored locally.
    Always check this tool first before using external web search.
    """
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "NO_LOCAL_DATA_FOUND"

        results = []
        for i, doc in enumerate(docs, 1):
            source_file = Path(doc.metadata.get("source", "PDF Manual")).name
            results.append(f"[Source {i}: {source_file}]\n{doc.page_content.strip()}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error reading local vector store: {str(e)}"