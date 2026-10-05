import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# =====================================================================
# PREVIOUS GROQ IMPLEMENTATION (COMMENTED OUT TO AVOID RATE LIMITS / TPM)
# =====================================================================
# def get_text_llm():
#     """Text and voice queries via Groq."""
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         raise ValueError("GROQ_API_KEY is not set in .env file.")
#     return ChatGroq(
#         model="openai/gpt-oss-120b",
#         groq_api_key=groq_api_key,
#         temperature=0.2,
#     )
def get_text_llm():
    """
    High-speed text & advisory generation using Groq.
    Using 'openai/gpt-oss-20b' prevents the 8k TPM limits of the 120b model
    while remaining fully active on your account.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in .env file.")
        
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0.0,
        max_tokens=2048,
    )

# Maintain aliases so all existing imports in agri_agents.py work smoothly
get_chat_model = get_text_llm

# =====================================================================
# ACTIVE LOCAL OLLAMA IMPLEMENTATION (UNLIMITED CONTEXT, NO RATE LIMITS)
# =====================================================================
# def get_text_llm():
#     """
#     Text and voice queries via local Ollama instance.
#     Defaults to 'llama3.1:8b' or whatever model is defined in .env.
#     """
#     ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#     ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
#     return ChatOllama(
#         model=ollama_model,
#         base_url=ollama_base_url,
#         temperature=0.2,
#     )

def get_vision_llm():
    """Leaf and crop image diagnosis via Gemini Vision API."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.0,
       max_output_tokens=3072,
    )

def get_llm(has_image: bool = False):
    """
    Routes to Gemini Vision if an image is attached,
    otherwise routes to local Ollama for unlimited RAG text generation.
    """
    if has_image:
        return get_vision_llm()
    return get_text_llm()