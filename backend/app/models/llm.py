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

HF_MODEL_URL = "https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

def _decode_base64_image(image_data: str) -> bytes:
    """Strips data URI header and returns raw image bytes."""
    clean_b64 = image_data.strip()
    if "," in clean_b64:
        _, clean_b64 = clean_b64.split(",", 1)
    clean_b64 = re.sub(r"\s+", "", clean_b64)
    return base64.b64decode(clean_b64)

def detect_plant_disease(image_data: str, confidence_threshold: float = 0.45) -> dict:
    """
    Classifies crop leaf images via lightweight MobileNetV2.
    Returns: {"label": str, "confidence": float, "is_confident": bool}
    """
    try:
        image_bytes = _decode_base64_image(image_data)
    except Exception as e:
        print(f"[CV CLASSIFIER ERROR] Invalid base64 image: {e}")
        return {"label": None, "confidence": 0.0, "is_confident": False}

    hf_token = os.getenv("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        response = requests.post(
            HF_MODEL_URL,
            headers=headers,
            data=image_bytes,
            timeout=10
        )
        if response.status_code == 200:
            predictions = response.json()
            if isinstance(predictions, list) and len(predictions) > 0:
                top = predictions[0]
                raw_label = top.get("label", "").replace("___", " - ").replace("_", " ")
                score = float(top.get("score", 0.0))
                print(f">>> [CV MODEL DETECTED]: '{raw_label}' (Confidence: {score:.2%})")
                return {
                    "label": raw_label,
                    "confidence": round(score, 3),
                    "is_confident": score >= confidence_threshold
                }
        else:
            print(f"[CV API WARNING] Status {response.status_code}: {response.text[:120]}")
    except Exception as e:
        print(f"[CV CLASSIFIER FALLBACK] Request failed: {e}")

    return {"label": None, "confidence": 0.0, "is_confident": False}

