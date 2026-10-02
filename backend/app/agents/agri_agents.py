import sys
from pathlib import Path
import os
import re
import base64
import tempfile
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.llm import get_llm, get_text_llm
from backend.app.rag.vectorstore import search_local_handbooks

from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from groq import Groq

SYSTEM_INSTRUCTION = """You are 'Kisan Mitra' (किसान मित्र), an expert agronomist and friendly conversational crop advisor.
You assist farmers, agricultural officers, and growers with practical, field-tested guidance.

STRICT CONVERSATIONAL & OUTPUT GUARDRAILS:
1. Speak DIRECTLY to the farmer in natural, supportive dialogue.
2. NEVER output internal reasoning, thought process, meta-explanations, or raw JSON tool calls.
3. NEVER say things like "Since the user asked...", "I will call function...", or "Here is the JSON object:".
4. GREETINGS & SHORT INPUTS:
   - If the user says "hello", "hi", "namaste", "नमस्कार", or gives a general greeting, reply warmly as Kisan Mitra in their language and politely ask which crop, disease, or farming issue they need assistance with today. Do not trigger technical advisories on bare greetings.

VOICE & AUDIO (SPEECH-FRIENDLY) OPTIMIZATION:
1. Since responses may be spoken aloud via Text-to-Speech (TTS) to farmers:
   - Use natural conversational phrasing, short sentences, and clean punctuation (periods and commas) to ensure smooth audio breathing pauses.
   - Do NOT use heavy markdown nesting, complex tables, or excessive symbols (avoid '/', '~', multiple asterisks, or raw math symbols).
   - Write dosages and units phonetically and clearly (e.g., write "2 grams per liter of water" or "दोन ग्रॅम प्रति लिटर पाणी" rather than "2g/L").
   - Spell out abbreviations where ambiguity exists (e.g., say "कृषी विज्ञान केंद्र" instead of just "KVK").
2. SPEECH-TO-TEXT ROBUSTNESS:
   - Recognize that user queries coming from voice microphones may have minor transcription slips, dialect variations, or missing punctuation. Interpret the intended agricultural context generously.

MULTIMODAL VISION & CROP DIAGNOSIS RULES:
When an image of a leaf, fruit, pest, or crop is provided:
1. ANATOMICAL & SYMPTOM LOCATION VERIFICATION:
   - Always determine the exact anatomical position of damage relative to the fruit stem/calyx attachment:
     * In tomatoes, peppers, and eggplants:
       - If a sunken, leathery, or blackened patch is strictly at the BOTTOM tip (opposite the stem) -> Diagnose BLOSSOM END ROT (calcium mobility or moisture stress).
       - If a bleached, papery, yellowish-white, or blistered leathery lesion is on the UPPER SHOULDER or SIDE wall facing sunlight -> Diagnose SUNSCALD (solar heat damage / loss of leaf foliage canopy), NOT Blossom End Rot.
     * In leaves and foliage:
       - Concentric target-board rings with yellow halos -> Early Blight (लवकर येणारा करपा).
       - Water-soaked dark lesions spreading rapidly with white fungal growth underneath in cool/humid weather -> Late Blight (उशिरा येणारा करपा).
       - Interveinal chlorosis, leaf curl virus, or mosaic patterns -> Differentiate nutrient deficiencies from viral/sucking pest attacks.
2. Provide a structured visual advisory:
   - Symptoms Identified (लक्षणे)
   - Probable Cause (संभाव्य कारण - बुरशी, कीड, किंवा शारीरिक विकृती)
   - Recommended Spray / Dosage (शिफारस केलेली फवारणी व प्रमाण)
   - Cultural Management & Preventive Steps (प्रतिबंधात्मक उपाय)

CORE OBJECTIVES:
1. Converse naturally in an easy-to-understand, friendly tone while maintaining scientific accuracy.
2. Respond in the EXACT same language or dialect as the user's message (e.g., Marathi, Hindi, English, Hinglish).
3. Seamlessly handle diverse topics: crop varieties, seed rate, sowing time, soil preparation, symptoms, fertilizer scheduling, pest/disease management, and follow-up queries.
4. DYNAMIC FORMATTING:
   - For variety/general inquiries: Provide concise, bulleted explanations highlighting yield, maturity duration, and regional suitability.
   - For pest/disease/spray queries: Provide clear actionable steps including Symptoms, Dosage per Liter of Water or per 15-liter pump, Application timing, and Safety waiting periods.
   - Use clean bolding and bullet points for high readability on mobile screens.
5. STRICT PRIVACY & SOURCE CLEANLINESS:
   - NEVER mention file names, PDF paths, database IDs, or internal markers (e.g., do NOT say "According to DiseaseofFieldHorticultureCrops.pdf" or "[Source 1]").
   - Present information as direct agronomy expertise or cite official bodies generally (e.g., "कृषी विद्यापीठाच्या शिफारशीनुसार" or "As per ICAR package of practices").

WEATHER & GENERAL CROP CONSULTATION FALLBACK:
- If the user asks about climate, temperature, soil pH, sowing seasons, or crops not detailed in local manuals (e.g., Capsicum / Bell Pepper, Exotic Vegetables, Floriculture):
    * Provide standard ICAR agronomy recommendations directly.
    * Include Day/Night Temperature ranges, Humidity, Rainfall/Sunlight needs, Soil drainage/pH requirements, and Common Risks (e.g., flower drop during extreme heat/cold).

DOSAGE & CHEMICAL ACCURACY GUARDRAILS:
1. BRAND & TECHNICAL NAME PAIRING:
   - Always mention BOTH the popular commercial brand name and its active technical chemical ingredient (e.g., "Nativo containing Tebuconazole 50% plus Trifloxystrobin 25% WG" or "Amistar Top containing Azoxystrobin plus Difenoconazole").
2. EXACT DUAL PUMP MEASURES:
   - Always state pesticide/fungicide dilution rates in two explicit measures:
     * Per 1 Liter of water (e.g., "1 milliliter per liter of water" or "१ मिली प्रति लिटर पाणी").
     * Per standard 15-Liter knapsack pump tank (e.g., "15 milliliters per 15-liter pump" or "१५ मिली प्रति १५ लिटर पंप").
3. SPRAY TIMING & MANDATORY INTERVAL:
   - State the optimal time of day: Early morning (before 9:30 AM) or late afternoon (after 4:30 PM) to avoid high heat evaporation and pollinator harm.
   - Specify a mandatory waiting gap between applications (e.g., "Repeat after 10 to 14 days only if active symptoms persist").
   - NEVER advise daily spraying.
4. EVALUATION MILESTONE (RECOVERY CHECK):
   - Always give the farmer clear, practical indicators to look for after 4 to 5 days:
     * Example: "Check the plants after 4 to 5 days: active fungal lesions should dry up and turn pale grey or brown with no new yellow halos spreading, and new top leaves should emerge clean."
5. FERTILIZER & SEED TREATMENT ACCURACY:
   - NEVER hallucinate micro-dosages for field fertilizers (standard field fertilizers are applied in tens or hundreds of kilograms per hectare).
   - For seed treatment (बीज प्रक्रिया): Recommend standard fungicides like Carbendazim (कार्बेंडाझिम २ ग्रॅम प्रति किलो बियाणे), Mancozeb (मॅन्कोझेब २.५ ग्रॅम प्रति किलो बियाणे), or bio-agents like Trichoderma (ट्रायकोडर्मा ५ ग्रॅम प्रति किलो बियाणे).
   - NEVER confuse insecticides (e.g., Carbaryl) with fungicides/bactericides.
   - If exact fertilizer dosages per hectare are not explicitly retrieved from verified vectorstore handbooks, DO NOT invent numbers—explicitly advise: "खतांचे प्रमाण माती परीक्षणानुसार व स्थानिक कृषी विद्यापीठाच्या शिफारशीनुसार द्यावे (स्थानिक कृषी विज्ञान केंद्राचा सल्ला घ्यावा)."

CONTEXT GROUNDING:
- Answer the user's question using the retrieved agronomy context provided in the conversation.
- If the retrieved context contains relevant crop, pest, or chemical recommendations, prioritize those details accurately.
"""

WHITELISTED_DOMAINS = [
    "agritech.tnau.ac.in",
    "cibrc.gov.in",
    "icar.gov.in",
    "fao.org"
]

web_search_tool = TavilySearch(
    name="search_verified_websites",
    description="SECONDARY FALLBACK. Searches whitelisted university and ICAR agronomy resources.",
    max_results=3,
    include_domains=WHITELISTED_DOMAINS
)

tools = [search_local_handbooks, web_search_tool]
tool_map = {tool.name: tool for tool in tools}

AGRI_KEYWORDS = {
    "रोग", "औषध", "फवारणी", "खत", "कीड", "बियाणे", "पाणी", "लागवड",
    "disease", "spray", "blight", "rot", "dose", "fertilizer", "pest", "seed"
}

GREETINGS = {
    "hi", "hello", "hey", "namaste", "namaskar",
    "नमस्कार", "नमस्ते", "ram ram", "राम राम", "pranam"
}

def is_simple_greeting(text: str) -> bool:
    if not text:
        return False
    clean = text.strip().lower()
    return clean in GREETINGS or clean.startswith(("hi ", "hello ", "hey "))

def needs_rag(query: str) -> bool:
    """Bypasses ChromaDB vector search for casual greetings or non-agri banter."""
    clean_query = query.lower()
    if any(keyword in clean_query for keyword in AGRI_KEYWORDS):
        return True
    return len(query.strip().split()) > 4

def _clean_content(content) -> str:
    """Converts LLM response blocks to a clean text string."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        text = "\n".join(text_parts)
    else:
        text = str(content)
        
    cleaned = re.sub(
        r"^(Since the user|Based on the user|I will respond|As an agronomist|Here is).*?:\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    return cleaned.strip()

def transcribe_audio_base64(audio_base64: str) -> str:
    """Transcribes incoming farmer voice audio using Groq Whisper."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[AUDIO WARNING] GROQ_API_KEY is not set. Skipping audio transcription.")
        return ""
    
    client = Groq(api_key=groq_api_key)

    ext = "webm"
    if "," in audio_base64:
        header, audio_base64 = audio_base64.split(",", 1)
        if "wav" in header:
            ext = "wav"
        elif "mp4" in header or "m4a" in header:
            ext = "m4a"
        elif "ogg" in header:
            ext = "ogg"

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        print(f"[AUDIO ERROR] Base64 decode failed: {e}")
        return ""

    if len(audio_bytes) < 1000:
        print("[AUDIO WARNING] Audio clip was too short or empty.")
        return ""

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(f"recording.{ext}", file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        
        result_text = getattr(transcription, "text", None) or (
            transcription.get("text") if isinstance(transcription, dict) else str(transcription)
        )
        print(f"\n[VOICE TRANSCRIBED SUCCESS]: '{result_text}'\n")
        return result_text.strip()
    except Exception as e:
        print(f"\n[VOICE TRANSCRIPTION ERROR]: {e}\n")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def call_groq(prompt: str, history: list = None) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "⚠️ GROQ_API_KEY is not set in environment."
    client = Groq(api_key=groq_api_key)
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if history:
        for turn in history:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content

def call_ollama(prompt: str, history: list = None, model_name: str = "llama3") -> str:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if history:
        for turn in history:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        else:
            return f"Ollama error: HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is unreachable. Verify Ollama is running and OLLAMA_BASE_URL is reachable."

def run_farmer_assistant(
    query: str = "", 
    language: str = "mr",
    history: list = None, 
    image_data: str = None, 
    audio_data: str = None, 
    max_iterations: int = 4,
    provider: str = "gemini",
    **kwargs
) -> str:
    """Main agricultural advisory agent supporting audio, text, RAG, and image input."""
    print(f">>> [ASSISTANT ENTRY]: query='{query}', lang='{language}', provider='{provider}'")

    # 1. Process voice audio if submitted
    transcribed_text = ""
    if audio_data and isinstance(audio_data, str) and len(audio_data.strip()) > 0:
        print("\n>>> [STEP 1] Transcribing incoming audio with Groq Whisper...")
        transcribed_text = transcribe_audio_base64(audio_data)
        print(f">>> [STEP 2] Transcribed Text: '{transcribed_text}'\n")

    # Merge query text and transcribed audio
    final_query = f"{query.strip()} {transcribed_text.strip()}".strip()

    # Fast fallback if audio was unintelligible and no query was typed
    if not final_query and not image_data:
        fallback_msg = {
            "mr": "माफ करा, तुमचा आवाज स्पष्ट ऐकू आला नाही. कृपया पुन्हा माईक दाबून बोला किंवा टाईप करा.",
            "hi": "क्षमा करें, आपकी आवाज स्पष्ट नहीं आई। कृपया पुनः माइक दबाकर बोलें या टाइप करें।",
            "en": "Sorry, I couldn't hear any speech. Please tap the mic and speak clearly or type your query."
        }
        return fallback_msg.get(language, fallback_msg["mr"])

    # 2. Fast-Path: Simple greetings without image inspection
    if is_simple_greeting(final_query) and not image_data:
        if language == "mr":
            return "नमस्कार शेतकरी बंधू! मी आपला कृषी सल्लागार किसान मित्र आहे. आपल्या पिकाबद्दल किंवा फवारणीबद्दल काय प्रश्न आहे?"
        elif language == "hi":
            return "नमस्ते किसान साथी! मैं आपका कृषि सलाहकार किसान मित्र हूँ। आज आपकी फसल या कीट नियंत्रण में क्या मदद कर सकता हूँ?"
        else:
            return "Hello! I am Kisan Mitra, your crop advisor. How can I assist you with your crops, disease diagnosis, or spray schedules today?"

    has_image = bool(image_data and isinstance(image_data, str) and image_data.startswith("data:image"))

    # 3. Retrieve handbook context via RAG for pure text agricultural topics
    # CLOUD SAFETY: Skip local ChromaDB on image uploads to avoid 512MB RAM OOM crash
    rag_context = ""
    if not has_image and needs_rag(final_query):
        print(">>> [RAG GATE]: Agricultural text query. Querying handbook vectorstore...")
        try:
            if hasattr(search_local_handbooks, "invoke"):
                rag_context = search_local_handbooks.invoke({"query": final_query})
            elif hasattr(search_local_handbooks, "func"):
                rag_context = search_local_handbooks.func(final_query)
            else:
                rag_context = search_local_handbooks(final_query)
        except Exception as e:
            print(f">>> [RAG RETRIEVAL WARNING]: {e}. Continuing with general agronomy knowledge.")
            rag_context = ""
    elif has_image:
        print(">>> [VISION GATE]: Leaf image detected. Relying directly on Gemini Multimodal Pathology.")

    # 4. Prepare language and contextual prompt
    lang_map = {
        "mr": "Respond completely in Marathi (मराठी).",
        "hi": "Respond completely in Hindi (हिन्दी).",
        "en": "Respond completely in English."
    }
    lang_note = lang_map.get(language, "Respond in the language chosen by user.")

    context_block = f"\n\n[Reference Data from Handbooks]:\n{rag_context}" if rag_context else ""
    prompt_with_instructions = (
        f"[Language Requirement: {lang_note}]\n"
        f"Farmer Question: {final_query if final_query else 'कृपया या पानाचे/फळाचे निरीक्षण करून रोग व फवारणी औषधांची मात्रा सांगा.'}"
        f"{context_block}"
    )

    # 5. Route to text-only providers when no image is uploaded
    if not has_image:
        if provider.startswith("ollama"):
            return _clean_content(call_ollama(prompt_with_instructions, history=history))
        elif provider.startswith("groq"):
            return _clean_content(call_groq(prompt_with_instructions, history=history))

    # 6. Multimodal Vision Execution (Gemini)
    llm = get_llm(has_image=has_image)

    messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]

    # Append past conversation turns
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Append current turn
    if has_image:
        user_content = [
            {"type": "text", "text": prompt_with_instructions},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
        messages.append(HumanMessage(content=user_content))
        # Direct multimodal invocation without tool binding conflicts
        response = llm.invoke(messages)
        return _clean_content(response.content)

    # 7. Text-based Tool Calling loop for complex queries
    llm_with_tools = llm.bind_tools(tools)
    messages.append(HumanMessage(content=prompt_with_instructions))

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return _clean_content(response.content)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            try:
                tool_to_use = tool_map.get(tool_name)
                tool_output = tool_to_use.invoke(tool_args) if tool_to_use else f"Unknown tool: {tool_name}"
            except Exception as e:
                tool_output = f"Tool failure: {str(e)}"

            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

    final_response = llm.invoke(messages)
    return _clean_content(final_response.content)