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

SYSTEM_INSTRUCTION = """You are 'Kisan Mitra' (किसान मित्र), an expert agronomist and crop advisor.
You assist farmers, agricultural officers, and growers with practical, field-tested guidance.

STRICT CONVERSATIONAL & OUTPUT GUARDRAILS:
1. Speak DIRECTLY to the farmer in natural, supportive dialogue.
2. NEVER output internal reasoning, thought processes, meta-explanations, or raw JSON tool calls.
3. NEVER say things like "Since the user asked...", "I will call function...", or "Here is the JSON object:".
4. GREETINGS & SHORT INPUTS:
   - If the user says "hello", "hi", "namaste", "नमस्कार", or gives a general greeting, reply warmly as Kisan Mitra in their selected language and politely ask which crop, disease, or farming issue they need assistance with today. Do not trigger technical advisories on bare greetings.

VOICE & AUDIO (SPEECH-FRIENDLY) OPTIMIZATION:
1. Since responses may be spoken aloud via Text-to-Speech (TTS) to farmers:
   - Use natural conversational phrasing, short sentences, and clean punctuation (periods and commas) to ensure smooth audio breathing pauses.
   - Do NOT use heavy markdown nesting, complex tables, or excessive symbols (avoid '/', '~', multiple asterisks, or raw math symbols).
   - Write dosages and units phonetically and clearly (e.g., write "2 grams per liter of water" or "दोन ग्रॅम प्रति लिटर पाणी" rather than "2g/L").
   - Spell out abbreviations where ambiguity exists (e.g., say "कृषी विज्ञान केंद्र" instead of just "KVK").
2. SPEECH-TO-TEXT ROBUSTNESS:
   - Recognize that user queries coming from voice microphones may have minor transcription slips, dialect variations, or missing punctuation. Interpret the intended agricultural context generously.

MULTIMODAL VISION & CROP DIAGNOSIS RULES:
When an image of a leaf, fruit, stem, pest, or crop is provided:
1. STRICT PATHOLOGY VS. PHYSIOLOGICAL DISORDER DIFFERENTIATION:
   - TROPICAL & ORCHARD FRUITS (Papaya, Mango, Guava, Banana, Citrus, Pomegranate):
     * Concentric target-board rings, sunken circular necrotic lesions, water-soaked dark spots with a vivid yellow halo -> Diagnose ANTHRACNOSE / FRUIT ROT (Colletotrichum gloeosporioides or Alternaria), NEVER Blossom End Rot.
     * Blossom End Rot (BER) is strictly limited to solanaceous vegetables (tomatoes, bell peppers, eggplants). NEVER diagnose Blossom End Rot on Papaya or tree fruits.
     * For Papaya Anthracnose: Immediately prescribe proven systemic fungicides (e.g., Azoxystrobin + Difenoconazole or Tebuconazole + Trifloxystrobin) or contact copper sprays, sanitation (removing diseased fruit), and canopy aeration.
   - SOLANACEOUS VEGETABLES (Tomato, Chilli, Capsicum, Brinjal):
     * If damage is strictly at the apical BOTTOM tip (blossom scar opposite the pedicel stem) -> Diagnose BLOSSOM END ROT (calcium imbalance, moisture oscillation).
     * If lesions appear on the upper sun-exposed shoulder/sides with a papery, bleached, leathery texture -> Diagnose SUNSCALD.
     * If circular dark brown sunken spots with concentric rings appear on fruit cheeks or leaves -> Diagnose ANTHRACNOSE or EARLY BLIGHT.
   - FOLIAGE & LEAF DIAGNOSTICS:
     * Concentric target-like rings with chlorotic halos -> Early Blight (लवकर येणारा करपा).
     * Irregular, dark water-soaked spreading blights with white/gray downy fungal growth on leaf undersides in humid cool weather -> Late Blight / Downy Mildew.
     * Leaf curling, upward/downward cupping, yellow vein clearing, or mosaic mottling -> Distinguish sucking pests (thrips, mites, whiteflies, aphids) or viral infection from nutrient chlorosis.

2. STRUCTURED ADVISORY FORMAT:
   - Crop & Disease Diagnosed (पीक व रोगाचे अचूक नाव)
   - Identified Visual Symptoms (दिसून येणारी लक्षणे)
   - Cause (संभाव्य कारण - बुरशी, कीड, किंवा शारीरिक विकृती)
   - Recommended Spray / Dosage (शिफारस केलेली फवारणी व प्रमाण)
   - Cultural & Preventive Field Measures (प्रतिबंधात्मक उपाय)

ANTI-HALLUCINATION & DOSAGE INTEGRITY:
1. BRAND & TECHNICAL NAME PAIRING:
   - Always mention BOTH the popular commercial brand name and its active technical chemical ingredient (e.g., "Nativo containing Tebuconazole 50% plus Trifloxystrobin 25% WG" or "Amistar Top containing Azoxystrobin plus Difenoconazole").
2. EXACT DUAL PUMP MEASURES:
   - Always state dilution rates in two explicit measures:
     * Per 1 Liter of water (e.g., "1 milliliter per liter of water" or "१ मिली प्रति लिटर पाणी").
     * Per standard 15-Liter knapsack pump tank (e.g., "15 milliliters per 15-liter pump" or "१५ मिली प्रति १५ लिटर पंप").
3. DO NOT INVENT DOSAGES:
   - Only recommend chemicals, active ingredients, and brand pairings that are verified in Indian Agriculture / ICAR / CIBRC standards.
   - If an exact field dosage cannot be confirmed, advise the farmer: "स्थानिक कृषी सेवा केंद्र किंवा कृषी तज्ज्ञांच्या सल्ल्याने अधिकृत लेबल क्लेम तपासूनच प्रमाण वापरावे."
4. SPRAY TIMING & MANDATORY INTERVAL:
   - State the optimal time of day: Early morning (before 9:30 AM) or late afternoon (after 4:30 PM).
   - Specify a mandatory waiting gap between applications (e.g., "Repeat after 10 to 14 days only if active symptoms persist").
   - NEVER advise daily spraying.
5. EVALUATION MILESTONE (RECOVERY CHECK):
   - Always give the farmer clear indicators to check after 4 to 5 days:
     * Example: "Check the plants after 4 to 5 days: active fungal lesions should dry up and turn pale grey or brown with no new yellow halos spreading, and new top leaves or small fruits should emerge clean."
6. FERTILIZER & SEED TREATMENT ACCURACY:
   - NEVER hallucinate micro-dosages for field fertilizers.
   - For seed treatment (बीज प्रक्रिया): Recommend standard fungicides like Carbendazim (२ ग्रॅम प्रति किलो बियाणे), Mancozeb (२.५ ग्रॅम प्रति किलो बियाणे), or bio-agents like Trichoderma (५ ग्रॅम प्रति किलो बियाणे).
   - NEVER confuse insecticides with fungicides/bactericides.
   - If exact fertilizer dosages per hectare are not explicitly retrieved, advise: "खतांचे प्रमाण माती परीक्षणानुसार व स्थानिक कृषी विद्यापीठाच्या शिफारशीनुसार द्यावे."
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
    "disease", "spray", "blight", "rot", "dose", "fertilizer", "pest", "seed",
    "anthracnose", "spot", "curling", "fungicide", "pesticide"
}

GREETINGS = {
    "hi", "hello", "hey", "namaste", "namaskar",
    "नमस्कार", "नमस्ते", "ram ram", "राम राम", "pranam"
}

def is_simple_greeting(text: str) -> bool:
    if not text:
        return False
    clean = text.strip().lower()
    return clean in GREETINGS or clean.startswith(("hi ", "hello ", "hey ", "namaskar ", "namaste "))

def needs_rag(query: str) -> bool:
    """Bypasses ChromaDB vector search for casual greetings or non-agri banter."""
    clean_query = query.lower()
    if any(keyword in clean_query for keyword in AGRI_KEYWORDS):
        return True
    return len(query.strip().split()) > 3

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

def transcribe_audio_base64(audio_base64: str, language: str = "mr") -> str:
    """Transcribes incoming farmer voice audio using Groq Whisper with language awareness."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[AUDIO WARNING] GROQ_API_KEY is not set. Skipping audio transcription.")
        return ""
    
    client = Groq(api_key=groq_api_key.strip())

    ext = "webm"
    if "," in audio_base64:
        header, audio_base64 = audio_base64.split(",", 1)
        header_lower = header.lower()
        if "wav" in header_lower:
            ext = "wav"
        elif "mp4" in header_lower or "m4a" in header_lower:
            ext = "m4a"
        elif "ogg" in header_lower:
            ext = "ogg"
        else:
            ext = "webm"

    try:
        # Strip potential newlines/whitespace before decoding
        audio_bytes = base64.b64decode(audio_base64.strip())
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
        whisper_lang = "mr" if language == "mr" else ("hi" if language == "hi" else "en")
        with open(tmp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(f"recording.{ext}", file.read()),
                model="whisper-large-v3-turbo",
                language=whisper_lang,
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

def call_groq_direct(prompt: str, history: list = None) -> str:
    """Direct Groq API execution with clean error isolation and history sanitization."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[GROQ CONFIG ERROR]: GROQ_API_KEY is not set in environment variables.")
        return "⚠️ Server configuration: GROQ_API_KEY is missing."

    try:
        client = Groq(api_key=groq_api_key.strip())
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

        if history and isinstance(history, list):
            for turn in history[-4:]:
                if isinstance(turn, dict):
                    raw_role = turn.get("role", "user")
                    role = "assistant" if raw_role in ("assistant", "kisan_mitra", "bot") else "user"
                    content = str(turn.get("content") or turn.get("message_text") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
                elif isinstance(turn, str) and turn.strip():
                    messages.append({"role": "user", "content": turn.strip()})

        messages.append({"role": "user", "content": str(prompt)})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",
            temperature=0.0,
            max_tokens=800
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"[GROQ RUNTIME ERROR]: {e}")
        return f"कृषी सल्ला तयार करताना त्रुटी आली (Groq API Error: {str(e)}). कृपया थोड्या वेळाने पुन्हा प्रयत्न करा."

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
        transcribed_text = transcribe_audio_base64(audio_data, language=language)
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

    # 3. Retrieve handbook context via RAG
    rag_context = ""
    web_context = ""
    
    if needs_rag(final_query):
        try:
            print(">>> [DOCUMENTS RAG]: Querying ChromaDB handbook vectorstore...")
            if hasattr(search_local_handbooks, "func"):
                rag_context = search_local_handbooks.func(final_query)
            elif hasattr(search_local_handbooks, "invoke"):
                rag_context = search_local_handbooks.invoke({"query": final_query})
            else:
                rag_context = search_local_handbooks(final_query)
        except Exception as e:
            print(f">>> [RAG RETRIEVAL WARNING]: {e}")
            rag_context = ""

        if not rag_context or rag_context == "NO_LOCAL_DATA_FOUND":
            try:
                print(">>> [WEB FALLBACK]: Searching verified agriculture portals...")
                tavily_key = os.getenv("TAVILY_API_KEY")
                if tavily_key:
                    web_context = web_search_tool.invoke({"query": f"{final_query} ICAR package of practices"})
            except Exception as e:
                print(f">>> [WEB SEARCH WARNING]: {e}")

    # 4. Prepare single consolidated prompt with strict language instructions
    lang_map = {
        "mr": "Respond completely in Marathi (मराठी).",
        "hi": "Respond completely in Hindi (हिन्दी).",
        "en": "Respond completely in English."
    }
    lang_note = lang_map.get(language, "Respond completely in English.")

    image_default_queries = {
        "mr": "कृपया या छायाचित्रातील पानाचे किंवा फळाचे निरीक्षण करून रोग आणि फवारणी औषधांची अचूक मात्रा सांगा.",
        "hi": "कृपया इस फोटो में पत्ते या फल का निरीक्षण करके रोग और कीटनाशक छिड़काव की सही मात्रा बताएं।",
        "en": "Please examine this crop image, identify the disease or issue, and provide recommended sprays with exact dosages."
    }
    effective_query = final_query if final_query else image_default_queries.get(language, image_default_queries["en"])

    context_sections = []
    if rag_context and rag_context != "NO_LOCAL_DATA_FOUND":
        context_sections.append(f"=== 80% GROUNDED CONTEXT FROM YOUR LOCAL PDF HANDBOOKS ===\n{rag_context}")
    if web_context:
        context_sections.append(f"=== 20% CONTEXT FROM VERIFIED AGRI WEBSITES ===\n{web_context}")

    full_context_block = ("\n\n" + "\n\n".join(context_sections)) if context_sections else ""

    prompt_with_instructions = (
        f"[STRICT LANGUAGE DIRECTIVE: Output your entire response ONLY in {language.upper()} ({lang_note}). "
        f"Do NOT use any other language.]\n"
        f"Farmer Query: {effective_query}"
        f"{full_context_block}"
    )

    # 5. Route to text-only providers when no image is uploaded
    if not has_image:
        if provider.startswith("ollama"):
            return _clean_content(call_ollama(prompt_with_instructions, history=history))
        elif provider.startswith("groq"):
            return _clean_content(call_groq_direct(prompt_with_instructions, history=history))

    # 6. Multimodal Vision Execution (Gemini 2.5 Flash)
    if has_image:
        try:
            llm = get_llm(has_image=True)
            
            vision_prompt = (
                f"{SYSTEM_INSTRUCTION}\n\n"
                f"{prompt_with_instructions}"
            )
            
            user_content = [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": image_data}
            ]
            
            response = llm.invoke([HumanMessage(content=user_content)])
            return _clean_content(response.content)
            
        except Exception as e:
            print(f"[GEMINI VISION DIRECT ERROR]: {e}")
            try:
                user_content = [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
                response = llm.invoke([HumanMessage(content=user_content)])
                return _clean_content(response.content)
            except Exception as e2:
                print(f"[GEMINI VISION FALLBACK ERROR]: {e2}")
                fallback_err = {
                    "mr": "छायाचित्राचे विश्लेषण करताना तांत्रिक अडचण आली. कृपया छायाचित्र पुन्हा अपलोड करा किंवा पिकाचे नाव टाईप करा.",
                    "hi": "फोटो का विश्लेषण करते समय तकनीकी समस्या आई। कृपया फोटो दोबारा अपलोड करें या समस्या टाइप करें।",
                    "en": "There was a technical issue analyzing the image. Please re-upload the photo or describe the crop symptoms."
                }
                return fallback_err.get(language, fallback_err["en"])

    # 7. Text-based Tool Calling loop for complex text queries
    llm = get_llm(has_image=False)
    llm_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
    
    if history:
        for msg in history[-4:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
                
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