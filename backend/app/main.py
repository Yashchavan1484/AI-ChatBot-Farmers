import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount frontend static directory

        
# Set project root before importing internal backend modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.cache import advisory_cache
from backend.app.agents.agri_agents import run_farmer_assistant
from backend.app.db.sessions import init_db, get_db
from backend.app.api.auth_routes import router as auth_router
from backend.app.services.auth import get_current_user_optional
from backend.app.db.models import ChatMessage as DBChatMessage, ChatSession, User


app = FastAPI(
    title="Kisan Mitra - Multilingual Crop Advisory Engine",
    version="2.2.0"
)

# Startup lifecycle
@app.on_event("startup")
def on_startup():
    init_db()
    print(">>> [DATABASE]: SQLite tables initialized successfully.")

# Include Auth & Session History Endpoints
app.include_router(auth_router)

# CORS Middleware (permits Live Server on 5500 and local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class ChatHistoryItem(BaseModel):
    role: str = Field(default="user", json_schema_extra={"example": "user"})
    content: str = Field(default="", json_schema_extra={"example": "Hello"})

class ChatRequest(BaseModel):
    query: Optional[str] = ""
    language: Optional[str] = "mr"
    session_id: Optional[int] = None
    image_data: Optional[str] = None
    audio_data: Optional[str] = None
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: str
    answer: Optional[str] = None
    transcription: Optional[str] = ""
    status: str = "success"

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Kisan Mitra Advisory"}

GREETING_WORDS = {
    "hello", "hi", "hey", "namaste", "namaskar",
    "नमस्कार", "नमस्ते", "ram ram", "राम राम", "pranam"
}

def is_simple_greeting(text: str) -> bool:
    if not text:
        return False
    clean = text.strip().lower().rstrip(".!?")
    return clean in GREETING_WORDS or clean.startswith(("hi ", "hello ", "hey "))

def clean_llm_output(text: str) -> str:
    """Strips meta-reasoning, thought preambles, and surrounding quotes from LLMs."""
    if not text:
        return ""
    cleaned = re.sub(
        r"^(Since the user|Based on the user|I will respond|As an agronomist|Here is).*?:\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = re.sub(r"^Since the user's greeting.*?\.\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip('"\'')

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    try:
        lang = request.language or "mr"
        user_query = (request.query or "").strip()
        has_media = bool(getattr(request, "image_data", None) or getattr(request, "audio_data", None))

        # Helper to persist turns if user is logged in with an active session
        def persist_turn(reply_text: str):
            if current_user and request.session_id:
                session_obj = db.query(ChatSession).filter(
                    ChatSession.id == request.session_id, 
                    ChatSession.user_id == current_user.id
                ).first()
                
                # Auto-rename session on the first real question
                if session_obj and ("नवीन" in session_obj.title or "New" in session_obj.title):
                    session_obj.title = (user_query[:40] + "...") if len(user_query) > 40 else user_query
                
                # Save conversation turns using the SQLAlchemy DBChatMessage model
                if user_query:
                    db.add(DBChatMessage(session_id=request.session_id, sender="user", message_text=user_query))
                db.add(DBChatMessage(session_id=request.session_id, sender="kisan_mitra", message_text=reply_text))
                db.commit()

        # 0. Fast-Path: Greetings
        if is_simple_greeting(user_query) and not has_media:
            if lang == "mr":
                greeting_reply = "नमस्कार शेतकरी बंधू! मी आपला कृषी सल्लागार किसान मित्र आहे. आपल्या पिकाबद्दल किंवा फवारणीबद्दल काय प्रश्न आहे?"
            elif lang == "hi":
                greeting_reply = "नमस्ते किसान साथी! मैं आपका कृषि सलाहकार किसान मित्र हूँ। आज आपकी फसल या कीट नियंत्रण में क्या मदद कर सकता हूँ?"
            else:
                greeting_reply = "Hello! I am Kisan Mitra, your crop advisor. How can I assist you with your crops, disease diagnosis, or spray schedules today?"

            persist_turn(greeting_reply)
            return ChatResponse(
                response=greeting_reply,
                answer=greeting_reply,
                transcription=user_query,
                status="success"
            )
        
        # 1. Cache hit check
        is_cacheable = not has_media and not getattr(request, "history", None)
        if is_cacheable and user_query:
            cached_answer = advisory_cache.get(user_query, lang)
            if cached_answer:
                persist_turn(cached_answer)
                return ChatResponse(
                    response=cached_answer,
                    answer=cached_answer,
                    transcription=user_query,
                    status="success"
                )

        # 2. Prepare query & language
        lang_instruction_map = {
            "mr": "Respond completely in Marathi (मराठी).",
            "hi": "Respond completely in Hindi (हिन्दी).",
            "en": "Respond completely in English."
        }
        lang_note = lang_instruction_map.get(lang, "Respond in the language chosen by user.")
        full_query = f"[Language Requirement: {lang_note}]\nUser question: {user_query}"

        # 3. Extract history
        history_dicts = []
        if getattr(request, "history", None):
            history_dicts = [
                m.model_dump() if hasattr(m, "model_dump") else dict(m)
                for m in request.history
            ]

        # 4. Run LLM Agent
        raw_answer = run_farmer_assistant(
            query=full_query,
            history=history_dicts,
            image_data=getattr(request, "image_data", None),
            audio_data=getattr(request, "audio_data", None)
        )

        answer = clean_llm_output(str(raw_answer or ""))
        if not answer.strip():
            answer = "माफ करा, या प्रश्नासाठी माहिती गोळा करताना अडचण आली. कृपया प्रश्न थोडा वेगळ्या शब्दांत विचारा."

        # Cache & persist
        if is_cacheable and user_query:
            advisory_cache.set(user_query, lang, answer)
        persist_turn(answer)

        return ChatResponse(
            response=answer,
            answer=answer,
            transcription=user_query,
            status="success"
        )

    except Exception as e:
        print("CRITICAL SERVER ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_homepage():
        return FileResponse(FRONTEND_DIR / "index.html")
      
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)