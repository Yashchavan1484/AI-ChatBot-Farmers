from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime

from backend.app.db.sessions import get_db
from backend.app.db.models import User, ChatSession, ChatMessage
from backend.app.services.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user_optional,
)

router = APIRouter(prefix="/api", tags=["Auth & History"])

# --- Pydantic Schemas ---
class RegisterRequest(BaseModel):
    phone_number: str
    password: str
    full_name: Optional[str] = "शेतकरी मित्र"
    preferred_language: Optional[str] = "mr"

class LoginRequest(BaseModel):
    phone_number: str
    password: str

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "नवीन कृषी चर्चा"


# --- Auth Endpoints ---
@router.post("/auth/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    clean_phone = req.phone_number.strip()
    existing_user = db.query(User).filter(User.phone_number == clean_phone).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="हा फोन नंबर आधीपासून नोंदणीकृत आहे (Phone number already registered)."
        )

    user = User(
        phone_number=clean_phone,
        full_name=req.full_name,
        hashed_password=get_password_hash(req.password),
        preferred_language=req.preferred_language,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.phone_number})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "phone_number": user.phone_number,
            "preferred_language": user.preferred_language,
        }
    }


@router.post("/auth/login")
def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    clean_phone = req.phone_number.strip()
    user = db.query(User).filter(User.phone_number == clean_phone).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="चुकीचा फोन नंबर किंवा पासवर्ड (Invalid credentials)."
        )

    token = create_access_token({"sub": user.phone_number})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "phone_number": user.phone_number,
            "preferred_language": user.preferred_language,
        }
    }


# --- Chat History & Session Endpoints ---
@router.get("/history/sessions")
def list_sessions(
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if not current_user:
        return []

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for s in sessions
    ]


@router.post("/history/sessions")
def create_session(
    req: SessionCreateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    new_session = ChatSession(user_id=current_user.id, title=req.title)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"id": new_session.id, "title": new_session.title}


@router.get("/history/sessions/{session_id}")
def get_session_messages(
    session_id: int,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    session_obj = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "sender": m.sender,
            "text": m.message_text,
            "created_at": m.created_at.strftime("%H:%M"),
        }
        for m in messages
    ]

@router.delete("/history/sessions/{session_id}")
def delete_session(
    session_id: int,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    session_obj = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.delete(session_obj)
    db.commit()
    return {"status": "success", "message": "Session deleted successfully."}

