import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kisan_mitra.db")

from sqlalchemy.orm import sessionmaker, declarative_base

# Read DATABASE_URL from Render environment

if DATABASE_URL:
    # Render provides postgres://, but SQLAlchemy 1.4+ requires postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Local fallback for your PC
    engine = create_engine(
        "sqlite:///./kisan_mitra.db", 
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes tables on startup."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI dependency for database session lifecycle."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()