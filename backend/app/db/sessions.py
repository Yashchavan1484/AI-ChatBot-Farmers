import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    engine = create_engine(
        "sqlite:///./kisan_mitra.db", 
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    # Explicitly import all DB models so Base.metadata discovers them
    try:
        from backend.app.models.user import User  # Adjust path if your model is in user.py / models.py
    except ImportError:
        pass
    try:
        from backend.app.db.models import User, ChatSession, ChatMessage
    except ImportError:
        pass

    Base.metadata.create_all(bind=engine)
    print(">>> [DATABASE]: PostgreSQL / SQLite tables verified and created successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()