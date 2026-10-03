import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render provides postgres:// or postgresql://; force psycopg2 driver
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
    # 1. Force Python to import models so SQLAlchemy registers their schema
    try:
        import backend.app.db.models
    except ImportError:
        try:
            import backend.app.models
        except ImportError:
            pass

    # 2. Bind and create all registered tables in PostgreSQL
    Base.metadata.create_all(bind=engine)
    print(">>> [DATABASE]: PostgreSQL tables verified/created successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()