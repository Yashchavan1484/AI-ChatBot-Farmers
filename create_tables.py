import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app.db.sessions import engine, Base
import backend.app.db.models

def create_all_tables():
    db_url = os.getenv("DATABASE_URL", "sqlite:///./kisan_mitra.db")
    print(f"^>^>^> [MIGRATION]: Connecting to database: {db_url.split('@'^)[-1] if '@' in db_url else db_url}")
    print(f"^>^>^> [MIGRATION]: Found models: {list(Base.metadata.tables.keys(^)^)}")
    Base.metadata.create_all(bind=engine)
    print("^>^>^> [MIGRATION SUCCESS]: All database tables initialized in PostgreSQL!")

if __name__ == "__main__":
    create_all_tables()
