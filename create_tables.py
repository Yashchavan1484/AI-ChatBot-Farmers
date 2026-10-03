import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app.db.sessions import engine
import backend.app.db.models as models

def create_all_tables():
    db_url = os.getenv("DATABASE_URL", "sqlite:///./kisan_mitra.db")
    masked_url = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"[MIGRATION] Connecting to database: {masked_url}")
    
    # Use the Base that contains the actual models
    target_base = getattr(models, "Base", None)
    if target_base and hasattr(target_base, "metadata"):
        print(f"[MIGRATION] Models discovered: {list(target_base.metadata.tables.keys())}")
        target_base.metadata.create_all(bind=engine)
    else:
        from backend.app.db.sessions import Base as SessionBase
        print(f"[MIGRATION] Models discovered: {list(SessionBase.metadata.tables.keys())}")
        SessionBase.metadata.create_all(bind=engine)
        
    print("[MIGRATION SUCCESS] All tables created successfully!")

if __name__ == "__main__":
    create_all_tables()
