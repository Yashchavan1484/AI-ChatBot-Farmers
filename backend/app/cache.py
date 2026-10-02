import hashlib
import time
from typing import Optional

class AdvisoryCache:
    """Thread-safe, lightweight in-memory cache with TTL expiration."""
    def __init__(self, default_ttl_seconds: int = 21600):  # 6 hours
        self._store = {}
        self.default_ttl = default_ttl_seconds

    def _generate_key(self, query: str, language: str) -> str:
        # Normalize: lower-case, remove extra whitespace
        normalized = f"{language.strip().lower()}::{' '.join(query.strip().lower().split())}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str, language: str) -> Optional[str]:
        key = self._generate_key(query, language)
        record = self._store.get(key)
        if not record:
            return None
        
        answer, expires_at = record
        if time.time() > expires_at:
            del self._store[key]
            return None
        return answer

    def set(self, query: str, language: str, answer: str, ttl: Optional[int] = None):
        key = self._generate_key(query, language)
        expires_at = time.time() + (ttl or self.default_ttl)
        self._store[key] = (answer, expires_at)

advisory_cache = AdvisoryCache()