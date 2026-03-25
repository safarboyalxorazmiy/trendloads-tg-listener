"""SHA-256 keyed extraction cache backed by aiosqlite with TTL expiry."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite
from loguru import logger


class ExtractionCache:
    """Async SQLite-backed cache for GPT extraction results.

    Keys are SHA-256 hashes of the raw message text.
    Values expire after *ttl* seconds (default 24 h).
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS extraction_cache (
            text_hash TEXT PRIMARY KEY,
            result    TEXT    NOT NULL,
            created   REAL   NOT NULL
        )
    """
    _GET = "SELECT result, created FROM extraction_cache WHERE text_hash = ?"
    _SET = "INSERT OR REPLACE INTO extraction_cache (text_hash, result, created) VALUES (?, ?, ?)"
    _DELETE_EXPIRED = "DELETE FROM extraction_cache WHERE created < ?"
    _COUNT = "SELECT COUNT(*) FROM extraction_cache"

    def __init__(self, db_path: str, ttl: int = 86400) -> None:
        self._db_path = db_path
        self._ttl = ttl
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Open the database and ensure the table exists."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(self._CREATE_TABLE)
        await self._db.commit()
        logger.info(f"Extraction cache initialised at {self._db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Return cached result dict or None if miss / expired."""
        if not self._db:
            return None
        text_hash = self._hash(text)
        async with self._db.execute(self._GET, (text_hash,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        result_json, created = row
        if time.time() - created > self._ttl:
            # Expired — delete lazily
            await self._db.execute(
                "DELETE FROM extraction_cache WHERE text_hash = ?", (text_hash,)
            )
            await self._db.commit()
            return None
        try:
            return json.loads(result_json)
        except json.JSONDecodeError:
            return None

    async def set(self, text: str, result: Dict[str, Any]) -> None:
        """Store a result in the cache."""
        if not self._db:
            return
        text_hash = self._hash(text)
        await self._db.execute(self._SET, (text_hash, json.dumps(result), time.time()))
        await self._db.commit()

    async def purge_expired(self) -> int:
        """Delete all expired entries. Returns count of deleted rows."""
        if not self._db:
            return 0
        cutoff = time.time() - self._ttl
        cursor = await self._db.execute(self._DELETE_EXPIRED, (cutoff,))
        await self._db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info(f"Purged {deleted} expired cache entries")
        return deleted

    async def count(self) -> int:
        """Return the total number of cached entries."""
        if not self._db:
            return 0
        async with self._db.execute(self._COUNT) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
