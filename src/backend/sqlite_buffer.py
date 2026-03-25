"""SQLite-backed buffer for loads that failed to send to the backend."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from loguru import logger


@dataclass
class BufferedLoad:
    id: int
    payload: Dict[str, Any]
    created: float
    attempts: int


class SQLiteBuffer:
    """Async SQLite buffer for unsent load payloads.

    Loads are stored when the backend is unreachable and flushed
    when connectivity is restored.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS load_buffer (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            payload  TEXT    NOT NULL,
            created  REAL    NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0
        )
    """
    _INSERT = "INSERT INTO load_buffer (payload, created, attempts) VALUES (?, ?, ?)"
    _SELECT_ALL = "SELECT id, payload, created, attempts FROM load_buffer ORDER BY created ASC"
    _SELECT_BATCH = (
        "SELECT id, payload, created, attempts FROM load_buffer ORDER BY created ASC LIMIT ?"
    )
    _DELETE_IDS = "DELETE FROM load_buffer WHERE id IN ({placeholders})"
    _INCREMENT_ATTEMPTS = "UPDATE load_buffer SET attempts = attempts + 1 WHERE id IN ({placeholders})"
    _COUNT = "SELECT COUNT(*) FROM load_buffer"

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Open database and create table if needed."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(self._CREATE_TABLE)
        await self._db.commit()
        count = await self.count()
        logger.info(f"SQLite buffer initialised at {self._db_path} ({count} pending loads)")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def add(self, payload: Dict[str, Any]) -> None:
        """Buffer a single load payload."""
        if not self._db:
            raise RuntimeError("Buffer not initialised")
        await self._db.execute(self._INSERT, (json.dumps(payload), time.time(), 0))
        await self._db.commit()
        logger.debug("Load buffered to SQLite")

    async def get_all(self) -> List[BufferedLoad]:
        """Return all buffered loads ordered by creation time."""
        if not self._db:
            return []
        rows: List[BufferedLoad] = []
        async with self._db.execute(self._SELECT_ALL) as cursor:
            async for row in cursor:
                rows.append(
                    BufferedLoad(
                        id=row[0],
                        payload=json.loads(row[1]),
                        created=row[2],
                        attempts=row[3],
                    )
                )
        return rows

    async def get_batch(self, limit: int) -> List[BufferedLoad]:
        """Return up to *limit* buffered loads."""
        if not self._db:
            return []
        rows: List[BufferedLoad] = []
        async with self._db.execute(self._SELECT_BATCH, (limit,)) as cursor:
            async for row in cursor:
                rows.append(
                    BufferedLoad(
                        id=row[0],
                        payload=json.loads(row[1]),
                        created=row[2],
                        attempts=row[3],
                    )
                )
        return rows

    async def delete(self, ids: List[int]) -> None:
        """Delete buffered loads by their IDs."""
        if not self._db or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        sql = self._DELETE_IDS.format(placeholders=placeholders)
        await self._db.execute(sql, ids)
        await self._db.commit()
        logger.debug(f"Deleted {len(ids)} loads from buffer")

    async def increment_attempts(self, ids: List[int]) -> None:
        """Increment the attempt counter for given IDs."""
        if not self._db or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        sql = self._INCREMENT_ATTEMPTS.format(placeholders=placeholders)
        await self._db.execute(sql, ids)
        await self._db.commit()

    async def count(self) -> int:
        """Return total buffered load count."""
        if not self._db:
            return 0
        async with self._db.execute(self._COUNT) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
