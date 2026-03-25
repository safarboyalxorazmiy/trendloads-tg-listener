"""Backend HTTP client with retry logic and SQLite fallback buffer."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from src.backend.sqlite_buffer import SQLiteBuffer
from src.config import BackendConfig
from src.utils.metrics import metrics


class BackendClient:
    """Async HTTP client for pushing extracted loads to the backend API.

    On failure the payload is buffered to SQLite and retried later via
    flush_buffer().
    """

    def __init__(self, config: BackendConfig, buffer: SQLiteBuffer) -> None:
        self._config = config
        self._buffer = buffer
        self._session: Optional[aiohttp.ClientSession] = None
        self._url = f"{config.url.rstrip('/')}{config.endpoint}"

    async def init(self) -> None:
        """Create the aiohttp session."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.token:
            headers["Authorization"] = f"Bearer {self._config.token}"

        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        logger.info(f"Backend client initialised, target={self._url}")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def send(self, loads: List[Dict[str, Any]]) -> bool:
        """POST load(s) to the backend API.

        Returns True on success.  On failure, buffers the payloads to SQLite.
        """
        if not loads:
            return True

        # Send bare array — Java controller expects List<TelegramLoadDto> directly
        success = await self._post_with_retry(loads)

        if success:
            metrics.inc_sent(len(loads))
            logger.info(f"Sent {len(loads)} load(s) to backend")
            # Opportunistically try flushing any buffered loads
            await self.flush_buffer()
            return True

        # Buffer each load individually so partial retries work
        for load in loads:
            await self._buffer.add(load)
            metrics.inc_buffered()
        metrics.inc_backend_error()
        logger.warning(f"Buffered {len(loads)} load(s) to SQLite after send failure")
        return False

    async def _post_with_retry(self, payload: Any) -> bool:
        """Attempt to POST with exponential-backoff retry."""
        if not self._session:
            logger.error("Backend session not initialised")
            return False

        delay = self._config.retry_base_delay
        for attempt in range(1, self._config.max_retries + 1):
            try:
                async with self._session.post(
                    self._url, data=json.dumps(payload)
                ) as resp:
                    if 200 <= resp.status < 300:
                        return True
                    body = await resp.text()
                    logger.warning(
                        f"Backend returned {resp.status} on attempt "
                        f"{attempt}/{self._config.max_retries}: {body[:500]}"
                    )
                    # Don't retry client errors (4xx) except 429
                    if 400 <= resp.status < 500 and resp.status != 429:
                        return False

            except aiohttp.ClientError as exc:
                logger.warning(
                    f"Backend connection error on attempt "
                    f"{attempt}/{self._config.max_retries}: {exc}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Backend timeout on attempt {attempt}/{self._config.max_retries}"
                )
            except Exception as exc:
                logger.exception(
                    f"Unexpected backend error on attempt {attempt}/{self._config.max_retries}"
                )
                break  # Don't retry unknown errors

            if attempt < self._config.max_retries:
                logger.debug(f"Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._config.retry_max_delay)

        return False

    async def flush_buffer(self) -> None:
        """Attempt to send all buffered loads to the backend."""
        buffered = await self._buffer.get_batch(self._config.batch_size)
        if not buffered:
            return

        logger.info(f"Flushing {len(buffered)} buffered load(s)")
        # Send bare array — Java controller expects List<TelegramLoadDto> directly
        payloads = [b.payload for b in buffered]
        success = await self._post_with_retry(payloads)

        if success:
            ids = [b.id for b in buffered]
            await self._buffer.delete(ids)
            metrics.inc_sent(len(buffered))
            logger.info(f"Flushed {len(buffered)} buffered load(s) successfully")
        else:
            ids = [b.id for b in buffered]
            await self._buffer.increment_attempts(ids)
            logger.warning("Buffer flush failed, will retry later")
