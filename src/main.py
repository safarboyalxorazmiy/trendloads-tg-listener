"""Entry point: Pyrogram Client setup, handler registration, graceful shutdown."""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Optional

from loguru import logger
from pyrogram import Client
from pyrogram.errors import FloodWait
from pyrogram.handlers import MessageHandler

from src.backend.client import BackendClient
from src.backend.sqlite_buffer import SQLiteBuffer
from src.cache.extraction_cache import ExtractionCache
from src.config import AppConfig, load_config
from src.extractor.gpt_extractor import GPTExtractor
from src.filters.message_filter import MessageFilter
from src.listener import LoadListener
from src.utils.logger import setup_logger
from src.utils.metrics import metrics


class Application:
    """Top-level application orchestrator."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client: Optional[Client] = None
        self._buffer: Optional[SQLiteBuffer] = None
        self._cache: Optional[ExtractionCache] = None
        self._backend: Optional[BackendClient] = None
        self._listener: Optional[LoadListener] = None
        self._shutdown_event = asyncio.Event()
        self._flush_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._cache_purge_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise all components and connect the Pyrogram client."""
        logger.info("Starting Telegram Listener application")

        # SQLite buffer
        self._buffer = SQLiteBuffer(self._config.db_path)
        await self._buffer.init()

        # Extraction cache
        self._cache = ExtractionCache(
            self._config.cache_db_path,
            ttl=self._config.extraction.cache_ttl,
        )
        await self._cache.init()

        # Backend HTTP client
        self._backend = BackendClient(self._config.backend, self._buffer)
        await self._backend.init()

        # Extractor
        extractor = GPTExtractor(self._config.extraction, self._cache)

        # Message filter
        msg_filter = MessageFilter(self._config.filters)

        # Pyrogram client (phone set at construction — no interactive prompt)
        self._client = Client(
            name=self._config.telegram.session_name,
            api_id=self._config.telegram.api_id,
            api_hash=self._config.telegram.api_hash,
            phone_number=self._config.telegram.phone,
        )

        # Listener (needs client reference for download_media)
        self._listener = LoadListener(msg_filter, extractor, self._backend, self._client)

        # Register handler — receives ALL messages (groups, channels, DMs)
        self._client.add_handler(MessageHandler(self._listener.handle_new_message))

        # Connect with auto-reconnect and flood-wait handling
        await self._connect_with_retry()

        # Start background tasks
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self._stats_task = asyncio.create_task(self._periodic_stats())
        self._cache_purge_task = asyncio.create_task(self._periodic_cache_purge())

        logger.info("Application started successfully — listening for messages")

    async def _connect_with_retry(self) -> None:
        """Connect the Pyrogram client with exponential backoff."""
        assert self._client is not None
        delay = self._config.telegram.reconnect_base_delay
        max_delay = self._config.telegram.reconnect_max_delay
        max_attempts = self._config.telegram.max_reconnect_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                await self._client.start()
                me = await self._client.get_me()
                logger.info(
                    f"Connected to Telegram as {me.first_name} "
                    f"(id={me.id}) on attempt {attempt}"
                )
                return
            except FloodWait as exc:
                wait = exc.value
                if wait > self._config.telegram.flood_wait_threshold:
                    logger.error(
                        f"Flood wait of {wait}s exceeds threshold "
                        f"({self._config.telegram.flood_wait_threshold}s), aborting"
                    )
                    raise
                logger.warning(f"Flood wait: sleeping {wait}s before retry")
                await asyncio.sleep(wait)
            except Exception as exc:
                logger.warning(
                    f"Connection attempt {attempt}/{max_attempts} failed: {exc}"
                )
                if attempt == max_attempts:
                    logger.error("Max reconnect attempts reached, giving up")
                    raise
                logger.info(f"Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def run_until_shutdown(self) -> None:
        """Block until a shutdown signal is received."""
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Graceful shutdown: cancel tasks, flush buffer, close connections."""
        logger.info("Shutting down application")
        self._shutdown_event.set()

        for task in (self._flush_task, self._stats_task, self._cache_purge_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        metrics.print_stats()

        if self._backend:
            try:
                await self._backend.flush_buffer()
            except Exception:
                logger.exception("Error during final buffer flush")

        if self._backend:
            await self._backend.close()
        if self._cache:
            await self._cache.close()
        if self._buffer:
            await self._buffer.close()
        if self._client:
            await self._client.stop()

        logger.info("Application stopped")

    # ── Background Tasks ──────────────────────────────────────────────────

    async def _periodic_flush(self) -> None:
        interval = self._config.backend.flush_interval
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)
                if self._backend:
                    await self._backend.flush_buffer()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in periodic buffer flush")

    async def _periodic_stats(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)
                metrics.print_stats()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in periodic stats")

    async def _periodic_cache_purge(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)
                if self._cache:
                    await self._cache.purge_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in periodic cache purge")


def _setup_signal_handlers(app: Application, loop: asyncio.AbstractEventLoop) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_signal_handler(app, s)))


async def _signal_handler(app: Application, sig: signal.Signals) -> None:
    logger.info(f"Received signal {sig.name}, initiating shutdown")
    await app.stop()


def main() -> None:
    """CLI entry point."""
    try:
        config = load_config()
    except Exception as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logger(config.log_level)
    logger.info("Configuration loaded")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = Application(config)

    try:
        _setup_signal_handlers(app, loop)
        loop.run_until_complete(app.start())
        loop.run_until_complete(app.run_until_shutdown())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
    finally:
        loop.run_until_complete(app.stop())
        loop.close()


if __name__ == "__main__":
    main()
