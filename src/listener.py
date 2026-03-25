"""Pyrogram message handler: filter -> extract -> send to backend.

Handles:
  - Text messages: keyword-filtered, then GPT text extraction
  - Single photo messages: vision extraction with optional caption as contact
  - Album / grouped photos: collect all photos in the album, extract each via
    vision, apply the shared caption (contact info) to every extracted load
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger
from pyrogram import Client
from pyrogram.types import Message

from src.backend.client import BackendClient
from src.extractor.gpt_extractor import ExtractedLoad, GPTExtractor
from src.filters.message_filter import MessageFilter
from src.utils.distance import estimate_load_distances
from src.utils.metrics import metrics

# Regex for Telegram @handles in caption text
_HANDLE_RE = re.compile(r"@[\w]+")
# Regex for phone numbers
_PHONE_RE = re.compile(r"[\(]?\d{3}[\)]?[\s\-.]?\d{3}[\s\-.]?\d{4}")


def _extract_contact_from_caption(caption: str) -> Optional[str]:
    """Pull the most useful contact from a caption."""
    if not caption:
        return None
    handles = _HANDLE_RE.findall(caption)
    if handles:
        return handles[-1]
    phones = _PHONE_RE.findall(caption)
    if phones:
        return phones[0]
    stripped = caption.strip()
    if len(stripped) < 60:
        return stripped
    return None


class LoadListener:
    """Bridges the Telegram message stream with extraction and backend delivery."""

    def __init__(
        self,
        message_filter: MessageFilter,
        extractor: GPTExtractor,
        backend: BackendClient,
        client: Client,
    ) -> None:
        self._filter = message_filter
        self._extractor = extractor
        self._backend = backend
        self._client = client

        # Buffer for grouped media (albums): media_group_id -> state
        self._album_buffer: Dict[str, _AlbumState] = {}

    async def handle_new_message(self, client: Client, message: Message) -> None:
        """Entry point called by Pyrogram. Returns immediately — processing is background."""
        # ── Raw incoming log (synchronous, no API calls) ─────────────
        chat_title = getattr(message.chat, 'title', None) or 'DM'
        text_preview = (message.text or message.caption or '')[:80]
        has_photo = message.photo is not None
        logger.info(
            f"[INCOMING] chat='{chat_title}' | "
            f"photo={has_photo} | text='{text_preview}'"
        )

        # ── Filter (fast, no I/O) ────────────────────────────────────
        if not self._filter.passes(message):
            return

        # ── Process in background — don't block the event loop ───────
        asyncio.ensure_future(self._process_message_safe(message))

    async def _process_message_safe(self, message: Message) -> None:
        """Background task: process a single message or album photo."""
        try:
            grouped_id = message.media_group_id
            if grouped_id and message.photo:
                await self._handle_album_photo(message, grouped_id)
                return
            await self._process_single_message(message)
        except Exception:
            logger.exception(f"Unhandled error processing message {message.id}")
            metrics.inc_extraction_error()

    async def _handle_album_photo(self, message: Message, grouped_id: str) -> None:
        """Buffer album photos and process the full album after a short delay."""
        if grouped_id not in self._album_buffer:
            self._album_buffer[grouped_id] = _AlbumState()
            asyncio.get_event_loop().call_later(
                2.0,
                lambda gid=grouped_id: asyncio.ensure_future(self._process_album(gid)),
            )

        state = self._album_buffer[grouped_id]
        state.messages.append(message)

        caption = message.text or message.caption or ""
        if caption.strip() and not state.caption:
            state.caption = caption.strip()

    async def _process_album(self, grouped_id: str) -> None:
        """Process all photos in a collected album."""
        state = self._album_buffer.pop(grouped_id, None)
        if not state or not state.messages:
            return

        caption = state.caption or ""
        contact = _extract_contact_from_caption(caption)
        msg_list = state.messages

        first = msg_list[0]
        chat_id, message_id, chat_title, sender_name, message_timestamp = self._extract_metadata(first)

        logger.info(
            f"Processing album (grouped_id={grouped_id}): {len(msg_list)} photo(s), "
            f"caption={caption!r}, contact={contact!r}"
        )

        all_loads: List[ExtractedLoad] = []

        for idx, msg in enumerate(msg_list):
            try:
                bio = await self._client.download_media(msg, in_memory=True)
                photo_bytes = bio.getvalue() if bio else None
                if not photo_bytes or len(photo_bytes) < 5000:
                    continue

                ev_message_id = msg.id or message_id

                logger.info(f"  Album photo {idx + 1}/{len(msg_list)} ({len(photo_bytes)} bytes)")
                result = await self._extractor.extract_from_image(
                    image_bytes=photo_bytes,
                    mime_type="image/jpeg",
                    chat_id=chat_id,
                    message_id=ev_message_id,
                    chat_title=chat_title,
                    sender_name=sender_name,
                    message_timestamp=message_timestamp,
                )

                if result and isinstance(result, list):
                    all_loads.extend(result)
            except Exception as img_err:
                logger.warning(f"  Album photo {idx + 1} extraction failed: {img_err}")

        if not all_loads:
            logger.debug(f"No loads extracted from album {grouped_id}")
            return

        if contact:
            for load in all_loads:
                if not load.contact:
                    load.contact = contact
        if caption:
            for load in all_loads:
                if load.rawText:
                    load.rawText = f"{load.rawText} | {caption}"
                else:
                    load.rawText = caption

        valid_loads = [l for l in all_loads if l.confidence >= 0.3]
        if not valid_loads:
            logger.debug(f"All album loads below confidence threshold (grouped_id={grouped_id})")
            return

        logger.info(
            f"Album {grouped_id}: extracted {len(valid_loads)} load(s) from {len(msg_list)} photo(s)"
        )

        await estimate_load_distances(valid_loads)
        payloads = [load.to_dict() for load in valid_loads]
        await self._backend.send(payloads)

    async def _process_single_message(self, message: Message) -> None:
        """Process a single text message or single (non-album) photo."""
        text = message.text or message.caption or ""
        chat_id, message_id, chat_title, sender_name, message_timestamp = self._extract_metadata(message)

        has_photo = message.photo is not None
        logger.info(
            f"Processing message {message_id} from chat {chat_id} "
            f"({chat_title!r}, {len(text)} chars, has_photo={has_photo})"
        )

        result = None

        if has_photo:
            try:
                bio = await self._client.download_media(message, in_memory=True)
                photo_bytes = bio.getvalue() if bio else None
                if photo_bytes and len(photo_bytes) > 5000:
                    logger.info(f"Extracting loads from screenshot ({len(photo_bytes)} bytes)")
                    result = await self._extractor.extract_from_image(
                        image_bytes=photo_bytes,
                        mime_type="image/jpeg",
                        chat_id=chat_id,
                        message_id=message_id,
                        chat_title=chat_title,
                        sender_name=sender_name,
                        message_timestamp=message_timestamp,
                    )
                    if result and text:
                        contact = _extract_contact_from_caption(text)
                        if contact:
                            loads_list = result if isinstance(result, list) else [result]
                            for load in loads_list:
                                if not load.contact:
                                    load.contact = contact
            except Exception as img_err:
                logger.warning(f"Screenshot extraction failed: {img_err}")

        if result is None and text:
            result = await self._extractor.extract(
                text=text,
                chat_context=None,
                chat_id=chat_id,
                message_id=message_id,
                chat_title=chat_title,
                sender_name=sender_name,
                message_timestamp=message_timestamp,
            )

        if result is None:
            logger.debug(f"No load extracted from message {message_id}")
            return

        loads: List[ExtractedLoad]
        if isinstance(result, list):
            loads = result
        else:
            loads = [result]

        valid_loads = [l for l in loads if l.confidence >= 0.3]
        if not valid_loads:
            logger.debug(f"All loads below confidence threshold for message {message_id}")
            return

        logger.info(
            f"Extracted {len(valid_loads)} load(s) from message {message_id} "
            f"(confidence: {[round(l.confidence, 2) for l in valid_loads]})"
        )

        await estimate_load_distances(valid_loads)
        payloads = [load.to_dict() for load in valid_loads]
        await self._backend.send(payloads)

    def _extract_metadata(
        self, message: Message
    ) -> Tuple[int, int, str, str, Optional[datetime]]:
        """Extract chat_id, message_id, chat_title, sender_name, message_timestamp.
        All synchronous — Pyrogram populates these on the message object directly.
        """
        chat_id: int = 0
        message_id: int = 0
        chat_title: str = ""
        sender_name: str = ""
        message_timestamp: Optional[datetime] = None

        try:
            chat_id = message.chat.id if message.chat else 0
            message_id = message.id or 0
            message_timestamp = message.date
            if message_timestamp:
                # Pyrogram may return local or UTC — always normalize to UTC
                if message_timestamp.tzinfo is None:
                    message_timestamp = message_timestamp.replace(tzinfo=timezone.utc)
                else:
                    message_timestamp = message_timestamp.astimezone(timezone.utc)
        except Exception:
            pass

        try:
            chat_title = message.chat.title or "" if message.chat else ""
        except Exception:
            pass

        try:
            user = message.from_user
            if user:
                first = user.first_name or ""
                last = user.last_name or ""
                username = user.username or ""
                sender_name = f"{first} {last}".strip() or (f"@{username}" if username else "")
        except Exception:
            pass

        return chat_id, message_id, chat_title, sender_name, message_timestamp


class _AlbumState:
    """Internal state for collecting album photos before processing."""

    __slots__ = ("messages", "caption")

    def __init__(self) -> None:
        self.messages: List[Message] = []
        self.caption: str = ""
