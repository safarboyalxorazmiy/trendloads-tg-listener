"""Filter incoming Telegram messages before extraction."""

from __future__ import annotations

import re
from typing import List, Optional

from loguru import logger
from pyrogram.types import Message

from src.config import FiltersConfig
from src.utils.metrics import metrics


class MessageFilter:
    """Determine whether an incoming Telegram message should be processed."""

    def __init__(self, config: FiltersConfig) -> None:
        self._config = config
        # Pre-compile keyword patterns for speed
        self._keyword_regexes: List[re.Pattern[str]] = []
        for pattern in config.keyword_patterns:
            try:
                self._keyword_regexes.append(
                    re.compile(pattern, re.IGNORECASE | re.UNICODE)
                )
            except re.error as exc:
                logger.warning(f"Invalid keyword regex '{pattern}': {exc}")

    def passes(self, message: Message) -> bool:
        """Return True if the message should be forwarded to extraction."""
        metrics.inc_received()

        # ── 1. Group ID filtering ────────────────────────────────────────
        chat_id = self._get_chat_id(message)
        if chat_id is not None:
            if self._config.whitelist_group_ids:
                if chat_id not in self._config.whitelist_group_ids:
                    logger.debug(f"Message from chat {chat_id} not in whitelist, skipping")
                    metrics.inc_filtered()
                    return False
            if chat_id in self._config.blacklist_group_ids:
                logger.debug(f"Message from chat {chat_id} in blacklist, skipping")
                metrics.inc_filtered()
                return False

        # ── 2. Skip bot senders ──────────────────────────────────────────
        if self._config.skip_bots and self._is_bot(message):
            logger.debug("Message from bot, skipping")
            metrics.inc_filtered()
            return False

        # ── 3. Photo messages (loadboard screenshots) — always pass ──────
        has_photo = message.photo is not None
        if has_photo:
            logger.debug("Photo message detected — passing for vision extraction")
            return True

        # ── 4. Media-only filter (non-photo media like voice, docs) ─────
        text = message.text or message.caption or ""
        if self._config.skip_media_only and not text.strip() and message.media:
            logger.debug("Media-only message (non-photo), skipping")
            metrics.inc_filtered()
            return False

        # ── 5. Minimum length ────────────────────────────────────────────
        if len(text.strip()) < self._config.min_length:
            logger.debug(
                f"Message too short ({len(text.strip())} < {self._config.min_length}), skipping"
            )
            metrics.inc_filtered()
            return False

        # ── 6. Keyword match ─────────────────────────────────────────────
        if self._keyword_regexes:
            if not self._matches_keywords(text):
                logger.debug("No keyword match, skipping")
                metrics.inc_filtered()
                return False

        return True

    def _matches_keywords(self, text: str) -> bool:
        """Return True if text matches at least one keyword pattern."""
        for regex in self._keyword_regexes:
            if regex.search(text):
                return True
        return False

    @staticmethod
    def _get_chat_id(message: Message) -> Optional[int]:
        """Extract the numeric chat/group ID from the message."""
        try:
            return message.chat.id if message.chat else None
        except Exception:
            return None

    @staticmethod
    def _is_bot(message: Message) -> bool:
        """Check if the sender is a bot."""
        try:
            return bool(message.from_user and message.from_user.is_bot)
        except Exception:
            return False
