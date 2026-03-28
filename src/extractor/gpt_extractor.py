"""GPT-based load extraction with caching and city-code pre-processing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from src.cache.extraction_cache import ExtractionCache
from src.config import ExtractionConfig
from src.extractor.city_codes import CITY_CODES, resolve
from src.extractor.prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
from src.utils.metrics import metrics


@dataclass
class ExtractedLoad:
    """Flat structure matching the Java TelegramLoadDto exactly (camelCase keys)."""

    # --- Set by listener, NOT by GPT ---
    chatId: int = 0
    chatTitle: str = ""
    messageId: int = 0
    senderName: str = ""
    rawText: str = ""
    messageTimestamp: Optional[str] = None  # ISO-8601

    # --- Set by GPT extraction ---
    originCity: Optional[str] = None
    originState: Optional[str] = None
    originCode: Optional[str] = None
    destCity: Optional[str] = None
    destState: Optional[str] = None
    destCode: Optional[str] = None
    equipmentType: Optional[str] = None
    driverType: Optional[str] = None
    rate: Optional[float] = None
    ratePerMile: Optional[float] = None
    stops: Optional[int] = None
    pickupTime: Optional[str] = None  # ISO-8601 or None
    readyStatus: Optional[str] = None
    contact: Optional[str] = None  # plain string — @handle, phone, or name
    confidence: float = 0.0

    # --- GPT rate predictions (when rate not in message) ---
    estimatedRate: Optional[float] = None  # GPT-predicted total payout
    estimatedRatePerMile: Optional[float] = None  # GPT-predicted $/mi

    # --- Distance & payout estimation (set by listener post-processing) ---
    estimatedDistance: Optional[float] = None  # miles
    distancePredicted: bool = False  # True = AI-estimated, False = from message
    estimatedPayout: Optional[float] = None  # dollars
    payoutPredicted: bool = False  # True = AI-estimated, False = from message

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dict with camelCase keys matching Java TelegramLoadDto."""
        return {
            "chatId": self.chatId,
            "chatTitle": self.chatTitle,
            "messageId": self.messageId,
            "senderName": self.senderName,
            "rawText": self.rawText,
            "messageTimestamp": self.messageTimestamp,
            "originCity": self.originCity,
            "originState": self.originState,
            "originCode": self.originCode,
            "destCity": self.destCity,
            "destState": self.destState,
            "destCode": self.destCode,
            "equipmentType": self.equipmentType,
            "driverType": self.driverType,
            "rate": self.rate,
            "ratePerMile": self.ratePerMile,
            "stops": self.stops,
            "pickupTime": self.pickupTime,
            "readyStatus": self.readyStatus,
            "contact": self.contact,
            "confidence": self.confidence,
            "estimatedDistance": self.estimatedDistance,
            "distancePredicted": self.distancePredicted,
            "estimatedPayout": self.estimatedPayout,
            "payoutPredicted": self.payoutPredicted,
        }


def _parse_response(raw: dict, text: str) -> ExtractedLoad:
    """Convert a GPT JSON response dict into an ExtractedLoad dataclass.

    GPT returns nested origin/destination objects — we flatten them here.
    Contact may come as a string or object — we normalise to string.
    """
    # --- Flatten origin ---
    origin = raw.get("origin") or {}
    if isinstance(origin, str):
        origin = {"city": origin}
    origin_city = origin.get("city")
    origin_state = origin.get("state")
    origin_code = origin.get("code")

    # --- Flatten destination ---
    dest = raw.get("destination") or {}
    if isinstance(dest, str):
        dest = {"city": dest}
    dest_city = dest.get("city")
    dest_state = dest.get("state")
    dest_code = dest.get("code")

    # --- Normalise contact to plain string ---
    contact_raw = raw.get("contact")
    contact_str: Optional[str] = None
    if isinstance(contact_raw, dict):
        # Pick the most useful field: telegram handle > phone > name
        contact_str = (
            contact_raw.get("telegram")
            or contact_raw.get("phone")
            or contact_raw.get("name")
        )
    elif isinstance(contact_raw, str) and contact_raw:
        contact_str = contact_raw

    return ExtractedLoad(
        rawText=text,
        originCity=origin_city,
        originState=origin_state,
        originCode=origin_code,
        destCity=dest_city,
        destState=dest_state,
        destCode=dest_code,
        equipmentType=raw.get("equipmentType"),
        driverType=raw.get("driverType"),
        rate=_safe_float(raw.get("rate")),
        ratePerMile=_safe_float(raw.get("ratePerMile")),
        stops=_safe_int(raw.get("stops")),
        pickupTime=raw.get("pickupTime"),
        readyStatus=raw.get("readyStatus"),
        contact=contact_str,
        confidence=float(raw.get("confidence", 0.0)),
        estimatedRate=_safe_float(raw.get("estimatedRate")),
        estimatedRatePerMile=_safe_float(raw.get("estimatedRatePerMile")),
    )


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _preprocess_city_codes(text: str) -> str:
    """Annotate known city codes in the text so GPT has extra context."""
    tokens = re.split(r"(\s+|->|~>|-->|-|>)", text)
    annotated_parts: list[str] = []
    for token in tokens:
        clean = token.strip().upper()
        resolved = resolve(clean) if clean else None
        if resolved:
            city, state = resolved
            annotated_parts.append(f"{token}({city}, {state})")
        else:
            annotated_parts.append(token)
    return "".join(annotated_parts)


class GPTExtractor:
    """Async GPT-based load extractor with caching."""

    def __init__(self, config: ExtractionConfig, cache: ExtractionCache) -> None:
        self._config = config
        self._cache = cache
        self._client = AsyncOpenAI(api_key=config.openai_api_key)

    def _build_messages(self, text: str, chat_context: Optional[str] = None) -> list[dict]:
        """Build the OpenAI chat messages list including few-shot examples."""
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add few-shot examples
        for user_msg, assistant_msg in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        # Pre-process city codes for extra context
        annotated = _preprocess_city_codes(text)

        # Sanitize: remove null bytes and surrogates that break JSON serialization
        annotated = annotated.replace('\x00', '').encode('utf-8', errors='replace').decode('utf-8')
        user_content = f"Parse this load message:\n\n{annotated}"
        if chat_context:
            user_content += f"\n\nChat context (recent messages for reference):\n{chat_context}"

        messages.append({"role": "user", "content": user_content})
        return messages

    async def extract(
        self,
        text: str,
        chat_context: Optional[str] = None,
        chat_id: int = 0,
        message_id: int = 0,
        chat_title: str = "",
        sender_name: str = "",
        message_timestamp: Optional[datetime] = None,
    ) -> Optional[ExtractedLoad | List[ExtractedLoad]]:
        """Extract structured load data from raw message text.

        Returns ExtractedLoad, a list of ExtractedLoad (for multi-load messages),
        or None if extraction fails.
        """
        # Normalize to UTC and use Z suffix for Jackson Instant compatibility
        ts_dt = message_timestamp or datetime.now(timezone.utc)
        if ts_dt.tzinfo is not None:
            ts_dt = ts_dt.astimezone(timezone.utc)
        ts_iso = ts_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        def _enrich(load: ExtractedLoad) -> ExtractedLoad:
            """Fill in the listener-level fields that GPT doesn't know about."""
            load.chatId = chat_id
            load.chatTitle = chat_title
            load.messageId = message_id
            load.senderName = sender_name
            load.rawText = text
            load.messageTimestamp = ts_iso
            return load

        # Check cache first
        cached = await self._cache.get(text)
        if cached is not None:
            metrics.inc_cache_hit()
            logger.debug("Cache hit for message text")
            return _enrich(_parse_response(cached, text))

        metrics.inc_cache_miss()

        messages = self._build_messages(text, chat_context)

        last_error: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    logger.warning("Empty response from OpenAI")
                    metrics.inc_extraction_error()
                    return None

                parsed = json.loads(content)

                # Handle array response (multiple loads)
                if isinstance(parsed, list):
                    loads: List[ExtractedLoad] = []
                    for item in parsed:
                        loads.append(_enrich(_parse_response(item, text)))
                    # Cache the first item for dedup purposes
                    if loads:
                        await self._cache.set(text, parsed[0])
                    metrics.inc_extracted()
                    return loads

                # Handle {"loads": [...]} wrapper from GPT
                if isinstance(parsed, dict) and "loads" in parsed and isinstance(parsed["loads"], list):
                    loads = []
                    for item in parsed["loads"]:
                        loads.append(_enrich(_parse_response(item, text)))
                    if loads:
                        await self._cache.set(text, parsed["loads"][0])
                    metrics.inc_extracted()
                    return loads

                # Single load
                load = _enrich(_parse_response(parsed, text))
                await self._cache.set(text, parsed)
                metrics.inc_extracted()
                return load

            except json.JSONDecodeError as exc:
                logger.warning(
                    f"JSON parse error on attempt {attempt}/{self._config.max_retries}: {exc}"
                )
                last_error = exc
            except RateLimitError as exc:
                logger.warning(f"OpenAI rate limit on attempt {attempt}: {exc}")
                last_error = exc
                import asyncio
                await asyncio.sleep(2 ** attempt)
            except APITimeoutError as exc:
                logger.warning(f"OpenAI timeout on attempt {attempt}: {exc}")
                last_error = exc
            except APIError as exc:
                logger.error(f"OpenAI API error on attempt {attempt}: {exc}")
                last_error = exc
            except Exception as exc:
                logger.exception(f"Unexpected extraction error on attempt {attempt}")
                last_error = exc
                break

        metrics.inc_extraction_error()
        logger.error(f"Extraction failed after {self._config.max_retries} attempts: {last_error}")
        return None

    async def extract_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        chat_id: int = 0,
        message_id: int = 0,
        chat_title: str = "",
        sender_name: str = "",
        message_timestamp: Optional[datetime] = None,
    ) -> Optional[List[ExtractedLoad]]:
        """Extract structured load data from a loadboard screenshot.

        Uses GPT-4o vision to OCR the table and return structured JSON.
        Returns a list of ExtractedLoad (one per table row) or None on failure.
        """
        import base64

        ts_dt = message_timestamp or datetime.now(timezone.utc)
        if ts_dt.tzinfo is not None:
            ts_dt = ts_dt.astimezone(timezone.utc)
        ts_iso = ts_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Use gpt-4o for vision (gpt-4o-mini doesn't support images well)
        vision_model = "gpt-4o"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_image}"

        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all load rows from this loadboard screenshot. Return JSON only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            },
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=vision_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    logger.warning("Empty vision response from OpenAI")
                    return None

                parsed = json.loads(content)

                # Expect {"loads": [...]} from vision prompt
                raw_loads: list
                if isinstance(parsed, list):
                    raw_loads = parsed
                elif isinstance(parsed, dict) and "loads" in parsed:
                    raw_loads = parsed["loads"]
                else:
                    raw_loads = [parsed]

                results: List[ExtractedLoad] = []
                for i, item in enumerate(raw_loads):
                    load = _parse_response(item, json.dumps(item, ensure_ascii=False)[:200])
                    load.chatId = chat_id
                    load.chatTitle = chat_title
                    # Each row gets a unique messageId: base + row index
                    load.messageId = message_id * 1000 + i
                    load.senderName = sender_name
                    load.messageTimestamp = ts_iso
                    # Build rawText from extracted fields for display
                    origin = f"{load.originCity or ''}, {load.originState or ''}".strip(", ")
                    dest = f"{load.destCity or ''}, {load.destState or ''}".strip(", ")
                    rate_str = f"${load.rate:.0f}" if load.rate else ""
                    load.rawText = f"{origin} → {dest} {load.equipmentType or ''} {rate_str}".strip()
                    results.append(load)

                if results:
                    metrics.inc_extracted()
                    logger.info(f"Extracted {len(results)} load(s) from screenshot")
                return results

            except json.JSONDecodeError as exc:
                logger.warning(f"Vision JSON parse error on attempt {attempt}: {exc}")
                last_error = exc
            except RateLimitError as exc:
                logger.warning(f"OpenAI rate limit on vision attempt {attempt}: {exc}")
                last_error = exc
                import asyncio
                await asyncio.sleep(2 ** attempt)
            except APIError as exc:
                logger.error(f"OpenAI vision API error on attempt {attempt}: {exc}")
                last_error = exc
            except Exception as exc:
                logger.exception(f"Unexpected vision extraction error on attempt {attempt}")
                last_error = exc
                break

        metrics.inc_extraction_error()
        logger.error(f"Vision extraction failed after {self._config.max_retries} attempts: {last_error}")
        return None
