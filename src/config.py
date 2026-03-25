"""Load YAML config + environment variables into typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class TelegramConfig:
    api_id: int
    api_hash: str
    phone: str
    session_name: str = "loadboard_listener"
    max_reconnect_attempts: int = 10
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 300.0
    flood_wait_threshold: int = 60


@dataclass(frozen=True)
class FiltersConfig:
    whitelist_group_ids: List[int] = field(default_factory=list)
    blacklist_group_ids: List[int] = field(default_factory=list)
    skip_bots: bool = True
    min_length: int = 15
    skip_media_only: bool = True
    keyword_patterns: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionConfig:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    cache_ttl: int = 86400
    max_retries: int = 3


@dataclass(frozen=True)
class BackendConfig:
    url: str = "http://localhost:8000"
    token: str = ""
    endpoint: str = "/api/telegram-loads"
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    batch_size: int = 10
    flush_interval: int = 30


@dataclass(frozen=True)
class AppConfig:
    telegram: TelegramConfig
    filters: FiltersConfig
    extraction: ExtractionConfig
    backend: BackendConfig
    log_level: str = "INFO"
    db_path: str = "data/buffer.db"
    cache_db_path: str = "data/cache.db"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file and environment variables.

    Environment variables take precedence over YAML values.
    """
    load_dotenv()

    path = Path(config_path or os.getenv("CONFIG_PATH", "config.yaml"))
    yaml_data: dict = {}
    if path.exists():
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

    tg_yaml = yaml_data.get("telegram", {})
    filters_yaml = yaml_data.get("filters", {})
    extraction_yaml = yaml_data.get("extraction", {})
    backend_yaml = yaml_data.get("backend", {})

    api_id_raw = os.getenv("API_ID", "")
    if not api_id_raw:
        raise ValueError("API_ID environment variable is required")

    api_hash = os.getenv("API_HASH", "")
    if not api_hash:
        raise ValueError("API_HASH environment variable is required")

    phone = os.getenv("PHONE", "")
    if not phone:
        raise ValueError("PHONE environment variable is required")

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    telegram_config = TelegramConfig(
        api_id=int(api_id_raw),
        api_hash=api_hash,
        phone=phone,
        session_name=tg_yaml.get("session_name", "loadboard_listener"),
        max_reconnect_attempts=tg_yaml.get("max_reconnect_attempts", 10),
        reconnect_base_delay=tg_yaml.get("reconnect_base_delay", 1.0),
        reconnect_max_delay=tg_yaml.get("reconnect_max_delay", 300.0),
        flood_wait_threshold=tg_yaml.get("flood_wait_threshold", 60),
    )

    filters_config = FiltersConfig(
        whitelist_group_ids=filters_yaml.get("whitelist_group_ids", []),
        blacklist_group_ids=filters_yaml.get("blacklist_group_ids", []),
        skip_bots=filters_yaml.get("skip_bots", True),
        min_length=filters_yaml.get("min_length", 15),
        skip_media_only=filters_yaml.get("skip_media_only", True),
        keyword_patterns=filters_yaml.get("keyword_patterns", []),
    )

    extraction_config = ExtractionConfig(
        openai_api_key=openai_api_key,
        model=extraction_yaml.get("model", "gpt-4o-mini"),
        temperature=extraction_yaml.get("temperature", 0.0),
        max_tokens=extraction_yaml.get("max_tokens", 1024),
        cache_ttl=extraction_yaml.get("cache_ttl", 86400),
        max_retries=extraction_yaml.get("max_retries", 3),
    )

    backend_url = os.getenv("BACKEND_URL", backend_yaml.get("url", "http://localhost:8000"))
    backend_token = os.getenv("BACKEND_TOKEN", "")

    backend_config = BackendConfig(
        url=backend_url,
        token=backend_token,
        endpoint=backend_yaml.get("endpoint", "/api/telegram-loads"),
        max_retries=backend_yaml.get("max_retries", 3),
        retry_base_delay=backend_yaml.get("retry_base_delay", 1.0),
        retry_max_delay=backend_yaml.get("retry_max_delay", 30.0),
        batch_size=backend_yaml.get("batch_size", 10),
        flush_interval=backend_yaml.get("flush_interval", 30),
    )

    return AppConfig(
        telegram=telegram_config,
        filters=filters_config,
        extraction=extraction_config,
        backend=backend_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        db_path=os.getenv("DB_PATH", "data/buffer.db"),
        cache_db_path=os.getenv("CACHE_DB_PATH", "data/cache.db"),
    )
