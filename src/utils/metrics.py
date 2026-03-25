"""Simple in-process counters for observability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class Metrics:
    messages_received: int = 0
    messages_filtered: int = 0
    loads_extracted: int = 0
    loads_sent: int = 0
    loads_buffered: int = 0
    extraction_errors: int = 0
    backend_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def inc_received(self) -> None:
        self.messages_received += 1

    def inc_filtered(self) -> None:
        self.messages_filtered += 1

    def inc_extracted(self) -> None:
        self.loads_extracted += 1

    def inc_sent(self, count: int = 1) -> None:
        self.loads_sent += count

    def inc_buffered(self) -> None:
        self.loads_buffered += 1

    def inc_extraction_error(self) -> None:
        self.extraction_errors += 1

    def inc_backend_error(self) -> None:
        self.backend_errors += 1

    def inc_cache_hit(self) -> None:
        self.cache_hits += 1

    def inc_cache_miss(self) -> None:
        self.cache_misses += 1

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def print_stats(self) -> None:
        """Log current metric values."""
        uptime = self.uptime_seconds
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            "Metrics snapshot | "
            f"uptime={hours:02d}h{minutes:02d}m{seconds:02d}s | "
            f"received={self.messages_received} | "
            f"filtered={self.messages_filtered} | "
            f"extracted={self.loads_extracted} | "
            f"sent={self.loads_sent} | "
            f"buffered={self.loads_buffered} | "
            f"cache_hits={self.cache_hits} | "
            f"cache_misses={self.cache_misses} | "
            f"extraction_errors={self.extraction_errors} | "
            f"backend_errors={self.backend_errors}"
        )

    def as_dict(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "messages_received": self.messages_received,
            "messages_filtered": self.messages_filtered,
            "loads_extracted": self.loads_extracted,
            "loads_sent": self.loads_sent,
            "loads_buffered": self.loads_buffered,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "extraction_errors": self.extraction_errors,
            "backend_errors": self.backend_errors,
        }


# Global singleton
metrics = Metrics()
