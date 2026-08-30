"""Shared plumbing for data providers.

Every provider fetches from a public source and writes a raw copy into the
on-disk cache. The cache is what makes the demo reproducible: once populated,
the whole pipeline runs with the network unplugged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Repo root -> data/cache. Kept next to the code so a clone is self-contained.
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshot"

DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class ProviderError(RuntimeError):
    """A provider could not produce data from either the network or the cache."""


@dataclass
class FetchResult:
    """Payload plus where it came from.

    `source` is surfaced all the way up to the agent so an answer can say
    "this came from cached data" instead of quietly implying it is live.
    """

    payload: Any
    source: str  # "live" | "cache" | "snapshot"
    fetched_at: float
    url: str | None = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


@dataclass
class HttpCache:
    """Content-addressed response cache.

    Keyed by a hash of the full request URL so distinct queries against the
    same ArcGIS layer do not collide.
    """

    cache_dir: Path = field(default=CACHE_DIR)
    ttl: float = DEFAULT_TTL_SECONDS

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def read(self, key: str, *, ignore_ttl: bool = False) -> FetchResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt cache entry, ignoring: %s", path)
            return None

        fetched_at = envelope.get("fetched_at", 0.0)
        if not ignore_ttl and (time.time() - fetched_at) > self.ttl:
            return None
        return FetchResult(
            payload=envelope["payload"],
            source="cache",
            fetched_at=fetched_at,
            url=envelope.get("url"),
        )

    def write(self, key: str, payload: Any, url: str | None = None) -> FetchResult:
        fetched_at = time.time()
        envelope = {"fetched_at": fetched_at, "url": url, "payload": payload}
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope))
        tmp.replace(self._path(key))
        return FetchResult(payload=payload, source="live", fetched_at=fetched_at, url=url)


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    cache: HttpCache | None = None,
    refresh: bool = False,
    timeout: float = 45.0,
) -> FetchResult:
    """GET JSON, preferring the cache unless `refresh` is set.

    On a network failure we fall back to a stale cache entry rather than
    failing the request - a slightly old number beats no answer, as long as
    the staleness is reported (which `FetchResult.source` does).
    """
    cache = cache or HttpCache()
    key = httpx.URL(url, params=params or {}).__str__()

    if not refresh:
        hit = cache.read(key)
        if hit is not None:
            log.debug("cache hit: %s", key[:120])
            return hit

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        stale = cache.read(key, ignore_ttl=True)
        if stale is not None:
            log.warning("fetch failed (%s); serving stale cache for %s", exc, key[:80])
            return stale
        raise ProviderError(f"fetch failed and no cache available: {url}") from exc

    return cache.write(key, payload, url=key)


def fetch_text(
    url: str,
    *,
    cache: HttpCache | None = None,
    refresh: bool = False,
    timeout: float = 90.0,
) -> FetchResult:
    """GET a text body (used for the OurAirports CSV)."""
    cache = cache or HttpCache()
    key = url

    if not refresh:
        hit = cache.read(key)
        if hit is not None:
            return hit

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.text
    except httpx.HTTPError as exc:
        stale = cache.read(key, ignore_ttl=True)
        if stale is not None:
            log.warning("fetch failed (%s); serving stale cache for %s", exc, url)
            return stale
        raise ProviderError(f"fetch failed and no cache available: {url}") from exc

    return cache.write(key, payload, url=url)
