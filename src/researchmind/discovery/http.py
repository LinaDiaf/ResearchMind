"""HTTP access for scholarly APIs: per-host rate limiting, retries with backoff, disk cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from researchmind import __version__


class DiscoveryError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


class RateLimitedClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_s: int = 7 * 24 * 3600,
        min_interval_s: dict[str, float] | None = None,
        contact_email: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        ua = f"ResearchMind/{__version__}"
        if contact_email:
            ua += f" (mailto:{contact_email})"
        self._client = client or httpx.Client(
            timeout=30.0, headers={"User-Agent": ua}, follow_redirects=True
        )
        self.cache_dir = cache_dir
        self.cache_ttl_s = cache_ttl_s
        self.min_interval = min_interval_s or {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        interval = self.min_interval.get(host, 0.0)
        with self._lock:
            now = time.monotonic()
            wait = self._last.get(host, 0.0) + interval - now
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(json.dumps([url, params or {}], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / "http" / f"{key}.json"

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None, headers: dict | None) -> httpx.Response:
        self._wait(httpx.URL(url).host)
        r = self._client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> Any:
        return json.loads(self.get_text(url, params, headers))

    def get_text(self, url: str, params: dict | None = None, headers: dict | None = None) -> str:
        path = self._cache_path(url, params)
        if path and path.exists() and time.time() - path.stat().st_mtime < self.cache_ttl_s:
            return json.loads(path.read_text())["body"]
        try:
            body = self._get(url, params, headers).text
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"request to {httpx.URL(url).host} failed: {exc}") from exc
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "body": body}))
        return body

    def get_bytes(self, url: str, max_bytes: int = 60 * 1024 * 1024) -> bytes:
        self._wait(httpx.URL(url).host)
        try:
            with self._client.stream("GET", url) as r:
                r.raise_for_status()
                chunks, size = [], 0
                for part in r.iter_bytes():
                    size += len(part)
                    if size > max_bytes:
                        raise DiscoveryError("file too large")
                    chunks.append(part)
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"download failed: {exc}") from exc
        return b"".join(chunks)
