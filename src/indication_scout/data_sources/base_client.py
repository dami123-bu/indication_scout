"""
Base client for all external data source clients.

Provides: caching, rate limiting, retry with exponential backoff,
structured logging, temporal filtering support, and graceful degradation.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger("indication_scout.data_sources")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Retry behaviour for failed requests."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    backoff_factor: float = 2.0
    retryable_status_codes: set[int] = {429, 500, 502, 503, 504}


class RateLimitConfig(BaseModel):
    """Token-bucket rate limiter settings."""

    requests_per_second: float = 5.0
    burst: int = 10


class CacheConfig(BaseModel):
    """Disk cache settings."""

    enabled: bool = True
    directory: Path = Path("_cache")
    ttl_seconds: int = 86400  # 24 hours default


class ClientConfig(BaseModel):
    """Top-level config aggregating retry, rate limit, and cache."""

    retry: RetryConfig = RetryConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    cache: CacheConfig = CacheConfig()
    timeout_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Rate limiter (async token bucket)
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """
    Async token-bucket rate limiter.

    Allows `burst` requests immediately, then refills at
    `requests_per_second`.  Callers await `acquire()` before
    making a request — it sleeps only when the bucket is empty.
    """

    def __init__(self, config: RateLimitConfig):
        self.rate = config.requests_per_second
        self.max_tokens = config.burst
        self.tokens = float(config.burst)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                logger.debug("Rate limiter: sleeping %.2fs", wait)
                await asyncio.sleep(wait)
                self.tokens = 0.0
                self.last_refill = time.monotonic()
            else:
                self.tokens -= 1.0


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class DiskCache:
    """
    Simple JSON disk cache keyed by a deterministic hash of the request.

    Each entry is a JSON file containing {"data": ..., "cached_at": ..., "ttl": ...}.
    Expired entries are treated as misses and overwritten on next store.
    """

    def __init__(self, config: CacheConfig):
        self.enabled = config.enabled
        self.directory = config.directory
        self.ttl = config.ttl_seconds
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(namespace: str, params: dict[str, Any]) -> str:
        raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    async def get(self, namespace: str, params: dict[str, Any]) -> Any | None:
        if not self.enabled:
            return None
        key = self._make_key(namespace, params)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(entry["cached_at"])
            age = (datetime.now() - cached_at).total_seconds()
            if age > entry.get("ttl", self.ttl):
                logger.debug("Cache expired for %s (age=%.0fs)", namespace, age)
                path.unlink(missing_ok=True)
                return None
            logger.debug("Cache hit for %s", namespace)
            return entry["data"]
        except (json.JSONDecodeError, KeyError, ValueError):
            path.unlink(missing_ok=True)
            return None

    async def set(
        self,
        namespace: str,
        params: dict[str, Any],
        data: Any,
        ttl: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        key = self._make_key(namespace, params)
        entry = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "ttl": ttl or self.ttl,
        }
        self._path(key).write_text(json.dumps(entry, default=str))

    async def invalidate(self, namespace: str, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        key = self._make_key(namespace, params)
        self._path(key).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Request context (for structured logging)
# ---------------------------------------------------------------------------


class RequestContext(BaseModel):
    """Metadata attached to every outgoing request for logging."""

    source: str  # e.g. "open_targets", "pubmed"
    method: str  # e.g. "get_target_associations"
    agent: str | None = None  # e.g. "biology", "critique" — set by caller
    params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataSourceError(Exception):
    """Base exception for data source failures."""

    def __init__(self, source: str, message: str, status_code: int | None = None):
        self.source = source
        self.status_code = status_code
        super().__init__(f"[{source}] {message}")


class RateLimitError(DataSourceError):
    """Raised when rate limit is exceeded and retries are exhausted."""

    pass


class TemporalFilterError(DataSourceError):
    """Raised when a temporal filter cannot be applied by the API."""

    pass


# ---------------------------------------------------------------------------
# Partial result wrapper
# ---------------------------------------------------------------------------


class PartialResult(BaseModel):
    """
    Wraps a response that may be incomplete due to errors or timeouts.

    Agents check `is_complete` and `errors` to decide how much to trust
    the data.  The Supervisor can log degraded responses.
    """

    data: Any
    is_complete: bool = True
    errors: list[str] = []
    cached: bool = False
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------


class BaseClient(ABC):
    """
    Abstract base for Open Targets, PubMed, and ClinicalTrials.gov clients.

    Subclasses implement `_source_name` and their own typed methods that
    call `_request()` or `_graphql()` / `_rest_get()` depending on API type.
    """

    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self.rate_limiter = TokenBucketRateLimiter(self.config.rate_limit)
        self.cache = DiskCache(self.config.cache)
        self._session: aiohttp.ClientSession | None = None

    @property
    @abstractmethod
    def _source_name(self) -> str:
        """Identifier for this data source, e.g. 'open_targets'."""
        ...

    # -- Session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- Core request with retry + cache + rate limiting ---------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_namespace: str | None = None,
        cache_params: dict[str, Any] | None = None,
        cache_ttl: int | None = None,
        context: RequestContext | None = None,
    ) -> PartialResult:
        """
        Make an HTTP request with caching, rate limiting, and retry.

        Parameters
        ----------
        method : str
            HTTP method — "GET" or "POST".
        url : str
            Full URL.
        params : dict, optional
            Query string parameters (for GET).
        json_body : dict, optional
            JSON body (for POST — used by GraphQL).
        headers : dict, optional
            Additional HTTP headers.
        cache_namespace : str, optional
            Cache key namespace.  If None, caching is skipped.
        cache_params : dict, optional
            Parameters used to build the cache key.
        cache_ttl : int, optional
            Override default cache TTL for this request.
        context : RequestContext, optional
            Logging context.
        """
        ctx = context or RequestContext(source=self._source_name, method="unknown")

        # --- Check cache first ---
        if cache_namespace and cache_params:
            cached = await self.cache.get(cache_namespace, cache_params)
            if cached is not None:
                return PartialResult(data=cached, cached=True)

        # --- Retry loop ---
        last_error: Exception | None = None
        start = time.monotonic()

        for attempt in range(self.config.retry.max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                session = await self._get_session()

                logger.info(
                    "Request [%s.%s] attempt=%d url=%s",
                    ctx.source,
                    ctx.method,
                    attempt + 1,
                    url,
                )

                if method.upper() == "GET":
                    resp = await session.get(url, params=params, headers=headers)
                else:
                    resp = await session.post(
                        url, json=json_body, params=params, headers=headers
                    )

                # --- Handle HTTP errors ---
                if resp.status in self.config.retry.retryable_status_codes:
                    body = await resp.text()
                    logger.warning(
                        "Retryable %d from %s.%s: %s",
                        resp.status,
                        ctx.source,
                        ctx.method,
                        body[:200],
                    )
                    last_error = DataSourceError(
                        ctx.source,
                        f"HTTP {resp.status}: {body[:200]}",
                        status_code=resp.status,
                    )
                    if resp.status == 429:
                        # Respect Retry-After header if present
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            await asyncio.sleep(float(retry_after))
                            continue

                    delay = min(
                        self.config.retry.base_delay
                        * (self.config.retry.backoff_factor**attempt),
                        self.config.retry.max_delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status >= 400:
                    body = await resp.text()
                    raise DataSourceError(
                        ctx.source,
                        f"HTTP {resp.status}: {body[:500]}",
                        status_code=resp.status,
                    )

                # --- Success ---
                data = await resp.json()
                elapsed = time.monotonic() - start

                logger.info(
                    "Success [%s.%s] elapsed=%.2fs cached=False",
                    ctx.source,
                    ctx.method,
                    elapsed,
                )

                # Store in cache
                if cache_namespace and cache_params:
                    await self.cache.set(
                        cache_namespace, cache_params, data, ttl=cache_ttl
                    )

                return PartialResult(data=data, elapsed_seconds=elapsed)

            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                last_error = DataSourceError(
                    ctx.source, f"Timeout after {elapsed:.1f}s"
                )
                logger.warning(
                    "Timeout [%s.%s] attempt=%d elapsed=%.1fs",
                    ctx.source,
                    ctx.method,
                    attempt + 1,
                    elapsed,
                )

            except aiohttp.ClientError as e:
                last_error = DataSourceError(ctx.source, f"Connection error: {e}")
                logger.warning(
                    "Connection error [%s.%s] attempt=%d: %s",
                    ctx.source,
                    ctx.method,
                    attempt + 1,
                    e,
                )

            # Exponential backoff before next attempt
            if attempt < self.config.retry.max_retries:
                delay = min(
                    self.config.retry.base_delay
                    * (self.config.retry.backoff_factor**attempt),
                    self.config.retry.max_delay,
                )
                await asyncio.sleep(delay)

        # --- All retries exhausted: graceful degradation ---
        elapsed = time.monotonic() - start
        logger.error(
            "All retries exhausted [%s.%s] after %.1fs: %s",
            ctx.source,
            ctx.method,
            elapsed,
            last_error,
        )
        return PartialResult(
            data=None,
            is_complete=False,
            errors=[str(last_error)],
            elapsed_seconds=elapsed,
        )

    # -- Convenience methods for subclasses ----------------------------------

    async def _rest_get(
        self,
        url: str,
        params: dict[str, Any],
        *,
        cache_namespace: str | None = None,
        cache_ttl: int | None = None,
        context: RequestContext | None = None,
    ) -> PartialResult:
        """Convenience wrapper for REST GET requests (PubMed, ClinicalTrials.gov)."""
        return await self._request(
            "GET",
            url,
            params=params,
            cache_namespace=cache_namespace,
            cache_params=params,
            cache_ttl=cache_ttl,
            context=context,
        )

    async def _graphql(
        self,
        url: str,
        query: str,
        variables: dict[str, Any],
        *,
        cache_namespace: str | None = None,
        cache_ttl: int | None = None,
        context: RequestContext | None = None,
    ) -> PartialResult:
        """Convenience wrapper for GraphQL POST requests (Open Targets)."""
        json_body = {"query": query, "variables": variables}
        result = await self._request(
            "POST",
            url,
            json_body=json_body,
            headers={"Content-Type": "application/json"},
            cache_namespace=cache_namespace,
            cache_params={
                "query_hash": hashlib.md5(query.encode()).hexdigest(),
                **variables,
            },
            cache_ttl=cache_ttl,
            context=context,
        )

        # Check for GraphQL errors
        if result.data and "errors" in result.data:
            ctx = context or RequestContext(source=self._source_name, method="unknown")
            error_messages = [e.get("message", str(e)) for e in result.data["errors"]]
            raise DataSourceError(
                ctx.source,
                f"GraphQL errors: {error_messages}",
            )

        return result

    # -- Temporal filtering helper -------------------------------------------

    @staticmethod
    def apply_date_filter(
        params: dict[str, Any],
        date_before: date | None,
        date_param_name: str,
        date_format: str = "%Y-%m-%d",
    ) -> dict[str, Any]:
        """
        Add a temporal cutoff to request params if date_before is set.

        This is the hook for temporal holdout evaluation — every client
        uses this to enforce the "pretend it's before this date" constraint.
        """
        if date_before is not None:
            params = {**params, date_param_name: date_before.strftime(date_format)}
        return params
