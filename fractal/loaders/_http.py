"""Shared HTTP client used by every loader.

A single ``requests.Session`` with retry/backoff and ``Retry-After``
support. Loaders MUST go through this module so transport behavior
(timeouts, retries, error messages) stays consistent.

Key redaction: transports whose URL or query carries a secret (a keyed
RPC endpoint, an ``apikey`` parameter) go through :func:`call_redacted`
or :class:`RedactingTransport`. Failures are re-raised as redacted copies
detached from the key-bearing original, and urllib3 log records that
print request paths are scrubbed while the request runs.
:meth:`HttpClient._check` itself is unchanged: redaction wraps it.
"""
import contextlib
import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


@dataclass
class HttpConfig:
    timeout: float = 15.0
    max_retries: int = 5
    backoff_factor: float = 0.5
    status_forcelist: tuple = (408, 425, 429, 500, 502, 503, 504)


class LoaderHttpError(RuntimeError):
    """Wraps ``requests.HTTPError`` with the response body for easier debugging."""


class HttpClient:
    """A retrying ``requests.Session`` wrapper.

    Methods accept full URLs so per-protocol clients can carry their own
    base URL or override timeout per call.
    """

    def __init__(self, cfg: HttpConfig | None = None) -> None:
        self.cfg = cfg or HttpConfig()
        self.session = requests.Session()

        retry = Retry(
            total=self.cfg.max_retries,
            read=self.cfg.max_retries,
            connect=self.cfg.max_retries,
            status=self.cfg.max_retries,
            status_forcelist=self.cfg.status_forcelist,
            allowed_methods=("GET", "POST"),
            backoff_factor=self.cfg.backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _check(self, resp: requests.Response, url: str, params: Any) -> Any:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LoaderHttpError(
                f"HTTP {resp.status_code} for {url} params={params}: {resp.text[:300]}"
            ) from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise LoaderHttpError(
                f"Non-JSON response from {url}: {resp.text[:300]}"
            ) from exc

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        resp = self.session.get(
            url,
            params=params or {},
            timeout=timeout or self.cfg.timeout,
        )
        return self._check(resp, url, params)

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        resp = self.session.post(
            url,
            json=json,
            timeout=timeout or self.cfg.timeout,
            headers=headers or {"Content-Type": "application/json"},
        )
        return self._check(resp, url, json)


# ------------------------------------------------------------- redaction
REDACTED = "<redacted>"
# urllib3 loggers whose records can carry the request path (and so the key).
_REDACTED_LOGGERS = ("urllib3.connectionpool", "urllib3.util.retry", "urllib3.poolmanager")
# URL parts shorter than this are treated as routing ("v3", "eth"), not keys.
_MIN_SECRET_LEN = 8

_T = TypeVar("_T")


class KeyRedactingFilter(logging.Filter):
    """Scrub the secrets of in-flight requests from urllib3 log records.

    One shared instance (:data:`KEY_REDACTOR`) is installed once on every
    logger in ``_REDACTED_LOGGERS``; each request registers its secrets
    only while it runs. Installing once, instead of adding and removing a
    filter per request, avoids racing with a concurrent request whose
    record is being filtered while the logger's filter list changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._secrets: Counter[str] = Counter()
        self._installed = False

    @staticmethod
    def scrub(text: str, *secrets: str) -> str:
        """``text`` with every non-empty secret replaced, longest first."""
        for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        return text

    @contextlib.contextmanager
    def redacting(self, *secrets: str) -> Iterator[None]:
        """Redact ``secrets`` from urllib3 log records inside the block."""
        live = [s for s in secrets if s]
        with self._lock:
            if not self._installed:
                for name in _REDACTED_LOGGERS:
                    logging.getLogger(name).addFilter(self)
                self._installed = True
            self._secrets.update(live)
        try:
            yield
        finally:
            with self._lock:
                for secret in live:
                    self._secrets[secret] -= 1
                    if self._secrets[secret] <= 0:
                        del self._secrets[secret]

    def filter(self, record: logging.LogRecord) -> bool:
        with self._lock:
            secrets = list(self._secrets)
        if not secrets:
            return True
        message = record.getMessage()
        redacted = self.scrub(message, *secrets)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


KEY_REDACTOR = KeyRedactingFilter()


def url_secrets(url: str) -> tuple[str, ...]:
    """Parts of ``url`` that may carry a credential.

    Userinfo, every path segment and query value of at least eight
    characters (API keys are long; ``v3`` or ``eth`` are routing). When no
    part qualifies but the URL has a path or query, the whole path+query
    is returned so the URL form is still masked. A bare ``scheme://host``
    yields ``()``.
    """
    parts = urlsplit(url)
    found: list[str] = [p for p in (parts.username, parts.password) if p]
    found += [seg for seg in parts.path.split("/") if len(seg) >= _MIN_SECRET_LEN]
    found += [value for _, value in parse_qsl(parts.query) if len(value) >= _MIN_SECRET_LEN]
    if not found and (parts.path.strip("/") or parts.query):
        found.append(parts.path.rstrip("/") + (f"?{parts.query}" if parts.query else ""))
    return tuple(dict.fromkeys(found))


def _scrubbed_copy(exc: Exception, secrets: tuple[str, ...]) -> Exception:
    message = KeyRedactingFilter.scrub(str(exc), *secrets)
    try:
        return type(exc)(message)
    except Exception:  # pylint: disable=broad-exception-caught
        return LoaderHttpError(message)  # the type needs more than a message


def call_redacted(secrets: Iterable[str], fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """``fn(*args, **kwargs)`` with ``secrets`` kept out of errors and urllib3 logs.

    A failure is re-raised as a same-type copy whose message is scrubbed,
    with neither ``__cause__`` nor ``__context__`` pointing at the
    key-bearing original (its URL, params or response body may echo the
    key). Other sinks, e.g. third-party HTTP debugging hooks, are not
    covered. With no secrets the call is passed through untouched.
    """
    live = tuple(s for s in secrets if s)
    if not live:
        return fn(*args, **kwargs)
    failure: Exception | None = None
    with KEY_REDACTOR.redacting(*live):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failure = _scrubbed_copy(exc, live)
    # Raised outside the ``except`` block: ``from None`` alone still leaves
    # ``__context__`` set to the original.
    raise failure


class RedactingTransport:
    """An :class:`HttpClient`-shaped wrapper that routes ``get``/``post``
    through :func:`call_redacted`; other attributes pass through."""

    def __init__(self, inner: Any, secrets: Iterable[str]) -> None:
        self._inner = inner
        self._secrets = tuple(s for s in secrets if s)

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        return call_redacted(self._secrets, self._inner.get, url, params=params, timeout=timeout)

    def post(self, url: str, json: dict[str, Any] | None = None, timeout: float | None = None,
             headers: dict[str, str] | None = None) -> Any:
        return call_redacted(self._secrets, self._inner.post, url, json=json, timeout=timeout, headers=headers)

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":  # not set yet (copy/pickle): avoid recursing
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"RedactingTransport({self._inner.__class__.__name__}, secrets={REDACTED})"
