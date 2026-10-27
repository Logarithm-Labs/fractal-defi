"""Offline tests for the shared key redaction in :mod:`fractal.loaders._http`.

Regression: a key embedded in an RPC URL or in a query parameter must never
reach a raised error message, its exception chain, or a log record. Keys
here are synthetic; the "endpoints" are a closed local port or a patched
session, so nothing leaves the machine.
"""
import logging
import traceback

import pytest
import requests

from fractal.loaders import EtherscanClient, JsonRpcClient
from fractal.loaders._http import (
    KEY_REDACTOR,
    REDACTED,
    HttpClient,
    HttpConfig,
    LoaderHttpError,
    RedactingTransport,
    call_redacted,
    url_secrets,
)
from fractal.loaders.defillama import defillama as defillama_module

RPC_KEY = "synthetic0rpc0key0000000000000001"
ES_KEY = "SYNTHETICETHERSCANKEY000000000002"


def _assert_clean(err: BaseException, *secrets: str) -> None:
    rendered = "".join(traceback.format_exception(err))
    for secret in secrets:
        assert secret not in str(err)
        assert secret not in rendered
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.core
@pytest.mark.parametrize(("url", "expected"), [
    ("https://rpc.example", ()),
    ("https://rpc.example/", ()),
    (f"https://mainnet.example/v3/{RPC_KEY}", (RPC_KEY,)),
    (f"https://rpc.example/eth?apikey={RPC_KEY}&x=1", (RPC_KEY,)),
    ("https://user:longpassword@rpc.example", ("user", "longpassword")),
    ("https://rpc.example/eth/abc12", ("/eth/abc12",)),  # short key: whole path masked
])
def test_url_secrets(url, expected):
    assert url_secrets(url) == expected


@pytest.mark.core
def test_scrub_replaces_longest_secret_first():
    from fractal.loaders._http import KeyRedactingFilter
    assert KeyRedactingFilter.scrub("a /v3/KEYKEYKEY b KEYKEYKEY", "/v3/KEYKEYKEY", "KEYKEYKEY", "") == (
        f"a {REDACTED} b {REDACTED}")


@pytest.mark.core
def test_defillama_keeps_the_shared_redactor():
    assert defillama_module._KEY_REDACTOR is KEY_REDACTOR  # pylint: disable=protected-access


@pytest.mark.core
def test_call_redacted_copies_the_error_without_chain():
    def boom():
        raise LoaderHttpError(f"HTTP 401 for https://h/{RPC_KEY}") from ValueError(RPC_KEY)

    with pytest.raises(LoaderHttpError) as caught:
        call_redacted((RPC_KEY,), boom)
    assert REDACTED in str(caught.value)
    _assert_clean(caught.value, RPC_KEY)


@pytest.mark.core
def test_call_redacted_falls_back_when_the_type_needs_more_args():
    class Picky(Exception):
        def __init__(self, a, b):
            super().__init__(f"{a} {b}")

    def boom():
        raise Picky("x", RPC_KEY)

    with pytest.raises(LoaderHttpError, match=REDACTED):
        call_redacted((RPC_KEY,), boom)


@pytest.mark.core
def test_keyless_rpc_url_keeps_the_transport_untouched():
    fake = object()
    assert JsonRpcClient("https://rpc.example", http=fake)._http is fake  # pylint: disable=protected-access
    keyed = JsonRpcClient(f"https://rpc.example/v3/{RPC_KEY}", http=fake)
    assert isinstance(keyed._http, RedactingTransport)  # pylint: disable=protected-access
    assert RPC_KEY not in repr(keyed._http)  # pylint: disable=protected-access


@pytest.mark.core
def test_keyed_rpc_url_never_reaches_transport_errors_or_logs(caplog):
    """Closed local port: urllib3 retries (logging the path) then fails."""
    http = HttpClient(HttpConfig(timeout=0.5, max_retries=2, backoff_factor=0.0))
    rpc = JsonRpcClient(f"http://127.0.0.1:9/v3/{RPC_KEY}", http=http)
    with caplog.at_level(logging.DEBUG, logger="urllib3"), pytest.raises(requests.ConnectionError) as caught:
        rpc.block_number()
    assert any(r.name.startswith("urllib3") for r in caplog.records)
    assert RPC_KEY not in caplog.text
    for record in caplog.records:
        assert RPC_KEY not in record.getMessage()
    assert REDACTED in str(caught.value)
    _assert_clean(caught.value, RPC_KEY)


def _http_error_session(monkeypatch, http: HttpClient, method: str) -> None:
    """Patch ``session.<method>`` to answer HTTP 401 with a body echoing the request."""

    def respond(url, params=None, json=None, timeout=None, headers=None):
        resp = requests.Response()
        resp.status_code = 401
        resp.url = url
        resp._content = f"invalid key in {url} {params}".encode()  # pylint: disable=protected-access
        return resp

    monkeypatch.setattr(http.session, method, respond)


@pytest.mark.core
def test_keyed_rpc_url_never_reaches_http_status_errors(monkeypatch):
    http = HttpClient()
    _http_error_session(monkeypatch, http, "post")
    rpc = JsonRpcClient(f"https://mainnet.example/v3/{RPC_KEY}", http=http)
    with pytest.raises(LoaderHttpError) as caught:
        rpc.eth_call("0x" + "11" * 20, "0x")
    assert "HTTP 401" in str(caught.value)
    _assert_clean(caught.value, RPC_KEY)


@pytest.mark.core
def test_etherscan_query_key_never_reaches_errors_or_logs(monkeypatch, caplog):
    http = HttpClient()
    _http_error_session(monkeypatch, http, "get")
    es = EtherscanClient(api_key=ES_KEY, http=http, min_interval=0.0)
    with caplog.at_level(logging.DEBUG), pytest.raises(LoaderHttpError) as caught:
        es.block_by_time(1_700_000_000)
    assert "params=" in str(caught.value)  # HttpClient._check echoes params: still scrubbed
    _assert_clean(caught.value, ES_KEY)
    assert ES_KEY not in caplog.text
    assert ES_KEY not in repr(es)


@pytest.mark.core
def test_redactor_scrubs_registered_secrets_only_while_active():
    record = logging.LogRecord("urllib3.connectionpool", logging.DEBUG, __file__, 1,
                               '"POST %s HTTP/1.1" 401', (f"/v3/{RPC_KEY}",), None)
    with KEY_REDACTOR.redacting(RPC_KEY, ""):
        KEY_REDACTOR.filter(record)
    assert RPC_KEY not in record.getMessage()
    later = logging.LogRecord("urllib3.connectionpool", logging.DEBUG, __file__, 1, "%s", ("other",), None)
    KEY_REDACTOR.filter(later)
    assert later.getMessage() == "other"


class _RecordingTransport:
    """Records each ``post`` call's keyword names; fails on demand."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def post(self, url, **kwargs):
        self.calls.append((url, sorted(kwargs)))
        if self.fail is not None:
            raise self.fail
        return {"jsonrpc": "2.0", "id": kwargs["json"]["id"], "result": "0x10"}


@pytest.mark.core
def test_keyless_rpc_url_behaviour_is_unchanged():
    """Key-less URLs (the existing Uniswap/Morpho/Pendle flows in tests) see the
    original transport: same call shape, same exception object and chain."""
    ok = _RecordingTransport()
    rpc = JsonRpcClient("https://rpc.example", http=ok)
    assert rpc.block_number() == 16
    assert ok.calls == [("https://rpc.example", ["json"])]  # no extra timeout/headers kwargs

    cause = ValueError("upstream")
    error = LoaderHttpError("HTTP 500 for https://rpc.example params={}")
    error.__cause__ = cause
    rpc = JsonRpcClient("https://rpc.example", http=_RecordingTransport(fail=error))
    with pytest.raises(LoaderHttpError) as caught:
        rpc.block_number()
    assert caught.value is error and caught.value.__cause__ is cause
