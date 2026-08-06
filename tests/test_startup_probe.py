"""Startup probe behavior: unreachable host must not kill the server.

Run with: python tests/test_startup_probe.py
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mealie.client import MealieClient  # noqa: E402


def _client(handler):
    """MealieClient whose transport is driven by ``handler``."""
    original = httpx.Client

    def patched(**kwargs):
        return original(transport=httpx.MockTransport(handler), **kwargs)

    httpx.Client = patched
    try:
        return MealieClient(base_url="http://mealie.test", api_key="k")
    finally:
        httpx.Client = original


def _timeout(request):
    raise httpx.ConnectTimeout("timed out", request=request)


def _unauthorized(request):
    return httpx.Response(401, json={"detail": "nope"})


def _bad_scheme(request):
    raise httpx.UnsupportedProtocol("no scheme", request=request)


def _ok(request):
    return httpx.Response(200, json={"version": "2.0.0"})


# Unreachable Mealie: construct anyway, retry per request.
client = _client(_timeout)
try:
    client._handle_request("GET", "/api/app/about")
    raise AssertionError("expected the request itself to fail")
except ConnectionError as e:
    assert "Connection error" in str(e), e

# Bad credentials still fail fast at startup.
try:
    _client(_unauthorized)
    raise AssertionError("expected HTTP 401 to abort startup")
except ConnectionError as e:
    assert "MEALIE_API_KEY" in str(e), e

# A misconfigured base URL is permanent, not transient: still fatal.
try:
    _client(_bad_scheme)
    raise AssertionError("expected UnsupportedProtocol to abort startup")
except httpx.UnsupportedProtocol:
    pass

# Happy path.
assert _client(_ok)._handle_request("GET", "/api/app/about")["version"] == "2.0.0"

print("ok")
