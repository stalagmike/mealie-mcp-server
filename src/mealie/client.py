import json
import logging
import re
import traceback
from typing import Any, Dict

import httpx
from httpx import HTTPStatusError, ReadTimeout

logger = logging.getLogger("mealie-mcp")

# Keys whose values must never appear in logs. Matched case-insensitively as a
# substring of the key name, so e.g. "currentPassword" and "password_confirm"
# are both caught by "password".
_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "token",
    "secret",
    "apikey",
    "api_key",
    "authorization",
)
_REDACTED = "***REDACTED***"

# Endpoints whose response bodies contain freshly minted credentials; we redact
# their full bodies rather than trying to inspect arbitrary fields.
_SENSITIVE_RESPONSE_PATHS = (
    "/api/users/api-tokens",
    "/api/users/password",
    "/api/users/forgot-password",
    "/api/users/reset-password",
    "/api/auth/token",
    "/api/auth/refresh",
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _REDACTED if _is_sensitive_key(k) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _is_sensitive_url(url: str) -> bool:
    return any(path in url for path in _SENSITIVE_RESPONSE_PATHS)


_BYTES_KV_RE = re.compile(rb'"([^"]+)"\s*:\s*"[^"]*"')


def _redact_request_bytes(content: bytes | None) -> str:
    """Best-effort redaction of a captured request body for debug logging.

    We try JSON first (the common case for this client). If that fails — e.g.
    multipart uploads — we fall back to a regex pass that masks any string
    value whose key matches a sensitive fragment, without trying to fully
    parse the multipart envelope.
    """
    if not content:
        return ""
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        def _mask(match: re.Match[bytes]) -> bytes:
            key = match.group(1).decode("utf-8", errors="replace")
            if _is_sensitive_key(key):
                return f'"{key}":"{_REDACTED}"'.encode()
            return match.group(0)

        return _BYTES_KV_RE.sub(_mask, content).decode("utf-8", errors="replace")
    return json.dumps(_redact(parsed))


class MealieApiError(Exception):
    """Custom exception for Mealie API errors with status code and response details."""

    def __init__(self, status_code: int, message: str, response_text: str = None):
        self.status_code = status_code
        self.message = message
        self.response_text = response_text
        super().__init__(f"{message} (Status Code: {status_code})")


class MealieClient:

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        if not base_url:
            raise ValueError("Base URL cannot be empty")
        if not api_key:
            raise ValueError("API key cannot be empty")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")

        logger.debug(
            {
                "message": "Initializing MealieClient",
                "base_url": base_url,
                "timeout": timeout,
            }
        )
        try:
            self._client = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    # Don't set Content-Type here - let httpx set it per request
                    # This allows multipart/form-data uploads to work correctly
                },
                timeout=timeout,
            )
            # Test connection
            logger.debug({"message": "Testing connection to Mealie API"})
            response = self._client.get("/api/app/about")
            response.raise_for_status()
            logger.info({"message": "Successfully connected to Mealie API"})
        except HTTPStatusError as e:
            status_code = e.response.status_code
            hint = (
                "verify MEALIE_API_KEY is valid"
                if status_code in (401, 403)
                else "verify MEALIE_BASE_URL points at a Mealie instance"
            )
            error_msg = (
                f"Mealie API at {base_url} returned HTTP {status_code} during "
                f"startup connectivity check; {hint}."
            )
            logger.error({"message": error_msg})
            logger.debug(
                {"message": "Error traceback", "traceback": traceback.format_exc()}
            )
            raise ConnectionError(error_msg) from e
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            # Transient network failure (connect timeout, DNS, refused, read
            # timeout): the Mealie host is often just not reachable *yet* —
            # VPN still connecting, container still booting, laptop just
            # resumed. Taking the whole MCP server down for that means the
            # client never reconnects; every tool call re-attempts the request
            # and reports the failure itself, so start up degraded instead.
            logger.warning(
                {
                    "message": (
                        f"Mealie API at {base_url} is not reachable yet "
                        f"({type(e).__name__}: {e}); starting anyway and will "
                        "retry on the first request."
                    )
                }
            )
        except Exception as e:
            error_msg = f"Error initializing Mealie client: {str(e)}"
            logger.error({"message": error_msg})
            logger.debug(
                {"message": "Error traceback", "traceback": traceback.format_exc()}
            )
            raise

    def _handle_request(self, method: str, url: str, **kwargs) -> Dict[str, Any] | str:
        """Common request handler with error handling for all API calls.

        Supports:
        - JSON requests via json= parameter
        - Multipart uploads via files= parameter
        - Form data via data= parameter
        """
        try:
            logger.debug(
                {
                    "message": "Making API request",
                    "method": method,
                    "url": url,
                    "body": _redact(kwargs.get("json")),
                    "has_files": "files" in kwargs,
                }
            )

            if "params" in kwargs:
                logger.debug(
                    {"message": "Request parameters", "params": _redact(kwargs["params"])}
                )
            if "json" in kwargs:
                logger.debug(
                    {"message": "Request payload", "payload": _redact(kwargs["json"])}
                )
            if "files" in kwargs:
                logger.debug({"message": "Request has file upload"})

            # For JSON requests, explicitly set Content-Type
            # For multipart requests (files), httpx will set the correct Content-Type with boundary
            if "json" in kwargs and "files" not in kwargs:
                if "headers" not in kwargs:
                    kwargs["headers"] = {}
                kwargs["headers"]["Content-Type"] = "application/json"

            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses

            logger.debug(
                {"message": "Request successful", "status_code": response.status_code}
            )

            # Handle empty responses (common for DELETE operations)
            if response.status_code == 204 or (not response.content or len(response.content) == 0):
                logger.debug({"message": "Response has no content (likely successful DELETE)"})
                return {"success": True, "message": "Operation completed successfully"}

            # Log the response content at debug level
            try:
                response_data = response.json()

                # Normalize JSON null to success dict (common for DELETE operations)
                if response_data is None:
                    logger.debug(
                        {"message": "Response content is null; normalizing to success payload"}
                    )
                    return {"success": True, "message": "Operation completed successfully"}

                if _is_sensitive_url(url):
                    logger.debug(
                        {"message": "Response content", "data": _REDACTED, "url": url}
                    )
                else:
                    logger.debug(
                        {"message": "Response content", "data": _redact(response_data)}
                    )
                return response_data
            except json.JSONDecodeError:
                # If we can't decode JSON but got a successful response, treat as success
                logged_text = _REDACTED if _is_sensitive_url(url) else response.text
                if response.status_code >= 200 and response.status_code < 300:
                    logger.debug(
                        {"message": "Response content (non-JSON but successful)", "content": logged_text}
                    )
                    if not response.text or response.text.strip() == "":
                        return {"success": True, "message": "Operation completed successfully"}
                    return response.text
                else:
                    logger.debug(
                        {"message": "Response content (non-JSON)", "content": logged_text}
                    )
                    return response.text

        except HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = f"HTTP Error {status_code}"

            # Try to parse error details from response
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text

            error_msg = f"API error for {method} {url}: {error_detail}"
            logger.error(
                {
                    "message": "API request failed",
                    "method": method,
                    "url": url,
                    "status_code": status_code,
                    "error_detail": error_detail,
                }
            )
            logger.debug(
                {
                    "message": "Failed Request body",
                    "content": _redact_request_bytes(e.request.content),
                }
            )
            raise MealieApiError(status_code, error_msg, e.response.text) from e

        except ReadTimeout:
            error_msg = f"Request timeout for {method} {url}"
            logger.error({"message": error_msg, "method": method, "url": url})
            logger.debug(
                {"message": "Error traceback", "traceback": traceback.format_exc()}
            )
            raise TimeoutError(error_msg)

        except httpx.TransportError as e:
            # Covers ConnectError, ConnectTimeout and friends — without this a
            # connect timeout escaped as a bare httpx error whose message is
            # just "timed out".
            error_msg = f"Connection error for {method} {url}: {type(e).__name__}: {e}"
            logger.error(
                {"message": error_msg, "method": method, "url": url, "error": str(e)}
            )
            logger.debug(
                {"message": "Error traceback", "traceback": traceback.format_exc()}
            )
            raise ConnectionError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error for {method} {url}: {str(e)}"
            logger.error(
                {"message": error_msg, "method": method, "url": url, "error": str(e)}
            )
            logger.debug(
                {"message": "Error traceback", "traceback": traceback.format_exc()}
            )
            raise
