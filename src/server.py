import argparse
import ipaddress
import logging
import os
import stat
import sys
import traceback

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional convenience dependency
    # ``python-dotenv`` only loads a local ``.env`` file for convenience. It is
    # optional: launchers such as ``uv run --env-file ...`` or a real process
    # manager already inject the environment before this module is imported, so
    # a missing dependency must not take the whole server down.
    def load_dotenv(*args, **kwargs) -> bool:  # type: ignore[misc]
        return False


from mcp.server.fastmcp import FastMCP

from mealie import MealieFetcher
from prompts import register_prompts
from tools import register_all_tools

# Load environment variables first (no-op if python-dotenv is unavailable).
load_dotenv()

# Get log level from environment variable with INFO as default
log_level_name = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name.upper(), logging.INFO)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mealie_mcp_server.log")

# Configure logging. The stream handler is pinned to stderr: stdout is
# reserved for the JSON-RPC protocol on the stdio transport, so any log line
# written there would corrupt the message stream.
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path)],
)
logger = logging.getLogger("mealie-mcp")


# ---------------------------------------------------------------------------
# Transport configuration
# ---------------------------------------------------------------------------
# The server defaults to HTTP/SSE bound to 127.0.0.1:8765 so it can run as a
# long-lived daemon (e.g. via launchd) and be proxied into Claude Desktop with
# `mcp-remote`. The legacy stdio transport is preserved behind a CLI flag and
# an environment variable so existing configs keep working.

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VALID_TRANSPORTS = {"sse", "stdio", "streamable-http"}

# Opt-in flag for binding to a non-loopback interface. The server exposes the
# Mealie API key's full authority and local-file upload tools to whoever can
# reach it, so we refuse public binds unless the operator explicitly accepts
# the risk and has fronted the server with their own auth/proxy layer.
ALLOW_REMOTE_BIND_ENV = "MEALIE_MCP_ALLOW_REMOTE_BIND"


def _parse_port(raw: str | None) -> int:
    if raw is None or raw == "":
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as e:
        raise ValueError(
            f"MEALIE_MCP_PORT must be an integer in 1..65535, got {raw!r}"
        ) from e
    if not 1 <= port <= 65535:
        raise ValueError(
            f"MEALIE_MCP_PORT must be in 1..65535, got {port}"
        )
    return port


def _is_loopback_host(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Hostnames other than "localhost" are treated as non-loopback; we
        # cannot know what they resolve to without a DNS round-trip and
        # resolution is racy anyway.
        return False


def _validate_host(host: str) -> str:
    if _is_loopback_host(host):
        return host
    if os.getenv(ALLOW_REMOTE_BIND_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.warning(
            {
                "message": "Binding MCP server to non-loopback interface",
                "host": host,
                "note": "Ensure an auth/proxy layer is in front of this server.",
            }
        )
        return host
    raise ValueError(
        f"Refusing to bind MEALIE_MCP_HOST={host!r}: non-loopback bind requires "
        f"{ALLOW_REMOTE_BIND_ENV}=1 and a trusted auth/proxy layer in front of "
        "the server."
    )


def _parse_timeout(raw: str | None) -> float:
    if raw is None or raw == "":
        return 30.0
    try:
        timeout = float(raw)
    except ValueError as e:
        raise ValueError(
            f"MEALIE_TIMEOUT must be a positive number of seconds, got {raw!r}"
        ) from e
    if timeout <= 0:
        raise ValueError(
            f"MEALIE_TIMEOUT must be a positive number of seconds, got {timeout}"
        )
    return timeout


def _stdin_kind() -> str:
    """Classify fd 0 so transport auto-detection is explainable from the logs."""
    try:
        fd = sys.stdin.fileno()
        mode = os.fstat(fd).st_mode
    except (OSError, ValueError, AttributeError):
        # No usable stdin: closed, or replaced by a non-file object.
        return "none"
    if stat.S_ISFIFO(mode):
        return "pipe"
    if stat.S_ISSOCK(mode):
        return "socket"
    if os.isatty(fd):
        return "tty"
    return "other"


# stdin kinds that mean "a parent process wired this up and is talking to us".
# Node — which is what Claude Desktop, Cowork and Code spawn MCP servers from —
# uses a socketpair rather than a pipe for child stdio on POSIX, so both count.
# A daemon under launchd/systemd gets /dev/null ("other"), an interactive run
# gets a "tty", and neither is mistaken for a stdio client.
_CLIENT_SPAWNED_STDIN = frozenset({"pipe", "socket"})


def _launched_over_stdio_pipe() -> bool:
    return _stdin_kind() in _CLIENT_SPAWNED_STDIN


def _resolve_transport(cli_stdio: bool, cli_transport: str | None) -> str:
    """Resolve which transport to run.

    Precedence (highest first):
      1. ``--stdio`` CLI flag
      2. ``--transport`` CLI flag
      3. ``MEALIE_MCP_TRANSPORT`` env var
      4. ``stdio`` when an MCP client spawned us with a pipe on stdin
      5. Default: ``sse``
    """
    if cli_stdio:
        return "stdio"
    if cli_transport:
        return cli_transport
    env_transport = os.getenv("MEALIE_MCP_TRANSPORT")
    if env_transport:
        return env_transport.strip().lower()
    if _launched_over_stdio_pipe():
        return "stdio"
    return "sse"


def _build_server() -> FastMCP:
    """Construct the FastMCP server with Mealie tools/prompts registered."""
    host = _validate_host(os.getenv("MEALIE_MCP_HOST", DEFAULT_HOST))
    port = _parse_port(os.getenv("MEALIE_MCP_PORT"))

    mcp = FastMCP("mealie", host=host, port=port)

    base_url = os.getenv("MEALIE_BASE_URL")
    api_key = os.getenv("MEALIE_API_KEY")
    if not base_url or not api_key:
        raise ValueError(
            "MEALIE_BASE_URL and MEALIE_API_KEY must be set in environment variables."
        )

    timeout = _parse_timeout(os.getenv("MEALIE_TIMEOUT"))

    try:
        mealie = MealieFetcher(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as e:
        logger.error({"message": "Failed to initialize Mealie client", "error": str(e)})
        logger.debug({"message": "Error traceback", "traceback": traceback.format_exc()})
        raise

    register_prompts(mcp)
    register_all_tools(mcp, mealie)
    return mcp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mealie-mcp-server",
        description="Mealie MCP server (HTTP/SSE by default, stdio fallback available).",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run on stdio (legacy behavior). Overrides --transport and MEALIE_MCP_TRANSPORT.",
    )
    parser.add_argument(
        "--transport",
        choices=sorted(VALID_TRANSPORTS),
        default=None,
        help="Transport protocol to use. Defaults to 'sse' unless overridden by env or --stdio.",
    )
    return parser.parse_args(argv)


def _pin_uvicorn_logging_to_stderr() -> None:
    """Route uvicorn's log streams to stderr.

    FastMCP runs the SSE / streamable-http transports on uvicorn, which
    configures its own logging via ``dictConfig`` when the server starts. By
    default uvicorn sends its ``access`` logger to **stdout** (and ``default``
    to stderr), so request lines like ``INFO: ... "GET /sse" 200`` land on
    stdout and bypass our root stderr handler. We mutate the default config in
    place — it is the same object uvicorn falls back to when no ``log_config``
    is passed (which is the case here) — so both handlers write to stderr and
    stdout stays clean.
    """
    try:
        from uvicorn.config import LOGGING_CONFIG
    except Exception:  # pragma: no cover - uvicorn absent (e.g. stdio-only)
        return
    for handler in LOGGING_CONFIG.get("handlers", {}).values():
        if handler.get("stream") == "ext://sys.stdout":
            handler["stream"] = "ext://sys.stderr"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    transport = _resolve_transport(cli_stdio=args.stdio, cli_transport=args.transport)
    logger.info({"message": "Resolved transport", "transport": transport, "stdin": _stdin_kind()})
    if transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. Expected one of: {sorted(VALID_TRANSPORTS)}"
        )

    mcp = _build_server()

    try:
        if transport == "stdio":
            logger.info({"message": "Starting Mealie MCP Server", "transport": "stdio"})
            mcp.run(transport="stdio")
        else:
            _pin_uvicorn_logging_to_stderr()
            logger.info(
                {
                    "message": "Starting Mealie MCP Server",
                    "transport": transport,
                    "host": mcp.settings.host,
                    "port": mcp.settings.port,
                }
            )
            mcp.run(transport=transport)
    except Exception as e:
        logger.critical({"message": "Fatal error in Mealie MCP Server", "error": str(e)})
        logger.debug({"message": "Error traceback", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
