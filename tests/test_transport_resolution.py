"""Transport auto-detection: a client-spawned copy must speak stdio.

Uses real subprocesses so the stdin mode is the genuine article (pipe vs
/dev/null), not a mock.

Run with: python tests/test_transport_resolution.py
"""

import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

PROBE = (
    "import sys; sys.path.insert(0, %r);"
    "import server;"
    "print(server._resolve_transport(cli_stdio=False, cli_transport=None))" % SRC
)


def resolve(stdin, env=None):
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        stdin=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.strip()


# Spawned by an MCP client (Claude Desktop, Cowork, Code): stdin is a pipe.
assert resolve(subprocess.PIPE) == "stdio"

# Daemon under launchd/systemd: stdin is /dev/null, keep the SSE default.
assert resolve(subprocess.DEVNULL) == "sse"

# An explicit env var still wins over the heuristic.
import os  # noqa: E402

assert resolve(subprocess.PIPE, env={**os.environ, "MEALIE_MCP_TRANSPORT": "sse"}) == "sse"

print("ok")
