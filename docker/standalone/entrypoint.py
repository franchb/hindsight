"""Entrypoint for distroless api-only container (no shell available).

Handles:
- Optional dependency waiting (HINDSIGHT_WAIT_FOR_DEPS=true)
- SIGTERM/SIGINT graceful shutdown
- Direct invocation of hindsight_api.main:main()
"""

import os
import signal
import socket
import sys
import time
import urllib.request


def _check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_http(url: str, timeout: float = 5.0) -> bool:
    """Check if an HTTP endpoint returns a successful response."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _parse_db_host_port(db_url: str) -> tuple[str, int] | None:
    """Extract host and port from a PostgreSQL URL."""
    # postgresql://user:pass@host:port/dbname
    try:
        at_idx = db_url.index("@")
        rest = db_url[at_idx + 1 :]
        # Strip /dbname and ?params
        slash_idx = rest.index("/")
        host_port = rest[:slash_idx]
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            return host, int(port_str)
        return host_port, 5432
    except (ValueError, IndexError):
        return None


def _wait_for_deps() -> None:
    """Wait for database and LLM dependencies to become available."""
    llm_base_url = os.environ.get(
        "HINDSIGHT_API_LLM_BASE_URL", "http://host.docker.internal:1234/v1"
    )
    max_retries = int(os.environ.get("HINDSIGHT_RETRY_MAX", "0"))  # 0 = infinite
    retry_interval = int(os.environ.get("HINDSIGHT_RETRY_INTERVAL", "10"))

    db_url = os.environ.get("HINDSIGHT_API_DATABASE_URL", "")
    db_host_port = _parse_db_host_port(db_url) if db_url else None

    print("Waiting for dependencies to be ready...", flush=True)
    attempt = 1

    while True:
        db_ok = db_host_port is None or _check_tcp(*db_host_port)
        llm_ok = _check_http(f"{llm_base_url.rstrip('/')}/models")

        if db_ok and llm_ok:
            print("Dependencies ready.", flush=True)
            return

        if max_retries > 0 and attempt >= max_retries:
            print(
                f"Max retries ({max_retries}) reached. Dependencies not available.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        db_status = "ok" if db_ok else "waiting"
        llm_status = "ok" if llm_ok else "waiting"
        print(
            f"  Attempt {attempt}: DB={db_status}, LLM={llm_status}",
            flush=True,
        )
        time.sleep(retry_interval)
        attempt += 1


def _signal_handler(signum: int, _frame: object) -> None:
    """Forward signals for graceful shutdown."""
    print(f"Received signal {signum}, shutting down...", flush=True)
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if os.environ.get("HINDSIGHT_WAIT_FOR_DEPS", "false").lower() == "true":
        _wait_for_deps()

    # Import and run the hindsight API main function
    from hindsight_api.main import main as api_main

    api_main()


if __name__ == "__main__":
    main()
