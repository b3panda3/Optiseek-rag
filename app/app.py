#!/usr/bin/env python3
"""app.py — Thin client for Optiseek RAG.

INVOCATION CONTRACT (two phases):

Phase 1 — index (once, before any graded clock):
    python3 /app/app.py --index /app/corpus

Phase 2 — query (once per question, ten times):
    python3 /app/app.py --corpus /app/corpus \\
                        --query-id query_01 \\
                        --query "What is the maximum junction temperature?"

OUTPUT:
    /app/output/<query-id>_output.json
    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

ARCHITECTURE:
    This script is a thin client. The heavy lifting (model loading, retrieval,
    generation) happens in a long-running daemon process started by the
    container's CMD. This script connects to the daemon via a unix socket
    and sends either an INDEX or QUERY command.

    This is critical because the harness starts a NEW PROCESS for each
    question. If we loaded the model inside this script, we'd load it 10 times
    and blow the 30-second per-question budget. The daemon holds the model
    resident across all queries.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("OPTISEEK_RAG_OUTPUT_DIR", "/app/output"))
SOCKET_PATH = os.environ.get("OPTISEEK_RAG_SOCKET", "/app/daemon.sock")
DAEMON_READY_TIMEOUT = 120  # seconds to wait for daemon to come up


def send_to_daemon(command: dict, timeout: int = 120) -> dict:
    """Connect to the daemon socket, send a JSON command, return the response.

    Retries for up to DAEMON_READY_TIMEOUT seconds in case the daemon is
    still loading the model at container startup.
    """
    deadline = time.time() + DAEMON_READY_TIMEOUT
    last_err = None

    while time.time() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(SOCKET_PATH)
            sock.sendall((json.dumps(command) + "\n").encode("utf-8"))

            # Read response until newline
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
            sock.close()

            return json.loads(buf.decode("utf-8").strip())
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout) as e:
            last_err = e
            time.sleep(1)
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"Could not connect to daemon at {SOCKET_PATH}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Optiseek RAG — thin client")
    ap.add_argument("--index", type=Path, help="index this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    # Phase 1: INDEX
    if args.index is not None:
        try:
            result = send_to_daemon(
                {"cmd": "index", "corpus": str(args.index)},
                timeout=600,  # 10 min for indexing
            )
            if result.get("status") == "ok":
                print(f"Index complete: {result.get('files_indexed', 0)} files, "
                      f"{result.get('chunks_indexed', 0)} chunks",
                      file=sys.stderr)
                return 0
            else:
                print(f"Index failed: {result.get('error', 'unknown')}",
                      file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Index error: {e}", file=sys.stderr)
            return 1

    # Phase 2: QUERY
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    try:
        result = send_to_daemon(
            {
                "cmd": "query",
                "corpus": str(args.corpus),
                "query": args.query,
            },
            timeout=30,  # per-question budget
        )
    except Exception as e:
        # On any error, write a valid (but empty) response so the harness
        # doesn't see a missing file. A malformed response scores zero, but
        # so does a missing one — and at least this way the harness knows we
        # tried.
        result = {"answer": "", "citations": [], "confidence": 0.0,
                  "error": str(e)}

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (args.query_id + "_output.json")
    payload = {
        "answer": result.get("answer", ""),
        "citations": list(result.get("citations", [])),
        "confidence": float(result.get("confidence", 0.0)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
