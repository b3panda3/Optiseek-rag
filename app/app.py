#!/usr/bin/env python3
"""Thin client for Optiseek RAG. Connects to daemon via unix socket."""
from __future__ import annotations
import argparse, json, os, socket, sys, time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("OPTISEEK_RAG_OUTPUT_DIR", "/app/output"))
SOCKET_PATH = os.environ.get("OPTISEEK_RAG_SOCKET", "/app/daemon.sock")

def send_to_daemon(command: dict, timeout: int = 120) -> dict:
    deadline = time.time() + 120
    last_err = None
    while time.time() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(SOCKET_PATH)
            sock.sendall((json.dumps(command) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk: break
                buf += chunk
            sock.close()
            return json.loads(buf.decode("utf-8").strip())
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout) as e:
            last_err = e
            time.sleep(1)
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(f"Could not connect to daemon: {last_err}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()

    if args.index is not None:
        try:
            result = send_to_daemon({"cmd": "index", "corpus": str(args.index)}, timeout=600)
            if result.get("status") == "ok":
                print(f"Index: {result.get('files_indexed',0)} files, {result.get('chunks_indexed',0)} chunks", file=sys.stderr)
                return 0
            else:
                print(f"Index failed: {result.get('error','unknown')}", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Index error: {e}", file=sys.stderr)
            return 1

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    try:
        result = send_to_daemon({"cmd": "query", "corpus": str(args.corpus), "query": args.query}, timeout=30)
    except Exception as e:
        result = {"answer": "", "citations": [], "confidence": 0.0, "error": str(e)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (args.query_id + "_output.json")
    payload = {"answer": result.get("answer", ""), "citations": list(result.get("citations", [])), "confidence": float(result.get("confidence", 0.0))}
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())
