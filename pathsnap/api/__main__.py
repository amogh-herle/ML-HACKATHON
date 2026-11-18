"""Start the API:  python -m pathsnap.api [--partition evaluator_test] [--host 0.0.0.0] [--port 8000]

0.0.0.0 makes the API reachable from the UI laptop on the same network. Request logs are
printed in this terminal (and written to logs/api.log) for the evaluator.

Partition served to the UI, highest priority first:
    --partition flag  >  PATHSNAP_PARTITION env var  >  evaluator_test if present, else student_test
"""
import argparse
import os
import socket
import sys

import uvicorn

from pathsnap.config import DATA_ROOT


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def available_partitions() -> list[str]:
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()) if DATA_ROOT.is_dir() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PathSnap API (Laptop 1)")
    ap.add_argument("--partition", help="data folder to serve, e.g. evaluator_test (demo) or "
                                        "student_test (practice). Overrides PATHSNAP_PARTITION.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    if not DATA_ROOT.is_dir():
        sys.exit(f"Data folder not found: {DATA_ROOT}\n"
                 f"Put 'Student folder' next to the repo folder or set PATHSNAP_DATA.")
    if a.partition:
        if not (DATA_ROOT / a.partition).is_dir():
            sys.exit(f"Partition '{a.partition}' not found in {DATA_ROOT}. "
                     f"Available: {', '.join(available_partitions()) or 'none'}")
        env = os.environ.get("PATHSNAP_PARTITION")
        if env and env != a.partition:
            print(f"Note: --partition {a.partition} overrides PATHSNAP_PARTITION={env}")
        os.environ["PATHSNAP_PARTITION"] = a.partition

    os.environ["PATHSNAP_PUBLIC_URL"] = f"http://{local_ip()}:{a.port}"  # shown in the start-up banner
    uvicorn.run("pathsnap.api.app:app", host=a.host, port=a.port, log_level="info")
