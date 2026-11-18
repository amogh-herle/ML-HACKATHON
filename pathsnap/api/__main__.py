"""Start the API:  python -m pathsnap.api [--host 0.0.0.0] [--port 8000]

0.0.0.0 makes the API reachable from the UI laptop on the same network. Request logs are
printed in this terminal (and written to logs/api.log) for the evaluator.
"""
import argparse
import socket

import uvicorn


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"PathSnap API -> http://{local_ip()}:{a.port}  (docs: /docs)  - give this address to the UI laptop")
    uvicorn.run("pathsnap.api.app:app", host=a.host, port=a.port, log_level="info")
