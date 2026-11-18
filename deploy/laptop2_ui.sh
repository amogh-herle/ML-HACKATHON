#!/usr/bin/env bash
# PathSnap - Laptop 2 (evaluator UI). Serves ui/ locally and opens it in the browser.
#
#   deploy/laptop2_ui.sh                       open the UI (it remembers the last API address)
#   deploy/laptop2_ui.sh 192.168.1.20:8000     open the UI already connected to Laptop 1
#
# Needs only python3 (preinstalled on macOS) and a browser; no venv, no data, no model.
# Stop with Ctrl+C. Windows: double-click ui\index.html instead.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-5500}"
url="http://localhost:$PORT/"
[ $# -ge 1 ] && url="${url}?api=$1"

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory ui &
server=$!
trap 'kill $server 2>/dev/null' EXIT INT TERM
sleep 1
echo "PathSnap UI: $url   (Ctrl+C to stop)"
case "$(uname)" in
  Darwin) open "$url" ;;
  *) xdg-open "$url" >/dev/null 2>&1 || echo "Open $url in a browser." ;;
esac
wait "$server"
