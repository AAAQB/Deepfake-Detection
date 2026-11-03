#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it and install the project with web and legacy extras."
  exit 1
fi

if [[ ! -d "web/node_modules" ]]; then
  echo "Missing web dependencies. Run: cd web && npm install"
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

.venv/bin/python -m uvicorn deepfake_detection.web_api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cd web
npm run dev
