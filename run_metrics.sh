#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv311/bin/activate
./scripts/run_models_and_push.sh "$@"
