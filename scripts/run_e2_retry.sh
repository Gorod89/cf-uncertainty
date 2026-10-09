#!/usr/bin/env bash
# Run scripts/run_e2.py until it finishes without failures (the runner resumes from its rollout tables).
# Needed on a GPU shared with other jobs: a driver reset ("CUDA error: unknown error") kills every CUDA
# context at once, the process must be restarted.
#   scripts/run_e2_retry.sh <log file> <max attempts> <run_e2.py arguments...>
set -u
log="$1"; attempts="$2"; shift 2
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$log")"   # a fresh root has no logs directory yet (the redirect below would fail)
export PYTHONUTF8=1
for attempt in $(seq 1 "$attempts"); do
  echo "=== attempt $attempt of $attempts: $(date '+%Y-%m-%d %H:%M:%S')  run_e2.py $*" >> "$log"
  ./.venv/Scripts/python.exe scripts/run_e2.py "$@" >> "$log" 2>&1
  code=$?
  echo "=== attempt $attempt finished with exit code $code: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log"
  if [ "$code" -eq 0 ]; then exit 0; fi
  sleep 120
done
exit 1
