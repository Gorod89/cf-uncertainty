#!/usr/bin/env bash
# GPU queue of the strengthening items (extension/docs/PLAN.md, section S), one step after the other on the shared GPU:
# S3 re-planning grid -> S4 factorial preview cells -> S6 reference units (waits for the mode-calibrated checkpoints
# and the config written by the offline steps) -> analyses. A failed step does not stop the queue (every step can be
# repeated by hand through the retry runners, which resume from the unit tables).
#   bash extension/scripts/queue2.sh            (log: extension/artifacts/queue2.log; marker extension/artifacts/queue2.done)
set -u
cd "$(dirname "$0")/../.."
export PYTHONUTF8=1
PY=./.venv/Scripts/python.exe
LOG=extension/artifacts/queue2.log
mkdir -p extension/artifacts
rm -f extension/artifacts/queue2.done
step() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

step "S3 re-planning grid: platoon cells"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_replan/logs/run.out 8 --config extension/configs/experiment/ext_e3_replan.yaml
step "S3: analysis (exit of the run: $?)"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_replan --quiet > extension/artifacts/ext_e3_replan/logs/analysis.out 2>&1
step "S4 factorial preview cells (exit of the analysis: $?)"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_preview/logs/run_c.out 8 --config extension/configs/experiment/ext_e3_preview_c.yaml
step "S4: analysis of ext_e3_preview (exit of the run: $?)"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_preview --quiet > extension/artifacts/ext_e3_preview/logs/analysis_c.out 2>&1
step "S6: waiting for the mode-calibrated references (exit of the analysis: $?)"
for i in $(seq 1 720); do
  if [ -f extension/artifacts/ext_idm_modes/checkpoints/DONE ] && [ -f extension/configs/experiment/ext_e3_idmref.yaml ]; then break; fi
  sleep 30
done
if [ -f extension/artifacts/ext_idm_modes/checkpoints/DONE ] && [ -f extension/configs/experiment/ext_e3_idmref.yaml ]; then
  step "S6 reference units"
  bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_idmref/logs/run.out 8 --config extension/configs/experiment/ext_e3_idmref.yaml
  step "S6: analysis (exit of the run: $?)"
  $PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_idmref --quiet > extension/artifacts/ext_e3_idmref/logs/analysis.out 2>&1
  step "S6 analysis finished (exit $?)"
else
  step "S6 skipped: references or config not present after 6 h"
fi
step "queue2 finished"
touch extension/artifacts/queue2.done
