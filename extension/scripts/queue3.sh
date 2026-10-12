#!/usr/bin/env bash
# GPU queue of the strengthening items S7 and S8 (extension/docs/PLAN.md, section S), started after queue2.sh: waits for
# its marker, then the closed-loop fine-tunings (highD, pooled, OpenACC held out) with their one-step analyses, the E2
# rollouts and the platoon cells of the fine-tuned laws, and the plan-sharing cells of S8. A failed step does not stop
# the queue (every step resumes through the retry runners).
#   bash extension/scripts/queue3.sh            (log: extension/artifacts/queue3.log; marker extension/artifacts/queue3.done)
set -u
cd "$(dirname "$0")/../.."
export PYTHONUTF8=1
PY=./.venv/Scripts/python.exe
LOG=extension/artifacts/queue3.log
rm -f extension/artifacts/queue3.done
step() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

step "waiting for queue2"
for i in $(seq 1 2880); do [ -f extension/artifacts/queue2.done ] && break; sleep 30; done
[ -f extension/artifacts/queue2.done ] || step "queue2 marker not seen after 24 h: starting anyway"

for r in highd pool lodo; do
  step "S7 fine-tuning ext_cl_$r"
  bash scripts/run_e1_retry.sh extension/artifacts/ext_cl_$r/logs/run.out 8 --config extension/configs/experiment/ext_cl_$r.yaml
  step "S7 ext_cl_$r finished (exit $?); analysis"
  $PY scripts/analyze_e1.py --root extension/artifacts/ext_cl_$r --config extension/configs/experiment/ext_cl_$r.yaml > extension/artifacts/ext_cl_$r/logs/analysis.out 2>&1
  step "S7 ext_cl_$r analysis finished (exit $?)"
done
step "S7 E2 rollouts of the fine-tuned laws"
bash scripts/run_e2_retry.sh extension/artifacts/ext_e2_cl/logs/run.out 8 --config extension/configs/experiment/ext_e2_cl.yaml
step "S7 E2 finished (exit $?); analysis"
$PY scripts/analyze_e2.py --root extension/artifacts/ext_e2_cl > extension/artifacts/ext_e2_cl/logs/analysis.out 2>&1
step "S7 E2 analysis finished (exit $?); platoon cells"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_cl/logs/run.out 8 --config extension/configs/experiment/ext_e3_cl.yaml
step "S7 platoon cells finished (exit $?); analysis"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_cl --quiet > extension/artifacts/ext_e3_cl/logs/analysis.out 2>&1
step "S8 plan-sharing cells (exit of the analysis: $?)"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_replan/logs/run_b.out 8 --config extension/configs/experiment/ext_e3_replan_b.yaml
step "S8 cells finished (exit $?); analysis of ext_e3_replan"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_replan --quiet > extension/artifacts/ext_e3_replan/logs/analysis_b.out 2>&1
step "queue3 finished (exit of the analysis: $?)"
touch extension/artifacts/queue3.done
