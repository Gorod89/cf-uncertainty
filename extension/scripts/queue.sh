#!/usr/bin/env bash
# Queue of the extension runs on the shared GPU, one after the other (extension/docs/PLAN.md, order of work):
# X1 stage A (preview sweep on the stored checkpoints) -> X2 cost benchmark (GPU free) -> X1 stage B (four diffusion
# trainings at a preview budget, then their platoon cells) -> X6 (NGSIM I-80 training root, its analysis, its platoon
# cells). Every run goes through the retry runners (a CUDA reset restarts the runner, which resumes); a failed step
# does not stop the queue (the next steps do not need it, and every step can be repeated by hand).
#   bash extension/scripts/queue.sh            (log: extension/artifacts/queue.log)
set -u
cd "$(dirname "$0")/../.."
export PYTHONUTF8=1
PY=./.venv/Scripts/python.exe
LOG=extension/artifacts/queue.log
mkdir -p extension/artifacts
step() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

step "X1-A preview sweep: platoon cells"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_preview/logs/run.out 8 --config extension/configs/experiment/ext_e3_preview.yaml
step "X1-A: analysis (exit of the run: $?)"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_preview --quiet > extension/artifacts/ext_e3_preview/logs/analysis_a.out 2>&1
step "X2: cost benchmark (exit of the analysis: $?)"
if [ -f extension/scripts/cost_table.py ]; then
  $PY extension/scripts/cost_table.py > extension/artifacts/ext_cost.log 2>&1
  step "X2 done (exit $?)"
else
  step "X2 skipped: extension/scripts/cost_table.py not present yet"
fi
for r in pool_prev1 pool_prev2 lodo_prev1 lodo_prev2; do
  step "X1-B training $r"
  bash scripts/run_e1_retry.sh extension/artifacts/ext_$r/logs/run.out 8 --config extension/configs/experiment/ext_$r.yaml
  step "X1-B training $r finished (exit $?)"
done
step "X1-B: platoon cells"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_preview/logs/run_b.out 8 --config extension/configs/experiment/ext_e3_preview_b.yaml
step "X1-B: analysis of every cell (exit of the run: $?)"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_preview --quiet > extension/artifacts/ext_e3_preview/logs/analysis_b.out 2>&1
step "X6: NGSIM I-80 training root (exit of the analysis: $?)"
bash scripts/run_e1_retry.sh extension/artifacts/ext_e1_ngsim/logs/run.out 8 --config extension/configs/experiment/ext_e1_ngsim.yaml
step "X6: E1 analysis (exit of the run: $?)"
$PY scripts/analyze_e1.py --root extension/artifacts/ext_e1_ngsim --config extension/configs/experiment/ext_e1_ngsim.yaml --in-domain ngsim --placebo ngsim_val > extension/artifacts/ext_e1_ngsim/logs/analysis.out 2>&1
step "X6: platoon cells (exit of the analysis: $?)"
bash scripts/run_e3_retry.sh extension/artifacts/ext_e3_ngsim/logs/run.out 8 --config extension/configs/experiment/ext_e3_ngsim.yaml
step "X6: E3 analysis (exit of the run: $?)"
$PY scripts/analyze_e3.py --root extension/artifacts/ext_e3_ngsim --quiet > extension/artifacts/ext_e3_ngsim/logs/analysis.out 2>&1
step "queue finished (exit of the analysis: $?)"
