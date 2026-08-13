#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-"$ROOT/build-release/decision_mesh"}
MODE=${2:-bh}
OUT=${3:-"$ROOT/benchmark-output/$MODE"}
SEED=${4:-7}
DATA=${DMESH_BENCH_DATA:-"$ROOT/data/ginnie_design.csv"}
if [[ "$MODE" != bh && "$MODE" != lindsey ]]; then
  echo "mode must be bh or lindsey" >&2; exit 2
fi
mkdir -p "$OUT"
export DMESH_DATA="$DATA" DMESH_SPLIT=1 DMESH_SPLIT_MODE=${DMESH_SPLIT_MODE:-local} DMESH_PL_SOLVER=1 DMESH_HIER_FIT=2
export DMESH_HEIGHT_LIMIT_LOGIT=12 DMESH_IRLS_STEP_CAP=4 DMESH_HIER_MAP=1
export DMESH_QUAD_PRIOR=1
export DMESH_SIGMA0_MIN=0.3 DMESH_SIGMA0_MAX=6 DMESH_PI0_MIN=0.05
export DMESH_TAU_FLOOR_LOGIT_SD=0.005 DMESH_DUMP="$OUT/run"
export DMESH_STAGE_COUNT=3 DMESH_MAX_GATE_ROUNDS=80
export DMESH_PARENT_VAR_SCALE=0.25
export DMESH_GATE_EXTRA_VAR=0.35 DMESH_B3_USE_SCORE=1 DMESH_B3_STAGE0_EXACT=0
unset DMESH_STOP_AFTER_ADAPTATION DMESH_HIST_NULL DMESH_HIST_TOTAL_VAR
unset DMESH_HIST_MIX_SHAPE DMESH_B3_FIXED_EMPIRICAL_NULL
unset DMESH_CANDIDATE_BH DMESH_ONELAW DMESH_POINT_VAR DMESH_LAW_WEIGHTS
if [[ "$MODE" == bh ]]; then export DMESH_CANDIDATE_BH=1; fi
/usr/bin/time -f '%e' -o "$OUT/wall_seconds.txt" \
  "$BIN" 0 0.10 1 24 "$SEED" >"$OUT/stdout.log" 2>"$OUT/stderr.log"
