#!/usr/bin/env bash
set -euo pipefail

# Run one of the paper experiment suites.
#
# Example:
#   bash scripts/run_reproduce_suite.sh \
#     --suite main \
#     --model-config configs/model/mobileclip2_s0_fetal.json \
#     --pretrained /path/to/mobileclip2_s0.pt \
#     --teacher /path/to/FetalCLIP_weights.pt

SUITE="main"
BASE_CONFIG="configs/default.yaml"
MODEL_CONFIG="configs/model/mobileclip2_s0_fetal.json"
PRETRAINED=""
TEACHER=""
TRAIN_DATA=""
OUTPUT_ROOT="outputs/repro"
PROJECT_NAME="mobile_fetal_clip"
MASTER_CSV="outputs/experiments_master.csv"
SEED="42"
DRY_RUN="0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --master-csv) MASTER_CSV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD=(
  python -m mobile_fetal_clip.cli reproduce
  --suite "$SUITE"
  --base-config "$BASE_CONFIG"
  --model-config "$MODEL_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --project-name "$PROJECT_NAME"
  --master-csv "$MASTER_CSV"
  --seed "$SEED"
)

if [[ -n "$PRETRAINED" ]]; then
  CMD+=(--pretrained "$PRETRAINED")
fi
if [[ -n "$TEACHER" ]]; then
  CMD+=(--teacher "$TEACHER")
fi
if [[ -n "$TRAIN_DATA" ]]; then
  CMD+=(--train-data "$TRAIN_DATA")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Running suite: $SUITE"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
