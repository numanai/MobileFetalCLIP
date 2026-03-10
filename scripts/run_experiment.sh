#!/usr/bin/env bash
set -euo pipefail

# Run one configured experiment.
#
# Example:
#   bash scripts/run_experiment.sh \
#     --experiment-id static-kd \
#     --model-config configs/model/mobileclip2_s0_fetal.json \
#     --pretrained /path/to/mobileclip2_s0.pt \
#     --teacher /path/to/FetalCLIP_weights.pt

EXPERIMENT_ID=""
EXPERIMENT_CONFIG=""
BASE_CONFIG="configs/default.yaml"
MODEL_CONFIG="configs/model/mobileclip2_s0_fetal.json"
PRETRAINED=""
TEACHER=""
TRAIN_DATA=""
OUTPUT_ROOT="outputs/experiments"
PROJECT_NAME="mobile_fetal_clip"
MASTER_CSV="outputs/experiments_master.csv"
SEED="42"
DRY_RUN="0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
    --experiment-config) EXPERIMENT_CONFIG="$2"; shift 2 ;;
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
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EXPERIMENT_CONFIG" ]]; then
  if [[ -z "$EXPERIMENT_ID" ]]; then
    echo "Either --experiment-id or --experiment-config is required." >&2
    exit 1
  fi
  EXPERIMENT_CONFIG="configs/experiments/${EXPERIMENT_ID}.yaml"
fi

if [[ ! -f "$EXPERIMENT_CONFIG" ]]; then
  echo "Experiment config not found: $EXPERIMENT_CONFIG" >&2
  exit 1
fi

if [[ -z "$EXPERIMENT_ID" ]]; then
  EXPERIMENT_ID="$(basename "$EXPERIMENT_CONFIG" .yaml)"
fi

OUTPUT_DIR="${OUTPUT_ROOT%/}/${EXPERIMENT_ID}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD=(
  python -m mobile_fetal_clip.cli train
  --base-config "$BASE_CONFIG"
  --experiment-config "$EXPERIMENT_CONFIG"
  --model-config "$MODEL_CONFIG"
  --output-dir "$OUTPUT_DIR"
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

echo "Launching experiment: $EXPERIMENT_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
