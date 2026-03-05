# Reproducibility Guide

## Canonical experiment execution

Use one canonical experiment config per run:

```bash
bash scripts/run_experiment.sh \
  --experiment-id <experiment-id> \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

## Suite execution

Main:

```bash
bash scripts/run_reproduce_suite.sh --suite main \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

Ablation:

```bash
bash scripts/run_reproduce_suite.sh --suite ablation \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

## Results outputs

Each run emits:
- `train.log`
- checkpoints (`epoch_*.pt`, `best.pt`)
- `results.json`
- `experiments.csv`

`results.json` includes `schema_version: 1.0.0` and key metrics required for table generation.
