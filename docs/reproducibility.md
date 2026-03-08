# Reproducibility Guide

The release ships a small set of experiment configs in `configs/experiments/`.

## Run one experiment

The main paper checkpoint is `selective-beta2-to-neg-0p8`:

```bash
bash scripts/run_experiment.sh \
  --experiment-id selective-beta2-to-neg-0p8 \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

Other released experiment ids:

- `no-kd`
- `static-kd`
- `repulsive-r-neg-0p5`
- `repulsive-r-neg-0p8`
- `coupled-beta2-to-neg-0p8`
- `confidence-penalty`
- `selective-beta1-to-neg-0p8`
- `selective-beta2-to-neg-0p8`
- `selective-beta4-to-neg-0p8`
- `selective-beta8-to-neg-0p8`

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
