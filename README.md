# MobileFetalCLIP (ECCV Reproducibility Release)

Clean, anonymized, and reproducible code release for the MobileFetalCLIP ECCV submission.

This repository focuses on:
- canonical training/evaluation code paths,
- full paper ablation experiment registry,
- reproducibility scripts and table generation,
- access-gated dataset preparation and validation.

No personal paths, machine-specific scripts, or credentials are included.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Validate Data Layout

If you already have licensed data access, first map your assets into this repo layout:

```bash
python tools/prepare_access_gated_dataset.py \
  --source-train-shards "/path/to/shard_{0000000001..0000000025}.tar" \
  --source-fetal-planes-images "/path/to/FETAL_PLANES_DB/Images" \
  --source-fetal-planes-csv "/path/to/FETAL_PLANES_DB_data.csv" \
  --source-hc18-images "/path/to/HC18/training_set" \
  --source-hc18-csv "/path/to/training_set_pixel_size_and_HC.csv"
```

Then validate:

```bash
python tools/validate_dataset_contract.py --strict
```

## 3. Run a Single Training Experiment

```bash
bash scripts/run_experiment.sh \
  --experiment-id static-kd \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

Outputs are saved in `outputs/experiments/<experiment-id>/`.

## 4. Reproduce Paper Suites

Run main suite:

```bash
bash scripts/run_reproduce_suite.sh \
  --suite main \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

Run full ablation suite:

```bash
bash scripts/run_reproduce_suite.sh \
  --suite ablation \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --pretrained checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --teacher checkpoints/FetalCLIP_weights.pt
```

## 5. Evaluate and Benchmark

Evaluate a checkpoint:

```bash
python -m mobile_fetal_clip.cli eval \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --base-checkpoint checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --checkpoint outputs/experiments/selective-beta2-to-neg-0p8/checkpoints/best.pt \
  --eval-config configs/default.yaml \
  --output-json outputs/eval/selective_beta2_eval.json
```

Benchmark:

```bash
bash scripts/benchmark_inference.sh \
  --model-id mobileclip2_s0 \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --base-ckpt checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --finetuned-ckpt outputs/experiments/selective-beta2-to-neg-0p8/checkpoints/best.pt \
  --device cpu \
  --scope both \
  --batch-sizes 1,16 \
  --output-json outputs/benchmarks/mobileclip2_s0_cpu.json
```

## Checkpoints

Checkpoints are not committed. See [checkpoints/README.md](checkpoints/README.md) for expected filenames and placement.

## Additional Docs

- [Dataset Contract](docs/dataset_contract.md)
- [Reproducibility](docs/reproducibility.md)
- [Experiment Registry](docs/experiment_registry.md)
- [Benchmark Protocol](docs/benchmark_protocol.md)
- [Checkpoint Policy](docs/checkpoint_policy.md)
