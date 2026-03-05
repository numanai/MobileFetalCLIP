# Benchmark Protocol

Use `tools/benchmark_inference.py` with:
- fixed warmup/iters/repeats,
- explicit device and precision,
- explicit batch sizes,
- schema-versioned JSON output.

Recommended baseline command:

```bash
python tools/benchmark_inference.py \
  --model-id mobileclip2_s0 \
  --model-config configs/model/mobileclip2_s0_fetal.json \
  --base-ckpt checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt \
  --finetuned-ckpt outputs/experiments/selective-beta2-to-neg-0p8/checkpoints/best.pt \
  --device cpu \
  --scope both \
  --batch-sizes 1,16 \
  --warmup 20 \
  --iters 100 \
  --repeats 3 \
  --output-json outputs/benchmarks/mobileclip2_s0_cpu.json
```
