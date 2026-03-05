# Experiment Registry

Canonical experiment configs live in `configs/experiments/`.

## Main suite
- `no-kd`
- `static-kd`
- `repulsive-r-neg-0p8`
- `selective-beta2-to-neg-0p8`

## Full ablation suite
- `no-kd`
- `static-kd`
- `confidence-penalty`
- `repulsive-r-neg-0p5`
- `repulsive-r-neg-0p8`
- `coupled-beta2-to-neg-0p8`
- `selective-beta1-to-neg-0p8`
- `selective-beta2-to-neg-0p8`
- `selective-beta4-to-neg-0p8`
- `selective-beta8-to-neg-0p8`

Each file contains:
- `id`
- `description`
- `teacher_required`
- `args` (CLI argument overrides)
