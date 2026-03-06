# Checkpoint Placeholders

Checkpoints are intentionally not committed.

## Expected files

- `MobileCLIP2-S0/mobileclip2_s0.pt`
- `MobileCLIP2-S2/mobileclip2_s2.pt`
- `MobileCLIP2-B/mobileclip2_b.pt`
- `FetalCLIP_weights.pt`

## Public weights

Public checkpoints are hosted on Hugging Face:

- Model hub: `https://huggingface.co/numansaeed/MobileFetalCLIP`
- Files tab: `https://huggingface.co/numansaeed/MobileFetalCLIP/tree/main`

Download the required artifacts from the Hugging Face repo and place them under `checkpoints/`
using the expected filenames above. The release includes:

- Base student checkpoints
- Teacher checkpoint
- Fine-tuned MobileFetalCLIP weights used for the paper/release

## Checksum policy

When checkpoints are released, include a `SHA256SUMS.txt` file with:

```text
<sha256>  <relative/path/to/checkpoint.pt>
```
