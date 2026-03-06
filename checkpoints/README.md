# Checkpoint Placeholders

Checkpoints are intentionally not committed.

## Expected files

- `MobileCLIP2-S0/mobileclip2_s0.pt`
- `MobileCLIP2-S2/mobileclip2_s2.pt`
- `MobileCLIP2-B/mobileclip2_b.pt`
- `FetalCLIP_weights.pt`

## Public weights

Public checkpoints are listed in the Hugging Face collection:

- Collection: `https://huggingface.co/collections/numansaeed/fetal-ultrasound-models`

Open the MobileFetalCLIP entry from the collection, download the required artifacts, and place
them under `checkpoints/` using the expected filenames above. The release includes:

- Base student checkpoints
- Teacher checkpoint
- Fine-tuned MobileFetalCLIP weights used for the paper/release

## Checksum policy

When checkpoints are released, include a `SHA256SUMS.txt` file with:

```text
<sha256>  <relative/path/to/checkpoint.pt>
```
