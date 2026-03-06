# Checkpoint Policy

This repository does not commit model checkpoints.

## Expected local placement

- `checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt`
- `checkpoints/MobileCLIP2-S2/mobileclip2_s2.pt`
- `checkpoints/MobileCLIP2-B/mobileclip2_b.pt`
- `checkpoints/FetalCLIP_weights.pt`

## Release note

Public weights are hosted on Hugging Face:

- `https://huggingface.co/numansaeed/MobileFetalCLIP`
- `https://huggingface.co/numansaeed/MobileFetalCLIP/tree/main`

Download the required files from the Hugging Face repo and place them locally with the expected
filenames above. Checksums should be tracked in `checkpoints/SHA256SUMS.txt` for any reproduced
or redistributed setup.
