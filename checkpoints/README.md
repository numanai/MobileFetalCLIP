# Checkpoints

Model weights are not stored in this repository. Download them from the
Hugging Face collection:

- `https://huggingface.co/collections/numansaeed/fetal-ultrasound-models`

Place downloaded files under `checkpoints/` with these names:

- `MobileCLIP2-S0/mobileclip2_s0.pt`
- `MobileCLIP2-S2/mobileclip2_s2.pt`
- `MobileCLIP2-B/mobileclip2_b.pt`
- `FetalCLIP_weights.pt`

Training runs will write fine-tuned checkpoints under
`outputs/experiments/<experiment-id>/checkpoints/`.
