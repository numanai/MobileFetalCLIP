"""Model factory: create MobileCLIP2 models for fetal ultrasound fine-tuning.

Uses open_clip library for model creation with custom MCI timm models registered.
The factory registers JSON model configs with open_clip so that create_model() can
resolve them by name, then delegates all model construction to open_clip.
"""
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import open_clip
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from mobile_fetal_clip.models.mci_registry import register_all_mci_models

logger = logging.getLogger(__name__)

# Path to bundled model configs
_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "model"


def _resolve_config_path(model_config_path: str) -> Path:
    """Resolve model config path, checking CWD first then bundled configs."""
    p = Path(model_config_path)
    if p.exists():
        return p
    bundled = _CONFIG_DIR / model_config_path
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"Model config not found at '{model_config_path}' or '{bundled}'"
    )


def create_fetal_clip_model(
    model_config_path: str,
    pretrained: Optional[str] = None,
    precision: str = "amp",
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Any, Any]:
    """Create a MobileCLIP2 model configured for fetal ultrasound data.

    Registers the JSON config with open_clip, then calls
    ``open_clip.create_model_and_transforms`` so all standard OpenCLIP
    logic (CustomTextCLIP instantiation, TimmModel vision tower, transforms,
    precision handling) is reused.

    Args:
        model_config_path: Path to model JSON config file.
        pretrained: Path to pretrained checkpoint, or None.
        precision: Training precision ('amp', 'fp32', 'fp16', 'bf16').
        device: Device string.

    Returns:
        Tuple of (model, preprocess_train, preprocess_val).
    """
    # Ensure MCI models are registered with timm
    register_all_mci_models()

    config_path = _resolve_config_path(model_config_path)
    model_name = config_path.stem  # e.g. "mobileclip2_s0_fetal"
    logger.info(f"Creating model '{model_name}' from config: {config_path}")

    # Register the config's parent directory so open_clip can find the JSON
    # by its stem name. add_model_config accepts a directory or a single file.
    open_clip.add_model_config(config_path.parent)

    # Create model via open_clip (handles CLIP vs CustomTextCLIP, timm vision
    # tower, text tower, transforms, precision casting, etc.).
    # ImageNet mean/std are passed explicitly so the val preprocess generated
    # here matches the train preprocess in student_preprocess.py — both use
    # the stats MobileCLIP2-S0 was pretrained with (not open_clip's OpenAI defaults).
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device=device,
        output_dict=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created: {n_params / 1e6:.1f}M total params "
        f"({n_trainable / 1e6:.1f}M trainable)"
    )

    return model, preprocess_train, preprocess_val


def get_tokenizer(model_config_path: str):
    """Get the tokenizer for a given model config.

    Registers the config dir with open_clip so ``get_tokenizer`` can resolve
    context_length from the JSON file.
    """
    config_path = _resolve_config_path(model_config_path)
    open_clip.add_model_config(config_path.parent)
    return open_clip.get_tokenizer(config_path.stem)


def _remap_mobileclip2_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remap Apple MobileCLIP2 checkpoint keys to OpenCLIP CustomTextCLIP format.

    Apple format                              OpenCLIP format
    ─────────────────────────────────────     ──────────────────────────────────────
    image_encoder.model.*                  →  visual.trunk.*
    text_encoder.embedding_layer.weight    →  text.token_embedding.weight
    text_encoder.final_layer_norm.*        →  text.ln_final.*
    text_encoder.positional_embedding.     →  text.positional_embedding
      pos_embed.pos_embed
    text_encoder.projection_layer          →  text.text_projection
    text_encoder.transformer.N.            →  text.transformer.resblocks.N.
      pre_norm_mha.0.*                           ln_1.*
      pre_norm_mha.1.qkv_proj.*                  attn.in_proj_*
      pre_norm_mha.1.out_proj.*                   attn.out_proj.*
      pre_norm_ffn.0.*                            ln_2.*
      pre_norm_ffn.1.*                            mlp.c_fc.*
      pre_norm_ffn.4.*                            mlp.c_proj.*
    logit_scale                            →  logit_scale  (unchanged)
    """
    import re

    new_sd: Dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = key

        # ── Vision head: proj [out, in] → head.weight [in, out] (transposed) ──
        if key in ("image_encoder.model.head.proj", "image_encoder.model.classifier.proj"):
            new_sd["visual.trunk.head.weight"] = value.T
            new_sd["visual.trunk.head.bias"] = value.new_zeros((value.shape[1],))
            continue

        # ── Vision encoder (ViT-B style) ──
        elif key == "image_encoder.model.pos_embed.pos_embed.pos_embed":
            new_key = "visual.trunk.pos_embed"
            # Apple stores this as [1, 1, num_patches+1, dim] for ViT-B.
            if value.dim() == 4 and value.shape[0] == 1:
                value = value.squeeze(0)

        elif key.startswith("image_encoder.model.patch_emb."):
            suffix = key[len("image_encoder.model.patch_emb."):]
            m = re.match(r"(\d+)\.block\.(.*)", suffix)
            if m:
                idx, rest = m.group(1), m.group(2)
                if rest.startswith("norm."):
                    # Apple conv-bn block uses "norm", timm ViT stem uses "bn"
                    rest = "bn." + rest[len("norm."):]
                new_key = f"visual.trunk.patch_embed.backbone.{idx}.{rest}"
            else:
                logger.warning(f"Unmapped vision patch embedding key: {key}")

        elif key.startswith("image_encoder.model.post_transformer_norm."):
            new_key = "visual.trunk.norm." + key[len("image_encoder.model.post_transformer_norm."):]

        elif key.startswith("image_encoder.model.transformer."):
            suffix = key[len("image_encoder.model.transformer."):]
            m = re.match(r"(\d+)\.(.*)", suffix)
            if m:
                idx, rest = m.group(1), m.group(2)
                prefix = f"visual.trunk.blocks.{idx}."

                if rest.startswith("pre_norm_mha.0."):
                    new_key = prefix + "norm1." + rest[len("pre_norm_mha.0."):]
                elif rest.startswith("pre_norm_mha.1.qkv_proj."):
                    new_key = prefix + "attn.qkv." + rest[len("pre_norm_mha.1.qkv_proj."):]
                elif rest.startswith("pre_norm_mha.1.out_proj."):
                    new_key = prefix + "attn.proj." + rest[len("pre_norm_mha.1.out_proj."):]
                elif rest.startswith("pre_norm_ffn.0."):
                    new_key = prefix + "norm2." + rest[len("pre_norm_ffn.0."):]
                elif rest.startswith("pre_norm_ffn.1."):
                    new_key = prefix + "mlp.fc1." + rest[len("pre_norm_ffn.1."):]
                elif rest.startswith("pre_norm_ffn.4."):
                    new_key = prefix + "mlp.fc2." + rest[len("pre_norm_ffn.4."):]
                else:
                    logger.warning(f"Unmapped vision transformer key: {key}")

        # ── Vision encoder (fallback prefix remap, e.g. S0/S2) ──
        elif key.startswith("image_encoder.model."):
            new_key = "visual.trunk." + key[len("image_encoder.model."):]

        # ── Text encoder: non-transformer parts ──
        elif key == "text_encoder.embedding_layer.weight":
            new_key = "text.token_embedding.weight"

        elif key.startswith("text_encoder.final_layer_norm."):
            new_key = "text.ln_final." + key[len("text_encoder.final_layer_norm."):]

        elif key == "text_encoder.positional_embedding.pos_embed.pos_embed":
            new_key = "text.positional_embedding"

        elif key == "text_encoder.projection_layer":
            new_key = "text.text_projection"

        # ── Text encoder: transformer blocks ──
        elif key.startswith("text_encoder.transformer."):
            suffix = key[len("text_encoder.transformer."):]
            # Extract block index: "N.rest"
            m = re.match(r"(\d+)\.(.*)", suffix)
            if m:
                idx, rest = m.group(1), m.group(2)
                prefix = f"text.transformer.resblocks.{idx}."

                if rest.startswith("pre_norm_mha.0."):
                    # LayerNorm before attention → ln_1
                    new_key = prefix + "ln_1." + rest[len("pre_norm_mha.0."):]

                elif rest.startswith("pre_norm_mha.1.qkv_proj."):
                    # qkv_proj → attn.in_proj_  (weight/bias)
                    param = rest[len("pre_norm_mha.1.qkv_proj."):]  # "weight" or "bias"
                    new_key = prefix + "attn.in_proj_" + param

                elif rest.startswith("pre_norm_mha.1.out_proj."):
                    new_key = prefix + "attn.out_proj." + rest[len("pre_norm_mha.1.out_proj."):]

                elif rest.startswith("pre_norm_ffn.0."):
                    # LayerNorm before FFN → ln_2
                    new_key = prefix + "ln_2." + rest[len("pre_norm_ffn.0."):]

                elif rest.startswith("pre_norm_ffn.1."):
                    # First linear → mlp.c_fc
                    new_key = prefix + "mlp.c_fc." + rest[len("pre_norm_ffn.1."):]

                elif rest.startswith("pre_norm_ffn.4."):
                    # Second linear → mlp.c_proj
                    new_key = prefix + "mlp.c_proj." + rest[len("pre_norm_ffn.4."):]
                else:
                    logger.warning(f"Unmapped text transformer key: {key}")

        # logit_scale stays as-is

        new_sd[new_key] = value

    logger.info(f"Remapped {len(new_sd)} keys from MobileCLIP2 → OpenCLIP format")
    return new_sd


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load pretrained weights with key mapping for MobileCLIP2 → OpenCLIP format.

    Handles:
      - Different checkpoint envelope formats (state_dict / model / bare dict)
      - ``module.`` prefix from DDP
      - Apple MobileCLIP2 key remapping to OpenCLIP CustomTextCLIP format
      - Text positional-embedding resizing (e.g. 77 → 117)

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the .pt checkpoint file.
        strict: Whether to enforce strict key matching.

    Returns:
        Dict with ``missing_keys`` and ``unexpected_keys``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap checkpoint envelope
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip DDP 'module.' prefix
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # Detect Apple MobileCLIP2 format and remap keys
    if any(k.startswith("image_encoder.") for k in state_dict):
        logger.info("Detected Apple MobileCLIP2 checkpoint format, remapping keys...")
        state_dict = _remap_mobileclip2_keys(state_dict)

    # Resize text positional embeddings if context_length changed (77 → 117)
    model_state = model.state_dict()
    pe_key = "text.positional_embedding"
    if pe_key in state_dict and pe_key in model_state:
        old_pe = state_dict[pe_key]
        target_shape = model_state[pe_key].shape

        # Apple stores pos_embed as [1, 1, seq_len, dim] — squeeze to [seq_len, dim]
        while old_pe.dim() > 2:
            old_pe = old_pe.squeeze(0)

        if old_pe.shape != target_shape:
            new_len = target_shape[0]
            logger.info(
                f"Resizing {pe_key}: {list(old_pe.shape)} → {list(target_shape)}"
            )
            # [seq_len, dim] → interpolate along seq dim
            new_pe = torch.nn.functional.interpolate(
                old_pe.unsqueeze(0).transpose(1, 2),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).squeeze(0)
            state_dict[pe_key] = new_pe
        else:
            state_dict[pe_key] = old_pe

    result = model.load_state_dict(state_dict, strict=strict)
    logger.info(
        f"Loaded pretrained weights from {checkpoint_path} — "
        f"missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}"
    )
    if result.missing_keys:
        logger.warning(f"Missing keys (first 10): {result.missing_keys[:10]}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys (first 10): {result.unexpected_keys[:10]}")

    return {
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
    }
