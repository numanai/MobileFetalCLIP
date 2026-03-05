"""MobileFetalCLIP: Main entry point for training.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \
        -m mobile_fetal_clip.main \
        --config configs/default.yaml \
        --model-config configs/model/mobileclip2_s0_fetal.json \
        --pretrained /path/to/mobileclip2_s0.pt
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import open_clip
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

# Register MCI models before any open_clip imports
from mobile_fetal_clip.models import register_all_mci_models
register_all_mci_models()

from mobile_fetal_clip.data.fetal_dataset import create_fetal_dataloader
from mobile_fetal_clip.data.kd_coupled_preprocess import create_coupled_kd_train_preprocess
from mobile_fetal_clip.data.kd_coupled_preprocess import TEACHER_IMAGE_SIZE
from mobile_fetal_clip.data.student_preprocess import create_student_train_preprocess, DEFAULT_STUDENT_IMAGE_SIZE
from mobile_fetal_clip.models.factory import create_fetal_clip_model, get_tokenizer, load_pretrained_weights
from mobile_fetal_clip.training.trainer import CLIPTrainer
from mobile_fetal_clip.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MobileFetalCLIP Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to training config YAML file.",
    )
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Path to model config JSON file (e.g., configs/model/mobileclip2_s0_fetal.json).",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained MobileCLIP2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--train-data", type=str, default=None,
        help="Override train data path from config.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="Path to teacher model weights for knowledge distillation (e.g. FetalCLIP_weights.pt).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--distill-weight", type=float, default=None,
        help="Override distillation loss weight.",
    )
    parser.add_argument(
        "--distill-temperature", type=float, default=None,
        help="Override distillation temperature.",
    )
    parser.add_argument(
        "--feature-kd-weight", type=float, default=None,
        help="Override Feature-Space KD (MSE projection) weight.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--project-name", type=str, default=None,
        help="Override tracker project name.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name for local + W&B logging.",
    )
    parser.add_argument(
        "--run-group", type=str, default=None,
        help="W&B run group.",
    )
    parser.add_argument(
        "--run-tags", type=str, default=None,
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--exp-id", type=str, default=None,
        help="Stable experiment identifier used in local summaries.",
    )
    parser.add_argument(
        "--master-csv", type=str, default=None,
        help="Path to master experiments CSV for cross-run comparison.",
    )
    parser.add_argument(
        "--loss-type", type=str, default=None, choices=["clip", "siglip"],
        help="Contrastive loss type: 'clip' (softmax CE, default) or 'siglip' (sigmoid binary CE).",
    )
    parser.add_argument(
        "--distill-weight-decay", action="store_true", default=False,
        help="Enable linear decay of distill_weight over training.",
    )
    parser.add_argument(
        "--distill-weight-min-ratio", type=float, default=None,
        help="End distill_weight = initial × this ratio. Only used when --distill-weight-decay is set.",
    )
    parser.add_argument(
        "--feature-kd-type", type=str, default=None, choices=["mse", "cosine"],
        help="Feature KD loss type: 'mse' (default) or 'cosine' (1 - cos_sim).",
    )
    parser.add_argument(
        "--decay-feature-kd", action="store_true", default=False,
        help="Enable linear decay of feature_kd_weight over training (same schedule as logit decay).",
    )
    parser.add_argument(
        "--legacy-decay", action="store_true", default=False,
        help="Replicate the original legacy overshoot decay schedule where "
             "_decay_max_steps used optimizer steps instead of micro-batches, "
             "causing decay_frac to reach ~2.0 with accum=2. This makes the KD "
             "weight cross zero and go negative. Used for apple-to-apple scaling "
             "comparisons with original S0 runs.",
    )
    parser.add_argument(
        "--confidence-penalty", type=float, default=None,
        help="Label smoothing applied to teacher KL targets (0.0 = off).",
    )
    parser.add_argument(
        "--decoupled-kd", action="store_true", default=False,
        help="Enable DKD (Decoupled KD): split KL into diagonal (TCKD, always positive) "
             "and off-diagonal (NCKD, goes repulsive via distill-weight decay). "
             "Without this flag, standard coupled KL is used (identical to previous experiments).",
    )
    parser.add_argument(
        "--logit-standardization", action="store_true", default=False,
        help="Z-score normalize teacher and student logits per-row before softmax "
             "(Sun et al., CVPR 2024). Removes magnitude mismatch, focuses KD on ranking. "
             "Without this flag, raw logits are used (identical to previous experiments).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override per-GPU batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=None,
        help="Override gradient accumulation steps.",
    )
    return parser.parse_args()


def _read_image_size_from_config(model_config_path: str) -> int:
    """Read image_size from a model JSON config file.

    Falls back to DEFAULT_STUDENT_IMAGE_SIZE (256) if the field is absent.
    This ensures preprocessing is correct for all MobileCLIP2 variants:
      S0 / S2 (FastViT) → 256,  B (ViT-B) → 224.
    """
    try:
        with open(model_config_path) as f:
            cfg = json.load(f)
        size = cfg.get("vision_cfg", {}).get("image_size", DEFAULT_STUDENT_IMAGE_SIZE)
        return int(size)
    except Exception:
        return DEFAULT_STUDENT_IMAGE_SIZE


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.train_data:
        cfg.data.train_data = args.train_data
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.distill_weight is not None:
        cfg.training.distill_weight = args.distill_weight
    if args.distill_temperature is not None:
        cfg.training.distill_temperature = args.distill_temperature
    if args.feature_kd_weight is not None:
        cfg.training.feature_kd_weight = args.feature_kd_weight
    if args.seed is not None:
        cfg.seed = args.seed
    if args.project_name:
        cfg.training.project_name = args.project_name
    if args.run_name:
        cfg.training.run_name = args.run_name
    if args.run_group:
        cfg.training.run_group = args.run_group
    if args.run_tags:
        cfg.training.run_tags = [t.strip() for t in args.run_tags.split(",") if t.strip()]
    if args.exp_id:
        cfg.training.exp_id = args.exp_id
    if args.master_csv:
        cfg.training.master_csv = args.master_csv
    if args.loss_type:
        cfg.training.loss_type = args.loss_type
    if args.distill_weight_decay:
        cfg.training.distill_weight_decay = True
    if args.distill_weight_min_ratio is not None:
        cfg.training.distill_weight_min_ratio = args.distill_weight_min_ratio
    if args.feature_kd_type is not None:
        cfg.training.feature_kd_type = args.feature_kd_type
    if args.decay_feature_kd:
        cfg.training.decay_feature_kd = True
    if args.legacy_decay:
        cfg.training.legacy_decay = True
    if args.confidence_penalty is not None:
        cfg.training.confidence_penalty = args.confidence_penalty
    if args.decoupled_kd:
        cfg.training.decoupled_kd = True
    if args.logit_standardization:
        cfg.training.logit_standardization = True
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps

    # Store model config path in cfg for reference
    cfg.model = OmegaConf.create({
        "config_path": args.model_config,
        "pretrained": args.pretrained,
        "teacher": args.teacher,
    })

    # Initialize accelerator
    log_with = list(cfg.training.get("log_with", ["tensorboard"]))
    accelerator = Accelerator(
        mixed_precision=cfg.training.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        log_with=log_with,
        project_dir=cfg.training.get("output_dir", "outputs"),
    )

    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file=os.path.join(cfg.training.output_dir, "train.log") if accelerator.is_main_process else None,
        rank=accelerator.process_index,
    )

    # Set seed
    set_seed(cfg.get("seed", 42))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    if accelerator.is_main_process:
        outdir = Path(cfg.training.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
        (outdir / "cli_args.json").write_text(json.dumps(vars(args), indent=2))
        cmd = " ".join(sys.argv)
        (outdir / "command_main.txt").write_text(cmd + "\n")

    # Create model (open_clip handles val preprocess automatically from the JSON config)
    logger.info(f"Creating model from: {args.model_config}")
    model, _, preprocess_val = create_fetal_clip_model(
        model_config_path=args.model_config,
        pretrained=None,  # weights loaded separately below for better control
        precision="fp32",  # Accelerate handles mixed precision
        device="cpu",  # Accelerate handles device placement
    )

    # ── Preprocessing ──────────────────────────────────────────────────────────
    #
    # image_size is read from the model JSON config so preprocessing is correct
    # for all MobileCLIP2 variants (S0/S2 → 256, B → 224).
    #
    # Without KD (--teacher not passed):
    #   Student train: Resize(N,BICUBIC) → CenterCrop(N) → RandomAffine
    #                  → ColorJitter → ToTensor → Normalize(ImageNet)
    #
    # With KD (--teacher passed):
    #   Coupled preprocess: one PIL image → (student_tensor, teacher_tensor)
    #   Both branches share the same randomly-sampled affine + jitter params.
    #   Student: N×N, ImageNet stats.  Teacher: 224×224, OpenAI stats.
    #
    # Eval (both modes): open_clip val transform from the model JSON
    #   → Resize(N,BICUBIC) → CenterCrop(N) → ToTensor → Normalize(ImageNet)
    # ───────────────────────────────────────────────────────────────────────────
    student_image_size = _read_image_size_from_config(args.model_config)
    cfg.data.student_image_size = student_image_size
    if args.teacher:
        paired_preprocess_train = create_coupled_kd_train_preprocess(
            student_image_size=student_image_size,
        )
        preprocess_train = create_student_train_preprocess(image_size=student_image_size)
        logger.info(
            "KD mode — coupled preprocess: "
            "student 3×%d×%d (ImageNet stats), teacher 3×%d×%d (OpenAI stats); "
            "RandomAffine(7°, translate=0.05) + ColorJitter(0.15,0.15,0.15) shared.",
            student_image_size, student_image_size,
            TEACHER_IMAGE_SIZE, TEACHER_IMAGE_SIZE,
        )
    else:
        paired_preprocess_train = None
        preprocess_train = create_student_train_preprocess(image_size=student_image_size)
        logger.info(
            "Standard mode — student train preprocess: "
            "Resize(%d,BICUBIC) → CenterCrop(%d) → RandomAffine(7°, translate=0.05) "
            "→ ColorJitter(0.15,0.15,0.15) → ToTensor → Normalize(ImageNet).",
            student_image_size, student_image_size,
        )

    # Load pretrained weights if specified
    if args.pretrained:
        logger.info(f"Loading pretrained weights from: {args.pretrained}")
        load_pretrained_weights(model, args.pretrained, strict=False)

    # Get tokenizer (registers config dir with open_clip internally)
    tokenizer = get_tokenizer(args.model_config)

    # Create train dataloader
    logger.info(f"Creating train dataloader: {cfg.data.train_data}")
    train_dataloader, num_train_batches = create_fetal_dataloader(
        data_path=cfg.data.train_data,
        preprocess_fn=preprocess_train,
        paired_preprocess_fn=paired_preprocess_train,
        tokenizer=tokenizer,
        num_samples=cfg.data.train_num_samples,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 4),
        is_train=True,
        seed=cfg.get("seed", 42),
    )

    # Create val dataloader (optional)
    val_dataloader = None
    num_val_batches = 0
    if cfg.data.get("val_data"):
        logger.info(f"Creating val dataloader: {cfg.data.val_data}")
        val_dataloader, num_val_batches = create_fetal_dataloader(
            data_path=cfg.data.val_data,
            preprocess_fn=preprocess_val,
            paired_preprocess_fn=None,
            tokenizer=tokenizer,
            num_samples=cfg.data.get("val_num_samples", 0),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.get("num_workers", 4),
            is_train=False,
        )

    # Load teacher model for knowledge distillation (optional)
    teacher_model = None
    if args.teacher:
        logger.info(f"Loading teacher model (ViT-L/14) from: {args.teacher}")
        teacher_model = open_clip.create_model(
            "ViT-L-14", pretrained=None, force_context_length=117,
            output_dict=True,
        )
        ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        teacher_model.load_state_dict(sd, strict=True)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        logger.info(f"Teacher model loaded (epoch {ckpt.get('epoch', '?')}), "
                    f"{sum(p.numel() for p in teacher_model.parameters())/1e6:.1f}M params, frozen.")

    # Create trainer
    trainer = CLIPTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        num_train_batches=num_train_batches,
        val_dataloader=val_dataloader,
        num_val_batches=num_val_batches,
        accelerator=accelerator,
        tokenizer=tokenizer,
        preprocess_val=preprocess_val,
        teacher_model=teacher_model,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrapped = accelerator.unwrap_model(trainer.model)
        unwrapped.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        if "global_step" in checkpoint:
            trainer.global_step = checkpoint["global_step"]
        if "epoch" in checkpoint:
            trainer.start_epoch = int(checkpoint["epoch"])
        if "img_proj_head" in checkpoint and trainer.img_proj_head is not None:
            accelerator.unwrap_model(trainer.img_proj_head).load_state_dict(checkpoint["img_proj_head"])
            accelerator.unwrap_model(trainer.txt_proj_head).load_state_dict(checkpoint["txt_proj_head"])
            logger.info("Restored projection head weights from checkpoint.")
        logger.info(
            f"Resumed from epoch {checkpoint.get('epoch', '?')} "
            f"(next epoch index: {trainer.start_epoch})"
        )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
