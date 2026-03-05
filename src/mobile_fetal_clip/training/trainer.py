"""Training loop for MobileCLIP2 fetal fine-tuning using HuggingFace Accelerate."""
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm

from mobile_fetal_clip.training.loss import (
    CLIPLoss,
    DistillCLIPLoss,
    SigLIPLoss,
    DistillSigLIPLoss,
    ProjectionHead,
)
from mobile_fetal_clip.training.schedules import linear_weight_decay
from mobile_fetal_clip.training.scheduler import cosine_warmup_scheduler
from mobile_fetal_clip.evaluation.zero_shot import zero_shot_eval
SCHEMA_VERSION = "1.0.0"

logger = logging.getLogger(__name__)

IMPORTANT_ZEROSHOT_KEYS = (
    "five_planes/f1",
    "brain_subplanes/f1",
    "hc18/validity_rate",
    "hc18/ga_hc_spearman",
)
FIVE_PLANES_CLASS_COUNT = 5.0
BRAIN_SUBPLANES_CLASS_COUNT = 3.0


class CLIPTrainer:
    """Trains a CLIP model on fetal ultrasound data using HF Accelerate.

    Handles distributed training, mixed precision, gradient accumulation,
    checkpointing, and logging (wandb + tensorboard).
    """

    @staticmethod
    def _safe_dataloader_len(dataloader, fallback: int) -> int:
        """Return dataloader length when available, otherwise use fallback."""
        length = fallback
        try:
            length = len(dataloader)
        except TypeError:
            pass
        return max(int(length), 1)

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_dataloader,
        num_train_batches: int,
        val_dataloader=None,
        num_val_batches: int = 0,
        accelerator: Optional[Accelerator] = None,
        tokenizer=None,
        preprocess_val=None,
        teacher_model: Optional[torch.nn.Module] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_train_batches = num_train_batches
        self.num_train_batches_local = num_train_batches
        self.val_dataloader = val_dataloader
        self.num_val_batches = num_val_batches
        self.num_val_batches_local = num_val_batches

        # Accelerator
        self.accelerator = accelerator or Accelerator(
            mixed_precision=cfg.training.get("mixed_precision", "fp16"),
            gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
            log_with=cfg.training.get("log_with", ["tensorboard"]),
            project_dir=cfg.training.get("output_dir", "outputs"),
        )

        # Training params
        self.epochs = cfg.training.epochs
        self.lr = cfg.training.lr
        self.weight_decay = cfg.training.weight_decay
        self.warmup_steps = cfg.training.warmup_steps
        self.grad_clip_norm = cfg.training.get("grad_clip_norm", 1.0)
        self.save_frequency = cfg.training.get("save_frequency", 1)
        self.log_every_n_steps = cfg.training.get("log_every_n_steps", 10)
        self.output_dir = Path(cfg.training.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Teacher model for knowledge distillation
        self.teacher_model = teacher_model

        # Loss — select based on loss_type config (default: "clip")
        loss_type = cfg.training.get("loss_type", "clip").lower()
        if self.teacher_model is not None:
            if loss_type == "siglip":
                self.loss_fn = DistillSigLIPLoss(
                    contrastive_weight=cfg.training.get("contrastive_weight", 1.0),
                    distill_weight=cfg.training.get("distill_weight", 1.0),
                    distill_temperature=cfg.training.get("distill_temperature", 7.0),
                    gather_with_grad=cfg.training.get("gather_with_grad", True),
                    local_loss=cfg.training.get("local_loss", False),
                )
                logger.info(
                    f"Using DistillSigLIPLoss (SigLIP + KL on logits, "
                    f"contrastive_weight={cfg.training.get('contrastive_weight', 1.0)}, "
                    f"distill_weight={cfg.training.get('distill_weight', 1.0)}, "
                    f"distill_temperature={cfg.training.get('distill_temperature', 7.0)})"
                )
            else:
                self.loss_fn = DistillCLIPLoss(
                    contrastive_weight=cfg.training.get("contrastive_weight", 1.0),
                    distill_weight=cfg.training.get("distill_weight", 1.0),
                    distill_temperature=cfg.training.get("distill_temperature", 7.0),
                    feature_kd_weight=cfg.training.get("feature_kd_weight", 0.0),
                    feature_kd_type=cfg.training.get("feature_kd_type", "mse"),
                    confidence_penalty=cfg.training.get("confidence_penalty", 0.0),
                    decoupled_kd=cfg.training.get("decoupled_kd", False),
                    logit_standardization=cfg.training.get("logit_standardization", False),
                    gather_with_grad=cfg.training.get("gather_with_grad", True),
                    local_loss=cfg.training.get("local_loss", False),
                )
                logger.info(
                    f"Using DistillCLIPLoss (KL on logits, "
                    f"contrastive_weight={cfg.training.get('contrastive_weight', 1.0)}, "
                    f"distill_weight={cfg.training.get('distill_weight', 1.0)}, "
                    f"distill_temperature={cfg.training.get('distill_temperature', 7.0)}, "
                    f"feature_kd_weight={cfg.training.get('feature_kd_weight', 0.0)}, "
                    f"feature_kd_type={cfg.training.get('feature_kd_type', 'mse')}, "
                    f"confidence_penalty={cfg.training.get('confidence_penalty', 0.0)}, "
                    f"decoupled_kd={cfg.training.get('decoupled_kd', False)}, "
                    f"logit_standardization={cfg.training.get('logit_standardization', False)})"
                )
        else:
            if loss_type == "siglip":
                self.loss_fn = SigLIPLoss()
                logger.info("Using SigLIPLoss (sigmoid binary CE, no teacher)")
            else:
                self.loss_fn = CLIPLoss(
                    gather_with_grad=cfg.training.get("gather_with_grad", True),
                    local_loss=cfg.training.get("local_loss", False),
                )

        # Feature KD Projection Heads
        self.feature_kd_weight = cfg.training.get("feature_kd_weight", 0.0)
        self.img_proj_head = None
        self.txt_proj_head = None
        if self.feature_kd_weight > 0.0 and self.teacher_model is not None:
            student_dim = cfg.model.get("embed_dim", 512)
            teacher_dim = cfg.training.get("teacher_embed_dim", 768)
            self.img_proj_head = ProjectionHead(student_dim, teacher_dim)
            self.txt_proj_head = ProjectionHead(student_dim, teacher_dim)
            logger.info(f"Initialized ProjectionHeads for Feature KD ({student_dim} -> {teacher_dim})")

        # Optimizer
        self._setup_optimizer()

        # Prepare with accelerator (model/optimizer/dataloaders first).
        prepare_args = [self.model, self.optimizer, self.train_dataloader]
        if self.img_proj_head is not None:
            prepare_args.extend([self.img_proj_head, self.txt_proj_head])

        prepared = self.accelerator.prepare(*prepare_args)

        idx = 0
        self.model = prepared[idx]; idx += 1
        self.optimizer = prepared[idx]; idx += 1
        self.train_dataloader = prepared[idx]; idx += 1
        if self.img_proj_head is not None:
            self.img_proj_head = prepared[idx]; idx += 1
            self.txt_proj_head = prepared[idx]; idx += 1

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Scheduler
        # For iterable/dispatched dataloaders, per-rank micro-batch count can
        # differ from the global estimate. Compute scheduler/decay lengths from
        # post-prepare local length to keep single- and multi-GPU behavior
        # aligned in optimizer-step time.
        accum = max(1, int(cfg.training.get("gradient_accumulation_steps", 1)))
        world_size = max(1, int(self.accelerator.num_processes))

        train_local_fallback = math.ceil(self.num_train_batches / world_size)
        self.num_train_batches_local = self._safe_dataloader_len(
            self.train_dataloader, train_local_fallback
        )
        if self.val_dataloader is not None and self.num_val_batches > 0:
            val_local_fallback = math.ceil(self.num_val_batches / world_size)
            self.num_val_batches_local = self._safe_dataloader_len(
                self.val_dataloader, val_local_fallback
            )

        sync_steps_per_epoch = math.ceil(self.num_train_batches_local / accum)
        self.total_steps = max(1, sync_steps_per_epoch * self.epochs * world_size)
        self.warmup_steps = min(int(self.warmup_steps) * world_size, self.total_steps)
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer, self.warmup_steps, self.total_steps
        )
        self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        # Place teacher on device (frozen, not prepared by accelerator)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.accelerator.device)
            if self.accelerator.mixed_precision == "fp16":
                self.teacher_model = self.teacher_model.half()
            elif self.accelerator.mixed_precision == "bf16":
                self.teacher_model = self.teacher_model.bfloat16()
        self._warned_teacher_resize_fallback = False

        self.global_step = 0
        self.start_epoch = 0

        # Dynamic distill weight decay
        self._distill_weight_decay = cfg.training.get("distill_weight_decay", False)
        self._distill_weight_initial = cfg.training.get("distill_weight", 1.0)
        self._distill_weight_min_ratio = cfg.training.get("distill_weight_min_ratio", 0.1)
        self._decay_feature_kd = cfg.training.get("decay_feature_kd", False)
        self._feature_kd_initial = cfg.training.get("feature_kd_weight", 0.0)
        self._legacy_decay = cfg.training.get("legacy_decay", False)

        if self._legacy_decay:
            # Replicate original legacy overshoot behavior: _decay_max_steps used
            # optimizer steps (num_batches // accum * epochs) while global_step
            # counts micro-batches, causing decay_frac to reach ~2.0 with accum=2.
            self._decay_max_steps = (self.num_train_batches_local // accum) * self.epochs
        else:
            self._decay_max_steps = self.num_train_batches_local * self.epochs
        if self._distill_weight_decay and self.teacher_model is not None:
            logger.info(
                f"Dynamic distill weight decay enabled: "
                f"{self._distill_weight_initial:.2f} → "
                f"{self._distill_weight_initial * self._distill_weight_min_ratio:.2f} "
                f"over {self._decay_max_steps} micro-batches."
            )
        if self._decay_feature_kd and self.teacher_model is not None:
            logger.info(
                f"Dynamic feature KD weight decay enabled: "
                f"{self._feature_kd_initial:.2f} → "
                f"{self._feature_kd_initial * self._distill_weight_min_ratio:.2f} "
                f"over {self._decay_max_steps} micro-batches."
            )

        # Zero-shot eval config
        self.tokenizer = tokenizer
        self.preprocess_val = preprocess_val
        self.eval_cfg = dict(cfg.get("evaluation", {}))
        self.zeroshot_frequency = self.eval_cfg.get("zeroshot_frequency", 1)

    @staticmethod
    def _compute_classification_f1_true8(five_planes_f1: float, brain_subplanes_f1: float) -> float:
        """True 8-class macro F1 from grouped macro F1 values."""
        return (
            FIVE_PLANES_CLASS_COUNT * five_planes_f1
            + BRAIN_SUBPLANES_CLASS_COUNT * brain_subplanes_f1
        ) / (FIVE_PLANES_CLASS_COUNT + BRAIN_SUBPLANES_CLASS_COUNT)

    @staticmethod
    def _compute_classification_f1_groupavg(five_planes_f1: float, brain_subplanes_f1: float) -> float:
        """Legacy group-average F1 where each subgroup gets equal weight."""
        return (five_planes_f1 + brain_subplanes_f1) / 2.0

    def _setup_optimizer(self):
        """Set up AdamW with weight decay separation."""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(
                kw in name for kw in ("bn", "ln", "bias", "logit_scale", "norm")
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(self.cfg.training.get("beta1", 0.9), self.cfg.training.get("beta2", 0.98)),
            eps=self.cfg.training.get("eps", 1e-6),
        )

        # Add projection heads to optimizer if they exist
        if self.img_proj_head is not None and self.txt_proj_head is not None:
            proj_params = list(self.img_proj_head.parameters()) + list(self.txt_proj_head.parameters())
            self.optimizer.add_param_group({"params": proj_params, "weight_decay": self.weight_decay})

    def train(self):
        """Run the full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs, {self.total_steps} total steps")
        logger.info(f"  Batch size per device: {self.cfg.training.batch_size}")
        logger.info(f"  World size: {self.accelerator.num_processes}")
        logger.info(f"  Gradient accumulation: {self.cfg.training.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  Mixed precision: {self.accelerator.mixed_precision}")
        logger.info(
            f"  Train batches/epoch: global={self.num_train_batches}, "
            f"per-rank={self.num_train_batches_local}"
        )

        # Init trackers
        if self.accelerator.is_main_process:
            tracker_config = {
                "model": self.cfg.get("model", {}).get("name", "mobileclip2"),
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.cfg.training.batch_size,
                "seed": self.cfg.get("seed", 42),
                "exp_id": self.cfg.training.get("exp_id", ""),
                "distill_weight": self.cfg.training.get("distill_weight", 0.0),
                "distill_temperature": self.cfg.training.get("distill_temperature", 0.0),
                "teacher_enabled": self.teacher_model is not None,
                "student_image_size": self.cfg.data.get("student_image_size", 256),
                "student_preprocess": "Resize(N,BICUBIC)->CenterCrop(N)->RandomAffine->ColorJitter->Normalize(ImageNet)",
                "kd_preprocess": "coupled" if self.teacher_model is not None else "n/a",
            }
            wandb_init = {}
            run_name = self.cfg.training.get("run_name")
            run_group = self.cfg.training.get("run_group")
            run_tags = self.cfg.training.get("run_tags")
            if run_name:
                wandb_init["name"] = run_name
            if run_group:
                wandb_init["group"] = run_group
            if run_tags:
                wandb_init["tags"] = list(run_tags)
            init_kwargs = {"wandb": wandb_init} if wandb_init else None
            self.accelerator.init_trackers(
                project_name=self.cfg.training.get("project_name", "mobile_fetal_clip"),
                config=tracker_config,
                init_kwargs=init_kwargs,
            )
            # Configure W&B so zero-shot metrics are plotted against epoch while
            # all logs still use monotonically increasing global step.
            for tracker in self.accelerator.trackers:
                if hasattr(tracker, "run"):
                    try:
                        import wandb

                        wandb.define_metric("train/*", step_metric="trainer/global_step")
                        wandb.define_metric("val/*", step_metric="trainer/global_step")
                        wandb.define_metric("zeroshot/*", step_metric="zeroshot/epoch")
                    except Exception as e:
                        logger.warning(f"Could not configure W&B metric axes: {e}")

        best_val_loss = float("inf")
        best_zs_composite = 0.0
        best_zs_metrics = {}
        best_zs_epoch = -1

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_one_epoch(epoch)

            # Validation
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self._evaluate(epoch)
                if val_metrics.get("val_loss", float("inf")) < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self._save_checkpoint(epoch, is_best=True)

            # Zero-shot evaluation
            zs_metrics = {}
            if (
                self.tokenizer is not None
                and self.preprocess_val is not None
                and self.eval_cfg
                and ((epoch + 1) % self.zeroshot_frequency == 0 or epoch == self.epochs - 1)
            ):
                unwrapped = self.accelerator.unwrap_model(self.model)
                if self.accelerator.is_main_process:
                    zs_metrics = zero_shot_eval(
                        unwrapped,
                        self.tokenizer,
                        self.preprocess_val,
                        device=self.accelerator.device,
                        eval_cfg=self.eval_cfg,
                        epoch=epoch,
                    )
                    # Log only key zero-shot metrics to W&B/TB for cleaner dashboards.
                    zs_log = {
                        f"zeroshot/{k}": zs_metrics[k]
                        for k in IMPORTANT_ZEROSHOT_KEYS
                        if k in zs_metrics
                    }

                    # Save best based on composite score:
                    #   classification_f1 = true 8-class macro F1
                    #   composite = (classification_f1 + hc18_validity/100) / 2
                    five_f1 = zs_metrics.get("five_planes/f1", 0.0)
                    brain_f1 = zs_metrics.get("brain_subplanes/f1", 0.0)
                    classification_f1 = self._compute_classification_f1_true8(five_f1, brain_f1)
                    classification_f1_groupavg = self._compute_classification_f1_groupavg(five_f1, brain_f1)
                    zs_composite = (
                        classification_f1
                        + zs_metrics.get("hc18/validity_rate", 0.0) / 100.0
                    ) / 2.0
                    zs_log["zeroshot/classification_f1"] = classification_f1
                    zs_log["zeroshot/classification_f1_groupavg"] = classification_f1_groupavg
                    zs_log["zeroshot/composite_score"] = zs_composite
                    zs_log["zeroshot/epoch"] = epoch + 1
                    # Keep global step monotonic for logging backends.
                    self.accelerator.log(zs_log, step=self.global_step)
                    logger.info(f"  Composite score: {zs_composite:.4f}")
                    if zs_composite > best_zs_composite:
                        best_zs_composite = zs_composite
                        best_zs_metrics = zs_metrics.copy()
                        best_zs_epoch = epoch
                        self._save_checkpoint(epoch, is_best=True)

            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch)

            # Log epoch summary
            if self.accelerator.is_main_process:
                summary = {**train_metrics, **val_metrics, **zs_metrics, "epoch": epoch}
                logger.info(
                    f"Epoch {epoch}: "
                    + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in summary.items())
                )

        # Save results summary
        if self.accelerator.is_main_process and best_zs_metrics:
            self._save_results_summary(best_zs_metrics, best_zs_epoch, train_metrics)

        self.accelerator.end_training()
        logger.info("Training complete.")

    def _train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        batch_time = 0.0
        data_time = 0.0
        start = time.time()

        progress = tqdm(
            self.train_dataloader,
            total=self.num_train_batches_local,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process,
        )

        for batch_idx, batch in enumerate(progress):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                images, teacher_images, texts = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, texts = batch
                teacher_images = None
            else:
                raise ValueError(f"Unexpected batch structure with length={len(batch)}")

            data_time += time.time() - start

            # Dynamic weight decay: linearly decay weights from initial → initial×min_ratio
            if self.teacher_model is not None and (self._distill_weight_decay or self._decay_feature_kd):
                if self._distill_weight_decay:
                    dynamic_w = linear_weight_decay(
                        initial_weight=self._distill_weight_initial,
                        min_ratio=self._distill_weight_min_ratio,
                        step=self.global_step,
                        max_steps=self._decay_max_steps,
                        clamp=(not self._legacy_decay),
                    )
                    self.loss_fn.distill_weight = dynamic_w

                if self._decay_feature_kd:
                    dynamic_feature_w = linear_weight_decay(
                        initial_weight=self._feature_kd_initial,
                        min_ratio=self._distill_weight_min_ratio,
                        step=self.global_step,
                        max_steps=self._decay_max_steps,
                        clamp=(not self._legacy_decay),
                    )
                    self.loss_fn.feature_kd_weight = dynamic_feature_w

            with self.accelerator.accumulate(self.model):
                # Forward pass (autocast handles fp16 mixed precision)
                with self.accelerator.autocast():
                    # Run teacher first (no grad) to reduce peak memory. This avoids
                    # holding student autograd activations while doing ViT-L forward.
                    t_img = t_txt = t_logit_scale = None
                    if self.teacher_model is not None:
                        with torch.no_grad():
                            if teacher_images is None:
                                # Backward-compatible fallback for old dataloaders.
                                if not self._warned_teacher_resize_fallback:
                                    logger.warning(
                                        "Teacher-preprocessed images not found in batch. "
                                        "Falling back to bilinear resize from student images."
                                    )
                                    self._warned_teacher_resize_fallback = True
                                teacher_images = F.interpolate(
                                    images, size=(224, 224),
                                    mode="bilinear", align_corners=False,
                                )
                            teacher_out = self.teacher_model(teacher_images, texts)
                            t_img = teacher_out["image_features"]
                            t_txt = teacher_out["text_features"]
                            t_logit_scale = teacher_out["logit_scale"]

                    model_out = self.model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]

                    # Loss (with optional KD)
                    if self.teacher_model is not None:
                        # Project student features if feature KD is enabled
                        # Following CLIP-KD (Yang et al., CVPR 2024): project
                        # normalized student features, ProjectionHead outputs
                        # L2-normalized embeddings, teacher features are already
                        # normalized. MSE on unit vectors requires large weight.
                        proj_img, proj_txt = None, None
                        if self.img_proj_head is not None:
                            proj_img = self.img_proj_head(image_features)
                            proj_txt = self.txt_proj_head(text_features)

                        loss_out = self.loss_fn(
                            image_features, text_features, logit_scale,
                            t_img, t_txt, t_logit_scale,
                            proj_image_features=proj_img,
                            proj_text_features=proj_txt,
                        )
                        loss = loss_out["loss"]
                    else:
                        loss = self.loss_fn(image_features, text_features, logit_scale)

                # Backward
                self.accelerator.backward(loss)

                # Gradient clipping (only on sync steps to avoid double unscale with GradScaler)
                if self.grad_clip_norm is not None and self.accelerator.sync_gradients:
                    clip_params = list(self.model.parameters())
                    if self.img_proj_head is not None:
                        clip_params += list(self.img_proj_head.parameters()) + list(self.txt_proj_head.parameters())
                    self.accelerator.clip_grad_norm_(clip_params, self.grad_clip_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Clamp logit scale
            with torch.no_grad():
                unwrapped = self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped, "logit_scale"):
                    unwrapped.logit_scale.clamp_(0, math.log(100))

            batch_time += time.time() - start
            total_loss += loss.detach().item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.scheduler.get_last_lr()[0]
                log_scale = logit_scale.item() if logit_scale.dim() == 0 else logit_scale.mean().item()
                samples_per_sec = (
                    self.cfg.training.batch_size * self.accelerator.num_processes
                    / (batch_time / num_batches)
                )

                log_data = {
                    "trainer/global_step": float(self.global_step),
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/logit_scale": log_scale,
                    "train/samples_per_sec": samples_per_sec,
                    "train/data_time": data_time / num_batches,
                    "train/batch_time": batch_time / num_batches,
                }
                if self._distill_weight_decay and self.teacher_model is not None:
                    log_data["train/distill_weight"] = self.loss_fn.distill_weight
                if self._decay_feature_kd and self.teacher_model is not None:
                    log_data["train/feature_kd_weight"] = getattr(self.loss_fn, "feature_kd_weight", 0.0)
                if self.teacher_model is not None and isinstance(loss_out, dict):
                    # Log whichever contrastive loss key is present
                    if "clip_loss" in loss_out:
                        log_data["train/clip_loss"] = loss_out["clip_loss"].item()
                    if "siglip_loss" in loss_out:
                        log_data["train/siglip_loss"] = loss_out["siglip_loss"].item()
                    log_data["train/distill_loss"] = loss_out["distill_loss"].item()
                    if loss_out.get("feat_kd_loss") is not None:
                        log_data["train/feat_kd_loss"] = loss_out["feat_kd_loss"].item()
                    if loss_out.get("tckd_loss") is not None:
                        log_data["train/tckd_loss"] = loss_out["tckd_loss"].item()
                        log_data["train/nckd_loss"] = loss_out["nckd_loss"].item()
                self.accelerator.log(log_data, step=self.global_step)

                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    scale=f"{log_scale:.2f}",
                )

            if batch_idx >= self.num_train_batches_local - 1:
                break

            start = time.time()

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> dict:
        """Run evaluation and compute retrieval metrics."""
        self.model.eval()
        all_image_features = []
        all_text_features = []
        total_loss = 0.0
        num_batches = 0

        # Use base CLIP loss for validation (no teacher needed)
        val_loss_fn = (
            self.loss_fn.clip_loss
            if hasattr(self.loss_fn, "clip_loss")
            else self.loss_fn
        )

        for batch_idx, (images, texts) in enumerate(self.val_dataloader):
            with self.accelerator.autocast():
                model_out = self.model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]

                loss = val_loss_fn(image_features, text_features, logit_scale)
            total_loss += loss.item()
            num_batches += 1

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

            if batch_idx >= self.num_val_batches_local - 1:
                break

        if num_batches == 0:
            return {}

        avg_loss = total_loss / num_batches

        # Compute retrieval metrics
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        metrics = self._compute_retrieval_metrics(all_image_features, all_text_features)
        metrics["val_loss"] = avg_loss

        # Log
        log_data = {f"val/{k}": v for k, v in metrics.items()}
        self.accelerator.log(log_data, step=self.global_step)

        if self.accelerator.is_main_process:
            logger.info(
                f"Eval Epoch {epoch}: "
                + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

        return metrics

    @staticmethod
    def _compute_retrieval_metrics(
        image_features: torch.Tensor, text_features: torch.Tensor
    ) -> dict:
        """Compute image-text retrieval metrics (R@1, R@5, R@10)."""
        # Similarity matrix
        sim = image_features @ text_features.T
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        metrics = {}
        for name, logits in [("i2t", sim), ("t2i", sim.T)]:
            ranking = torch.argsort(logits, descending=True)
            preds = torch.where(ranking == ground_truth)[1].numpy()
            for k in [1, 5]:
                metrics[f"{name}_R@{k}"] = float(np.mean(preds < k))

        return metrics

    def _save_results_summary(self, best_zs_metrics: dict, best_epoch: int, last_train_metrics: dict):
        """Save experiment results as JSON, append to CSV, and set wandb summary."""
        five_f1 = best_zs_metrics.get("five_planes/f1", 0.0)
        brain_f1 = best_zs_metrics.get("brain_subplanes/f1", 0.0)
        classification_f1 = self._compute_classification_f1_true8(five_f1, brain_f1)
        classification_f1_groupavg = self._compute_classification_f1_groupavg(five_f1, brain_f1)
        composite_score = (
            classification_f1
            + best_zs_metrics.get("hc18/validity_rate", 0.0) / 100.0
        ) / 2.0
        wandb_run_id = "local"
        wandb_run_name = self.cfg.training.get("run_name", "local")
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, "run"):
                if hasattr(tracker.run, "id") and tracker.run.id:
                    wandb_run_id = str(tracker.run.id)
                if hasattr(tracker.run, "name") and tracker.run.name:
                    wandb_run_name = str(tracker.run.name)

        # --- Define the key metrics we care about ---
        try:
            output_dir_str = str(self.output_dir.relative_to(Path.cwd()))
        except ValueError:
            output_dir_str = str(self.output_dir)

        results = {
            "schema_version": SCHEMA_VERSION,
            "exp_id": self.cfg.training.get("exp_id", wandb_run_name),
            "run_name": wandb_run_name,
            "wandb_run_id": wandb_run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "backbone": self.cfg.get("model", {}).get("name", "mobileclip2_s0"),
            "teacher_enabled": bool(self.teacher_model is not None),
            "lr": self.lr,
            "distill_weight": self.cfg.training.get("distill_weight", 0.0),
            "distill_temperature": self.cfg.training.get("distill_temperature", 0.0),
            "seed": self.cfg.get("seed", 42),
            "batch_size": self.cfg.training.batch_size,
            "grad_accum": self.cfg.training.get("gradient_accumulation_steps", 1),
            "feature_kd_weight": self.cfg.training.get("feature_kd_weight", 0.0),
            "feature_kd_type": self.cfg.training.get("feature_kd_type", "mse"),
            "decay_feature_kd": self.cfg.training.get("decay_feature_kd", False),
            "confidence_penalty": self.cfg.training.get("confidence_penalty", 0.0),
            "student_image_size": self.cfg.data.get("student_image_size", 256),
            "student_preprocess": "Resize(N,BICUBIC)->CenterCrop(N)->RandomAffine->ColorJitter->Normalize(ImageNet)",
            "kd_preprocess": "coupled" if self.teacher_model is not None else "n/a",
            "epochs": self.epochs,
            "best_epoch": best_epoch + 1,
            "composite_score": composite_score,
            "classification_f1": classification_f1,
            "train_loss": last_train_metrics.get("train_loss", 0.0),
            "output_dir": output_dir_str,
            # Five-plane classification
            "five_planes_acc": best_zs_metrics.get("five_planes/acc", 0.0),
            "five_planes_f1": best_zs_metrics.get("five_planes/f1", 0.0),
            "five_planes_acc_top2": best_zs_metrics.get("five_planes/acc_top2", 0.0),
            "five_planes_acc_top3": best_zs_metrics.get("five_planes/acc_top3", 0.0),
            # Brain subplanes
            "brain_acc": best_zs_metrics.get("brain_subplanes/acc", 0.0),
            "brain_f1": best_zs_metrics.get("brain_subplanes/f1", 0.0),
            # HC18 gestational age
            "hc18_validity_rate": best_zs_metrics.get("hc18/validity_rate", 0.0),
            "hc18_validity_ci_lower": best_zs_metrics.get("hc18/validity_ci_lower", 0.0),
            "hc18_validity_ci_upper": best_zs_metrics.get("hc18/validity_ci_upper", 0.0),
            "hc18_spearman": best_zs_metrics.get("hc18/ga_hc_spearman", 0.0),
        }

        # --- 1. Save detailed JSON for this run ---
        results_json = self.output_dir / "results.json"
        full_results = {
            **results,
            "classification_f1_groupavg": classification_f1_groupavg,
            "all_zs_metrics": best_zs_metrics,
        }
        with open(results_json, "w") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Saved results: {results_json}")

        # --- 2. Append to experiments CSV ---
        csv_path = self.output_dir / "experiments.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        logger.info(f"Appended to: {csv_path}")

        # --- 2b. Append to global master CSV ---
        master_csv = self.cfg.training.get("master_csv", "outputs/experiments_master.csv")
        master_csv_path = Path(master_csv)
        if not master_csv_path.is_absolute():
            master_csv_path = Path.cwd() / master_csv_path
        master_csv_path.parent.mkdir(parents=True, exist_ok=True)
        master_exists = master_csv_path.exists()
        with open(master_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not master_exists:
                writer.writeheader()
            writer.writerow(results)
        logger.info(f"Appended to master: {master_csv_path}")

        # --- 3. Set wandb summary for Runs Table comparison ---
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, "run") and hasattr(tracker.run, "summary"):
                for k, v in results.items():
                    if isinstance(v, (int, float)):
                        tracker.run.summary[f"best/{k}"] = v
                tracker.run.summary["best/classification_f1_groupavg"] = classification_f1_groupavg
                logger.info("Set wandb summary metrics for cross-run comparison")
                # Add a compact per-run table to make sweep comparison easier in W&B.
                try:
                    import wandb

                    table_columns = [
                        "exp_id",
                        "run_name",
                        "best_epoch",
                        "composite_score",
                        "five_planes_f1",
                        "brain_f1",
                        "hc18_validity_rate",
                        "hc18_spearman",
                        "lr",
                        "distill_weight",
                        "distill_temperature",
                        "seed",
                        "feature_kd_type",
                        "decay_feature_kd",
                        "confidence_penalty",
                    ]
                    table_row = [results.get(c, "") for c in table_columns]
                    table = wandb.Table(columns=table_columns, data=[table_row])
                    tracker.run.log({"best/results_table": table})
                except Exception as e:
                    logger.warning(f"Could not log W&B results table: {e}")

        # --- 4. Print clean summary table ---
        logger.info("=" * 60)
        logger.info(f"BEST RESULTS (epoch {best_epoch + 1})")
        logger.info("=" * 60)
        logger.info(f"  Five-plane  | acc={results['five_planes_acc']:.4f}  f1={results['five_planes_f1']:.4f}  top2={results['five_planes_acc_top2']:.4f}  top3={results['five_planes_acc_top3']:.4f}")
        logger.info(f"  Brain sub   | acc={results['brain_acc']:.4f}  f1={results['brain_f1']:.4f}")
        logger.info(f"  HC18 GA     | validity={results['hc18_validity_rate']:.2f}% ({results['hc18_validity_ci_lower']:.2f}-{results['hc18_validity_ci_upper']:.2f}%)  spearman={results['hc18_spearman']:.4f}")
        logger.info("=" * 60)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }
        if self.img_proj_head is not None:
            checkpoint["img_proj_head"] = self.accelerator.unwrap_model(self.img_proj_head).state_dict()
            checkpoint["txt_proj_head"] = self.accelerator.unwrap_model(self.txt_proj_head).state_dict()

        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = ckpt_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        # Keep only last + best: remove other epoch_*.pt
        for f in ckpt_dir.glob("epoch_*.pt"):
            if f != path:
                f.unlink()
                logger.info(f"Removed old checkpoint: {f.name}")
