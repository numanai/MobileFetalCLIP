"""CLIP contrastive loss for image-text matching, with optional knowledge distillation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ProjectionHead(nn.Module):
    """Linear projection for Feature-Space Knowledge Distillation.

    Single linear layer matching CLIP-KD (Yang et al., CVPR 2024).
    Projects student embeddings to teacher dimension, then L2-normalizes.
    Use with MSE loss (reduction='mean') and a large weight (e.g. 2000)
    to compensate for the small magnitude of MSE between unit vectors.
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class CLIPLoss(nn.Module):
    """Symmetric cross-entropy loss for CLIP contrastive learning.

    Computes the standard CLIP loss: symmetric cross-entropy between
    image-to-text and text-to-image similarity matrices.
    Supports distributed training with all-gather across GPUs.
    """

    def __init__(
        self,
        gather_with_grad: bool = True,
        local_loss: bool = False,
        cache_labels: bool = True,
    ):
        super().__init__()
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self._labels = {}

    def _get_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get or create cached ground-truth labels."""
        if not self.cache_labels or batch_size not in self._labels:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.cache_labels:
                self._labels[batch_size] = labels
            return labels
        return self._labels[batch_size]

    def _gather_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """Gather features from all processes for distributed training."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return image_features, text_features

        if self.gather_with_grad:
            all_image = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image, image_features)
            dist.all_gather(gathered_text, text_features)
            if not self.local_loss:
                gathered_image[rank] = image_features
                gathered_text[rank] = text_features
            all_image = torch.cat(gathered_image, dim=0)
            all_text = torch.cat(gathered_text, dim=0)

        return all_image, all_text

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP contrastive loss.

        Args:
            image_features: Normalized image embeddings [B, D].
            text_features: Normalized text embeddings [B, D].
            logit_scale: Learnable temperature parameter (exp of log scale).

        Returns:
            Scalar loss value.
        """
        # Gather features across GPUs
        all_image_features, all_text_features = self._gather_features(
            image_features, text_features
        )

        # Compute similarity
        logits_per_image = logit_scale * image_features @ all_text_features.T
        logits_per_text = logit_scale * text_features @ all_image_features.T

        # Labels: diagonal is the positive pair
        batch_size = image_features.shape[0]
        if dist.is_initialized() and dist.get_world_size() > 1 and not self.local_loss:
            labels = self._get_labels(batch_size, image_features.device)
            labels = labels + batch_size * dist.get_rank()
        else:
            labels = self._get_labels(batch_size, image_features.device)

        # Symmetric cross-entropy
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss


class DistillCLIPLoss(nn.Module):
    """CLIP loss + KL-divergence knowledge distillation on similarity logits.

    Combines standard CLIP contrastive loss with soft-target distillation:
    the teacher's softmax distribution over the N×N similarity matrix serves
    as the target for the student's log-softmax distribution (KL divergence).
    This transfers relational knowledge (how all samples relate to each other)
    rather than just individual feature directions.

    No projection layers needed — operates on logits, not raw features.

    Args:
        contrastive_weight: Weight for the standard CLIP contrastive loss.
        distill_weight: Weight for the KL distillation loss.
        distill_temperature: Temperature to soften teacher's logits. Without this,
            a teacher with logit_scale ~100 produces near-one-hot distributions,
            making KD equivalent to hard-label CE (no soft knowledge transfer).
            Applied only to teacher logits (MobileCLIP approach).
        gather_with_grad: Whether to gather with gradients for distributed.
        local_loss: Use local loss in distributed setting.
        cache_labels: Cache label tensors.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        distill_weight: float = 1.0,
        distill_temperature: float = 1.0,
        feature_kd_weight: float = 0.0,
        feature_kd_type: str = "mse",
        confidence_penalty: float = 0.0,
        decoupled_kd: bool = False,
        logit_standardization: bool = False,
        gather_with_grad: bool = True,
        local_loss: bool = False,
        cache_labels: bool = True,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature
        self.feature_kd_weight = feature_kd_weight
        self.feature_kd_type = feature_kd_type.lower()
        self.confidence_penalty = confidence_penalty
        # DKD (Decoupled KD): split KL into diagonal (TCKD) + off-diagonal (NCKD).
        # When enabled, only NCKD goes repulsive via distill_weight; TCKD stays
        # positive (weight=1.0). Without this flag, behavior is identical to before.
        self.decoupled_kd = decoupled_kd
        # Logit standardization (Sun et al., CVPR 2024): z-score normalize logits
        # per-row before softmax to remove magnitude differences between teacher
        # and student, focusing KD on ranking structure. No effect when disabled.
        self.logit_standardization = logit_standardization
        self.clip_loss = CLIPLoss(
            gather_with_grad=gather_with_grad,
            local_loss=local_loss,
            cache_labels=cache_labels,
        )
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss

    def _gather_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """Gather features from all processes for distributed training."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return image_features, text_features

        if self.gather_with_grad:
            all_image = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image, image_features)
            dist.all_gather(gathered_text, text_features)
            if not self.local_loss:
                gathered_image[rank] = image_features
                gathered_text[rank] = text_features
            all_image = torch.cat(gathered_image, dim=0)
            all_text = torch.cat(gathered_text, dim=0)

        return all_image, all_text

    @staticmethod
    def _standardize_logits(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Z-score normalize logits per row (Logit Standardization, Sun et al. 2024).

        Removes magnitude differences between teacher and student logit scales,
        focusing KD gradient on ranking structure rather than absolute values.
        """
        mu = logits.mean(dim=1, keepdim=True)
        sigma = logits.std(dim=1, keepdim=True) + eps
        return (logits - mu) / sigma

    @staticmethod
    def _kl_distill_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        confidence_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Standard KL divergence: -(softmax(teacher) * log_softmax(student)).sum(dim=1).mean()

        This is the original (coupled) KL used when --decoupled-kd is NOT set.
        """
        teacher_probs = teacher_logits.softmax(dim=1)
        if confidence_penalty > 0.0:
            num_classes = teacher_probs.size(1)
            teacher_probs = (1.0 - confidence_penalty) * teacher_probs + (confidence_penalty / num_classes)

        return -(
            teacher_probs * student_logits.log_softmax(dim=1)
        ).sum(dim=1).mean(dim=0)

    @staticmethod
    def _decoupled_kl_distill_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> tuple:
        """Decoupled KD (Zhao et al., CVPR 2022) adapted for contrastive learning.

        Splits KL divergence into two independent components:
          - TCKD (Target Class KD): KL contribution from diagonal entries (matched
            pairs). Always kept positive to preserve alignment on correct pairs.
          - NCKD (Non-Target Class KD): KL contribution from off-diagonal entries
            (all negative pairs). This component goes repulsive when distill_weight
            is negative, pushing the student to develop its own ranking structure
            for hard negatives.

        In classification DKD, "target class" = ground truth label.
        In our contrastive setting, "target" = diagonal of the N×N similarity matrix
        (the matched image-text pair for each row).

        Handles multi-GPU: logits are [B_local, N_gathered] where N_gathered = B_local * world_size.
        The "diagonal" for rank r starts at column r * B_local.

        Returns:
            (tckd_loss, nckd_loss) — both scalars, to be weighted independently.
        """
        B = teacher_logits.size(0)   # local batch size
        N = teacher_logits.size(1)   # gathered total (B * world_size)
        teacher_probs = teacher_logits.softmax(dim=1)
        student_log_probs = student_logits.log_softmax(dim=1)

        # Compute rank offset for the diagonal in multi-GPU setting
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank_offset = B * dist.get_rank()
        else:
            rank_offset = 0

        # Row indices and corresponding column indices for matched pairs
        row_idx = torch.arange(B, device=teacher_logits.device)
        col_idx = row_idx + rank_offset

        # TCKD: KL contribution from diagonal (matched pairs only)
        tckd = -(teacher_probs[row_idx, col_idx] * student_log_probs[row_idx, col_idx]).mean()

        # NCKD: KL contribution from off-diagonal (all negative pairs)
        # Build mask: True for all positions except the diagonal
        diag_mask = torch.zeros(B, N, dtype=torch.bool, device=teacher_logits.device)
        diag_mask[row_idx, col_idx] = True
        off_diag_mask = ~diag_mask
        nckd = -(teacher_probs[off_diag_mask] * student_log_probs[off_diag_mask]).sum() / B

        return tckd, nckd

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        teacher_image_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        teacher_logit_scale: torch.Tensor,
        proj_image_features: torch.Tensor = None,
        proj_text_features: torch.Tensor = None,
    ) -> dict:
        """Compute combined CLIP + KL distillation loss (+ Optional Feature KD).

        Args:
            image_features: Student image embeddings [B, D_s], L2-normalized.
            text_features: Student text embeddings [B, D_s], L2-normalized.
            logit_scale: Student's learnable temperature parameter.
            teacher_image_features: Teacher image embeddings [B, D_t], L2-normalized.
            teacher_text_features: Teacher text embeddings [B, D_t], L2-normalized.
            teacher_logit_scale: Teacher's temperature parameter.
            proj_image_features: Projected student image embeddings [B, D_t], L2-normalized.
            proj_text_features: Projected student text embeddings [B, D_t], L2-normalized.

        Returns:
            Dict with 'loss' (total), 'clip_loss', 'distill_loss', 'feature_kd_loss',
            and optionally 'tckd_loss'/'nckd_loss' when decoupled_kd is enabled.
        """
        # Standard CLIP contrastive loss
        clip_loss = self.clip_loss(image_features, text_features, logit_scale)

        # Gather all features for the N×N similarity matrices
        all_student_image, all_student_text = self._gather_features(
            image_features, text_features
        )
        all_teacher_image, all_teacher_text = self._gather_features(
            teacher_image_features, teacher_text_features
        )

        # Student logits (N×N similarity matrix)
        student_logits_i2t = logit_scale * image_features @ all_student_text.T
        student_logits_t2i = logit_scale * text_features @ all_student_image.T

        # Teacher logits (N×N similarity matrix) — no gradient
        # Divide by temperature to soften the teacher's distribution,
        # preventing near-one-hot outputs when teacher logit_scale is very high.
        with torch.no_grad():
            teacher_logits_i2t = (teacher_logit_scale * teacher_image_features @ all_teacher_text.T) / self.distill_temperature
            teacher_logits_t2i = (teacher_logit_scale * teacher_text_features @ all_teacher_image.T) / self.distill_temperature

        # Optional logit standardization: z-score per row before softmax.
        # Removes magnitude mismatch between teacher (logit_scale~100) and
        # student (~65), focusing KD on ranking structure. Applied to both
        # teacher and student logits symmetrically.
        if self.logit_standardization:
            teacher_logits_i2t = self._standardize_logits(teacher_logits_i2t)
            teacher_logits_t2i = self._standardize_logits(teacher_logits_t2i)
            student_logits_i2t = self._standardize_logits(student_logits_i2t)
            student_logits_t2i = self._standardize_logits(student_logits_t2i)

        # KL distillation: standard coupled or DKD decoupled
        tckd_loss_val = None
        nckd_loss_val = None
        if self.decoupled_kd:
            # DKD: split KL into diagonal (TCKD) + off-diagonal (NCKD).
            # TCKD is always positive (preserves alignment on matched pairs).
            # NCKD is weighted by distill_weight (goes repulsive when negative).
            tckd_i2t, nckd_i2t = self._decoupled_kl_distill_loss(teacher_logits_i2t, student_logits_i2t)
            tckd_t2i, nckd_t2i = self._decoupled_kl_distill_loss(teacher_logits_t2i, student_logits_t2i)
            tckd_loss = (tckd_i2t + tckd_t2i) / 2.0
            nckd_loss = (nckd_i2t + nckd_t2i) / 2.0
            # distill_weight only controls NCKD; TCKD stays at weight=1.0
            distill_loss = tckd_loss + self.distill_weight * nckd_loss
            tckd_loss_val = tckd_loss.detach()
            nckd_loss_val = nckd_loss.detach()
        else:
            # Standard coupled KL (original behavior, no change)
            distill_loss = (
                self._kl_distill_loss(teacher_logits_i2t, student_logits_i2t, self.confidence_penalty)
                + self._kl_distill_loss(teacher_logits_t2i, student_logits_t2i, self.confidence_penalty)
            ) / 2.0

        # Feature-Space KD — matching CLIP-KD (Yang et al., CVPR 2024).
        # Both proj_*_features (from ProjectionHead) and teacher_*_features
        # are L2-normalized. MSE between unit vectors is O(1/d) ≈ 0.002,
        # so feature_kd_weight should be large (e.g. 2000) to compensate.
        feature_kd_loss = torch.tensor(0.0, device=image_features.device)
        if self.feature_kd_weight > 0.0 and proj_image_features is not None and proj_text_features is not None:
            if self.feature_kd_type == "cosine":
                img_kd = 1.0 - (proj_image_features * teacher_image_features.detach()).sum(dim=-1).mean()
                txt_kd = 1.0 - (proj_text_features * teacher_text_features.detach()).sum(dim=-1).mean()
            else:
                img_kd = F.mse_loss(proj_image_features, teacher_image_features.detach())
                txt_kd = F.mse_loss(proj_text_features, teacher_text_features.detach())
            feature_kd_loss = img_kd + txt_kd

        # Total loss computation depends on whether DKD is active.
        # With DKD: distill_loss already incorporates the weight on NCKD only,
        # so we don't multiply by distill_weight again.
        # Without DKD: standard weighted sum as before.
        if self.decoupled_kd:
            total_loss = (
                self.contrastive_weight * clip_loss
                + distill_loss  # already weighted internally (TCKD + distill_weight * NCKD)
                + self.feature_kd_weight * feature_kd_loss
            )
        else:
            total_loss = (
                self.contrastive_weight * clip_loss
                + self.distill_weight * distill_loss
                + self.feature_kd_weight * feature_kd_loss
            )

        result = {
            "loss": total_loss,
            "clip_loss": clip_loss.detach(),
            "distill_loss": distill_loss.detach(),
            "feat_kd_loss": feature_kd_loss.detach() if self.feature_kd_weight > 0.0 else None,
        }
        if tckd_loss_val is not None:
            result["tckd_loss"] = tckd_loss_val
            result["nckd_loss"] = nckd_loss_val
        return result


class SigLIPLoss(nn.Module):
    """Sigmoid binary cross-entropy contrastive loss (SigLIP).

    Replace the softmax-based CLIP loss with a per-pair sigmoid binary loss.
    Each (image, text) pair is judged independently — no row-wise softmax
    normalization — making the loss much less sensitive to temperature and
    batch composition.

    Reference:
        Zhai et al., "Sigmoid Loss for Language Image Pre-Training", ICCV 2023.
        https://arxiv.org/abs/2303.15343

    Formula (per pair):
        y_{ij} = +1  if i == j (positive pair)
        y_{ij} = -1  if i != j (negative pair)
        loss = -mean_{i,j}[ log σ(y_{ij} · s_{ij}) ]
             = -mean_{i,j}[ log_sigmoid(y_{ij} · logit_scale · img_i · txt_j) ]

    This is equivalent to binary cross-entropy on each pair independently.
    """

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SigLIP contrastive loss.

        Args:
            image_features: Normalized image embeddings [B, D].
            text_features:  Normalized text embeddings  [B, D].
            logit_scale:    Learnable temperature (scalar or [1]).

        Returns:
            Scalar loss value.
        """
        # [B, B] similarity matrix
        logits = logit_scale * image_features @ text_features.T

        # Labels: +1 on diagonal (positive pairs), -1 off-diagonal (negatives)
        n = logits.shape[0]
        labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0

        # -log σ(y · s)  =  log(1 + exp(-y · s))  [numerically stable via logsigmoid]
        loss = -F.logsigmoid(labels * logits).mean()
        return loss


class DistillSigLIPLoss(nn.Module):
    """SigLIP contrastive loss + KL-divergence knowledge distillation.

    Identical to DistillCLIPLoss except the contrastive term uses SigLIP
    instead of softmax cross-entropy.  The KL distillation term is unchanged:
    it operates on the raw N×N logit matrices and is independent of the
    contrastive objective.

    Args:
        contrastive_weight:   Weight for the SigLIP contrastive loss.
        distill_weight:       Weight for the KL distillation loss.
        distill_temperature:  Softens teacher logits before KL target.
        gather_with_grad:     Whether to all-gather with gradients (distributed).
        local_loss:           Use local loss in distributed setting.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        distill_weight: float = 1.0,
        distill_temperature: float = 1.0,
        gather_with_grad: bool = True,
        local_loss: bool = False,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature
        self.siglip_loss = SigLIPLoss()
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss

    def _gather_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """Gather features from all processes for distributed training."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return image_features, text_features

        if self.gather_with_grad:
            all_image = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image, image_features)
            dist.all_gather(gathered_text, text_features)
            if not self.local_loss:
                gathered_image[rank] = image_features
                gathered_text[rank] = text_features
            all_image = torch.cat(gathered_image, dim=0)
            all_text = torch.cat(gathered_text, dim=0)

        return all_image, all_text

    @staticmethod
    def _kl_distill_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence: -(softmax(teacher) * log_softmax(student)).sum(dim=1).mean()"""
        return -(
            teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)
        ).sum(dim=1).mean(dim=0)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        teacher_image_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        teacher_logit_scale: torch.Tensor,
    ) -> dict:
        """Compute combined SigLIP + KL distillation loss.

        Args:
            image_features:         Student image embeddings [B, D_s], normalized.
            text_features:          Student text embeddings  [B, D_s], normalized.
            logit_scale:            Student temperature.
            teacher_image_features: Teacher image embeddings [B, D_t], normalized.
            teacher_text_features:  Teacher text embeddings  [B, D_t], normalized.
            teacher_logit_scale:    Teacher temperature.

        Returns:
            Dict with 'loss' (total), 'siglip_loss', and 'distill_loss'.
        """
        # SigLIP contrastive loss
        siglip_loss = self.siglip_loss(image_features, text_features, logit_scale)

        # Gather all features for the N×N KD similarity matrices
        all_student_image, all_student_text = self._gather_features(
            image_features, text_features
        )
        all_teacher_image, all_teacher_text = self._gather_features(
            teacher_image_features, teacher_text_features
        )

        # Student logits (N×N)
        student_logits_i2t = logit_scale * image_features @ all_student_text.T
        student_logits_t2i = logit_scale * text_features @ all_student_image.T

        # Teacher logits — softened by distill_temperature, no gradient
        with torch.no_grad():
            teacher_logits_i2t = (
                teacher_logit_scale * teacher_image_features @ all_teacher_text.T
            ) / self.distill_temperature
            teacher_logits_t2i = (
                teacher_logit_scale * teacher_text_features @ all_teacher_image.T
            ) / self.distill_temperature

        # KL distillation (unchanged from DistillCLIPLoss)
        distill_loss = (
            self._kl_distill_loss(teacher_logits_i2t, student_logits_i2t)
            + self._kl_distill_loss(teacher_logits_t2i, student_logits_t2i)
        ) / 2.0

        total_loss = (
            self.contrastive_weight * siglip_loss
            + self.distill_weight * distill_loss
        )

        return {
            "loss": total_loss,
            "siglip_loss": siglip_loss.detach(),
            "distill_loss": distill_loss.detach(),
        }
