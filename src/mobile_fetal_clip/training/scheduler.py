"""Learning rate schedulers for CLIP training."""
import math
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR. Default: 0.0.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
