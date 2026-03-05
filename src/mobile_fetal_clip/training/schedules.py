"""Small schedule helpers used by training and tests."""

from __future__ import annotations


def linear_weight_decay(
    initial_weight: float,
    min_ratio: float,
    step: int,
    max_steps: int,
    clamp: bool = True,
) -> float:
    """Linearly decay a weight from `initial_weight` to `initial_weight * min_ratio`.

    When `clamp=True`, the interpolation fraction is clipped to [0, 1].
    This corresponds to the corrected schedule behavior used in core runs.
    """
    if clamp:
        if step <= 0:
            return float(initial_weight)
        if step >= max_steps:
            return float(initial_weight) * float(min_ratio)

    denom = max(int(max_steps), 1)
    frac = float(step) / float(denom)
    if clamp:
        frac = min(1.0, max(0.0, frac))
    return float(initial_weight) * (1.0 - frac * (1.0 - float(min_ratio)))
