from mobile_fetal_clip.training.schedules import linear_weight_decay


def test_linear_weight_decay_clamped_endpoints() -> None:
    assert linear_weight_decay(1.0, 0.1, step=0, max_steps=100, clamp=True) == 1.0
    assert linear_weight_decay(1.0, 0.1, step=100, max_steps=100, clamp=True) == 0.1
    assert linear_weight_decay(1.0, 0.1, step=200, max_steps=100, clamp=True) == 0.1


def test_linear_weight_decay_unclamped_overshoot() -> None:
    # Legacy-style overshoot: frac=2.0 -> 1 - 2*(1-0.1) = -0.8
    val = linear_weight_decay(1.0, 0.1, step=200, max_steps=100, clamp=False)
    assert abs(val - (-0.8)) < 1e-9
