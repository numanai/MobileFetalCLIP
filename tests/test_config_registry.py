from pathlib import Path

from mobile_fetal_clip.experiment_registry import (
    ABLATION_EXPERIMENT_IDS,
    MAIN_EXPERIMENT_IDS,
    list_experiment_ids,
    load_experiment,
)


def test_registry_contains_main_and_ablation_ids() -> None:
    available = set(list_experiment_ids())
    for exp_id in MAIN_EXPERIMENT_IDS + ABLATION_EXPERIMENT_IDS:
        assert exp_id in available


def test_load_experiment_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = load_experiment(root / "configs" / "experiments" / "static-kd.yaml")
    assert exp.id == "static-kd"
    assert exp.teacher_required is True
    assert "distill_weight" in exp.args
