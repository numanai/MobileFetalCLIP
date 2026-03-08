"""Experiment definitions for MobileFetalCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExperimentConfig:
    """In-memory experiment definition loaded from YAML."""

    id: str
    description: str
    teacher_required: bool
    args: dict[str, Any]
    path: Path


MAIN_EXPERIMENT_IDS = [
    "no-kd",
    "static-kd",
    "repulsive-r-neg-0p8",
    "selective-beta2-to-neg-0p8",
]

ABLATION_EXPERIMENT_IDS = [
    "no-kd",
    "static-kd",
    "confidence-penalty",
    "repulsive-r-neg-0p5",
    "repulsive-r-neg-0p8",
    "coupled-beta2-to-neg-0p8",
    "selective-beta1-to-neg-0p8",
    "selective-beta2-to-neg-0p8",
    "selective-beta4-to-neg-0p8",
    "selective-beta8-to-neg-0p8",
]


def _default_experiments_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "experiments"


def list_experiment_ids(experiments_dir: Path | None = None) -> list[str]:
    """List available experiment IDs by reading YAML filenames."""
    root = experiments_dir or _default_experiments_dir()
    return sorted(p.stem for p in root.glob("*.yaml"))


def load_experiment(experiment_path: str | Path) -> ExperimentConfig:
    """Load one experiment config from YAML."""
    p = Path(experiment_path)
    cfg = OmegaConf.to_container(OmegaConf.load(p), resolve=True)

    return ExperimentConfig(
        id=str(cfg["id"]),
        description=str(cfg.get("description", "")),
        teacher_required=bool(cfg.get("teacher_required", True)),
        args=dict(cfg.get("args", {})),
        path=p,
    )


def suite_experiment_ids(suite: str) -> list[str]:
    """Return ordered experiment IDs for a suite."""
    if suite == "main":
        return MAIN_EXPERIMENT_IDS
    if suite == "ablation":
        return ABLATION_EXPERIMENT_IDS
    if suite == "all":
        extras = [
            exp_id
            for exp_id in list_experiment_ids()
            if exp_id not in ABLATION_EXPERIMENT_IDS
        ]
        return ABLATION_EXPERIMENT_IDS + sorted(extras)
    raise ValueError(f"Unknown suite: {suite}")
