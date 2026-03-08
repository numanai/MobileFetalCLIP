from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from mobile_fetal_clip import cli


def test_reproduce_passes_dry_run_to_subcommands(monkeypatch) -> None:
    root = Path("/tmp/mobile_fetal_clip")
    commands: list[tuple[list[str], bool]] = []

    monkeypatch.setattr(cli, "_repo_root", lambda: root)
    monkeypatch.setattr(cli, "suite_experiment_ids", lambda suite: ["no-kd", "static-kd"])

    def fake_run_command(cmd: list[str], dry_run: bool) -> int:
        commands.append((cmd, dry_run))
        return 0

    monkeypatch.setattr(cli, "_run_command", fake_run_command)

    args = Namespace(
        suite="main",
        base_config="configs/default.yaml",
        model_config="configs/model/mobileclip2_s0_fetal.json",
        pretrained="checkpoints/MobileCLIP2-S0/mobileclip2_s0.pt",
        teacher="checkpoints/FetalCLIP_weights.pt",
        train_data=None,
        output_root="outputs/repro",
        project_name="mobile_fetal_clip",
        master_csv="outputs/experiments_master.csv",
        seed=42,
        dry_run=True,
    )

    assert cli.cmd_reproduce(args) == 0
    assert len(commands) == 2
    assert all(dry_run is True for _, dry_run in commands)
    assert all("--dry-run" in cmd for cmd, _ in commands)
