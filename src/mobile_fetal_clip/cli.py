"""Top-level CLI for MobileFetalCLIP workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mobile_fetal_clip.experiment_registry import load_experiment, suite_experiment_ids


_BOOL_FLAGS = {
    "distill_weight_decay": "--distill-weight-decay",
    "decay_feature_kd": "--decay-feature-kd",
    "legacy_decay": "--legacy-decay",
    "decoupled_kd": "--decoupled-kd",
    "logit_standardization": "--logit-standardization",
}

_VALUE_FLAGS = {
    "lr": "--lr",
    "epochs": "--epochs",
    "distill_weight": "--distill-weight",
    "distill_temperature": "--distill-temperature",
    "feature_kd_weight": "--feature-kd-weight",
    "feature_kd_type": "--feature-kd-type",
    "distill_weight_min_ratio": "--distill-weight-min-ratio",
    "confidence_penalty": "--confidence-penalty",
    "loss_type": "--loss-type",
    "batch_size": "--batch-size",
    "gradient_accumulation_steps": "--gradient-accumulation-steps",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_remainder_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _append_experiment_args(cmd: list[str], exp_args: dict[str, object]) -> None:
    for key, value in exp_args.items():
        if key in _BOOL_FLAGS:
            if bool(value):
                cmd.append(_BOOL_FLAGS[key])
            continue

        if key in _VALUE_FLAGS and value is not None:
            cmd.extend([_VALUE_FLAGS[key], str(value)])
            continue

        raise ValueError(f"Unsupported experiment argument: {key}")


def _run_command(cmd: list[str], dry_run: bool) -> int:
    rendered = " ".join(cmd)
    print(rendered)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def cmd_train(args: argparse.Namespace) -> int:
    exp = load_experiment(args.experiment_config)
    if exp.teacher_required and not args.teacher:
        raise ValueError(
            f"Experiment '{exp.id}' requires --teacher checkpoint, but none was provided."
        )

    cmd = [
        sys.executable,
        "-m",
        "mobile_fetal_clip.main",
        "--config",
        str(args.base_config),
        "--model-config",
        str(args.model_config),
        "--output-dir",
        str(args.output_dir),
        "--exp-id",
        exp.id,
        "--run-name",
        args.run_name or exp.id,
        "--project-name",
        args.project_name,
        "--seed",
        str(args.seed),
        "--master-csv",
        str(args.master_csv),
    ]

    if args.pretrained:
        cmd.extend(["--pretrained", str(args.pretrained)])
    if args.teacher:
        cmd.extend(["--teacher", str(args.teacher)])
    if args.train_data:
        cmd.extend(["--train-data", str(args.train_data)])

    _append_experiment_args(cmd, exp.args)
    return _run_command(cmd, args.dry_run)


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from omegaconf import OmegaConf

    from mobile_fetal_clip.evaluation.zero_shot import zero_shot_eval
    from mobile_fetal_clip.models.factory import (
        create_fetal_clip_model,
        get_tokenizer,
        load_pretrained_weights,
    )
    SCHEMA_VERSION = "1.0.0"

    cfg = OmegaConf.to_container(OmegaConf.load(args.eval_config), resolve=True)
    eval_cfg = dict(cfg.get("evaluation", {}))

    model, _, preprocess_val = create_fetal_clip_model(
        model_config_path=str(args.model_config),
        pretrained=None,
        precision="fp32",
        device="cpu",
    )
    tokenizer = get_tokenizer(str(args.model_config))

    if args.base_checkpoint:
        load_pretrained_weights(model, str(args.base_checkpoint), strict=False)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)

    model = model.to(args.device)
    model.eval()

    metrics = zero_shot_eval(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess_val,
        device=torch.device(args.device),
        eval_cfg=eval_cfg,
        epoch=0,
    )

    five_f1 = float(metrics.get("five_planes/f1", 0.0))
    brain_f1 = float(metrics.get("brain_subplanes/f1", 0.0))
    hc18_validity = float(metrics.get("hc18/validity_rate", 0.0))
    classification_f1 = (5.0 * five_f1 + 3.0 * brain_f1) / 8.0
    composite = (classification_f1 + hc18_validity / 100.0) / 2.0

    payload = {
        "schema_version": SCHEMA_VERSION,
        "exp_id": args.exp_id,
        "composite_score": composite,
        "classification_f1": classification_f1,
        "hc18_validity_rate": hc18_validity,
        "five_planes_f1": five_f1,
        "brain_f1": brain_f1,
        "best_epoch": 0,
        "all_metrics": metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved evaluation results to {output_path}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    root = _repo_root()
    exp_dir = root / "configs" / "experiments"

    for exp_id in suite_experiment_ids(args.suite):
        exp_file = exp_dir / f"{exp_id}.yaml"
        out_dir = Path(args.output_root) / exp_id

        cmd = [
            sys.executable,
            "-m",
            "mobile_fetal_clip.cli",
            "train",
            "--base-config",
            str(args.base_config),
            "--experiment-config",
            str(exp_file),
            "--model-config",
            str(args.model_config),
            "--output-dir",
            str(out_dir),
            "--project-name",
            args.project_name,
            "--master-csv",
            str(args.master_csv),
            "--seed",
            str(args.seed),
        ]

        if args.pretrained:
            cmd.extend(["--pretrained", str(args.pretrained)])
        if args.teacher:
            cmd.extend(["--teacher", str(args.teacher)])
        if args.train_data:
            cmd.extend(["--train-data", str(args.train_data)])
        if args.dry_run:
            cmd.append("--dry-run")

        code = _run_command(cmd, dry_run=args.dry_run)
        if code != 0:
            return code

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    tool = _repo_root() / "tools" / "benchmark_inference.py"
    benchmark_args = _normalize_remainder_args(args.benchmark_args)
    cmd = [sys.executable, str(tool)] + benchmark_args
    return _run_command(cmd, dry_run=False)


def cmd_validate_data(args: argparse.Namespace) -> int:
    tool = _repo_root() / "tools" / "validate_dataset.py"
    validate_args = _normalize_remainder_args(args.validate_args)
    cmd = [sys.executable, str(tool)] + validate_args
    return _run_command(cmd, dry_run=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileFetalCLIP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Run one experiment")
    p_train.add_argument("--base-config", default="configs/default.yaml")
    p_train.add_argument("--experiment-config", required=True)
    p_train.add_argument("--model-config", required=True)
    p_train.add_argument("--pretrained", default=None)
    p_train.add_argument("--teacher", default=None)
    p_train.add_argument("--train-data", default=None)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--project-name", default="mobile_fetal_clip")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--master-csv", default="outputs/experiments_master.csv")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("eval", help="Run zero-shot evaluation")
    p_eval.add_argument("--model-config", required=True)
    p_eval.add_argument("--base-checkpoint", default=None)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--eval-config", default="configs/default.yaml")
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--exp-id", default="eval-only")
    p_eval.add_argument("--output-json", required=True)
    p_eval.set_defaults(func=cmd_eval)

    p_repro = subparsers.add_parser("reproduce", help="Run an experiment suite")
    p_repro.add_argument("--suite", choices=["main", "ablation", "all"], default="main")
    p_repro.add_argument("--base-config", default="configs/default.yaml")
    p_repro.add_argument("--model-config", required=True)
    p_repro.add_argument("--pretrained", default=None)
    p_repro.add_argument("--teacher", default=None)
    p_repro.add_argument("--train-data", default=None)
    p_repro.add_argument("--output-root", default="outputs/repro")
    p_repro.add_argument("--project-name", default="mobile_fetal_clip")
    p_repro.add_argument("--master-csv", default="outputs/experiments_master.csv")
    p_repro.add_argument("--seed", type=int, default=42)
    p_repro.add_argument("--dry-run", action="store_true")
    p_repro.set_defaults(func=cmd_reproduce)

    p_bench = subparsers.add_parser("benchmark", help="Run benchmark tool")
    p_bench.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    p_bench.set_defaults(func=cmd_benchmark)

    p_validate = subparsers.add_parser(
        "validate-data", help="Validate a dataset layout"
    )
    p_validate.add_argument("validate_args", nargs=argparse.REMAINDER)
    p_validate.set_defaults(func=cmd_validate_data)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
