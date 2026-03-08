#!/usr/bin/env python3
"""Validate the dataset layout expected by MobileFetalCLIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import braceexpand
import pandas as pd

REQUIRED_FETAL_PLANES_COLUMNS = {"Image_name", "Plane", "Brain_plane", "Train "}
REQUIRED_HC18_COLUMNS = {"filename", "pixel size(mm)", "head circumference (mm)"}


def _expand(pattern: str) -> list[Path]:
    return [Path(p) for p in braceexpand.braceexpand(pattern)]


def _check_exists(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")


def _validate_csv_columns(path: Path, required: set[str], label: str, errors: list[str]) -> None:
    if not path.exists():
        return
    try:
        df = pd.read_csv(path, sep=";" if "FETAL_PLANES_DB" in str(path) else ",")
    except Exception as exc:
        errors.append(f"Failed reading {label} CSV ({path}): {exc}")
        return

    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"{label} CSV missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset layout")
    parser.add_argument(
        "--train-shards",
        default="data/pretraining/shards/shard_{0000000001..0000000025}.tar",
        help="Brace-expandable shard pattern.",
    )
    parser.add_argument("--train-num-samples", type=int, default=246349)
    parser.add_argument("--fetal-planes-images", default="data/eval/FETAL_PLANES_DB/Images")
    parser.add_argument(
        "--fetal-planes-csv",
        default="data/eval/FETAL_PLANES_DB/FETAL_PLANES_DB_data.csv",
    )
    parser.add_argument("--hc18-images", default="data/eval/HC18/training_set")
    parser.add_argument(
        "--hc18-csv",
        default="data/eval/HC18/training_set_pixel_size_and_HC.csv",
    )
    parser.add_argument("--min-shards", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    shards = _expand(args.train_shards)
    if len(shards) < args.min_shards:
        errors.append(
            f"Shard expansion returned {len(shards)} files; expected at least {args.min_shards}."
        )

    existing_shards = [p for p in shards if p.exists()]
    if not existing_shards:
        errors.append("No training shards found from --train-shards pattern.")
    elif len(existing_shards) != len(shards):
        warnings.append(
            f"Only {len(existing_shards)}/{len(shards)} shards exist."
        )

    for shard in existing_shards[:3]:
        if shard.suffix != ".tar":
            warnings.append(f"Shard does not end with .tar: {shard}")

    fetal_images = Path(args.fetal_planes_images)
    fetal_csv = Path(args.fetal_planes_csv)
    hc18_images = Path(args.hc18_images)
    hc18_csv = Path(args.hc18_csv)

    _check_exists(fetal_images, "FETAL_PLANES_DB image directory", errors)
    _check_exists(fetal_csv, "FETAL_PLANES_DB CSV", errors)
    _check_exists(hc18_images, "HC18 image directory", errors)
    _check_exists(hc18_csv, "HC18 CSV", errors)

    _validate_csv_columns(
        fetal_csv,
        REQUIRED_FETAL_PLANES_COLUMNS,
        "FETAL_PLANES_DB",
        errors,
    )
    _validate_csv_columns(hc18_csv, REQUIRED_HC18_COLUMNS, "HC18", errors)

    if fetal_images.exists():
        png_count = len(list(fetal_images.glob("*.png")))
        if png_count == 0:
            errors.append("FETAL_PLANES_DB image directory contains no .png files.")
    if hc18_images.exists():
        img_count = len(list(hc18_images.glob("*.png"))) + len(list(hc18_images.glob("*.jpg")))
        if img_count == 0:
            warnings.append("HC18 image directory appears empty.")

    if args.strict and len(existing_shards) != len(shards):
        errors.append("Strict mode enabled and not all expanded shards exist.")

    print("Dataset check")
    print(f"- Expected samples: {args.train_num_samples}")
    print(f"- Found shard files: {len(existing_shards)}")

    for msg in warnings:
        print(f"WARNING: {msg}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}")
        return 1

    print("PASS: dataset layout looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
