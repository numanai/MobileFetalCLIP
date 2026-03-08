#!/usr/bin/env python3
"""Create the dataset layout expected by MobileFetalCLIP."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import braceexpand


def _link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset layout")
    parser.add_argument("--dest-root", default="data")
    parser.add_argument(
        "--source-train-shards",
        required=True,
        help="Brace-expandable source shard pattern.",
    )
    parser.add_argument("--source-fetal-planes-images", required=True)
    parser.add_argument("--source-fetal-planes-csv", required=True)
    parser.add_argument("--source-hc18-images", required=True)
    parser.add_argument("--source-hc18-csv", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symlink = not args.copy

    dest_root = Path(args.dest_root)
    train_dest = dest_root / "pretraining" / "shards"
    eval_fp_dest = dest_root / "eval" / "FETAL_PLANES_DB"
    eval_hc18_dest = dest_root / "eval" / "HC18"

    train_dest.mkdir(parents=True, exist_ok=True)
    eval_fp_dest.mkdir(parents=True, exist_ok=True)
    eval_hc18_dest.mkdir(parents=True, exist_ok=True)

    shard_paths = [Path(p) for p in braceexpand.braceexpand(args.source_train_shards)]
    existing = [p for p in shard_paths if p.exists()]
    if not existing:
        raise FileNotFoundError("No source training shards found.")

    for src in existing:
        dst = train_dest / src.name
        _link_or_copy(src, dst, symlink=symlink)

    _link_or_copy(Path(args.source_fetal_planes_csv), eval_fp_dest / "FETAL_PLANES_DB_data.csv", symlink)
    _link_or_copy(Path(args.source_hc18_csv), eval_hc18_dest / "training_set_pixel_size_and_HC.csv", symlink)

    # For image folders, link/copy directory root.
    fp_images_dst = eval_fp_dest / "Images"
    hc_images_dst = eval_hc18_dest / "training_set"

    for src_dir, dst_dir in [
        (Path(args.source_fetal_planes_images), fp_images_dst),
        (Path(args.source_hc18_images), hc_images_dst),
    ]:
        if dst_dir.exists() or dst_dir.is_symlink():
            if dst_dir.is_symlink() or dst_dir.is_file():
                dst_dir.unlink()
            else:
                shutil.rmtree(dst_dir)

        if symlink:
            dst_dir.symlink_to(src_dir.resolve(), target_is_directory=True)
        else:
            shutil.copytree(src_dir, dst_dir)

    print("Prepared dataset layout at:", dest_root)
    print("- Training shards:", train_dest)
    print("- FETAL_PLANES_DB:", eval_fp_dest)
    print("- HC18:", eval_hc18_dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
