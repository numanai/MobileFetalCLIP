#!/usr/bin/env python3
"""Benchmark inference latency and throughput for MobileFetalCLIP models."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT / "src"))

from mobile_fetal_clip.models.factory import create_fetal_clip_model, get_tokenizer, load_pretrained_weights

SCHEMA_VERSION = "1.0.0"
DEFAULT_PROMPTS = (
    PROJECT_ROOT / "src" / "mobile_fetal_clip" / "evaluation" / "prompts" / "five_planes_prompts.json"
)

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "mobileclip2_s0": {
        "label": "MobileCLIP2-S0",
        "kind": "mobile",
        "model_config": "configs/model/mobileclip2_s0_fetal.json",
        "input_size": 256,
    },
    "fetalclip_teacher_vitl14": {
        "label": "FetalCLIP (ViT-L/14)",
        "kind": "open_clip",
        "open_clip_model": "ViT-L-14",
        "open_clip_pretrained": None,
        "input_size": 224,
    },
    "clip_vitl14_openai": {
        "label": "CLIP (ViT-L/14)",
        "kind": "open_clip",
        "open_clip_model": "ViT-L-14",
        "open_clip_pretrained": "openai",
        "input_size": 224,
    },
    "biomedclip_vitb16": {
        "label": "BiomedCLIP (ViT-B/16)",
        "kind": "open_clip",
        "open_clip_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "open_clip_pretrained": None,
        "input_size": 224,
    },
    "unimedclip_vitb16": {
        "label": "UniMed-CLIP (ViT-B/16)",
        "kind": "open_clip",
        "open_clip_model": "hf-hub:WangLab/UniMed-CLIP-ViT-B-16",
        "open_clip_pretrained": None,
        "input_size": 224,
    },
}


@dataclass
class BenchCfg:
    model_id: str
    model_label: str
    scope: str
    device: str
    device_id: str
    precision: str
    batch_size: int
    input_size: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark model inference")
    p.add_argument("--model-id", required=True, help="Preset id, comma-separated ids, or 'all'.")
    p.add_argument("--list-models", action="store_true")

    p.add_argument("--model-config", default=None)
    p.add_argument("--base-ckpt", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--open-clip-model", default=None)
    p.add_argument("--open-clip-pretrained", default=None)

    p.add_argument("--device", default="cpu")
    p.add_argument("--device-id", default=None)
    p.add_argument("--scope", default="both", choices=["encoder", "e2e", "both"])
    p.add_argument("--batch-sizes", default="1,16")
    p.add_argument("--precision", default="fp32", choices=["fp16", "fp32", "bf16"])
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompts-json", default=str(DEFAULT_PROMPTS))
    p.add_argument("--notes", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_ids(model_id_arg: str) -> list[str]:
    if model_id_arg == "all":
        return list(MODEL_PRESETS.keys())
    ids = [x.strip() for x in model_id_arg.split(",") if x.strip()]
    unknown = [x for x in ids if x not in MODEL_PRESETS]
    if unknown:
        raise ValueError(f"Unknown model id(s): {unknown}")
    return ids


def _resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _load_mobile_model(
    model_config: str,
    base_ckpt: str | None,
    finetuned_ckpt: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, int]:
    model, _, preprocess = create_fetal_clip_model(
        model_config_path=model_config,
        pretrained=None,
        precision="fp32",
        device="cpu",
    )
    tokenizer = get_tokenizer(model_config)

    if base_ckpt and Path(base_ckpt).exists():
        load_pretrained_weights(model, base_ckpt, strict=False)

    if finetuned_ckpt and Path(finetuned_ckpt).exists():
        ckpt = torch.load(finetuned_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)

    model.eval().to(device)
    input_size = 224
    try:
        cfg = json.loads(Path(model_config).read_text())
        input_size = int(cfg.get("vision_cfg", {}).get("image_size", 224))
    except Exception:
        pass
    return model, tokenizer, input_size


def _load_open_clip_model(
    model_name: str,
    pretrained: str | None,
    base_ckpt: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, int]:
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp32",
        device=str(device),
        output_dict=True,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    if base_ckpt and Path(base_ckpt).exists():
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)

    model.eval().to(device)
    input_size = 224
    return model, tokenizer, input_size


def _precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.no_grad()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def _encode_text_cache(model, tokenizer, prompts_json: Path, device: torch.device) -> torch.Tensor:
    prompts = json.loads(prompts_json.read_text())
    class_names = list(prompts.keys())
    with torch.no_grad():
        text_vectors = []
        for class_name in class_names:
            tokens = tokenizer(prompts[class_name]).to(device)
            t = model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(dim=0, keepdim=True)
            t = t / t.norm(dim=-1, keepdim=True)
            text_vectors.append(t)
    return torch.cat(text_vectors, dim=0)


def _run_encoder_bench(
    model,
    device: torch.device,
    precision: str,
    batch_size: int,
    input_size: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[float, float, float, float, float]:
    samples = []
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(repeats):
        for _ in range(warmup):
            with _precision_context(device, precision):
                _ = model.encode_image(x)
        _device_sync(device)

        per_iter = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with _precision_context(device, precision):
                _ = model.encode_image(x)
            _device_sync(device)
            per_iter.append((time.perf_counter() - t0) * 1000.0)
        samples.extend(per_iter)

    mean_ms = float(statistics.mean(samples))
    std_ms = float(statistics.pstdev(samples))
    p50_ms = float(np.percentile(samples, 50))
    p90_ms = float(np.percentile(samples, 90))
    throughput = float(batch_size / (mean_ms / 1000.0))
    return mean_ms, std_ms, p50_ms, p90_ms, throughput


def _run_e2e_bench(
    model,
    text_cache: torch.Tensor,
    device: torch.device,
    precision: str,
    batch_size: int,
    input_size: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[float, float, float, float, float]:
    samples = []

    for _ in range(repeats):
        for _ in range(warmup):
            x = torch.randn(batch_size, 3, input_size, input_size, device=device)
            with _precision_context(device, precision):
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                logits = 100.0 * img @ text_cache.T
                _ = torch.argmax(logits, dim=-1)
        _device_sync(device)

        per_iter = []
        for _ in range(iters):
            x = torch.randn(batch_size, 3, input_size, input_size, device=device)
            t0 = time.perf_counter()
            with _precision_context(device, precision):
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                logits = 100.0 * img @ text_cache.T
                _ = torch.argmax(logits, dim=-1)
            _device_sync(device)
            per_iter.append((time.perf_counter() - t0) * 1000.0)
        samples.extend(per_iter)

    mean_ms = float(statistics.mean(samples))
    std_ms = float(statistics.pstdev(samples))
    p50_ms = float(np.percentile(samples, 50))
    p90_ms = float(np.percentile(samples, 90))
    throughput = float(batch_size / (mean_ms / 1000.0))
    return mean_ms, std_ms, p50_ms, p90_ms, throughput


def main() -> int:
    args = parse_args()

    if args.list_models:
        for model_id, cfg in MODEL_PRESETS.items():
            print(f"{model_id}: {cfg['label']}")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model_ids = _parse_ids(args.model_id)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    results: list[dict[str, Any]] = []

    for model_id in model_ids:
        preset = MODEL_PRESETS[model_id]
        kind = preset["kind"]

        if kind == "mobile":
            model_config = _resolve_path(args.model_config or preset["model_config"])
            base_ckpt = _resolve_path(args.base_ckpt)
            finetuned_ckpt = _resolve_path(args.finetuned_ckpt)
            model, tokenizer, model_input_size = _load_mobile_model(
                model_config=model_config,
                base_ckpt=base_ckpt,
                finetuned_ckpt=finetuned_ckpt,
                device=device,
            )
        else:
            model_name = args.open_clip_model or preset["open_clip_model"]
            pretrained = args.open_clip_pretrained or preset.get("open_clip_pretrained")
            base_ckpt = _resolve_path(args.base_ckpt)
            model, tokenizer, model_input_size = _load_open_clip_model(
                model_name=model_name,
                pretrained=pretrained,
                base_ckpt=base_ckpt,
                device=device,
            )

        input_size = int(args.input_size or preset.get("input_size", model_input_size))
        text_cache = _encode_text_cache(model, tokenizer, Path(args.prompts_json), device)

        scopes = ["encoder", "e2e"] if args.scope == "both" else [args.scope]
        for scope in scopes:
            for batch_size in batch_sizes:
                cfg = BenchCfg(
                    model_id=model_id,
                    model_label=preset["label"],
                    scope=scope,
                    device=device.type,
                    device_id=args.device_id or device.type,
                    precision=args.precision,
                    batch_size=batch_size,
                    input_size=input_size,
                )

                if scope == "encoder":
                    mean_ms, std_ms, p50_ms, p90_ms, throughput = _run_encoder_bench(
                        model,
                        device,
                        args.precision,
                        batch_size,
                        input_size,
                        args.warmup,
                        args.iters,
                        args.repeats,
                    )
                else:
                    mean_ms, std_ms, p50_ms, p90_ms, throughput = _run_e2e_bench(
                        model,
                        text_cache,
                        device,
                        args.precision,
                        batch_size,
                        input_size,
                        args.warmup,
                        args.iters,
                        args.repeats,
                    )

                results.append(
                    {
                        "model_id": cfg.model_id,
                        "model_label": cfg.model_label,
                        "device_id": cfg.device_id,
                        "device": cfg.device,
                        "backend": "torch",
                        "scope": cfg.scope,
                        "batch_size": cfg.batch_size,
                        "precision": cfg.precision,
                        "input_size": cfg.input_size,
                        "latency_mean_ms": mean_ms,
                        "latency_std_ms": std_ms,
                        "latency_p50_ms": p50_ms,
                        "latency_p90_ms": p90_ms,
                        "throughput_img_s": throughput,
                        "peak_memory_mb": None,
                        "params_total": int(sum(p.numel() for p in model.parameters())),
                        "run_timestamp": datetime.now(timezone.utc).isoformat(),
                        "notes": args.notes,
                    }
                )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "device_id": args.device_id or args.device,
            "scope": args.scope,
            "precision": args.precision,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "batch_sizes": args.batch_sizes,
        },
        "schema_version": SCHEMA_VERSION,
        "results": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_json}")

    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    fieldnames = list(results[0].keys()) if results else []
    if fieldnames:
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
