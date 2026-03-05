"""Zero-shot evaluation for fetal ultrasound models.

Implements three benchmarks from FetalCLIP:
  1. Five-plane classification (FETAL_PLANES_DB)
  2. Brain subplane classification (FETAL_PLANES_DB brain subset)
  3. Gestational age estimation (HC18)

Metrics and methodology match the official FetalCLIP evaluation protocol.
"""

import json
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_image_square_with_zero_padding(image: Image.Image) -> Image.Image:
    """Pad a non-square image to square with black borders (centered)."""
    width, height = image.size
    max_side = max(width, height)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    padding_color = (0, 0, 0) if image.mode == "RGB" else 0
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))

    return new_image


def _size_to_int(size: Any) -> Optional[int]:
    """Best-effort conversion of torchvision size fields to scalar int."""
    if isinstance(size, int):
        return size
    if isinstance(size, float):
        return int(size)
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) == 0:
            return None
        return int(size[0])
    return None


def infer_eval_input_size(preprocess, fallback: int = 224) -> int:
    """Infer eval crop size from preprocess; fallback keeps old behavior."""
    transforms = getattr(preprocess, "transforms", None)
    if not transforms:
        return fallback

    center_crop_size = None
    resize_size = None
    for tr in transforms:
        size = _size_to_int(getattr(tr, "size", None))
        if size is None:
            continue
        name = tr.__class__.__name__.lower()
        if "centercrop" in name:
            center_crop_size = size
        elif "resize" in name:
            resize_size = size

    if center_crop_size is not None:
        return center_crop_size
    if resize_size is not None:
        return resize_size
    return fallback


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FetalPlanesDBDataset(Dataset):
    """FETAL_PLANES_DB dataset for zero-shot plane classification.

    Matches the official FetalCLIP ``DatasetFetalPlanesDB`` class.

    Args:
        dir_images: Path to the ``Images/`` directory.
        path_csv: Path to ``FETAL_PLANES_DB_data.csv``.
        preprocess: Image transform (from open_clip).
        split: ``'all'``, ``'train'``, or ``'test'``.
        exclude_planes: List of plane names to exclude (e.g. ``['Other']``).
    """

    # Maps FETAL_PLANES_DB plane names → prompt-key names
    PLANE_MAP = {
        "Fetal brain": "brain",
        "Fetal abdomen": "abdomen",
        "Fetal thorax": "heart",
        "Fetal femur": "femur",
        "Maternal cervix": "cervix",
    }

    def __init__(
        self,
        dir_images: str,
        path_csv: str,
        preprocess,
        split: str = "all",
        exclude_planes: Optional[List[str]] = None,
    ):
        import pandas as pd

        self.root = dir_images
        self.preprocess = preprocess

        df = pd.read_csv(path_csv, sep=";")
        if split == "train":
            df = df[df["Train "] == 1]
        elif split == "test":
            df = df[df["Train "] == 0]

        if exclude_planes:
            for plane in exclude_planes:
                df = df[df["Plane"] != plane]

        self.data = []
        for _, row in df.iterrows():
            label = self.PLANE_MAP.get(row["Plane"], row["Plane"])
            self.data.append({
                "img": os.path.join(self.root, f"{row['Image_name']}.png"),
                "label": label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img = Image.open(item["img"])
        img = make_image_square_with_zero_padding(img)
        img = self.preprocess(img)
        return img, item["label"]


class FetalPlanesDBBrainDataset(Dataset):
    """FETAL_PLANES_DB brain-subplane dataset for zero-shot classification.

    Matches the official FetalCLIP ``DatasetFetalPlanesDBBrain`` class.
    Filters: ``Brain_plane != 'Not A Brain'`` and ``Brain_plane != 'Other'``.

    Args:
        dir_images: Path to the ``Images/`` directory.
        path_csv: Path to ``FETAL_PLANES_DB_data.csv``.
        preprocess: Image transform (from open_clip).
        split: ``'all'``, ``'train'``, or ``'test'``.
    """

    BRAIN_SUBPLANE_MAP = {
        "Trans-thalamic": "trans-thalamic",
        "Trans-cerebellum": "trans-cerebellum",
        "Trans-ventricular": "trans-ventricular",
    }

    def __init__(
        self,
        dir_images: str,
        path_csv: str,
        preprocess,
        split: str = "all",
    ):
        import pandas as pd

        self.root = dir_images
        self.preprocess = preprocess

        df = pd.read_csv(path_csv, sep=";")
        df = df[df["Brain_plane"] != "Not A Brain"]
        df = df[df["Brain_plane"] != "Other"]

        if split == "train":
            df = df[df["Train "] == 1]
        elif split == "test":
            df = df[df["Train "] == 0]

        self.data = []
        for _, row in df.iterrows():
            label = self.BRAIN_SUBPLANE_MAP[row["Brain_plane"]]
            self.data.append({
                "img": os.path.join(self.root, f"{row['Image_name']}.png"),
                "label": label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img = Image.open(item["img"])
        img = make_image_square_with_zero_padding(img)
        img = self.preprocess(img)
        return img, item["label"]


class HC18Dataset(Dataset):
    """HC18 dataset for zero-shot gestational age estimation.

    Args:
        root_dir: Path to ``training_set/`` directory.
        path_csv: Path to ``training_set_pixel_size_and_HC.csv``.
        preprocess: Image transform.
        input_size: Target image size (for pixel spacing calculation).
    """

    def __init__(self, root_dir: str, path_csv: str, preprocess, input_size: int = 224):
        import pandas as pd

        self.root_dir = root_dir
        self.preprocess = preprocess
        self.input_size = max(1, int(input_size))

        df = pd.read_csv(path_csv)
        self.data = df.to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        imagepath = os.path.join(self.root_dir, item["filename"])
        hc = item["head circumference (mm)"]

        image = Image.open(imagepath)
        pixel_spacing = max(image.size) / self.input_size * item["pixel size(mm)"]

        image = make_image_square_with_zero_padding(image)
        if self.preprocess:
            image = self.preprocess(image)

        return image, pixel_spacing, hc


# ---------------------------------------------------------------------------
# Zero-shot classifier helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_text_features(
    model: torch.nn.Module,
    tokenizer,
    prompts_dict: Dict[str, List[str]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Build per-class text features from prompt templates.

    Matches FetalCLIP: for each class, encode all prompts, normalize each,
    average, then re-normalize to get a single [1, embed_dim] vector.

    Returns:
        (list_text_features, class_names) — list of [1, D] tensors + ordered names.
    """
    list_text_features = []
    class_names = list(prompts_dict.keys())

    with torch.cuda.amp.autocast():
        for cls_name, prompts in prompts_dict.items():
            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0).unsqueeze(0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            list_text_features.append(text_features)

    return list_text_features, class_names


@torch.no_grad()
def _run_classification(
    model: torch.nn.Module,
    list_text_features: List[torch.Tensor],
    class_names: List[str],
    dataloader: DataLoader,
    device: torch.device,
    report_topk: Tuple[int, ...] = (1,),
) -> Dict[str, float]:
    """Run zero-shot classification and compute accuracy/F1 metrics.

    Matches FetalCLIP: computes softmax probabilities, then uses
    ``torchmetrics.Accuracy`` (per-class, macro-averaged) and
    ``torchmetrics.F1Score`` (per-class, macro-averaged).
    """
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    all_probs = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            list_text_logits = []
            for text_features in list_text_features:
                text_logits = (100.0 * image_features @ text_features.T).mean(dim=-1)[:, None]
                list_text_logits.append(text_logits)
            text_probs = torch.cat(list_text_logits, dim=1).softmax(dim=-1)

        targets = torch.tensor([name_to_idx[l] for l in labels])
        all_probs.append(text_probs.cpu())
        all_targets.append(targets)

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets)
    num_classes = len(class_names)

    # Identify which class indices actually appear in targets
    list_target_idxs = sorted(targets.unique().tolist())

    metrics = {}

    # --- Accuracy & F1 at each requested top-k ---
    for k in report_topk:
        suffix = "" if k == 1 else f"_top{k}"

        acc_per_class = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=k, average="none",
        )(probs, targets)
        acc_vals = [acc_per_class[i].item() for i in list_target_idxs]
        metrics[f"acc{suffix}"] = float(np.mean(acc_vals))

        f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, top_k=k, average="none",
        )(probs, targets)
        f1_vals = [f1_per_class[i].item() for i in list_target_idxs]
        metrics[f"f1{suffix}"] = float(np.mean(f1_vals))

        # Log per-class breakdown to console only
        per_class_lines = []
        for idx in list_target_idxs:
            per_class_lines.append(
                f"    {class_names[idx]}: acc={acc_per_class[idx].item():.4f}, f1={f1_per_class[idx].item():.4f}"
            )
        logger.info(f"  Top-{k} per-class breakdown:\n" + "\n".join(per_class_lines))

    return metrics


# ---------------------------------------------------------------------------
# HC18 gestational age estimation (matches FetalCLIP exactly)
# ---------------------------------------------------------------------------

GA_TEMPLATES = [
    "Ultrasound image at {weeks} weeks and {day} days gestation focusing on the fetal brain, highlighting anatomical structures with a pixel spacing of {pixel_spacing} mm/pixel.",
    "Fetal ultrasound image at {weeks} weeks, {day} days of gestation, focusing on the developing brain, with a pixel spacing of {pixel_spacing} mm/pixel, highlighting the structures of the fetal brain.",
    "Fetal ultrasound image at {weeks} weeks and {day} days gestational age, highlighting the developing brain structures with a pixel spacing of {pixel_spacing} mm/pixel, providing important visual insights for ongoing prenatal assessments.",
    "Ultrasound image at {weeks} weeks and {day} days gestation, highlighting the fetal brain structures with a pixel spacing of {pixel_spacing} mm/pixel.",
    "Fetal ultrasound at {weeks} weeks and {day} days, showing a clear view of the developing brain, with an image pixel spacing of {pixel_spacing} mm/pixel.",
]

# GA range used for building the dot-product map (14–39 weeks)
GA_DAYS_MAP = [weeks * 7 + days for weeks in range(14, 39) for days in range(7)]
# GA range used for postprocessing / validity check (14–38 weeks, per FetalCLIP)
GA_DAYS_POSTPROCESS = [weeks * 7 + days for weeks in range(14, 38) for days in range(7)]

# HC filtering thresholds (from FetalCLIP: REF https://srhr.org/fetalgrowthcalculator)
DATA_MIN_HC = 100  # Corresponding to GA 14 weeks (50th percentile)
DATA_MAX_HC = 342  # Corresponding to GA 40 weeks (50th percentile)

# Number of top probabilities to consider for median prediction
TOP_N_PROBS = 15


def _get_hc_from_days(t, quartile: str = "0.5") -> float:
    """Convert gestational age (days) → expected HC (mm) using Hadlock formula.

    Polynomial coefficients from INTERGROWTH-21st reference.
    """
    t = t / 7.0  # convert to weeks
    dict_params = {
        "0.025": [1.59317517131532e+0, 2.9459800552433e-1, -7.3860372566707e-3, 6.56951770216148e-5, 0e+0],
        "0.5":   [2.09924879247164e+0, 2.53373656106037e-1, -6.05647816678282e-3, 5.14256072059917e-5, 0e+0],
        "0.975": [2.50074069629423e+0, 2.20067854715719e-1, -4.93623111462443e-3, 3.89066000946519e-5, 0e+0],
    }
    b0, b1, b2, b3, b4 = dict_params[quartile]
    return float(np.exp(b0 + b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4))


def _find_median_from_top_n(text_dot_prods: np.ndarray, n: int = 15) -> int:
    """FetalCLIP's prediction method: take top-N highest dot products,
    sort by their day-index, return the median index."""
    assert text_dot_prods.ndim == 1
    indexed = [(i, t) for i, t in enumerate(text_dot_prods)]
    indexed = sorted(indexed, key=lambda x: x[1], reverse=True)[:n]
    indexed = sorted(indexed, key=lambda x: x[0])  # sort back by day index
    median_ind = indexed[n // 2][0]
    return median_ind


@torch.no_grad()
def _get_ga_text_features(
    template: str,
    pixel_spacing: float,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Encode all GA text prompts for a single template + pixel spacing."""
    prompts = []
    for weeks in range(14, 39):
        for days in range(7):
            prompt = template.replace("{weeks}", str(weeks))
            prompt = prompt.replace("{day}", str(days))
            prompt = prompt.replace("{pixel_spacing}", f"{pixel_spacing:.2f}")
            prompts.append(prompt)

    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def _get_unnormalized_dot_products(
    image_features: torch.Tensor,
    list_text_features: List[torch.Tensor],
) -> torch.Tensor:
    """Compute dot products and average across templates (matches FetalCLIP)."""
    text_features = torch.cat(list_text_features, dim=0)
    text_dot_prods = 100.0 * image_features @ text_features.T
    n_prompts = len(list_text_features)
    n_days = len(list_text_features[0])
    text_dot_prods = text_dot_prods.view(image_features.shape[0], n_prompts, n_days)
    text_dot_prods = text_dot_prods.mean(dim=1)
    return text_dot_prods


@torch.no_grad()
def _run_ga_estimation(
    model: torch.nn.Module,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run zero-shot gestational age estimation on HC18.

    Matches FetalCLIP's two-step evaluation:
      1. Build dot-product map (image features × GA text prompts).
      2. Predict GA via median-of-top-N, then compute validity rate
         against INTERGROWTH-21st Hadlock reference curves.
    """
    list_exp_outs = []

    for images, pixel_spacings, hcs in dataloader:
        assert images.shape[0] == 1, "HC18 eval requires batch_size=1"
        images = images.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        ps = pixel_spacings[0].item()
        values = [_get_ga_text_features(t, ps, tokenizer, model, device) for t in GA_TEMPLATES]
        text_dot_prods = _get_unnormalized_dot_products(image_features, values)

        list_exp_outs.append({
            "true_hc": hcs[0].item(),
            "text_dot_prods": text_dot_prods.detach().cpu().numpy(),
        })

    # --- Post-processing (matches FetalCLIP test_ga_postprocess_dot_prods.py) ---
    list_true_hc = [d["true_hc"] for d in list_exp_outs]
    list_dot_prods = [d["text_dot_prods"] for d in list_exp_outs]

    list_pred_days = []
    for text_dot_prods in list_dot_prods:
        med_idx = _find_median_from_top_n(text_dot_prods[0], TOP_N_PROBS)
        pred_days = GA_DAYS_MAP[med_idx]
        list_pred_days.append(pred_days)

    # Filter by HC range
    filtered_pred = [d for hc, d in zip(list_true_hc, list_pred_days) if DATA_MIN_HC <= hc <= DATA_MAX_HC]
    filtered_hc = [hc for hc in list_true_hc if DATA_MIN_HC <= hc <= DATA_MAX_HC]

    # Validity: check if true HC falls within 2.5th–97.5th percentile of predicted GA
    list_validity = []
    for true_hc, pred_days in zip(filtered_hc, filtered_pred):
        q_low = _get_hc_from_days(pred_days, "0.025")
        q_high = _get_hc_from_days(pred_days, "0.975")
        list_validity.append(1 if q_low <= true_hc <= q_high else 0)

    total = len(list_validity)
    valid_count = sum(list_validity)
    validity_rate = (valid_count / total * 100.0) if total > 0 else 0.0

    # Bootstrap 95% CI
    if total > 0:
        np_validity = np.array(list_validity)
        rng = np.random.default_rng(seed=42)
        boot_means = np.array([
            rng.choice(np_validity, size=len(np_validity), replace=True).mean()
            for _ in range(10000)
        ])
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_lower, ci_upper = 0.0, 0.0

    # Also report Spearman correlation as supplementary metric
    from scipy.stats import spearmanr
    pred_ga = np.array(filtered_pred)
    true_hc_arr = np.array(filtered_hc)
    corr, pval = spearmanr(pred_ga, true_hc_arr) if len(pred_ga) > 2 else (0.0, 1.0)
    if np.isnan(corr):
        corr = 0.0
    if np.isnan(pval):
        pval = 1.0

    metrics = {
        "validity_rate": validity_rate,
        "validity_ci_lower": ci_lower * 100.0,
        "validity_ci_upper": ci_upper * 100.0,
        "valid_count": float(valid_count),
        "total_filtered": float(total),
        "ga_hc_spearman": corr,
        "ga_hc_spearman_pval": pval,
        "ga_pred_mean_weeks": pred_ga.mean() / 7.0 if len(pred_ga) > 0 else 0.0,
        "ga_pred_std_weeks": pred_ga.std() / 7.0 if len(pred_ga) > 0 else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def zero_shot_eval(
    model: torch.nn.Module,
    tokenizer,
    preprocess,
    device: torch.device,
    eval_cfg: Dict[str, Any],
    epoch: int = 0,
) -> Dict[str, float]:
    """Run all configured zero-shot evaluations.

    Args:
        model: CLIP model (in eval mode).
        tokenizer: Text tokenizer.
        preprocess: Image preprocessing transform (val/test).
        device: Device to run on.
        eval_cfg: Dict with evaluation config, expecting keys like:
            - ``fetal_planes_db_images``: path to Images/ dir
            - ``fetal_planes_db_csv``: path to CSV
            - ``hc18_images``: path to training_set/ dir
            - ``hc18_csv``: path to CSV
            - ``batch_size``: eval batch size (default 16)
            - ``num_workers``: dataloader workers (default 4)
            - ``eval_five_planes``: bool (default True)
            - ``eval_brain_subplanes``: bool (default True)
            - ``eval_hc18_ga``: bool (default True)
        epoch: Current training epoch (for logging).

    Returns:
        Dict of all computed metrics with prefixed keys.
    """
    model.eval()
    metrics = {}
    batch_size = eval_cfg.get("batch_size", 16)
    num_workers = eval_cfg.get("num_workers", 4)

    # --- Five-plane classification ---
    if eval_cfg.get("eval_five_planes", True):
        planes_images = eval_cfg.get("fetal_planes_db_images")
        planes_csv = eval_cfg.get("fetal_planes_db_csv")
        if (
            planes_images
            and planes_csv
            and os.path.exists(planes_images)
            and os.path.exists(planes_csv)
        ):
            logger.info(f"[Epoch {epoch}] Running five-plane zero-shot classification...")
            prompts_path = _PROMPTS_DIR / "five_planes_prompts.json"
            with open(prompts_path) as f:
                prompts = json.load(f)

            list_text_features, class_names = _build_text_features(model, tokenizer, prompts, device)

            ds = FetalPlanesDBDataset(
                planes_images, planes_csv, preprocess,
                split="all", exclude_planes=["Other"],
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            plane_metrics = _run_classification(
                model, list_text_features, class_names, dl, device,
                report_topk=(1, 2, 3),
            )
            for k, v in plane_metrics.items():
                metrics[f"five_planes/{k}"] = v

            logger.info(
                f"  Five-plane acc: {plane_metrics['acc']:.4f} | "
                f"F1: {plane_metrics['f1']:.4f} | "
                f"acc_top2: {plane_metrics.get('acc_top2', 0):.4f} | "
                f"acc_top3: {plane_metrics.get('acc_top3', 0):.4f}"
            )
        else:
            logger.warning(
                "Skipping five-plane eval: missing paths (images=%s exists=%s, csv=%s exists=%s).",
                planes_images,
                bool(planes_images and os.path.exists(planes_images)),
                planes_csv,
                bool(planes_csv and os.path.exists(planes_csv)),
            )

    # --- Brain subplane classification ---
    if eval_cfg.get("eval_brain_subplanes", True):
        planes_images = eval_cfg.get("fetal_planes_db_images")
        planes_csv = eval_cfg.get("fetal_planes_db_csv")
        if (
            planes_images
            and planes_csv
            and os.path.exists(planes_images)
            and os.path.exists(planes_csv)
        ):
            logger.info(f"[Epoch {epoch}] Running brain subplane zero-shot classification...")
            prompts_path = _PROMPTS_DIR / "brain_subplanes_prompts.json"
            with open(prompts_path) as f:
                prompts = json.load(f)

            list_text_features, class_names = _build_text_features(model, tokenizer, prompts, device)

            ds = FetalPlanesDBBrainDataset(
                planes_images, planes_csv, preprocess,
                split="all",
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            brain_metrics = _run_classification(
                model, list_text_features, class_names, dl, device,
                report_topk=(1,),
            )
            for k, v in brain_metrics.items():
                metrics[f"brain_subplanes/{k}"] = v

            logger.info(
                f"  Brain subplane acc: {brain_metrics['acc']:.4f} | "
                f"F1: {brain_metrics['f1']:.4f}"
            )
        else:
            logger.warning(
                "Skipping brain subplane eval: missing paths (images=%s exists=%s, csv=%s exists=%s).",
                planes_images,
                bool(planes_images and os.path.exists(planes_images)),
                planes_csv,
                bool(planes_csv and os.path.exists(planes_csv)),
            )

    # --- HC18 gestational age estimation ---
    if eval_cfg.get("eval_hc18_ga", True):
        hc18_images = eval_cfg.get("hc18_images")
        hc18_csv = eval_cfg.get("hc18_csv")
        if (
            hc18_images
            and hc18_csv
            and os.path.exists(hc18_images)
            and os.path.exists(hc18_csv)
        ):
            logger.info(f"[Epoch {epoch}] Running HC18 zero-shot GA estimation...")
            hc18_input_size_cfg = eval_cfg.get("hc18_input_size")
            if hc18_input_size_cfg is not None:
                hc18_input_size = max(1, int(hc18_input_size_cfg))
            else:
                hc18_input_size = infer_eval_input_size(preprocess, fallback=224)
            logger.info(
                "  HC18 pixel spacing scale uses input_size=%d (derived from eval preprocess).",
                hc18_input_size,
            )

            ds = HC18Dataset(hc18_images, hc18_csv, preprocess, input_size=hc18_input_size)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)

            ga_metrics = _run_ga_estimation(model, tokenizer, dl, device)
            for k, v in ga_metrics.items():
                metrics[f"hc18/{k}"] = v

            logger.info(
                f"  HC18 validity rate: {ga_metrics['validity_rate']:.2f}% "
                f"(95% CI: {ga_metrics['validity_ci_lower']:.2f}–{ga_metrics['validity_ci_upper']:.2f}%) | "
                f"Spearman: {ga_metrics['ga_hc_spearman']:.4f}"
            )
        else:
            logger.warning(
                "Skipping HC18 GA eval: missing paths (images=%s exists=%s, csv=%s exists=%s).",
                hc18_images,
                bool(hc18_images and os.path.exists(hc18_images)),
                hc18_csv,
                bool(hc18_csv and os.path.exists(hc18_csv)),
            )

    return metrics
