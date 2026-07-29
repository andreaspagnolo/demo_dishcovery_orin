#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageFile

import orin_task1_pipeline as task1_siglip
import edgellm_qwen
from orin_measurements import PowerMonitor, build_benchmark_summary, print_benchmark_summary, query_nvpmodel


# This reproducibility package keeps the pipeline under <repository>/code.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET_ROOT = ROOT / "dataset/food500_subset"
DEFAULT_IMAGE_DIR = DEFAULT_SUBSET_ROOT / "images"
DEFAULT_MANIFEST = DEFAULT_SUBSET_ROOT / "manifest.csv"
DEFAULT_IMAGES_LIST = DEFAULT_SUBSET_ROOT / "images.txt"
DEFAULT_EVALUATION_JSON = ROOT / "labels/evaluation_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/task2_orin"
DEFAULT_SIGLIP_MODEL = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DEFAULT_SIGLIP_PRETRAINED = ""
DEFAULT_SIGLIP_ONNX_DIR = ROOT / "models/onnx/ViT-gopt-16-SigLIP2-384-ONNX"
DEFAULT_SIGLIP_TRT_CACHE_DIR = ROOT / "models/onnx/trt_cache"
DEFAULT_SIGLIP_TRT_ENGINE = Path("/home/danilopau/models_quant/siglip2_visual/visual_fp16.engine")
DEFAULT_RERANKER_SIZE = "8B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
RERANKER_MODEL_BY_SIZE = {
    "2B": "Qwen/Qwen3-VL-Reranker-2B",
    "8B": DEFAULT_RERANKER_MODEL,
}
RERANKER_REQUIRED_FILES = ("config.json", "modules.json", "tokenizer.json")
CAPTION_CACHE_VERSION = "orin_task2_siglip2_caption_bank_v1"
IMAGE_CACHE_VERSION = "orin_task2_siglip2_image_bank_v1"

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class Food500Row:
    cat: str
    filename: str
    caption: str


@dataclass(frozen=True)
class CaptionBankItem:
    caption_id: int
    cat: str
    filename: str
    caption: str
    text: str


@dataclass
class CaptionCandidate:
    caption_id: int
    cat: str
    filename: str
    caption: str
    text: str
    siglip_score: float
    rerank_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task 2 caption-alignment pipeline: SigLIP2 recalls candidate captions, "
            "then a Qwen3-VL reranker scores image-caption pairs."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run one image. Accepts a full path or a path relative to --image-dir.",
    )
    parser.add_argument("--eval-samples", type=int, default=0, help="Evaluate N images.")
    parser.add_argument("--eval-all", action="store_true", help="Evaluate every image found in --evaluation-json.")
    parser.add_argument("--eval-first", action="store_true", help="Use the first N rows instead of a seeded random sample.")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Run images without ground truth and skip accuracy metrics. "
            "Useful for Dishcovery Test2 with a flat captions.json bank."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--images-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited image list, relative to --image-dir. "
            f"For the 350-image benchmark use {DEFAULT_IMAGES_LIST}."
        ),
    )
    parser.add_argument("--evaluation-json", type=Path, default=DEFAULT_EVALUATION_JSON)
    parser.add_argument(
        "--caption-bank-json",
        type=Path,
        default=None,
        help=(
            "JSON containing candidate captions. Defaults to --evaluation-json. "
            "Each row must have cat, filename, caption."
        ),
    )
    parser.add_argument(
        "--caption-text-mode",
        choices=("caption", "class_caption"),
        default="class_caption",
        help="Text passed to SigLIP/Qwen. class_caption prefixes the food class before the caption.",
    )
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-pretrained", default=DEFAULT_SIGLIP_PRETRAINED)
    parser.add_argument("--siglip-image-size", type=int, default=384)
    parser.add_argument(
        "--siglip-backend",
        choices=("auto", "pytorch", "onnxruntime", "tensorrt", "tensorrt_engine", "cuda", "cpu"),
        default="pytorch",
        help=(
            "SigLIP2 runtime. pytorch keeps OpenCLIP. onnxruntime uses ONNX CPU. "
            "cuda and tensorrt use ONNX Runtime GPU providers when available. "
            "tensorrt_engine loads a serialized visual TensorRT engine and requires precomputed text embeddings."
        ),
    )
    parser.add_argument("--siglip-onnx-dir", type=Path, default=DEFAULT_SIGLIP_ONNX_DIR)
    parser.add_argument("--siglip-trt-cache-dir", type=Path, default=DEFAULT_SIGLIP_TRT_CACHE_DIR)
    parser.add_argument("--siglip-trt-engine", type=Path, default=DEFAULT_SIGLIP_TRT_ENGINE)
    parser.add_argument(
        "--siglip-torch-dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Torch dtype used only by PyTorch SigLIP2. auto follows --torch-dtype.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--torch-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument(
        "--siglip-top-k",
        type=int,
        default=5,
        help="Caption recall depth before Qwen reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="Number of caption candidates to send to Qwen per image.",
    )
    parser.add_argument(
        "--candidate-list-mode",
        choices=("fixed_topk", "dynamic_relative_delta"),
        default="fixed_topk",
        help=(
            "How to build the SigLIP2 caption candidate list. fixed_topk keeps "
            "--rerank-top-k captions. dynamic_relative_delta keeps captions close "
            "to the top SigLIP2 score, then uses the top1/top2 gap to keep "
            "confident images near the lower bound."
        ),
    )
    parser.add_argument(
        "--dynamic-candidate-relative-delta",
        type=float,
        default=0.35,
        help="For dynamic_relative_delta, initial recall window is scores >= top_score - abs(top_score) * this value.",
    )
    parser.add_argument(
        "--dynamic-candidate-min-k",
        type=int,
        default=3,
        help="Minimum caption candidate count for dynamic_relative_delta.",
    )
    parser.add_argument(
        "--dynamic-candidate-max-k",
        type=int,
        default=15,
        help="Maximum caption candidate count for dynamic_relative_delta. Use 0 for all captions.",
    )
    parser.add_argument(
        "--final-score-mode",
        choices=("rerank", "siglip", "blend_z", "siglip_guarded"),
        default="rerank",
        help=(
            "How to choose the final caption. siglip_guarded keeps SigLIP top-1 "
            "when it is confident and uses Qwen only for ambiguous images."
        ),
    )
    parser.add_argument("--rerank-weight", type=float, default=0.7, help="Qwen weight for --final-score-mode blend_z.")
    parser.add_argument(
        "--siglip-keep-gap",
        type=float,
        default=0.4,
        help=(
            "For --final-score-mode siglip_guarded: keep SigLIP top-1 without Qwen "
            "when its score gap over rank 2 is at least this value."
        ),
    )
    parser.add_argument(
        "--reranker-size",
        default=DEFAULT_RERANKER_SIZE,
        help=(
            "Qwen3-VL reranker size alias, for example 8B or 2B. "
            "Unknown values are expanded as Qwen/Qwen3-VL-Reranker-<size>. "
            "Ignored when --reranker-model is provided."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "Explicit reranker repo ID or local path. Overrides --reranker-size. "
            f"Default is {DEFAULT_RERANKER_MODEL}."
        ),
    )
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument(
        "--reranker-backend",
        choices=("sentence_transformers", "edgellm"),
        default="sentence_transformers",
        help=(
            "Qwen reranker runtime. sentence_transformers uses the original score-head path. "
            "edgellm uses the quantized TensorRT Edge-LLM engine."
        ),
    )
    parser.add_argument("--reranker-attn-implementation", default="sdpa")
    parser.add_argument("--reranker-local-files-only", action="store_true")
    parser.add_argument(
        "--reranker-edgellm-score-mode",
        choices=("logit_score", "generated_yes_no"),
        default="logit_score",
        help=(
            "How to score Edge-LLM reranker pairs. logit_score requires the patched persistent "
            "server and matches the Qwen3-VL-Reranker true/no logit method. generated_yes_no is "
            "a diagnostic fallback and is not comparable to the FP16 reranker."
        ),
    )
    parser.add_argument("--reranker-edgellm-true-token-id", type=int, default=9693)
    parser.add_argument("--reranker-edgellm-false-token-id", type=int, default=2152)
    parser.add_argument(
        "--reranker-edgellm-doc-max-chars",
        type=int,
        default=320,
        help=(
            "Maximum candidate-caption characters sent to the Edge-LLM reranker. "
            "Use with a visual engine capped below the LLM max input length, e.g. 384 image tokens for a 512-token LLM."
        ),
    )
    parser.add_argument("--reranker-edgellm-max-new-tokens", type=int, default=4)
    parser.add_argument("--reranker-edgellm-ambiguous-score", type=float, default=0.5)
    parser.add_argument("--edgellm-root", type=Path, default=Path(os.environ.get("EDGE_LLM_PATH", edgellm_qwen.DEFAULT_EDGE_LLM_PATH)))
    parser.add_argument("--edgellm-workspace", type=Path, default=Path(os.environ.get("WORKSPACE_DIR", edgellm_qwen.DEFAULT_WORKSPACE_DIR)))
    parser.add_argument("--edgellm-model-name", default=os.environ.get("MODEL_NAME", "Qwen3-VL-Reranker-2B"))
    parser.add_argument("--edgellm-llm-engine-profile", default=edgellm_qwen.DEFAULT_LLM_ENGINE_PROFILE)
    parser.add_argument("--edgellm-llm-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-visual-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-binary", type=Path, default=None)
    parser.add_argument("--edgellm-persistent-binary", type=Path, default=None)
    parser.add_argument("--edgellm-plugin-path", type=Path, default=None)
    parser.add_argument("--edgellm-runtime-mode", choices=("subprocess", "persistent"), default="persistent")
    parser.add_argument("--edgellm-timeout-sec", type=float, default=300.0)
    parser.add_argument("--edgellm-startup-timeout-sec", type=float, default=300.0)
    parser.add_argument("--edgellm-keep-io", action="store_true")
    parser.add_argument("--edgellm-dump-profile", action="store_true")
    parser.add_argument("--edgellm-warmup", type=int, default=0)
    parser.add_argument("--edgellm-temperature", type=float, default=0.0)
    parser.add_argument("--edgellm-top-p", type=float, default=1.0)
    parser.add_argument("--edgellm-top-k", type=int, default=1)
    parser.add_argument("--edgellm-enable-thinking", action="store_true")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Optional file containing a Hugging Face token.",
    )
    parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Do not load Qwen; use SigLIP scores as rerank scores. Useful for smoke tests.",
    )
    parser.add_argument(
        "--caption-cache",
        type=Path,
        default=ROOT / "outputs/cache/orin_task2_siglip2_caption_bank_cache.npz",
    )
    parser.add_argument(
        "--caption-cache-allow-mismatch",
        action="store_true",
        help="Load caption_embeddings from --caption-cache when shape matches even if the metadata hash differs.",
    )
    parser.add_argument(
        "--caption-cache-require-hit",
        action="store_true",
        help="Fail instead of running the text encoder when --caption-cache cannot be loaded.",
    )
    parser.add_argument(
        "--image-cache",
        type=Path,
        default=ROOT / "outputs/cache/orin_task2_siglip2_food500_subset_image_cache.npz",
    )
    parser.add_argument("--no-caption-cache", action="store_true")
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Use per-query evaluation and add latency, throughput, power, and energy summary.",
    )
    parser.add_argument("--measure-power", action="store_true", help="Sample Jetson tegrastats power during the per-query inference loop.")
    parser.add_argument("--measure-ram", action="store_true", help="Sample process-tree and system RAM before and during the run.")
    parser.add_argument("--ram-sample-interval-ms", type=int, default=200)
    parser.add_argument("--power-sample-interval-ms", type=int, default=200)
    parser.add_argument("--w-config", default="", help="Power-budget label for the run, for example 15W, 30W, or 50W.")
    args = parser.parse_args()
    resolve_reranker_model_args(args)
    return args


def normalize_reranker_size(value: str) -> str:
    text = str(value or DEFAULT_RERANKER_SIZE).strip()
    if not text:
        text = DEFAULT_RERANKER_SIZE
    return text.upper()


def reranker_model_from_size(size: str) -> str:
    normalized = normalize_reranker_size(size)
    return RERANKER_MODEL_BY_SIZE.get(normalized, f"Qwen/Qwen3-VL-Reranker-{normalized}")


def resolve_reranker_model_args(args: argparse.Namespace) -> None:
    args.reranker_size = normalize_reranker_size(args.reranker_size)
    if args.reranker_model:
        args.reranker_model = str(args.reranker_model).strip()
        return
    args.reranker_model = reranker_model_from_size(args.reranker_size)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def display_name(class_name: str) -> str:
    return normalize_text(class_name.replace("_", " "))


def make_caption_text(row: Food500Row, mode: str) -> str:
    caption = normalize_text(row.caption)
    if not row.cat:
        return caption
    if mode == "caption":
        return caption
    if mode == "class_caption":
        return normalize_text(f"{display_name(row.cat)}. {caption}")
    raise ValueError(f"Unsupported caption-text mode: {mode}")


def sha256_jsonable(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_food500_rows(path: Path) -> list[Food500Row]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list")
    rows: list[Food500Row] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"{path} row {idx} must be an object")
        cat = normalize_text(item.get("cat", ""))
        filename = normalize_text(item.get("filename", ""))
        caption = normalize_text(item.get("caption", ""))
        if not cat or not filename or not caption:
            raise ValueError(f"{path} row {idx} is missing cat, filename, or caption")
        rows.append(Food500Row(cat=cat, filename=filename, caption=caption))
    return rows


def read_manifest(path: Path) -> list[Food500Row]:
    if not path.exists():
        return []
    rows: list[Food500Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cat", "filename"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for item in reader:
            rows.append(
                Food500Row(
                    cat=normalize_text(item.get("cat", "")),
                    filename=normalize_text(item.get("filename", "")),
                    caption=normalize_text(item.get("caption", "")),
                )
            )
    return rows


def read_image_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    names = [normalize_text(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Image list contains duplicate entries: {duplicates[:10]}")
    return names


def discover_images(image_dir: Path, gt_by_filename: dict[str, Food500Row]) -> list[Food500Row]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    rows: list[Food500Row] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(image_dir).as_posix()
        gt = gt_by_filename.get(rel)
        if gt is not None:
            rows.append(gt)
    return rows


def resolve_image_name(image: Path, image_dir: Path) -> tuple[str, Path]:
    if image.exists():
        image_path = image
        try:
            image_name = image_path.relative_to(image_dir).as_posix()
        except ValueError:
            image_name = image_path.name
        return image_name, image_path
    candidate = image_dir / image
    if candidate.exists():
        return image.as_posix(), candidate
    candidate = image_dir / image.name
    if candidate.exists():
        return image.name, candidate
    raise FileNotFoundError(f"Image does not exist: {image}")


def select_eval_rows(
    args: argparse.Namespace,
    evaluation_rows: list[Food500Row],
    manifest_rows: list[Food500Row],
) -> list[Food500Row]:
    gt_by_filename = {row.filename: row for row in evaluation_rows}
    if args.images_list is not None:
        names = read_image_list(args.images_list)
        missing_gt = [name for name in names if name not in gt_by_filename]
        if missing_gt:
            raise FileNotFoundError(f"{len(missing_gt)} image-list entries are missing from {args.evaluation_json}: {missing_gt[:10]}")
        rows = [gt_by_filename[name] for name in names]
    elif manifest_rows:
        rows = [gt_by_filename[row.filename] for row in manifest_rows if row.filename in gt_by_filename]
    else:
        rows = evaluation_rows
    rows = [row for row in rows if (args.image_dir / row.filename).exists()]
    if not rows:
        source = str(args.images_list) if args.images_list is not None else str(args.manifest)
        raise FileNotFoundError(
            f"No evaluation images were found under {args.image_dir}. "
            f"Check --image-dir and --images-list/--manifest. Source: {source}"
        )
    if args.eval_all:
        return rows
    if args.eval_samples <= 0:
        raise SystemExit("Provide --image, --eval-samples N, or --eval-all.")
    if args.eval_samples >= len(rows):
        return rows
    if args.eval_first:
        return rows[: args.eval_samples]
    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.eval_samples)
    return sorted(selected, key=lambda row: row.filename)


def build_caption_bank(rows: list[Food500Row], caption_text_mode: str) -> list[CaptionBankItem]:
    bank: list[CaptionBankItem] = []
    for idx, row in enumerate(rows):
        bank.append(
            CaptionBankItem(
                caption_id=idx,
                cat=row.cat,
                filename=row.filename,
                caption=row.caption,
                text=make_caption_text(row, caption_text_mode),
            )
        )
    if not bank:
        raise ValueError("No candidate captions were built")
    return bank


def read_caption_bank_items(path: Path, caption_text_mode: str) -> list[CaptionBankItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list")
    if not raw:
        raise ValueError(f"{path} is empty")
    if all(isinstance(item, str) for item in raw):
        bank: list[CaptionBankItem] = []
        for idx, item in enumerate(raw):
            caption = normalize_text(item)
            if not caption:
                continue
            bank.append(
                CaptionBankItem(
                    caption_id=idx,
                    cat="",
                    filename=f"caption_{idx:06d}",
                    caption=caption,
                    text=caption,
                )
            )
        if not bank:
            raise ValueError(f"{path} does not contain usable captions")
        return bank
    return build_caption_bank(read_food500_rows(path), caption_text_mode)


def select_inference_rows(args: argparse.Namespace) -> list[Food500Row]:
    if args.images_list is not None:
        names = read_image_list(args.images_list)
    else:
        suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        names = sorted(path.name for path in args.image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    missing = [name for name in names if not (args.image_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image-list entries are missing from {args.image_dir}: {missing[:10]}")
    rows = [Food500Row(cat="", filename=name, caption="") for name in names]
    if not rows:
        raise FileNotFoundError(f"No inference images were found under {args.image_dir}")
    if args.eval_all:
        return rows
    if args.eval_samples <= 0:
        raise SystemExit("Provide --eval-samples N or --eval-all with --inference-only.")
    if args.eval_samples >= len(rows):
        return rows
    if args.eval_first:
        return rows[: args.eval_samples]
    rng = random.Random(args.seed)
    return sorted(rng.sample(rows, args.eval_samples), key=lambda row: row.filename)


def hf_hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_model_weights_complete(snapshot: Path) -> bool:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = snapshot / name
        if path.is_file():
            return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            return False
        shard_names = {str(name) for name in weight_map.values() if str(name)}
        return bool(shard_names) and all((snapshot / shard_name).is_file() for shard_name in shard_names)

    return False


def cached_hf_snapshot(repo_id: str, required_files: tuple[str, ...] = (), require_model_weights: bool = False) -> Path | None:
    cache_dir = hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates: list[Path] = []
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        try:
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots_dir / revision)
        except OSError:
            pass
    try:
        candidates.extend(sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True))
    except OSError:
        pass
    seen: set[Path] = set()
    for snapshot in candidates:
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if (
            snapshot.is_dir()
            and all((snapshot / name).exists() for name in required_files)
            and (not require_model_weights or snapshot_model_weights_complete(snapshot))
        ):
            return snapshot
    return None


def extract_env_value(line: str, key: str) -> str | None:
    raw = line.strip()
    if not raw or raw.startswith("#") or "=" not in raw:
        return None
    name, value = raw.split("=", 1)
    if name.strip() != key:
        return None
    value = value.strip().strip("'").strip('"')
    return value or None


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        for line in text.splitlines():
            value = extract_env_value(line, key)
            if value:
                return value
    return text.splitlines()[0].strip().strip("'").strip('"') or None


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    token_paths: list[Path] = []
    if args.hf_token_file is not None:
        token_paths.append(args.hf_token_file.expanduser())
    token_paths.extend([ROOT / ".env", ROOT / ".hf_token", Path.home() / ".cache/huggingface/token"])
    for path in token_paths:
        token = read_token_file(path)
        if token:
            return token
    return None


def configure_hf_token(args: argparse.Namespace) -> str | None:
    token = resolve_hf_token(args)
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        print("Using Hugging Face token from environment/file.")
    else:
        print("No Hugging Face token found; downloads will be unauthenticated.")
    return token


def resolve_open_clip_source(model_name: str) -> str:
    if not model_name.startswith("hf-hub:"):
        return model_name
    repo_id = model_name.removeprefix("hf-hub:")
    snapshot = cached_hf_snapshot(repo_id, ("open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json"))
    if snapshot is None:
        return model_name
    return f"local-dir:{snapshot}"


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "This pipeline requires PyTorch. On Jetson Orin, use an NVIDIA/JetPack-compatible "
            "PyTorch build, then install open_clip_torch, transformers, sentence-transformers, and pillow."
        ) from exc
    return torch


def torch_tensor_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().float().cpu().tolist(), dtype=np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def read_proc_status_kib(pid: int, key: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(f"{key}:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except Exception:
        return 0
    return 0


def process_children_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8", errors="ignore")
            after_name = text.rsplit(")", 1)[1].strip().split()
            ppid = int(after_name[1])
            pid = int(stat_path.parent.name)
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def process_tree_pids(root_pid: int) -> list[int]:
    children = process_children_by_parent()
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return sorted(seen)


def process_tree_rss_mib(root_pid: int) -> float:
    kib = sum(read_proc_status_kib(pid, "VmRSS") for pid in process_tree_pids(root_pid))
    return kib / 1024.0


def system_memory_used_mib() -> tuple[float, float, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1])
    except Exception:
        return 0.0, 0.0, 0.0
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = max(0, total - available)
    return used / 1024.0, total / 1024.0, available / 1024.0


class MemoryMonitor:
    def __init__(self, enabled: bool, interval_ms: int = 200) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = max(0.05, float(interval_ms) / 1000.0)
        self.pid = os.getpid()
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline: dict[str, float] = {}

    def sample(self) -> dict[str, float]:
        system_used, system_total, system_available = system_memory_used_mib()
        return {
            "time_sec": time.perf_counter(),
            "process_tree_rss_mib": process_tree_rss_mib(self.pid),
            "system_used_mib": system_used,
            "system_total_mib": system_total,
            "system_available_mib": system_available,
        }

    def start(self) -> None:
        if not self.enabled:
            return
        self.baseline = self.sample()
        self.samples = [dict(self.baseline)]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.samples.append(self.sample())

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))
        self.samples.append(self.sample())
        process_values = [sample["process_tree_rss_mib"] for sample in self.samples]
        system_values = [sample["system_used_mib"] for sample in self.samples]
        baseline_process = float(self.baseline.get("process_tree_rss_mib", process_values[0] if process_values else 0.0))
        baseline_system = float(self.baseline.get("system_used_mib", system_values[0] if system_values else 0.0))
        peak_process = max(process_values) if process_values else baseline_process
        peak_system = max(system_values) if system_values else baseline_system
        last = self.samples[-1] if self.samples else {}
        return {
            "enabled": True,
            "sample_count": len(self.samples),
            "interval_ms": int(self.interval_sec * 1000),
            "baseline_process_tree_rss_mib": baseline_process,
            "peak_process_tree_rss_mib": peak_process,
            "delta_process_tree_rss_mib": peak_process - baseline_process,
            "final_process_tree_rss_mib": float(last.get("process_tree_rss_mib", 0.0)),
            "baseline_system_used_mib": baseline_system,
            "peak_system_used_mib": peak_system,
            "delta_system_used_mib": peak_system - baseline_system,
            "final_system_used_mib": float(last.get("system_used_mib", 0.0)),
            "system_total_mib": float(last.get("system_total_mib", 0.0)),
            "system_available_mib": float(last.get("system_available_mib", 0.0)),
        }


class OpenCLIPSigLIP2Embedder:
    def __init__(self, model_name: str, pretrained: str, device: str, torch_dtype: str, image_size: int) -> None:
        self.torch = require_torch()
        try:
            import open_clip
        except Exception as exc:
            raise RuntimeError("Install open_clip_torch to use the SigLIP2 embedder.") from exc
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a CPU smoke test.")
        self.device = self.torch.device(device)
        self.autocast_dtype = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }[torch_dtype]
        self.torch_dtype = torch_dtype
        self.autocast_enabled = self.device.type == "cuda" and torch_dtype != "float32"
        model_source = resolve_open_clip_source(model_name)
        if model_source != model_name:
            print(f"Using cached OpenCLIP snapshot: {model_source.removeprefix('local-dir:')}")
        pretrained_arg = pretrained or None
        model, _, preprocess = open_clip.create_model_and_transforms(model_source, pretrained=pretrained_arg, device=self.device)
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_source)
        self.image_size = int(image_size)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokens = self.tokenizer(batch_texts).to(self.device)
                with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(torch_tensor_to_numpy(feats))
                print(f"Encoded candidate captions {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                paths = image_paths[start : start + batch_size]
                tensors = []
                for path in paths:
                    image = Image.open(path).convert("RGB")
                    tensors.append(self.preprocess_image_no_numpy(image))
                batch = self.torch.stack(tensors, dim=0).to(self.device)
                with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(torch_tensor_to_numpy(feats))
                print(f"Encoded images {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            tensor = self.preprocess_image_no_numpy(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalize_rows(torch_tensor_to_numpy(feats))[0]

    def preprocess_image_no_numpy(self, image: Image.Image) -> Any:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        image = image.convert("RGB").resize((self.image_size, self.image_size), resampling)
        raw = bytearray(image.tobytes())
        try:
            tensor = self.torch.frombuffer(raw, dtype=self.torch.uint8)
        except Exception:
            storage = self.torch.ByteStorage.from_buffer(raw)
            tensor = self.torch.ByteTensor(storage)
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        return (tensor - 0.5) / 0.5


class ONNXSigLIP2Embedder:
    def __init__(self, model_name: str, onnx_dir: Path, backend: str, trt_cache_dir: Path, image_size: int) -> None:
        self.base = task1_siglip.ONNXSigLIP2Embedder(
            model_name=model_name,
            onnx_dir=onnx_dir,
            backend=backend,
            trt_cache_dir=trt_cache_dir,
            image_size=image_size,
        )
        self.backend = self.base.backend
        self.onnx_provider = self.base.onnx_provider
        self.onnx_dir = self.base.onnx_dir
        self.session_providers = self.base.session_providers
        self.image_size = int(image_size)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self.base.encode_texts(texts, batch_size)

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            embeddings = []
            for path in paths:
                with Image.open(path) as image:
                    embeddings.append(self.base.encode_image(image.convert("RGB")))
            chunks.append(np.stack(embeddings, axis=0))
            print(f"Encoded images {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            return self.base.encode_image(image.convert("RGB"))


class TensorRTEngineSigLIP2VisualEmbedder:
    def __init__(self, engine_path: Path, image_size: int) -> None:
        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing SigLIP2 TensorRT visual engine: {self.engine_path}")
        self.torch = require_torch()
        if not self.torch.cuda.is_available():
            raise RuntimeError("The TensorRT visual engine backend requires CUDA.")
        try:
            import tensorrt as trt
        except Exception:
            system_site = "/usr/lib/python3.10/dist-packages"
            if system_site not in sys.path:
                sys.path.append(system_site)
            try:
                import tensorrt as trt
            except Exception as exc:
                raise RuntimeError("TensorRT Python bindings are required for --siglip-backend tensorrt_engine.") from exc

        self.trt = trt
        self.backend = "tensorrt_engine"
        self.onnx_provider = None
        self.onnx_dir = None
        self.session_providers: list[str] = []
        self.image_size = int(image_size)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = ""
        self.output_name = ""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
        if not self.input_name or not self.output_name:
            raise RuntimeError(f"Could not identify input/output tensors in {self.engine_path}")
        self.context.set_input_shape(self.input_name, (1, 3, self.image_size, self.image_size))
        print(f"Using TensorRT SigLIP2 visual engine: {self.engine_path}")

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        del texts, batch_size
        raise RuntimeError(
            "--siglip-backend tensorrt_engine loads only the visual encoder. "
            "Use --caption-cache with --caption-cache-require-hit for precomputed text embeddings."
        )

    def preprocess_image_tensor(self, image_path: Path) -> Any:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), resampling)
            raw = bytearray(image.tobytes())
        try:
            tensor = self.torch.frombuffer(raw, dtype=self.torch.uint8)
        except Exception:
            storage = self.torch.ByteStorage.from_buffer(raw)
            tensor = self.torch.ByteTensor(storage)
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor.unsqueeze(0).contiguous().cuda(non_blocking=True)

    def encode_image(self, image_path: Path) -> np.ndarray:
        tensor = self.preprocess_image_tensor(image_path)
        output = self.torch.empty((1, 1536), device="cuda", dtype=self.torch.float32)
        self.context.set_tensor_address(self.input_name, int(tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        stream = self.torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT SigLIP2 visual engine execution failed.")
        stream.synchronize()
        return normalize_rows(torch_tensor_to_numpy(output))[0]

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        del batch_size
        rows = []
        for idx, image_path in enumerate(image_paths, start=1):
            rows.append(self.encode_image(image_path))
            print(f"Encoded images {idx}/{len(image_paths)}", flush=True)
        return normalize_rows(np.stack(rows, axis=0))


def resolve_siglip_backend(args: argparse.Namespace) -> str:
    if str(args.siglip_backend) == "tensorrt_engine":
        return "tensorrt_engine"
    return task1_siglip.resolve_siglip_backend(args, image_size=int(args.siglip_image_size))


def siglip_torch_dtype(args: argparse.Namespace) -> str:
    value = str(getattr(args, "siglip_torch_dtype", "auto"))
    if value == "auto":
        return str(args.torch_dtype)
    return value


def build_siglip2_embedder(args: argparse.Namespace) -> Any:
    backend = resolve_siglip_backend(args)
    if backend == "tensorrt_engine":
        return TensorRTEngineSigLIP2VisualEmbedder(args.siglip_trt_engine, args.siglip_image_size)
    if backend in {"onnxruntime", "tensorrt", "cuda", "cpu"}:
        return ONNXSigLIP2Embedder(
            args.siglip_model,
            args.siglip_onnx_dir,
            backend,
            args.siglip_trt_cache_dir,
            args.siglip_image_size,
        )
    return OpenCLIPSigLIP2Embedder(
        args.siglip_model,
        args.siglip_pretrained,
        args.device,
        siglip_torch_dtype(args),
        args.siglip_image_size,
    )


def siglip_runtime_info(embedder: Any) -> dict[str, Any]:
    return {
        "siglip_backend": getattr(embedder, "backend", "pytorch"),
        "siglip_onnx_dir": str(getattr(embedder, "onnx_dir", "") or "") or None,
        "siglip_trt_engine": str(getattr(embedder, "engine_path", "") or "") or None,
        "siglip_onnx_provider": getattr(embedder, "onnx_provider", None),
        "siglip_session_providers": getattr(embedder, "session_providers", []),
        "siglip_torch_dtype": getattr(embedder, "torch_dtype", None),
    }


def load_or_build_caption_embeddings(
    embedder: OpenCLIPSigLIP2Embedder,
    caption_bank: list[CaptionBankItem],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_hash = sha256_jsonable(
        {
            "version": CAPTION_CACHE_VERSION,
            "model": args.siglip_model,
            "pretrained": args.siglip_pretrained,
            "siglip_backend": args.siglip_backend,
            "siglip_onnx_dir": str(args.siglip_onnx_dir),
            "siglip_torch_dtype": siglip_torch_dtype(args),
            "caption_text_mode": args.caption_text_mode,
            "captions": [
                {
                    "caption_id": item.caption_id,
                    "cat": item.cat,
                    "filename": item.filename,
                    "caption": item.caption,
                    "text": item.text,
                }
                for item in caption_bank
            ],
        }
    )
    info = {"path": str(args.caption_cache), "enabled": not args.no_caption_cache, "hit": False, "hash": cache_hash}
    if not args.no_caption_cache and args.caption_cache.exists():
        try:
            data = np.load(args.caption_cache, allow_pickle=False)
            cached_embeddings = normalize_rows(data["caption_embeddings"])
            if str(data["cache_hash"]) == cache_hash:
                info["hit"] = True
                return cached_embeddings, info
            if args.caption_cache_allow_mismatch and cached_embeddings.shape[0] == len(caption_bank):
                info["hit"] = True
                info["hash_mismatch_ignored"] = True
                info["cached_shape"] = list(cached_embeddings.shape)
                return cached_embeddings, info
            info["load_warning"] = "cache_hash_mismatch"
        except Exception as exc:
            info["load_warning"] = str(exc)
    if args.caption_cache_require_hit:
        raise RuntimeError(
            f"Caption embedding cache was required but not usable: {args.caption_cache}. "
            f"Reason: {info.get('load_warning', 'cache_missing')}"
        )
    texts = [item.text for item in caption_bank]
    embeddings = embedder.encode_texts(texts, args.text_batch_size)
    if not args.no_caption_cache:
        args.caption_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.caption_cache,
            caption_embeddings=embeddings.astype(np.float32),
            cache_version=np.asarray(CAPTION_CACHE_VERSION),
            cache_hash=np.asarray(cache_hash),
        )
    return embeddings, info


def load_image_cache(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info = {"path": str(args.image_cache), "enabled": not args.no_image_cache, "hit_count": 0}
    if args.no_image_cache or not args.image_cache.exists():
        return {}, info
    try:
        data = np.load(args.image_cache, allow_pickle=False)
        cached_backend = str(data["siglip_backend"]) if "siglip_backend" in data.files else "pytorch"
        cached_onnx_dir = str(data["siglip_onnx_dir"]) if "siglip_onnx_dir" in data.files else ""
        expected_onnx_dir = str(args.siglip_onnx_dir) if str(args.siglip_backend) != "pytorch" else ""
        if str(data["cache_version"]) != IMAGE_CACHE_VERSION:
            info["load_warning"] = "cache_version_mismatch"
            return {}, info
        if (
            str(data["siglip_model"]) != args.siglip_model
            or str(data["siglip_pretrained"]) != args.siglip_pretrained
            or int(data["siglip_image_size"]) != int(args.siglip_image_size)
            or cached_backend != str(args.siglip_backend)
            or cached_onnx_dir != expected_onnx_dir
        ):
            info["load_warning"] = "model_mismatch"
            return {}, info
        names = [str(x) for x in data["image_names"].tolist()]
        embeddings = normalize_rows(data["image_embeddings"])
        cache = {name: embeddings[idx] for idx, name in enumerate(names)}
        info["hit_count"] = len(cache)
        return cache, info
    except Exception as exc:
        info["load_warning"] = str(exc)
        return {}, info


def save_image_cache(path: Path, cache: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    if args.no_image_cache:
        return
    names = sorted(cache)
    if not names:
        return
    embeddings = np.stack([cache[name] for name in names], axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(IMAGE_CACHE_VERSION),
        siglip_model=np.asarray(args.siglip_model),
        siglip_pretrained=np.asarray(args.siglip_pretrained),
        siglip_image_size=np.asarray(int(args.siglip_image_size)),
        siglip_backend=np.asarray(str(args.siglip_backend)),
        siglip_onnx_dir=np.asarray(str(args.siglip_onnx_dir) if str(args.siglip_backend) != "pytorch" else ""),
        image_names=np.asarray(names),
        image_embeddings=embeddings,
    )


def load_or_build_image_embeddings(
    embedder: OpenCLIPSigLIP2Embedder,
    image_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache, info = load_image_cache(args)
    missing = [name for name in image_names if name not in cache]
    info["requested_count"] = len(image_names)
    info["missing_count"] = len(missing)
    if missing:
        paths = [args.image_dir / name for name in missing]
        encoded = embedder.encode_images(paths, args.image_batch_size)
        for name, row in zip(missing, encoded):
            cache[name] = row.astype(np.float32)
        save_image_cache(args.image_cache, cache, args)
    rows = [cache[name] for name in image_names]
    return normalize_rows(np.stack(rows, axis=0)), info


def build_caption_recall(
    scores: np.ndarray,
    caption_bank: list[CaptionBankItem],
    args: argparse.Namespace,
) -> list[CaptionCandidate]:
    if len(caption_bank) == 0:
        return []
    candidate_count = caption_candidate_count_for_scores(scores, len(caption_bank), args)
    depth = min(max(args.siglip_top_k, candidate_count), len(caption_bank))
    top_idx = np.argpartition(-scores, depth - 1)[:depth]
    ordered = top_idx[np.argsort(-scores[top_idx])]
    candidates: list[CaptionCandidate] = []
    for idx in ordered[:candidate_count]:
        item = caption_bank[int(idx)]
        candidates.append(
            CaptionCandidate(
                caption_id=item.caption_id,
                cat=item.cat,
                filename=item.filename,
                caption=item.caption,
                text=item.text,
                siglip_score=float(scores[int(idx)]),
            )
        )
    return candidates


def task2_dynamic_gap_cap(sorted_scores: np.ndarray, min_k: int, max_k: int) -> int:
    if sorted_scores.shape[0] <= 1:
        return min_k

    top_gap = float(sorted_scores[0] - sorted_scores[1])
    if top_gap >= 3.0:
        target = 3
    elif top_gap >= 2.0:
        target = 5
    elif top_gap >= 1.0:
        target = 7
    elif top_gap >= 0.5:
        target = 10
    else:
        target = 15
    return min(max(target, min_k), max_k)


def caption_candidate_count_for_scores(
    scores: np.ndarray,
    caption_count: int,
    args: argparse.Namespace,
) -> int:
    if caption_count <= 0:
        raise ValueError("No captions available for candidate selection")

    mode = str(args.candidate_list_mode)
    if mode == "fixed_topk":
        return min(max(1, int(args.rerank_top_k)), caption_count)

    if mode == "dynamic_relative_delta":
        relative_delta = float(args.dynamic_candidate_relative_delta)
        if relative_delta < 0.0:
            raise ValueError("--dynamic-candidate-relative-delta must be non-negative")
        min_k = max(1, int(args.dynamic_candidate_min_k))
        raw_max_k = int(args.dynamic_candidate_max_k)
        max_k = caption_count if raw_max_k <= 0 else min(max(1, raw_max_k), caption_count)
        if min_k > max_k:
            raise ValueError("--dynamic-candidate-min-k must be <= --dynamic-candidate-max-k")
        score_array = np.asarray(scores, dtype=np.float32)
        sorted_scores = np.sort(score_array)[::-1]
        top_score = float(sorted_scores[0])
        threshold = top_score - abs(top_score) * relative_delta
        raw_count = int((score_array >= threshold).sum())
        gap_cap = task2_dynamic_gap_cap(sorted_scores, min_k, max_k)
        count = min(max(raw_count, min_k), gap_cap)
        return min(max(count, min_k), max_k)

    raise ValueError(f"Unsupported candidate list mode: {mode}")


def patch_qwen_reranker_config(model_name_or_path: str, local_files_only: bool, hf_token: str | None) -> str:
    local_path = Path(model_name_or_path).expanduser()
    if not local_path.exists():
        cached = cached_hf_snapshot(model_name_or_path, RERANKER_REQUIRED_FILES, require_model_weights=True)
        if cached is not None:
            local_path = cached
        else:
            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:
                raise RuntimeError("Install huggingface_hub to download or locate the Qwen reranker model.") from exc
            local_path = Path(snapshot_download(repo_id=model_name_or_path, local_files_only=local_files_only, token=hf_token))
    score_head = local_path / "1_CausalScoreHead"
    score_head.mkdir(parents=True, exist_ok=True)
    cfg_path = score_head / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("true_token_id", 9693)
    cfg.setdefault("false_token_id", 2152)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return str(local_path)


def preflight_reranker_requirements(args: argparse.Namespace) -> None:
    if args.mock_reranker or args.final_score_mode == "siglip":
        return
    if args.reranker_backend == "edgellm":
        if args.reranker_edgellm_score_mode == "logit_score" and args.edgellm_runtime_mode != "persistent":
            raise RuntimeError("--reranker-edgellm-score-mode logit_score requires --edgellm-runtime-mode persistent.")
        edgellm_qwen.validate_config(edgellm_qwen.config_from_args(args))
        return
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency for Qwen3-VL reranking: sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    model_path = Path(args.reranker_model).expanduser()
    if (
        args.reranker_local_files_only
        and not model_path.exists()
        and cached_hf_snapshot(args.reranker_model, RERANKER_REQUIRED_FILES, require_model_weights=True) is None
    ):
        raise FileNotFoundError(
            f"Qwen reranker is not cached locally: {args.reranker_model}. "
            "Remove --reranker-local-files-only to download it, pass a local model path with --reranker-model, "
            "or choose a cached alias with --reranker-size."
        )


class QwenVLCaptionReranker:
    def __init__(self, args: argparse.Namespace, hf_token: str | None) -> None:
        torch = require_torch()
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL reranker models are loaded through sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the reranker, but torch.cuda.is_available() is false.")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.torch_dtype]
        patched = patch_qwen_reranker_config(args.reranker_model, args.reranker_local_files_only, hf_token)
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype if args.device == "cuda" else torch.float32}
        if args.reranker_attn_implementation:
            model_kwargs["attn_implementation"] = args.reranker_attn_implementation
        print(f"Loading Qwen caption reranker: {patched}")
        try:
            self.model = CrossEncoder(patched, device=args.device, model_kwargs=model_kwargs)
        except Exception as exc:
            if "attn_implementation" in model_kwargs:
                print(f"Retrying reranker load without attn_implementation after {type(exc).__name__}: {exc}")
                model_kwargs.pop("attn_implementation", None)
                self.model = CrossEncoder(patched, device=args.device, model_kwargs=model_kwargs)
            else:
                raise
        self.torch = torch
        self.activation = torch.nn.Sigmoid()
        self.prompt = (
            "Carefully compare the food image with the candidate caption. "
            "Score high only when the visible dish, ingredients, cooking method, sauce, plating, "
            "and other visual details match the caption."
        )

    @staticmethod
    def load_pair_image(path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(f"Reranker image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def make_pairs(self, image_paths: list[Path], candidate_texts: list[str], start: int, end: int) -> list[tuple[dict[str, Any], dict[str, str]]]:
        return [
            ({"image": self.load_pair_image(image_paths[idx])}, {"text": str(candidate_texts[idx])})
            for idx in range(start, end)
        ]

    @staticmethod
    def close_pair_images(pairs: list[tuple[dict[str, Any], dict[str, str]]]) -> None:
        for image_part, _ in pairs:
            image = image_part.get("image")
            if hasattr(image, "close"):
                image.close()

    def score_pairs(self, image_paths: list[Path], candidate_texts: list[str], batch_size: int) -> list[float]:
        if len(image_paths) != len(candidate_texts):
            raise ValueError("image_paths and candidate_texts must have the same length")
        scores: list[float] = []
        idx = 0
        batch_size = max(1, int(batch_size))
        total_pairs = len(candidate_texts)
        while idx < total_pairs:
            end = min(idx + batch_size, total_pairs)
            pairs = self.make_pairs(image_paths, candidate_texts, idx, end)
            try:
                with self.torch.inference_mode():
                    out = self.model.predict(
                        pairs,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        activation_fn=self.activation,
                        prompt=self.prompt,
                    )
                scores.extend(float(x) for x in np.asarray(out, dtype=np.float32).reshape(-1).tolist())
                print(f"Reranked image-caption pairs {end}/{total_pairs}", flush=True)
                idx = end
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"Reranker OOM; retrying with batch_size={batch_size}", flush=True)
                    continue
                raise
            finally:
                self.close_pair_images(pairs)
        return scores


class EdgeLLMCaptionReranker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = edgellm_qwen.config_from_args(args)
        edgellm_qwen.validate_config(self.config)
        self.scorer = edgellm_qwen.EdgeLLMQwenScorer(
            self.config,
            max_new_tokens=max(1, int(args.reranker_edgellm_max_new_tokens)),
        )
        self.score_mode = str(args.reranker_edgellm_score_mode)
        self.max_new_tokens = 1 if self.score_mode == "logit_score" else max(1, int(args.reranker_edgellm_max_new_tokens))
        self.ambiguous_score = float(args.reranker_edgellm_ambiguous_score)
        self.true_token_id = int(args.reranker_edgellm_true_token_id)
        self.false_token_id = int(args.reranker_edgellm_false_token_id)
        self.doc_max_chars = max(0, int(args.reranker_edgellm_doc_max_chars))
        self.instruction = (
            "Carefully compare the food image with the candidate caption. "
            "Score high only when the visible dish, ingredients, cooking method, sauce, plating, "
            "and other visual details match the caption."
        )
        self.prompt_prefix = (
            "Carefully compare the food image with the candidate caption. "
            "Answer only yes or no. The answer is yes only when the visible dish, ingredients, "
            "cooking method, sauce, plating, and other visual details match the caption.\n"
            "Candidate caption: "
        )

    def parse_score(self, output: str) -> float:
        text = " ".join(str(output).strip().lower().replace("<|im_end|>", " ").split())
        if text.startswith("yes") or text in {"y", "true"}:
            return 1.0
        if text.startswith("no") or text in {"n", "false"}:
            return 0.0
        return self.ambiguous_score

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def truncate_document(self, text: str) -> str:
        clean = " ".join(str(text).strip().split())
        if self.doc_max_chars <= 0 or len(clean) <= self.doc_max_chars:
            return clean
        return clean[: self.doc_max_chars].rsplit(" ", 1)[0].strip() or clean[: self.doc_max_chars].strip()

    def make_reranker_messages(self, image_path: Path, candidate_text: str) -> list[dict[str, Any]]:
        # Mirrors Qwen3-VL-Reranker format_mm_instruction: system prompt plus
        # Instruct/Query/Document fields. The image is the query and the caption is the document.
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Judge whether the Document meets the requirements based on the Query "
                            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<Instruct>: " + self.instruction},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "image", "image": str(Path(image_path).expanduser().resolve())},
                    {"type": "text", "text": "\n<Document>:"},
                    {"type": "text", "text": self.truncate_document(candidate_text)},
                ],
            },
        ]

    def score_pair_logit(self, image_path: Path, candidate_text: str) -> float:
        response = self.scorer.score_token_logits_messages(
            self.make_reranker_messages(image_path, candidate_text),
            [self.true_token_id, self.false_token_id],
            max_new_tokens=1,
        )
        responses = response.get("responses")
        if not isinstance(responses, list) or not responses:
            raise RuntimeError(f"Edge-LLM token-logit response has no responses: {response}")
        token_logits = responses[0].get("token_logits")
        if not isinstance(token_logits, list) or len(token_logits) < 2:
            raise RuntimeError(
                "Edge-LLM token-logit response does not contain token_logits. "
                "Apply patches/tensorrt_edgellm_reranker_logits.patch and rebuild llm_persistent_server."
            )
        true_logit = float(token_logits[0])
        false_logit = float(token_logits[1])
        return self.sigmoid(true_logit - false_logit)

    def score_pair_generated_yes_no(self, image_path: Path, candidate_text: str) -> tuple[float, str]:
        output = self.scorer.score(
            image_path,
            self.prompt_prefix + self.truncate_document(candidate_text),
            max_new_tokens=self.max_new_tokens,
        )
        return self.parse_score(output), output

    def score_pairs(self, image_paths: list[Path], candidate_texts: list[str], batch_size: int) -> list[float]:
        del batch_size
        if len(image_paths) != len(candidate_texts):
            raise ValueError("image_paths and candidate_texts must have the same length")
        scores: list[float] = []
        total_pairs = len(candidate_texts)
        for idx, (image_path, text) in enumerate(zip(image_paths, candidate_texts), start=1):
            if self.score_mode == "logit_score":
                score = self.score_pair_logit(image_path, str(text))
                output = "token_logits"
            else:
                score, output = self.score_pair_generated_yes_no(image_path, str(text))
            scores.append(score)
            print(f"EdgeLLM reranked image-caption pairs {idx}/{total_pairs}: score={score:.3f} output={output[:40]!r}", flush=True)
        return scores

    def close(self) -> None:
        self.scorer.close()


def normalize_score_values(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-12:
        return [0.0 for _ in values]
    return ((arr - float(arr.mean())) / std).astype(np.float32).tolist()


def siglip_top_gap(candidates: list[CaptionCandidate]) -> float:
    """Return the SigLIP score margin between the best and second-best captions."""
    if len(candidates) <= 1:
        return float("inf")
    ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))
    return float(ranked[0].siglip_score - ranked[1].siglip_score)


def siglip_is_confident(candidates: list[CaptionCandidate], args: argparse.Namespace) -> bool:
    return siglip_top_gap(candidates) >= float(args.siglip_keep_gap)


def final_candidate_scores(candidates: list[CaptionCandidate], args: argparse.Namespace) -> dict[int, float]:
    if not candidates:
        return {}
    if args.final_score_mode == "siglip":
        return {idx: float(candidate.siglip_score) for idx, candidate in enumerate(candidates)}
    if args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
        return {idx: float(candidate.siglip_score) for idx, candidate in enumerate(candidates)}

    rerank_values = [float(candidate.rerank_score if candidate.rerank_score is not None else candidate.siglip_score) for candidate in candidates]
    if args.final_score_mode in {"rerank", "siglip_guarded"}:
        return {idx: rerank_values[idx] for idx in range(len(candidates))}
    if args.final_score_mode == "blend_z":
        siglip_z = normalize_score_values([float(candidate.siglip_score) for candidate in candidates])
        rerank_z = normalize_score_values(rerank_values)
        weight = min(1.0, max(0.0, float(args.rerank_weight)))
        return {idx: float(weight * rerank_z[idx] + (1.0 - weight) * siglip_z[idx]) for idx in range(len(candidates))}
    raise ValueError(f"Unsupported final-score mode: {args.final_score_mode}")


def candidate_to_dict(candidate: CaptionCandidate, final_score: float | None = None) -> dict[str, Any]:
    return {
        "caption_id": candidate.caption_id,
        "cat": candidate.cat,
        "filename": candidate.filename,
        "caption": candidate.caption,
        "text": candidate.text,
        "siglip_score": candidate.siglip_score,
        "rerank_score": candidate.rerank_score,
        "final_score": final_score,
    }


def row_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def run_pipeline_per_query_measurement(args: argparse.Namespace) -> dict[str, Any]:
    t0_total = time.perf_counter()
    ram_monitor = MemoryMonitor(enabled=args.measure_ram, interval_ms=args.ram_sample_interval_ms)
    ram_monitor.start()
    hf_token = configure_hf_token(args)

    caption_bank_json = args.caption_bank_json or args.evaluation_json

    if args.inference_only:
        evaluation_rows: list[Food500Row] = []
        manifest_rows: list[Food500Row] = []
        gt_by_filename: dict[str, Food500Row] = {}
        caption_bank = read_caption_bank_items(caption_bank_json, args.caption_text_mode)
    else:
        evaluation_rows = read_food500_rows(args.evaluation_json)
        manifest_rows = read_manifest(args.manifest)
        gt_by_filename = {row.filename: row for row in evaluation_rows}
        caption_bank = read_caption_bank_items(caption_bank_json, args.caption_text_mode)
    print(f"Candidate captions: {len(caption_bank)}")
    preflight_reranker_requirements(args)

    single_image_name = None
    if args.image is not None:
        single_image_name, single_image_path = resolve_image_name(args.image, args.image_dir)
        target_rows = [gt_by_filename.get(single_image_name, Food500Row(cat="", filename=single_image_name, caption=""))]
        image_paths = [single_image_path]
    else:
        target_rows = select_inference_rows(args) if args.inference_only else select_eval_rows(args, evaluation_rows, manifest_rows)
        image_paths = [args.image_dir / row.filename for row in target_rows]
    image_names = [row.filename for row in target_rows]

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    embedder = build_siglip2_embedder(args)
    timings["load_siglip2_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    caption_embeddings, caption_cache = load_or_build_caption_embeddings(embedder, caption_bank, args)
    timings["caption_embeddings_sec"] = time.perf_counter() - t0

    reranker: Any | None = None
    t0 = time.perf_counter()
    if args.mock_reranker or args.final_score_mode == "siglip":
        reranker_mode = "mock_siglip_scores" if args.mock_reranker else "skipped_siglip_final_score"
    elif args.reranker_backend == "edgellm":
        reranker = EdgeLLMCaptionReranker(args)
        reranker_mode = (
            f"edgellm_{args.reranker_edgellm_score_mode}:{args.edgellm_model_name}"
            if args.final_score_mode != "siglip_guarded"
            else f"siglip_guarded:edgellm_{args.reranker_edgellm_score_mode}:{args.edgellm_model_name}"
        )
    else:
        reranker = QwenVLCaptionReranker(args, hf_token)
        reranker_mode = (
            f"sentence_transformers:{args.reranker_model}"
            if args.final_score_mode != "siglip_guarded"
            else f"siglip_guarded:sentence_transformers:{args.reranker_model}"
        )
    timings["load_reranker_sec"] = time.perf_counter() - t0

    power_monitor = PowerMonitor(enabled=args.measure_power, interval_ms=args.power_sample_interval_ms)
    power_monitor.start()
    measurement_t0 = time.perf_counter()
    measured_wall_sec = 0.0
    power_summary: dict[str, Any] = {}

    traces: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    correct_caption = 0
    correct_class = 0
    siglip_correct_caption = 0
    truth_caption_in_candidates = 0
    rerank_pair_count = 0
    rerank_skipped_by_siglip_guard = 0
    latencies_sec: list[float] = []

    try:
        for idx, (row, image_path) in enumerate(zip(target_rows, image_paths), start=1):
            query_t0 = time.perf_counter()

            t0 = time.perf_counter()
            image_embedding = embedder.encode_image(image_path)
            scores = 100.0 * (image_embedding @ caption_embeddings.T)
            candidates = build_caption_recall(scores, caption_bank, args)
            siglip_image_and_recall_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            rerank_applied = False
            if args.mock_reranker or args.final_score_mode == "siglip":
                for candidate in candidates:
                    candidate.rerank_score = candidate.siglip_score
            elif args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
                rerank_skipped_by_siglip_guard += 1
            else:
                if reranker is None:
                    raise RuntimeError("Reranker was not initialized")
                pair_paths = [image_path for _ in candidates]
                pair_texts = [candidate.text for candidate in candidates]
                rerank_pair_count += len(pair_paths)
                rerank_applied = True
                rerank_scores = reranker.score_pairs(pair_paths, pair_texts, args.reranker_batch_size)
                for candidate, score in zip(candidates, rerank_scores):
                    candidate.rerank_score = score
            rerank_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            final_scores = final_candidate_scores(candidates, args)
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (
                    -float(final_scores[int(item[0])]),
                    -float(item[1].siglip_score),
                    item[1].filename,
                    item[1].caption_id,
                ),
            )
            ranked_candidates = [candidate for _, candidate in ranked]
            ranked_scores = {id(candidate): float(final_scores[candidate_idx]) for candidate_idx, candidate in ranked}
            siglip_ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))

            pred = ranked_candidates[0] if ranked_candidates else None
            siglip_pred = siglip_ranked[0] if siglip_ranked else None

            is_caption_correct = None if args.inference_only else bool(pred and row.filename and pred.filename == row.filename)
            is_class_correct = None if args.inference_only else bool(pred and row.cat and pred.cat == row.cat)
            is_siglip_caption_correct = None if args.inference_only else bool(siglip_pred and row.filename and siglip_pred.filename == row.filename)
            has_truth_caption = None if args.inference_only else (any(candidate.filename == row.filename for candidate in candidates) if row.filename else False)

            if not args.inference_only:
                correct_caption += int(bool(is_caption_correct))
                correct_class += int(bool(is_class_correct))
                siglip_correct_caption += int(bool(is_siglip_caption_correct))
                truth_caption_in_candidates += int(bool(has_truth_caption))
            final_selection_sec = time.perf_counter() - t0
            total_image_sec = time.perf_counter() - query_t0
            latencies_sec.append(total_image_sec)

            trace = {
                "image": row.filename,
                "image_path": str(args.image_dir / row.filename) if single_image_name is None else str(image_path),
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                    "prediction_caption": pred.caption if pred else "",
                    "caption_correct": is_caption_correct,
                    "class_correct": is_class_correct,
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption": siglip_pred.caption if siglip_pred else "",
                "siglip_top1_caption_correct": is_siglip_caption_correct,
                "truth_caption_in_candidates": has_truth_caption,
                "siglip_top_gap": siglip_top_gap(candidates),
                "rerank_applied": rerank_applied,
                "timings_sec": {
                    "siglip_image_and_recall_sec": siglip_image_and_recall_sec,
                    "rerank_sec": rerank_sec,
                    "final_selection_sec": final_selection_sec,
                    "total_image_sec": total_image_sec,
                },
                "ranked_candidates": [candidate_to_dict(candidate, ranked_scores[id(candidate)]) for candidate in ranked_candidates],
            }
            traces.append(trace)
            csv_rows.append(
                {
                    "image": row.filename,
                    "truth_cat": row.cat,
                    "truth_caption": row.caption,
                    "prediction_filename": pred.filename if pred else "",
                    "prediction_cat": pred.cat if pred else "",
                    "prediction_caption": pred.caption if pred else "",
                    "caption_correct": "" if is_caption_correct is None else int(is_caption_correct),
                    "class_correct": "" if is_class_correct is None else int(is_class_correct),
                    "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                    "siglip_top1_caption_correct": "" if is_siglip_caption_correct is None else int(is_siglip_caption_correct),
                    "truth_caption_in_candidates": "" if has_truth_caption is None else int(has_truth_caption),
                    "siglip_top_gap": f"{siglip_top_gap(candidates):.6f}",
                    "rerank_applied": int(rerank_applied),
                    "prediction_rerank_score": f"{float(pred.rerank_score if pred and pred.rerank_score is not None else 0.0):.6f}",
                    "prediction_siglip_score": f"{float(pred.siglip_score if pred else 0.0):.6f}",
                    "prediction_final_score": f"{float(ranked_scores.get(id(pred), 0.0) if pred else 0.0):.6f}",
                    "siglip_image_and_recall_sec": f"{siglip_image_and_recall_sec:.6f}",
                    "rerank_sec": f"{rerank_sec:.6f}",
                    "final_selection_sec": f"{final_selection_sec:.6f}",
                    "total_image_sec": f"{total_image_sec:.6f}",
                    "ranked_candidates": "; ".join(f"{candidate.filename}:{ranked_scores[id(candidate)]:.4f}" for candidate in ranked_candidates),
                }
            )
            print(
                f"[{idx}/{len(target_rows)}] {row.filename} "
                f"{'prediction=' + pred.caption[:80] if args.inference_only and pred else 'caption_top1=' + str(int(bool(is_caption_correct))) + ' class_top1=' + str(int(bool(is_class_correct)))} "
                f"time={total_image_sec:.3f}s rerank={rerank_sec:.3f}s"
            )
    finally:
        measured_wall_sec = time.perf_counter() - measurement_t0
        power_summary = power_monitor.stop(wall_sec=measured_wall_sec)
        if reranker is not None and hasattr(reranker, "close"):
            reranker.close()

    ram_summary = ram_monitor.stop()

    total = len(target_rows)
    if args.inference_only:
        metrics = {
            "rows": total,
            "metric_available": False,
            "caption_accuracy": None,
            "caption_correct": None,
            "class_accuracy": None,
            "class_correct": None,
            "siglip_top1_caption_accuracy": None,
            "siglip_top1_caption_correct": None,
            "truth_caption_in_rerank_candidates_rate": None,
            "truth_caption_in_rerank_candidates": None,
            "rerank_pair_count": rerank_pair_count,
            "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
        }
    else:
        metrics = {
            "rows": total,
            "metric_available": True,
            "caption_accuracy": row_accuracy(correct_caption, total),
            "caption_correct": correct_caption,
            "class_accuracy": row_accuracy(correct_class, total),
            "class_correct": correct_class,
            "siglip_top1_caption_accuracy": row_accuracy(siglip_correct_caption, total),
            "siglip_top1_caption_correct": siglip_correct_caption,
            "truth_caption_in_rerank_candidates_rate": row_accuracy(truth_caption_in_candidates, total),
            "truth_caption_in_rerank_candidates": truth_caption_in_candidates,
            "rerank_pair_count": rerank_pair_count,
            "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
        }
    timings["measured_query_wall_sec"] = measured_wall_sec
    timings["total_sec"] = time.perf_counter() - t0_total

    benchmark = build_benchmark_summary(
        task_name="task2_caption_inference_only" if args.inference_only else "task2_food500_caption",
        query_count=total,
        latencies_sec=latencies_sec,
        measured_wall_sec=measured_wall_sec,
        task_metric_name="not_available_no_ground_truth" if args.inference_only else "top1_caption_accuracy",
        task_metric_value=None if args.inference_only else float(metrics["caption_accuracy"]),
        power_summary=power_summary,
        w_config=args.w_config,
        nvpmodel=query_nvpmodel(),
        extra_task_metrics={
            "metric_available": not args.inference_only,
            "class_top1_accuracy": metrics["class_accuracy"],
            "siglip_top1_caption_accuracy": metrics["siglip_top1_caption_accuracy"],
            "truth_caption_in_rerank_candidates_rate": metrics["truth_caption_in_rerank_candidates_rate"],
        },
    )

    return {
        "schema_version": "orin_task2_caption_reranker_per_query_measurement_v1",
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            **siglip_runtime_info(embedder),
            "reranker_size": args.reranker_size,
            "reranker_model": args.reranker_model,
            "reranker_backend": args.reranker_backend,
            "reranker_mode": reranker_mode,
            "reranker_edgellm_score_mode": args.reranker_edgellm_score_mode if args.reranker_backend == "edgellm" else None,
            "reranker_edgellm_doc_max_chars": args.reranker_edgellm_doc_max_chars if args.reranker_backend == "edgellm" else None,
            "edgellm_model_name": args.edgellm_model_name if args.reranker_backend == "edgellm" else None,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "caption_text_mode": args.caption_text_mode,
            "siglip_image_size": args.siglip_image_size,
            "siglip_top_k": args.siglip_top_k,
            "rerank_top_k": args.rerank_top_k,
            "candidate_list_mode": args.candidate_list_mode,
            "dynamic_candidate_relative_delta": args.dynamic_candidate_relative_delta,
            "dynamic_candidate_min_k": args.dynamic_candidate_min_k,
            "dynamic_candidate_max_k": args.dynamic_candidate_max_k,
            "final_score_mode": args.final_score_mode,
            "rerank_weight": args.rerank_weight,
            "siglip_keep_gap": args.siglip_keep_gap,
            "eval_first": args.eval_first,
            "eval_all": args.eval_all,
            "seed": args.seed,
            "measure": True,
            "inference_only": args.inference_only,
        },
        "paths": {
            "image_dir": str(args.image_dir),
            "manifest": str(args.manifest),
            "images_list": str(args.images_list) if args.images_list is not None else "",
            "evaluation_json": str(args.evaluation_json),
            "caption_bank_json": str(caption_bank_json),
        },
        "caption_bank_count": len(caption_bank),
        "caption_cache": caption_cache,
        "image_cache": {"enabled": False, "reason": "per_query_measurement_encodes_each_image"},
        "metrics": metrics,
        "benchmark": benchmark,
        "ram": ram_summary,
        "timings_sec": timings,
        "traces": traces,
        "csv_rows": csv_rows,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.measure or args.measure_power:
        return run_pipeline_per_query_measurement(args)
    if args.inference_only:
        raise SystemExit("--inference-only currently requires --measure or --measure-power.")

    t0_total = time.perf_counter()
    hf_token = configure_hf_token(args)

    evaluation_rows = read_food500_rows(args.evaluation_json)
    caption_bank_json = args.caption_bank_json or args.evaluation_json
    caption_bank_rows = read_food500_rows(caption_bank_json)
    manifest_rows = read_manifest(args.manifest)

    gt_by_filename = {row.filename: row for row in evaluation_rows}
    caption_bank = build_caption_bank(caption_bank_rows, args.caption_text_mode)
    print(f"Candidate captions: {len(caption_bank)}")
    preflight_reranker_requirements(args)

    single_image_name = None
    if args.image is not None:
        single_image_name, single_image_path = resolve_image_name(args.image, args.image_dir)
        target_rows = [gt_by_filename.get(single_image_name, Food500Row(cat="", filename=single_image_name, caption=""))]
        image_paths = [single_image_path]
    else:
        target_rows = select_eval_rows(args, evaluation_rows, manifest_rows)
        image_paths = [args.image_dir / row.filename for row in target_rows]
    image_names = [row.filename for row in target_rows]

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    embedder = build_siglip2_embedder(args)
    timings["load_siglip2_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    caption_embeddings, caption_cache = load_or_build_caption_embeddings(embedder, caption_bank, args)
    timings["caption_embeddings_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image_embeddings, image_cache = load_or_build_image_embeddings(embedder, image_names, args)
    timings["image_embeddings_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    score_matrix = 100.0 * (image_embeddings @ caption_embeddings.T)
    recall_by_image: dict[str, list[CaptionCandidate]] = {}
    for row_idx, image_name in enumerate(image_names):
        recall_by_image[image_name] = build_caption_recall(score_matrix[row_idx], caption_bank, args)
    timings["siglip_caption_recall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rerank_pair_count = 0
    rerank_skipped_by_siglip_guard = 0
    if args.mock_reranker or args.final_score_mode == "siglip":
        for candidates in recall_by_image.values():
            for candidate in candidates:
                candidate.rerank_score = candidate.siglip_score
        reranker_mode = "mock_siglip_scores" if args.mock_reranker else "skipped_siglip_final_score"
        if args.final_score_mode == "siglip":
            rerank_skipped_by_siglip_guard = len(recall_by_image)
    else:
        pair_paths: list[Path] = []
        pair_texts: list[str] = []
        pair_meta: list[tuple[str, int]] = []
        for image_name, image_path in zip(image_names, image_paths):
            candidates = recall_by_image[image_name]
            if args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args):
                rerank_skipped_by_siglip_guard += 1
                continue
            for cand_idx, candidate in enumerate(candidates):
                pair_paths.append(image_path)
                pair_texts.append(candidate.text)
                pair_meta.append((image_name, cand_idx))
        rerank_pair_count = len(pair_paths)
        if pair_paths:
            if args.reranker_backend == "edgellm":
                reranker = EdgeLLMCaptionReranker(args)
            else:
                reranker = QwenVLCaptionReranker(args, hf_token)
            scores = reranker.score_pairs(pair_paths, pair_texts, args.reranker_batch_size)
            if hasattr(reranker, "close"):
                reranker.close()
            for (image_name, cand_idx), score in zip(pair_meta, scores):
                recall_by_image[image_name][cand_idx].rerank_score = score
            if args.reranker_backend == "edgellm":
                reranker_mode = (
                    f"edgellm_{args.reranker_edgellm_score_mode}:{args.edgellm_model_name}"
                    if args.final_score_mode != "siglip_guarded"
                    else f"siglip_guarded:edgellm_{args.reranker_edgellm_score_mode}:{args.edgellm_model_name}"
                )
            else:
                reranker_mode = args.reranker_model if args.final_score_mode != "siglip_guarded" else f"siglip_guarded:{args.reranker_model}"
        else:
            reranker_mode = "skipped_all_by_siglip_guard" if args.final_score_mode == "siglip_guarded" else "skipped_no_pairs"
    timings["rerank_sec"] = time.perf_counter() - t0

    traces: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    correct_caption = 0
    correct_class = 0
    siglip_correct_caption = 0
    truth_caption_in_candidates = 0

    for row in target_rows:
        candidates = recall_by_image[row.filename]
        final_scores = final_candidate_scores(candidates, args)
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                -float(final_scores[int(item[0])]),
                -float(item[1].siglip_score),
                item[1].filename,
                item[1].caption_id,
            ),
        )
        ranked_candidates = [candidate for _, candidate in ranked]
        ranked_scores = {id(candidate): float(final_scores[idx]) for idx, candidate in ranked}
        siglip_ranked = sorted(candidates, key=lambda c: (-float(c.siglip_score), c.filename, c.caption_id))

        pred = ranked_candidates[0] if ranked_candidates else None
        siglip_pred = siglip_ranked[0] if siglip_ranked else None

        is_caption_correct = bool(pred and row.filename and pred.filename == row.filename)
        is_class_correct = bool(pred and row.cat and pred.cat == row.cat)
        is_siglip_caption_correct = bool(siglip_pred and row.filename and siglip_pred.filename == row.filename)
        has_truth_caption = any(candidate.filename == row.filename for candidate in candidates) if row.filename else False

        correct_caption += int(is_caption_correct)
        correct_class += int(is_class_correct)
        siglip_correct_caption += int(is_siglip_caption_correct)
        truth_caption_in_candidates += int(has_truth_caption)

        traces.append(
            {
                "image": row.filename,
                "image_path": str(args.image_dir / row.filename) if single_image_name is None else str(image_paths[0]),
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                "prediction_caption": pred.caption if pred else "",
                "caption_correct": is_caption_correct,
                "class_correct": is_class_correct,
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption": siglip_pred.caption if siglip_pred else "",
                "siglip_top1_caption_correct": is_siglip_caption_correct,
                "truth_caption_in_candidates": has_truth_caption,
                "siglip_top_gap": siglip_top_gap(candidates),
                "rerank_applied": not (args.mock_reranker or args.final_score_mode == "siglip" or (args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args))),
                "ranked_candidates": [candidate_to_dict(candidate, ranked_scores[id(candidate)]) for candidate in ranked_candidates],
            }
        )
        csv_rows.append(
            {
                "image": row.filename,
                "truth_cat": row.cat,
                "truth_caption": row.caption,
                "prediction_filename": pred.filename if pred else "",
                "prediction_cat": pred.cat if pred else "",
                "prediction_caption": pred.caption if pred else "",
                "caption_correct": int(is_caption_correct),
                "class_correct": int(is_class_correct),
                "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
                "siglip_top1_caption_correct": int(is_siglip_caption_correct),
                "truth_caption_in_candidates": int(has_truth_caption),
                "siglip_top_gap": f"{siglip_top_gap(candidates):.6f}",
                "rerank_applied": int(not (args.mock_reranker or args.final_score_mode == "siglip" or (args.final_score_mode == "siglip_guarded" and siglip_is_confident(candidates, args)))),
                "prediction_rerank_score": f"{float(pred.rerank_score if pred and pred.rerank_score is not None else 0.0):.6f}",
                "prediction_siglip_score": f"{float(pred.siglip_score if pred else 0.0):.6f}",
                "prediction_final_score": f"{float(ranked_scores.get(id(pred), 0.0) if pred else 0.0):.6f}",
                "ranked_candidates": "; ".join(
                    f"{candidate.filename}:{ranked_scores[id(candidate)]:.4f}"
                    for candidate in ranked_candidates
                ),
            }
        )

    total = len(target_rows)
    metrics = {
        "rows": total,
        "caption_accuracy": row_accuracy(correct_caption, total),
        "caption_correct": correct_caption,
        "class_accuracy": row_accuracy(correct_class, total),
        "class_correct": correct_class,
        "siglip_top1_caption_accuracy": row_accuracy(siglip_correct_caption, total),
        "siglip_top1_caption_correct": siglip_correct_caption,
        "truth_caption_in_rerank_candidates_rate": row_accuracy(truth_caption_in_candidates, total),
        "truth_caption_in_rerank_candidates": truth_caption_in_candidates,
        "rerank_pair_count": rerank_pair_count,
        "rerank_skipped_by_siglip_guard": rerank_skipped_by_siglip_guard,
    }
    timings["total_sec"] = time.perf_counter() - t0_total

    return {
        "schema_version": "orin_task2_caption_reranker_guarded_v1",
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            **siglip_runtime_info(embedder),
            "reranker_size": args.reranker_size,
            "reranker_model": args.reranker_model,
            "reranker_backend": args.reranker_backend,
            "reranker_mode": reranker_mode,
            "reranker_edgellm_score_mode": args.reranker_edgellm_score_mode if args.reranker_backend == "edgellm" else None,
            "reranker_edgellm_doc_max_chars": args.reranker_edgellm_doc_max_chars if args.reranker_backend == "edgellm" else None,
            "edgellm_model_name": args.edgellm_model_name if args.reranker_backend == "edgellm" else None,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "caption_text_mode": args.caption_text_mode,
            "siglip_image_size": args.siglip_image_size,
            "siglip_top_k": args.siglip_top_k,
            "rerank_top_k": args.rerank_top_k,
            "candidate_list_mode": args.candidate_list_mode,
            "dynamic_candidate_relative_delta": args.dynamic_candidate_relative_delta,
            "dynamic_candidate_min_k": args.dynamic_candidate_min_k,
            "dynamic_candidate_max_k": args.dynamic_candidate_max_k,
            "final_score_mode": args.final_score_mode,
            "rerank_weight": args.rerank_weight,
            "siglip_keep_gap": args.siglip_keep_gap,
            "eval_first": args.eval_first,
            "eval_all": args.eval_all,
            "seed": args.seed,
        },
        "paths": {
            "image_dir": str(args.image_dir),
            "manifest": str(args.manifest),
            "images_list": str(args.images_list) if args.images_list is not None else "",
            "evaluation_json": str(args.evaluation_json),
            "caption_bank_json": str(caption_bank_json),
        },
        "caption_bank_count": len(caption_bank),
        "caption_cache": caption_cache,
        "image_cache": image_cache,
        "metrics": metrics,
        "timings_sec": timings,
        "traces": traces,
        "csv_rows": csv_rows,
    }


def resolve_output_paths(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    rows = report["metrics"]["rows"]
    if args.image is not None:
        stem = Path(str(args.image)).stem
        default_json = args.output_dir / f"{stem}_caption_alignment.json"
        default_csv = args.output_dir / f"{stem}_caption_alignment.csv"
    elif args.eval_all:
        default_json = args.output_dir / "eval_all_caption_alignment.json"
        default_csv = args.output_dir / "eval_all_caption_alignment.csv"
    elif args.eval_first:
        default_json = args.output_dir / f"eval_first_{rows}_caption_alignment.json"
        default_csv = args.output_dir / f"eval_first_{rows}_caption_alignment.csv"
    else:
        default_json = args.output_dir / f"eval_{rows}_caption_alignment.json"
        default_csv = args.output_dir / f"eval_{rows}_caption_alignment.csv"

    out_json = args.output_json or default_json
    out_csv = args.predictions_csv or default_csv
    return out_json, out_csv


def print_final_summary(report: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    metrics = report["metrics"]
    if metrics.get("metric_available") is False:
        print(
            "Final metrics: inference_only_no_ground_truth "
            f"qwen_pairs={metrics['rerank_pair_count']} "
            f"siglip_guard_skips={metrics['rerank_skipped_by_siglip_guard']} "
            f"rows={metrics['rows']}",
            flush=True,
        )
    else:
        print(
            "Final metrics: "
            f"caption_accuracy={metrics['caption_accuracy']:.4f} "
            f"class_accuracy={metrics['class_accuracy']:.4f} "
            f"siglip_top1_caption={metrics['siglip_top1_caption_accuracy']:.4f} "
            f"truth_caption_in_candidates={metrics['truth_caption_in_rerank_candidates_rate']:.4f} "
            f"qwen_pairs={metrics['rerank_pair_count']} "
            f"siglip_guard_skips={metrics['rerank_skipped_by_siglip_guard']} "
            f"rows={metrics['rows']}",
            flush=True,
        )
    if "benchmark" in report:
        print_benchmark_summary(report["benchmark"])
    print(f"Planned Evaluation JSON: {out_json}", flush=True)
    print(f"Planned Predictions CSV: {out_csv}", flush=True)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json, out_csv = resolve_output_paths(report, args)
    json_payload = dict(report)
    csv_rows = json_payload.pop("csv_rows")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "truth_cat",
        "truth_caption",
        "prediction_filename",
        "prediction_cat",
        "prediction_caption",
        "caption_correct",
        "class_correct",
        "siglip_top1_filename",
        "siglip_top1_caption_correct",
        "truth_caption_in_candidates",
        "siglip_top_gap",
        "rerank_applied",
        "prediction_rerank_score",
        "prediction_siglip_score",
        "prediction_final_score",
        "siglip_image_and_recall_sec",
        "rerank_sec",
        "final_selection_sec",
        "total_image_sec",
        "ranked_candidates",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return out_json, out_csv


def main() -> None:
    args = parse_args()
    report = run_pipeline(args)
    out_json, out_csv = resolve_output_paths(report, args)
    print_final_summary(report, out_json, out_csv)
    out_json, out_csv = write_outputs(report, args)
    print(f"Evaluation JSON written: {out_json}")
    print(f"Predictions CSV written: {out_csv}")


if __name__ == "__main__":
    main()
