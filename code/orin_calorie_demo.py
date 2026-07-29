#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_CALORIES_CSV = ROOT / "dataset/captions_cleaned_calories_realistic_average_portions.csv"
DEFAULT_CLEANED_JSON = ROOT / "labels/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json"
DEFAULT_CAPTIONS = ROOT / "dataset/captions_cleaned.txt"
DEFAULT_SIGLIP_MODEL = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DEFAULT_SIGLIP_PRETRAINED = ""
DEFAULT_SIGLIP_ONNX_DIR = ROOT / "models/onnx/ViT-gopt-16-SigLIP2-384-ONNX"
DEFAULT_SIGLIP_TRT_CACHE_DIR = ROOT / "models/onnx/trt_cache"
DEFAULT_SIGLIP_TRT_ENGINE = Path("/home/danilopau/models_quant/siglip2_visual/visual_fp16.engine")
DEFAULT_TEXT_CACHE = ROOT / "embeddings/orin_siglip2_text_feats_cache.npz"
DEFAULT_CALORIE_CANDIDATE_TOP_K = 52
DEFAULT_CALORIE_CANDIDATE_LIST_MODE = "dynamic_relative_delta"
DEFAULT_CALORIE_DYNAMIC_CANDIDATE_RELATIVE_DELTA = 0.5
DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MIN_K = 12
DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MAX_K = 150
DEFAULT_CALORIE_SIGLIP_FILTER = True
CALORIE_ALIASES = {
    "parmesan": "parmesan cheese",
    "parmigiano": "parmesan cheese",
}
BUILTIN_CALORIES_PER_100G = {
    "olive oil": 884.0,
}
BUILTIN_AVERAGE_PORTIONS = {
    "olive oil": (14.0, 124.0),
}
PORTION_FACTORS = {
    "none": 0.0,
    "garnish": 0.25,
    "small": 0.5,
    "normal": 1.0,
    "large": 1.5,
    "double": 2.0,
}
PORTION_CATEGORIES = tuple(PORTION_FACTORS)
PORTION_CODE_TO_CATEGORY = {
    0: "none",
    1: "garnish",
    2: "small",
    3: "normal",
    4: "large",
    5: "double",
}
CONFIDENCE_CODE_TO_VALUE = {
    0: 0.30,
    1: 0.65,
    2: 0.95,
}
ABUNDANCE_PORTION_CATEGORIES = {"large", "double"}
MAX_INGREDIENT_PREDICTIONS = 8
COUNTING_LOGIC_LEGACY = "legacy"
COUNTING_LOGIC_CUT_AWARE = "cut-aware"
COUNTING_LOGIC_CHOICES = (COUNTING_LOGIC_LEGACY, COUNTING_LOGIC_CUT_AWARE)
CUT_WHOLE = 0
CUT_HALF = 1
CUT_QUARTER = 2
CUT_WEDGE = 3
CUT_SLICE_UNCLEAR = 4
CUT_CHOPPED_PILE = 5
CUT_LABELS = {
    CUT_WHOLE: "whole",
    CUT_HALF: "half",
    CUT_QUARTER: "quarter",
    CUT_WEDGE: "wedge_or_large_piece",
    CUT_SLICE_UNCLEAR: "slice_unclear",
    CUT_CHOPPED_PILE: "chopped_diced_pile",
}
COUNT_CONFIDENCE_THRESHOLD = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.75


def hf_hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_has_required_files(snapshot: Path, required_files: tuple[str, ...]) -> bool:
    if not all((snapshot / name).exists() for name in required_files):
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.exists():
            continue
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_payload.get("weight_map")
            if not isinstance(weight_map, dict):
                return False
            shard_names = {str(name) for name in weight_map.values()}
        except Exception:
            return False
        if not shard_names or not all((snapshot / name).exists() for name in shard_names):
            return False
    return True


def cached_hf_snapshot(repo_id: str, required_files: tuple[str, ...] = ()) -> Path | None:
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
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(snapshots)
    except OSError:
        pass

    seen: set[Path] = set()
    for snapshot in candidates:
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if snapshot.is_dir() and snapshot_has_required_files(snapshot, required_files):
            return snapshot
    return None


def resolve_transformers_source(model_id: str) -> tuple[str, bool]:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return str(local_path), True
    if "/" not in model_id:
        return model_id, False
    snapshot = cached_hf_snapshot(model_id, ("config.json", "tokenizer.json"))
    if snapshot is None:
        return model_id, False
    return str(snapshot), True


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "orin_calorie_demo.py requires PyTorch plus transformers, accelerate, qwen-vl-utils, and pillow."
        ) from exc
    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen-VL composition-estimation backend for the calorie task. Qwen outputs "
            "ingredient IDs, whole-object-equivalent counts when reliable, and portion categories; "
            "Python computes calories from per-object and average-portion tables."
        )
    )
    parser.add_argument("--image", type=Path, default=None, help="Run one image and write a trace JSON.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--ingredients-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--quantity-mode", default="", help=argparse.SUPPRESS)
    parser.add_argument("--calories-csv", type=Path, default=DEFAULT_CALORIES_CSV)
    parser.add_argument("--cleaned-json", type=Path, default=DEFAULT_CLEANED_JSON)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-pretrained", default=DEFAULT_SIGLIP_PRETRAINED)
    parser.add_argument(
        "--siglip-backend",
        choices=("auto", "pytorch", "onnx", "tensorrt_engine"),
        default="tensorrt_engine",
        help=(
            "SigLIP2 runtime. tensorrt_engine loads only the serialized FP16 visual encoder "
            "and requires precomputed text embeddings."
        ),
    )
    parser.add_argument("--siglip-onnx-dir", type=Path, default=DEFAULT_SIGLIP_ONNX_DIR)
    parser.add_argument("--siglip-onnx-provider", choices=("tensorrt", "cuda", "cpu"), default="tensorrt")
    parser.add_argument("--siglip-trt-cache-dir", type=Path, default=DEFAULT_SIGLIP_TRT_CACHE_DIR)
    parser.add_argument("--siglip-trt-engine", type=Path, default=DEFAULT_SIGLIP_TRT_ENGINE)
    parser.add_argument("--text-cache", type=Path, default=DEFAULT_TEXT_CACHE)
    parser.add_argument("--no-text-cache", action="store_true")
    parser.add_argument("--text-cache-require-hit", action="store_true")
    parser.add_argument(
        "--calorie-candidate-top-k",
        "--top-k",
        dest="calorie_candidate_top_k",
        type=int,
        default=DEFAULT_CALORIE_CANDIDATE_TOP_K,
        help=(
            "Number of SigLIP2 candidate ingredients allowed in the calorie-composition prompt "
            "when --calorie-candidate-list-mode fixed_topk is used."
        ),
    )
    parser.add_argument(
        "--calorie-candidate-list-mode",
        "--candidate-list-mode",
        dest="calorie_candidate_list_mode",
        choices=("fixed_topk", "dynamic_relative_delta"),
        default=DEFAULT_CALORIE_CANDIDATE_LIST_MODE,
        help=(
            "How to build the SigLIP2 calorie candidate list. fixed_topk keeps "
            "--calorie-candidate-top-k labels. dynamic_relative_delta keeps labels close "
            "to the top SigLIP score, clamped by --calorie-dynamic-candidate-min-k and "
            "--calorie-dynamic-candidate-max-k. Default matches the Task 1 dynamic policy."
        ),
    )
    parser.add_argument(
        "--calorie-dynamic-candidate-relative-delta",
        "--dynamic-candidate-relative-delta",
        dest="calorie_dynamic_candidate_relative_delta",
        type=float,
        default=DEFAULT_CALORIE_DYNAMIC_CANDIDATE_RELATIVE_DELTA,
        help="For dynamic_relative_delta, keep scores >= top_score - abs(top_score) * this value.",
    )
    parser.add_argument(
        "--calorie-dynamic-candidate-min-k",
        "--dynamic-candidate-min-k",
        dest="calorie_dynamic_candidate_min_k",
        type=int,
        default=DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MIN_K,
        help="Minimum candidate count for dynamic_relative_delta.",
    )
    parser.add_argument(
        "--calorie-dynamic-candidate-max-k",
        "--dynamic-candidate-max-k",
        dest="calorie_dynamic_candidate_max_k",
        type=int,
        default=DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MAX_K,
        help="Maximum candidate count for dynamic_relative_delta. Use 0 for no cap.",
    )
    parser.add_argument(
        "--calorie-siglip-filter",
        dest="calorie_siglip_filter",
        action="store_true",
        default=DEFAULT_CALORIE_SIGLIP_FILTER,
        help="Enable SigLIP2 top-k candidate filtering before the Qwen calorie-composition prompt. This is the default.",
    )
    parser.add_argument(
        "--no-calorie-siglip-filter",
        dest="calorie_siglip_filter",
        action="store_false",
        help="Use the full calorie ingredient table in the Qwen calorie-composition prompt.",
    )
    parser.add_argument("--defer-calorie-siglip-filter", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vlm-model",
        "--qwen-model",
        dest="qwen_model",
        default=DEFAULT_QWEN_MODEL,
        help=(
            "Qwen-VL instruction model used for dish composition estimation."
        ),
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--torch-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--vlm-max-new-tokens", "--qwen-max-new-tokens", dest="qwen_max_new_tokens", type=int, default=96)
    parser.add_argument("--vlm-min-pixels", "--qwen-min-pixels", dest="qwen_min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--vlm-max-pixels", "--qwen-max-pixels", dest="qwen_max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument(
        "--calorie-counting-logic",
        choices=COUNTING_LOGIC_CHOICES,
        default=COUNTING_LOGIC_CUT_AWARE,
        help=(
            "legacy keeps the original [id,count,many,portion] prompt. "
            "cut-aware uses compact [scene,[id,pieces,cut,whole_count,conf,portion],...] output and deterministic count correction."
        ),
    )
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token for model downloads.")
    parser.add_argument("--hf-token-file", type=Path, default=None, help="Optional file containing a Hugging Face token.")
    parser.add_argument("--mock-qwen-json", default="", help="Skip Qwen and parse this JSON. Useful for smoke tests.")
    parser.add_argument("--defer-calorie-qwen", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                return value.strip().strip("'").strip('"') or None
        else:
            return line.strip().strip("'").strip('"') or None
    return None


def configure_hf_token(args: argparse.Namespace) -> str | None:
    token = str(args.hf_token or "").strip()
    if not token and args.hf_token_file is not None:
        token = read_token_file(args.hf_token_file.expanduser()) or ""
    if not token:
        for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            value = os.environ.get(env_name)
            if value and value.strip():
                token = value.strip()
                break
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token or None


@dataclass(frozen=True)
class CalorieTableRow:
    ingredient_id: int
    ingredient: str
    calories_per_100g: float
    calories_per_single_object: float | None = None
    average_portion_g: float | None = None
    average_portion_kcal: float | None = None


@dataclass(frozen=True)
class IngredientPrediction:
    raw_name: str
    ingredient_id: int | None
    count: float | None
    portion_category: str | None = None
    many_instances: bool = False
    visible_pieces: float | None = None
    cut_code: int | None = None
    whole_count_vlm: float | None = None
    count_confidence: float | None = None


@dataclass
class CalorieRuntime:
    args: argparse.Namespace
    estimator: "QwenCompositionEstimator | None"
    calories_by_name: dict[str, CalorieTableRow]
    calories_by_id: dict[int, CalorieTableRow]
    candidate_filter: "CalorieCandidateFilter | None"
    model_load_timings: dict[str, float]


class CalorieCandidateFilter:
    def __init__(
        self,
        task1_module: Any,
        labels: list[str],
        text_embeddings: Any,
        embedder: Any,
        top_k: int,
        candidate_list_mode: str,
        dynamic_candidate_relative_delta: float,
        dynamic_candidate_min_k: int,
        dynamic_candidate_max_k: int,
        text_cache: dict[str, Any],
        source: str,
    ) -> None:
        self.task1 = task1_module
        self.labels = labels
        self.text_embeddings = text_embeddings
        self.embedder = embedder
        self.top_k = max(1, int(top_k))
        self.candidate_list_mode = str(candidate_list_mode)
        self.dynamic_candidate_relative_delta = float(dynamic_candidate_relative_delta)
        self.dynamic_candidate_min_k = int(dynamic_candidate_min_k)
        self.dynamic_candidate_max_k = int(dynamic_candidate_max_k)
        self.text_cache = text_cache
        self.source = source

    @property
    def filter_name(self) -> str:
        if self.candidate_list_mode == "fixed_topk":
            return "siglip2_topk"
        return f"siglip2_{self.candidate_list_mode}"

    def candidate_selection_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            candidate_list_mode=self.candidate_list_mode,
            top_k=self.top_k,
            dynamic_candidate_relative_delta=self.dynamic_candidate_relative_delta,
            dynamic_candidate_min_k=self.dynamic_candidate_min_k,
            dynamic_candidate_max_k=self.dynamic_candidate_max_k,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> tuple["CalorieCandidateFilter", dict[str, float]]:
        import orin_task1_pipeline as task1

        started = time.perf_counter()
        cleaned_rows = task1.read_cleaned_rows(args.cleaned_json)
        labels = task1.read_labels(args.captions, cleaned_rows)
        embedder = task1.build_siglip2_embedder(args)
        load_siglip2_sec = time.perf_counter() - started

        started = time.perf_counter()
        text_embeddings, text_cache = task1.load_or_build_text_embeddings(
            embedder,
            labels,
            args.siglip_model,
            args.siglip_pretrained,
            args.text_cache,
            use_cache=not args.no_text_cache,
            require_cache_hit=bool(
                args.text_cache_require_hit or getattr(embedder, "backend", "") == "tensorrt_engine"
            ),
        )
        text_embeddings_sec = time.perf_counter() - started
        return (
            cls(
                task1_module=task1,
                labels=labels,
                text_embeddings=text_embeddings,
                embedder=embedder,
                top_k=args.calorie_candidate_top_k,
                candidate_list_mode=args.calorie_candidate_list_mode,
                dynamic_candidate_relative_delta=args.calorie_dynamic_candidate_relative_delta,
                dynamic_candidate_min_k=args.calorie_dynamic_candidate_min_k,
                dynamic_candidate_max_k=args.calorie_dynamic_candidate_max_k,
                text_cache=text_cache,
                source="calorie_siglip2",
            ),
            {
                "load_siglip2_sec": load_siglip2_sec,
                "text_embeddings_sec": text_embeddings_sec,
            },
        )

    @classmethod
    def from_task1_runtime(
        cls,
        task1_runtime: Any,
        top_k: int,
        candidate_list_mode: str = DEFAULT_CALORIE_CANDIDATE_LIST_MODE,
        dynamic_candidate_relative_delta: float = DEFAULT_CALORIE_DYNAMIC_CANDIDATE_RELATIVE_DELTA,
        dynamic_candidate_min_k: int = DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MIN_K,
        dynamic_candidate_max_k: int = DEFAULT_CALORIE_DYNAMIC_CANDIDATE_MAX_K,
    ) -> "CalorieCandidateFilter":
        import orin_task1_pipeline as task1

        return cls(
            task1_module=task1,
            labels=list(task1_runtime.labels),
            text_embeddings=task1_runtime.text_embeddings,
            embedder=task1_runtime.embedder,
            top_k=top_k,
            candidate_list_mode=candidate_list_mode,
            dynamic_candidate_relative_delta=dynamic_candidate_relative_delta,
            dynamic_candidate_min_k=dynamic_candidate_min_k,
            dynamic_candidate_max_k=dynamic_candidate_max_k,
            text_cache=getattr(task1_runtime, "text_cache", {}),
            source="task1_runtime_siglip2",
        )

    def run(self, image_path: Path) -> tuple[list[Any], dict[str, float], dict[str, Any]]:
        started = time.perf_counter()
        image = self.task1.Image.open(image_path).convert("RGB")
        image_embedding = self.embedder.encode_image(image)
        visual_scores = 100.0 * (image_embedding @ self.text_embeddings.T)
        candidate_count, candidate_list_policy = self.task1.candidate_count_for_scores(
            visual_scores,
            self.candidate_selection_args(),
        )
        candidates = self.task1.build_candidates(self.labels, visual_scores, candidate_count)
        return candidates, {"siglip2_candidate_filter_sec": time.perf_counter() - started}, candidate_list_policy


class QwenCompositionEstimator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source = "calorie_runtime_qwen"
        self.torch = require_torch()
        if args.device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        model_path = Path(str(args.qwen_model)).expanduser()
        if "qwen" not in str(args.qwen_model).lower() and not model_path.exists():
            raise RuntimeError("Calorie estimation is configured to use only a Qwen-VL instruction model.")

        dtype = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }[args.torch_dtype]
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Qwen calorie estimation requires transformers, accelerate, and qwen-vl-utils."
            ) from exc

        self.process_vision_info = process_vision_info
        self.max_new_tokens = int(args.qwen_max_new_tokens)
        self.min_pixels = int(args.qwen_min_pixels)
        self.max_pixels = int(args.qwen_max_pixels)

        model_source, local_files_only = resolve_transformers_source(args.qwen_model)
        if model_source != args.qwen_model:
            print(f"Using cached Transformers snapshot: {model_source}")

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto" if args.device == "cuda" else None,
            "trust_remote_code": True,
        }
        if local_files_only:
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True

        self.processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs).eval()
        except TypeError:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs).eval()
        if args.device == "cpu":
            self.model.to("cpu")

    def generate(self, image_path: Path, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path), "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class SharedTask1QwenCompositionEstimator:
    def __init__(self, qwen: Any, args: argparse.Namespace) -> None:
        self.qwen = qwen
        self.max_new_tokens = int(args.qwen_max_new_tokens)
        self.min_pixels = int(args.qwen_min_pixels)
        self.max_pixels = int(args.qwen_max_pixels)
        self.source = "task1_runtime_qwen"

    def generate(self, image_path: Path, prompt: str) -> str:
        return self.qwen.score(
            image_path,
            prompt,
            max_new_tokens=self.max_new_tokens,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_optional_float(value: Any, field: str, ingredient: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} for {ingredient!r}: {value!r}") from exc


def parse_average_portion(value: Any, ingredient: str) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    match = re.search(
        r"(?P<grams>\d+(?:\.\d+)?)\s*g\s*(?:->|=>|-|:)\s*(?P<kcal>\d+(?:\.\d+)?)\s*kcal",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group("grams")), float(match.group("kcal"))
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    raise ValueError(f"Invalid average_portion_g_to_calories for {ingredient!r}: {value!r}")


def read_calorie_table(path: Path) -> dict[str, CalorieTableRow]:
    if not path.exists():
        raise FileNotFoundError(f"Calories CSV does not exist: {path}")
    rows: dict[str, CalorieTableRow] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ingredient", "calories_per_100g"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain columns: ingredient, calories_per_100g")
        for ingredient_id, raw in enumerate(reader):
            ingredient = str(raw.get("ingredient") or "").strip()
            if not ingredient:
                continue
            try:
                calories_per_100g = float(str(raw.get("calories_per_100g") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid calories_per_100g for {ingredient!r}: {raw}") from exc
            calories_per_single_object = parse_optional_float(
                raw.get("calories_per_single_object"),
                "calories_per_single_object",
                ingredient,
            )
            average_portion_g, average_portion_kcal = parse_average_portion(
                raw.get("average_portion_g_to_calories"),
                ingredient,
            )
            rows[normalize_name(ingredient)] = CalorieTableRow(
                ingredient_id=ingredient_id,
                ingredient=ingredient,
                calories_per_100g=calories_per_100g,
                calories_per_single_object=calories_per_single_object,
                average_portion_g=average_portion_g,
                average_portion_kcal=average_portion_kcal,
            )
    next_id = max((row.ingredient_id for row in rows.values()), default=-1) + 1
    for ingredient, calories_per_100g in sorted(BUILTIN_CALORIES_PER_100G.items()):
        key = normalize_name(ingredient)
        if key in rows:
            continue
        rows[key] = CalorieTableRow(
            ingredient_id=next_id,
            ingredient=ingredient,
            calories_per_100g=calories_per_100g,
            average_portion_g=BUILTIN_AVERAGE_PORTIONS.get(key, (None, None))[0],
            average_portion_kcal=BUILTIN_AVERAGE_PORTIONS.get(key, (None, None))[1],
        )
        next_id += 1
    return rows


def build_composition_prompt(
    calories_by_name: dict[str, CalorieTableRow],
    candidate_rows: list[CalorieTableRow] | None = None,
) -> str:
    rows = sorted(
        candidate_rows if candidate_rows else calories_by_name.values(),
        key=lambda row: row.ingredient_id,
    )
    candidate_scope = candidate_rows is not None
    allowed = "; ".join(f"{row.ingredient_id}: {row.ingredient}" for row in rows)
    countable = "; ".join(
        f"{row.ingredient_id}: {row.ingredient} ({format_number(row.calories_per_single_object)} kcal each)"
        for row in rows
        if row.calories_per_single_object is not None
    )
    average_portions = "; ".join(
        f"{row.ingredient_id}: {row.ingredient} ({format_number(row.average_portion_kcal)} kcal per normal portion)"
        for row in rows
        if row.average_portion_kcal is not None
    )
    alias_lines: list[str] = []
    for alias, target in sorted(CALORIE_ALIASES.items()):
        target_row = calories_by_name.get(normalize_name(target))
        if target_row is not None:
            alias_lines.append(f"{alias} -> {target_row.ingredient_id}: {target_row.ingredient}")
    aliases = "; ".join(alias_lines) if alias_lines else "none"
    rows_by_name = {normalize_name(row.ingredient): row for row in rows}
    serving_example = rows_by_name.get("pasta") or rows[0]
    count_example = next((row for row in rows if row.calories_per_single_object is not None), rows[min(1, len(rows) - 1)])
    abundance_example = rows_by_name.get("banana") or count_example
    count_example_count = "0.5" if count_example.calories_per_single_object is not None else "null"
    count_example_portion = "none" if count_example.calories_per_single_object is not None else "normal"
    return (
        "Analyze the food image and estimate dish composition for calorie calculation.\n\n"
        "Return only valid compact JSON. No markdown and no explanation.\n\n"
        "Output one JSON array, without extra keys.\n"
        "The first array item must be the quoted scene value: \"serving\" or \"abundance_display\".\n"
        "Each following item must be an ingredient row: [id,count,many,portion].\n"
        "Do not output the literal token scene.\n"
        "Example output:\n"
        f'[\"serving\",[{serving_example.ingredient_id},null,0,\"normal\"],'
        f'[{count_example.ingredient_id},{count_example_count},0,\"{count_example_portion}\"],'
        f'[{abundance_example.ingredient_id},null,1,\"double\"]]\n\n'
        "Field order:\n"
        "- scene: \"serving\" or \"abundance_display\".\n"
        "- id: allowed ingredient ID.\n"
        "- count: whole-object-equivalent count, decimal allowed, or null.\n"
        "- many: 1 for abundance/display repeated instances, else 0.\n"
        "- portion: one of none, garnish, small, normal, large, double.\n\n"
        "Rules:\n"
        + (
            "- The allowed ingredient list is a SigLIP2 top-candidate shortlist for this image.\n"
            if candidate_scope
            else ""
        )
        + "- Use only IDs and names from the allowed ingredient list.\n"
        "- Use scene serving for normal eating servings on plates, bowls, cups, pans, trays, or table/kitchen/restaurant containers.\n"
        "- Use scene abundance_display for market stalls, seafood displays on ice, baskets, crates, shop counters, bulk bags, or food displays clearly outside normal serving context.\n"
        "- Every ingredient prediction must be exactly [id,count,many,portion].\n"
        "- Include only visible edible ingredients or major components.\n"
        f"- Hard limit: output at most {MAX_INGREDIENT_PREDICTIONS} ingredient rows. For complex dishes, keep only the largest or most calorie-important visible components.\n"
        "- Never enumerate the allowed ingredient list or output consecutive IDs just because they appear in the list.\n"
        "- If many small toppings or mixed ingredients are visible, group the estimate into the main visible components instead of listing every possible item.\n"
        "- Never output count 0. Omit an ingredient if it is not visible or not selected.\n"
        "- For countable ingredients, count is the equivalent number of whole objects, not the number of pieces.\n"
        "- Whole objects use integers: one apple = 1, two eggs = 2.\n"
        "- Cut, sliced, wedged, or partial countable ingredients may use decimals when the fraction is visually clear: half lemon = 0.5, quarter tomato = 0.25, two tomato halves = 1, visible slices that look like half an apple = 0.5.\n"
        "- If scene is serving, many must be 0 unless there is clearly a market/display-scale abundance.\n"
        "- Use many 1 only for abundance/display scenes outside normal serving context: market stall, seafood display on ice, crate, basket, shop counter, bulk bag, or a table full of repeated food clearly too much for a few people. Set count to null and portion to large or double.\n"
        "- Do not use many 1 for ordinary cups, bowls, plates, or small servings of many small foods such as blueberries, grapes, cherry tomatoes, nuts, or berries. If the exact count is unreliable, set count to null, many to 0, and choose portion for the visible edible serving.\n"
        "- Set count to null for uncountable ingredients, mixed foods, sauces, chopped/blended food, occluded food, piles, grains, or slices where the whole-object fraction is not clear.\n"
        "- Use count only for ingredients in the countable list. Other ingredients must have count null.\n"
        "- Use portion none only when a reliable count is provided. If count is null for a visible ingredient, portion must be garnish, small, normal, large, or double.\n"
        "- For ingredients without a reliable count, choose the visible amount someone would likely eat: garnish for tiny decoration, small for about half a normal portion, normal for one normal portion, large or double for about two normal portions.\n\n"
        f"Allowed ingredient IDs and names: {allowed}\n\n"
        f"Countable ingredients with calories per single object: {countable}\n\n"
        f"Average normal portions for no-count fallback: {average_portions}\n\n"
        f"Common aliases to map into allowed IDs: {aliases}"
    )


def build_cut_aware_composition_prompt(
    calories_by_name: dict[str, CalorieTableRow],
    candidate_rows: list[CalorieTableRow] | None = None,
) -> str:
    rows = sorted(
        candidate_rows if candidate_rows else calories_by_name.values(),
        key=lambda row: row.ingredient_id,
    )
    candidate_scope = candidate_rows is not None
    allowed = "; ".join(f"{row.ingredient_id}: {row.ingredient}" for row in rows)
    countable = "; ".join(
        f"{row.ingredient_id}: {row.ingredient} ({format_number(row.calories_per_single_object)} kcal each)"
        for row in rows
        if row.calories_per_single_object is not None
    )
    average_portions = "; ".join(
        f"{row.ingredient_id}: {row.ingredient} ({format_number(row.average_portion_kcal)} kcal per normal portion)"
        for row in rows
        if row.average_portion_kcal is not None
    )
    alias_lines: list[str] = []
    for alias, target in sorted(CALORIE_ALIASES.items()):
        target_row = calories_by_name.get(normalize_name(target))
        if target_row is not None:
            alias_lines.append(f"{alias} -> {target_row.ingredient_id}: {target_row.ingredient}")
    aliases = "; ".join(alias_lines) if alias_lines else "none"
    rows_by_name = {normalize_name(row.ingredient): row for row in rows}
    serving_example = rows_by_name.get("pasta") or rows[0]
    count_example = next((row for row in rows if row.calories_per_single_object is not None), rows[min(1, len(rows) - 1)])
    abundance_example = rows_by_name.get("banana") or count_example

    def pick_row(names: tuple[str, ...], fallback: CalorieTableRow) -> CalorieTableRow:
        for name in names:
            row = rows_by_name.get(name)
            if row is not None:
                return row
        return fallback

    whole_example = pick_row(("apple", "egg", "banana"), count_example)
    half_example = pick_row(("tomato", "lemon", "apple"), count_example)
    quarter_example = pick_row(("lemon", "tomato", "lime"), count_example)
    wedge_example = pick_row(("watermelon", "melon", "orange"), count_example)
    slice_example = pick_row(("cucumber", "apple", "tomato"), count_example)
    chopped_example = pick_row(("onion", "garlic", "parsley"), serving_example)
    return (
        "Analyze the food image and estimate dish composition for calorie calculation.\n\n"
        "Return only one compact JSON array. No markdown, keys, words, or explanation.\n\n"
        "Format: [scene,[id,pieces,cut,whole_count,conf,portion],...]\n"
        "scene: 0 serving, 1 abundance_display.\n"
        "cut: 0 whole, 1 half, 2 quarter, 3 wedge_or_large_piece, 4 slice_unclear, 5 chopped_diced_pile.\n"
        "conf: 0 low, 1 medium, 2 high confidence in whole_count.\n"
        "portion: 0 none, 1 garnish, 2 small, 3 normal, 4 large, 5 double.\n"
        "Example:\n"
        f'[0,[{serving_example.ingredient_id},null,5,null,0,3],'
        f'[{whole_example.ingredient_id},1,0,1,2,0],'
        f'[{abundance_example.ingredient_id},null,5,null,0,5]]\n\n'
        "Counting policy for countable ingredients:\n"
        "- pieces is the number of visible physical pieces.\n"
        "- whole_count is the equivalent number of original whole food objects, not the number of pieces.\n"
        "- For whole objects: whole_count = pieces.\n"
        "- For halves: whole_count = pieces * 0.5.\n"
        "- For quarters: whole_count = pieces * 0.25.\n"
        "- For wedges or large pieces: estimate whole_count only if visually clear.\n"
        "- For thin slices, chopped, diced, shredded, piles, or unclear fragments: set whole_count=null and use portion.\n"
        "- Never count slices as whole objects.\n"
        "- If whole_count is uncertain, set whole_count=null, lower conf, and use portion.\n\n"
        "Cut examples using allowed IDs:\n"
        f'- one whole {whole_example.ingredient}: [{whole_example.ingredient_id},1,0,1,2,0]\n'
        f'- two {half_example.ingredient} halves: [{half_example.ingredient_id},2,1,1,2,0]\n'
        f'- four {quarter_example.ingredient} quarters: [{quarter_example.ingredient_id},4,2,1,2,0]\n'
        f'- one large {wedge_example.ingredient} wedge, roughly one eighth: [{wedge_example.ingredient_id},1,3,0.125,2,0]\n'
        f'- many {slice_example.ingredient} slices, unclear original amount: [{slice_example.ingredient_id},8,4,null,0,2]\n'
        f'- chopped {chopped_example.ingredient} pile: [{chopped_example.ingredient_id},null,5,null,0,3]\n\n'
        "Rules:\n"
        + (
            "- The allowed ingredient list is a SigLIP2 top-candidate shortlist for this image.\n"
            if candidate_scope
            else ""
        )
        + "- Use only IDs and names from the allowed ingredient list.\n"
        "- Use scene 0 for normal eating servings on plates, bowls, cups, pans, trays, or table/kitchen/restaurant containers.\n"
        "- Use scene 1 for market stalls, seafood displays on ice, baskets, crates, shop counters, bulk bags, or food displays clearly outside normal serving context.\n"
        "- Every ingredient prediction must be exactly [id,pieces,cut,whole_count,conf,portion].\n"
        "- Include only visible edible ingredients or major components.\n"
        f"- Hard limit: output at most {MAX_INGREDIENT_PREDICTIONS} ingredient rows. For complex dishes, keep only the largest or most calorie-important visible components.\n"
        "- Never enumerate the allowed ingredient list or output consecutive IDs just because they appear in the list.\n"
        "- If many small toppings or mixed ingredients are visible, group the estimate into the main visible components instead of listing every possible item.\n"
        "- Never output pieces 0 or whole_count 0. Omit an ingredient if it is not visible or not selected.\n"
        "- Use whole_count only for ingredients in the countable list. Other ingredients must have whole_count null.\n"
        "- Use portion 0 only when a reliable whole_count is provided. If whole_count is null for a visible ingredient, portion must be 1, 2, 3, 4, or 5.\n"
        "- For ingredients without a reliable whole_count, choose the visible amount someone would likely eat: 1 tiny garnish, 2 half portion, 3 normal portion, 4 large, 5 double.\n"
        "- For scene 1 abundance, set pieces and whole_count to null and portion to 4 or 5.\n\n"
        f"Allowed ingredient IDs and names: {allowed}\n\n"
        f"Countable ingredients with calories per single object: {countable or 'none'}\n\n"
        f"Average normal portions for no-count fallback: {average_portions or 'none'}\n\n"
        f"Common aliases to map into allowed IDs: {aliases}"
    )


def build_selected_composition_prompt(
    calories_by_name: dict[str, CalorieTableRow],
    candidate_rows: list[CalorieTableRow] | None,
    counting_logic: str,
) -> str:
    if counting_logic == COUNTING_LOGIC_CUT_AWARE:
        return build_cut_aware_composition_prompt(calories_by_name, candidate_rows)
    return build_composition_prompt(calories_by_name, candidate_rows)


def repair_compact_json_payload(payload: str) -> str | None:
    text = payload.strip()
    if not text.startswith("["):
        return None
    repaired = re.sub(
        r'^\[\s*scene\s*,\s*"(serving|abundance_display)"\s*,',
        r'["\1",',
        text,
        count=1,
    )
    if repaired != text:
        return repaired
    repaired = re.sub(
        r"^\[\s*scene\s*,",
        '["serving",',
        text,
        count=1,
    )
    if repaired != text:
        return repaired
    return None


def loads_json_with_compact_repair(payload: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(payload), None
    except Exception as original_exc:
        repaired = repair_compact_json_payload(payload)
        if repaired is not None:
            try:
                return json.loads(repaired), "repaired_unquoted_scene_token"
            except Exception:
                pass
        return None, f"invalid_json_payload:{original_exc}"


def balanced_json_value_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] not in "[{":
        return None
    closing_for = {"[": "]", "{": "}"}
    stack = [closing_for[text[start]]]
    in_string = False
    escape = False
    for idx in range(start + 1, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in closing_for:
            stack.append(closing_for[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return idx + 1
    return None


def recover_unterminated_compact_array(payload: str) -> tuple[Any | None, str | None]:
    text = payload.strip()
    repaired = repair_compact_json_payload(text)
    if repaired is not None:
        text = repaired
    match = re.match(r'\[\s*(?:"(?P<scene>serving|abundance_display)"|(?P<scene_code>[01]))\s*,?', text)
    if not match:
        return None, None
    scene = int(match.group("scene_code")) if match.group("scene_code") is not None else match.group("scene")
    idx = match.end()
    rows: list[Any] = []
    while idx < len(text) and len(rows) < MAX_INGREDIENT_PREDICTIONS:
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text) or text[idx] != "[":
            break
        end = balanced_json_value_end(text, idx)
        if end is None:
            break
        try:
            row = json.loads(text[idx:end])
        except Exception:
            break
        if isinstance(row, list) and len(row) >= 4:
            rows.append(row)
        idx = end
    if not rows:
        return None, None
    return [scene, *rows], f"recovered_unterminated_compact_array:{len(rows)}_rows"


def extract_json_object(text: str) -> tuple[Any | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"
    parsed, warning = loads_json_with_compact_repair(stripped)
    if parsed is not None:
        return parsed, warning
    starts = [idx for idx, char in enumerate(stripped) if char in "{["]
    if not starts:
        return None, "no_json_payload"
    last_error = ""
    for start in starts:
        end = balanced_json_value_end(stripped, start)
        if end is not None:
            parsed, warning = loads_json_with_compact_repair(stripped[start:end])
            if parsed is not None:
                return parsed, warning
            last_error = warning or "invalid_json_payload"
            continue
        recovered, recovery_warning = recover_unterminated_compact_array(stripped[start:])
        if recovered is not None:
            return recovered, recovery_warning
    if last_error:
        return None, last_error
    return None, "unterminated_json_payload"


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return max(0.0, float(match.group(0)))
    return None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d+", text):
            return int(text)
    return None


def coerce_count(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 0:
            return number
        return None
    if isinstance(value, str):
        text = normalize_name(value)
        if text in {"", "none", "null", "unknown", "unclear", "ambiguous", "many", "several", "few", "some"}:
            return None
        phrase_counts = {
            "a half": 0.5,
            "one half": 0.5,
            "half": 0.5,
            "a quarter": 0.25,
            "one quarter": 0.25,
            "quarter": 0.25,
            "one fourth": 0.25,
            "a fourth": 0.25,
            "fourth": 0.25,
            "three quarters": 0.75,
            "three fourths": 0.75,
            "two halves": 1.0,
            "one and a half": 1.5,
            "one and half": 1.5,
            "one-and-a-half": 1.5,
        }
        if text in phrase_counts:
            return phrase_counts[text]
        word_counts = {
            "a": 1.0,
            "an": 1.0,
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
            "nine": 9.0,
            "ten": 10.0,
            "eleven": 11.0,
            "twelve": 12.0,
        }
        if text in word_counts:
            return word_counts[text]
        if re.search(r"\d+\s*(?:-|to)\s*\d+", text):
            return None
        mixed_fraction = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
        if mixed_fraction:
            whole = float(mixed_fraction.group(1))
            numerator = float(mixed_fraction.group(2))
            denominator = float(mixed_fraction.group(3))
            if denominator > 0:
                return whole + numerator / denominator
        fraction = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
        if fraction:
            numerator = float(fraction.group(1))
            denominator = float(fraction.group(2))
            if denominator > 0 and numerator > 0:
                return numerator / denominator
        if re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", text):
            number = float(text)
            if number > 0:
                return number
    return None


def corrected_whole_count(
    pieces: float | None,
    cut_code: int | None,
    whole_count_vlm: float | None,
    confidence: float | None,
) -> float | None:
    if cut_code is None:
        return whole_count_vlm if confidence is not None and confidence >= COUNT_CONFIDENCE_THRESHOLD else None

    if pieces is None or pieces <= 0:
        if whole_count_vlm is not None and confidence is not None and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return whole_count_vlm
        return None

    if cut_code == CUT_WHOLE:
        return float(pieces)

    if cut_code == CUT_HALF:
        return float(pieces) * 0.5

    if cut_code == CUT_QUARTER:
        return float(pieces) * 0.25

    if cut_code == CUT_WEDGE:
        if whole_count_vlm is not None and confidence is not None and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return whole_count_vlm
        return None

    if cut_code in {CUT_SLICE_UNCLEAR, CUT_CHOPPED_PILE}:
        return None

    return None


def round_count_to_half_step(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    rounded = int(float(value) * 2.0 + 0.5) / 2.0
    return rounded if rounded > 0 else 0.5


def count_value_suggests_many(value: Any) -> bool:
    if isinstance(value, str):
        text = normalize_name(value)
        return text in {
            "many",
            "several",
            "few",
            "some",
            "bunch",
            "basket",
            "pile",
            "tray",
            "bag",
            "box",
            "dozens",
            "lots",
            "multiple",
        }
    return False


def count_value_is_zero(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        text = normalize_name(value)
        if re.fullmatch(r"0+(?:\.0+)?", text):
            return True
    return False


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = normalize_name(value)
        if text in {"true", "yes", "y", "1", "many", "multiple"}:
            return True
        if text in {"false", "no", "n", "0", "none", "null", ""}:
            return False
    return False


def coerce_confidence_code(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric in CONFIDENCE_CODE_TO_VALUE:
            return CONFIDENCE_CODE_TO_VALUE[int(numeric)]
        if 0.0 <= numeric <= 1.0:
            return numeric
        return None
    if isinstance(value, str):
        text = normalize_name(value)
        aliases = {
            "low": 0,
            "medium": 1,
            "med": 1,
            "high": 2,
        }
        if text in aliases:
            return CONFIDENCE_CODE_TO_VALUE[aliases[text]]
        parsed = coerce_float(value)
        if parsed is not None:
            return coerce_confidence_code(parsed)
    return None


def normalize_portion_category(value: Any, default: str = "normal") -> str:
    code = coerce_int(value)
    if code is not None and code in PORTION_CODE_TO_CATEGORY:
        return PORTION_CODE_TO_CATEGORY[code]
    text = normalize_name(value)
    aliases = {
        "": default,
        "null": default,
        "unknown": default,
        "no": "none",
        "zero": "none",
        "tiny": "garnish",
        "trace": "garnish",
        "little": "small",
        "half": "small",
        "medium": "normal",
        "regular": "normal",
        "standard": "normal",
        "big": "large",
        "extra": "large",
        "extra large": "double",
        "very large": "double",
        "two portions": "double",
    }
    text = aliases.get(text, text)
    if text in PORTION_FACTORS:
        return text
    return default


def coerce_range(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 2:
        low = coerce_float(value[0])
        high = coerce_float(value[1])
        if low is not None and high is not None:
            return sorted([low, high])
    if isinstance(value, str):
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))]
        if len(numbers) >= 2:
            return sorted(numbers[:2])
    return None


def match_calorie_row(name: str, calories_by_name: dict[str, CalorieTableRow]) -> tuple[CalorieTableRow | None, str | None]:
    key = normalize_name(name)
    if key in calories_by_name:
        return calories_by_name[key], None
    alias = CALORIE_ALIASES.get(key)
    if alias and alias in calories_by_name:
        return calories_by_name[alias], f"matched_alias:{name}->{calories_by_name[alias].ingredient}"
    if key in BUILTIN_CALORIES_PER_100G:
        return (
            CalorieTableRow(ingredient_id=-1, ingredient=key, calories_per_100g=BUILTIN_CALORIES_PER_100G[key]),
            f"matched_builtin_calories:{name}",
        )
    compact = key.replace("-", " ")
    if compact in calories_by_name:
        return calories_by_name[compact], f"matched_normalized:{name}->{calories_by_name[compact].ingredient}"
    matches = difflib.get_close_matches(compact, list(calories_by_name), n=1, cutoff=0.86)
    if matches:
        row = calories_by_name[matches[0]]
        return row, f"matched_fuzzy:{name}->{row.ingredient}"
    return None, f"missing_calories_per_100g:{name}"


def calorie_rows_from_candidates(
    candidates: list[Any],
    calories_by_name: dict[str, CalorieTableRow],
) -> tuple[list[CalorieTableRow], list[dict[str, Any]], list[str]]:
    rows: list[CalorieTableRow] = []
    trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[int] = set()
    for candidate in candidates:
        label = str(getattr(candidate, "label", "") or "").strip()
        row, warning = match_calorie_row(label, calories_by_name)
        if warning:
            warnings.append(f"candidate_{warning}")
        if row is not None and row.ingredient_id not in seen_ids:
            rows.append(row)
            seen_ids.add(row.ingredient_id)
        trace.append(
            {
                "rank": getattr(candidate, "rank", None),
                "label": label,
                "visual_score": getattr(candidate, "visual_score", None),
                "ingredient_id": row.ingredient_id if row is not None else None,
                "ingredient": row.ingredient if row is not None else None,
            }
        )
    return rows, trace, warnings


def count_key(value: Any) -> str:
    value_id = coerce_int(value)
    if value_id is not None:
        return f"id:{value_id}"
    return normalize_name(value)


def prediction_key(prediction: IngredientPrediction) -> str:
    if prediction.ingredient_id is not None:
        return f"id:{prediction.ingredient_id}"
    return normalize_name(prediction.raw_name)


def parse_count_map(parsed: Any) -> dict[str, float]:
    if not isinstance(parsed, dict):
        return {}
    raw = parsed.get("ingredient_counts")
    if raw is None:
        raw = parsed.get("object_counts")
    if raw is None:
        raw = parsed.get("counts")
    if not isinstance(raw, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in raw.items():
        count = coerce_count(value)
        if count is not None:
            values[count_key(key)] = count
    return values


def parse_portion_map(parsed: Any) -> dict[str, str]:
    if not isinstance(parsed, dict):
        return {}
    raw = parsed.get("ingredient_portions")
    if raw is None:
        raw = parsed.get("portion_categories")
    if raw is None:
        raw = parsed.get("portions")
    if not isinstance(raw, dict):
        return {}
    values: dict[str, str] = {}
    for key, value in raw.items():
        values[count_key(key)] = normalize_portion_category(value)
    return values


def parse_scene_context(parsed: Any) -> str:
    if isinstance(parsed, list) and parsed:
        if normalize_name(parsed[0]) == "scene" and len(parsed) >= 2:
            value = parsed[1]
        else:
            value = parsed[0]
        code = coerce_int(value)
        if code == 1:
            return "abundance_display"
        if code == 0:
            return "serving"
        text = normalize_name(value)
        if text in {"a", "abundance", "abundance display", "abundance_display"}:
            return "abundance_display"
        return "serving"
    if not isinstance(parsed, dict):
        return "serving"
    value = (
        parsed.get("scene_context")
        or parsed.get("scene_type")
        or parsed.get("context")
        or parsed.get("food_context")
    )
    text = normalize_name(value)
    if any(
        marker in text
        for marker in (
            "abundance",
            "market",
            "display",
            "stall",
            "shop",
            "counter",
            "crate",
            "basket",
            "bulk",
        )
    ):
        return "abundance_display"
    return "serving"


def looks_like_compact_ingredient_row(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 4 and coerce_int(value[0]) is not None


def normalize_compact_ingredient_rows(value: list[Any]) -> list[Any]:
    if looks_like_compact_ingredient_row(value):
        return [value]
    if len(value) == 1 and isinstance(value[0], list):
        first = value[0]
        if looks_like_compact_ingredient_row(first):
            return [first]
        return first
    return value


def compact_ingredients_payload(parsed: Any) -> list[Any] | None:
    if isinstance(parsed, list):
        if not parsed:
            return []
        start = 0
        if coerce_int(parsed[0]) in {0, 1}:
            start = 1
        elif isinstance(parsed[0], str):
            start = 1
            if normalize_name(parsed[0]) == "scene" and len(parsed) >= 2:
                start = 2 if isinstance(parsed[1], str) else 1
        rows = parsed[start:]
        return normalize_compact_ingredient_rows(rows)
    if isinstance(parsed, dict):
        for key in ("compact", "rows", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return normalize_compact_ingredient_rows(value)
    return None


def lookup_count(ingredient_id: int | None, raw_name: str, counts: dict[str, float]) -> float | None:
    if ingredient_id is not None:
        value = counts.get(f"id:{ingredient_id}")
        if value is not None:
            return value
    return counts.get(normalize_name(raw_name))


def lookup_portion_category(ingredient_id: int | None, raw_name: str, portions: dict[str, str]) -> str | None:
    if ingredient_id is not None:
        value = portions.get(f"id:{ingredient_id}")
        if value is not None:
            return value
    return portions.get(normalize_name(raw_name))


def merge_ingredient_prediction(
    existing: IngredientPrediction,
    *,
    raw_name: str,
    ingredient_id: int | None,
    count: float | None,
    portion_category: str | None,
    many_instances: bool,
    visible_pieces: float | None = None,
    cut_code: int | None = None,
    whole_count_vlm: float | None = None,
    count_confidence: float | None = None,
) -> IngredientPrediction:
    merged_count = existing.count
    if merged_count is not None and count is not None:
        merged_count += count
    elif merged_count is None:
        merged_count = count

    merged_visible_pieces = None
    if existing.visible_pieces is not None or visible_pieces is not None:
        merged_visible_pieces = float(existing.visible_pieces or 0.0) + float(visible_pieces or 0.0)

    merged_confidence = max(
        existing.count_confidence or 0.0,
        count_confidence or 0.0,
    ) or None

    return IngredientPrediction(
        raw_name=existing.raw_name or raw_name,
        ingredient_id=existing.ingredient_id if existing.ingredient_id is not None else ingredient_id,
        count=merged_count,
        portion_category=existing.portion_category or portion_category,
        many_instances=existing.many_instances or many_instances,
        visible_pieces=merged_visible_pieces,
        cut_code=existing.cut_code if existing.cut_code == cut_code else None,
        whole_count_vlm=None,
        count_confidence=merged_confidence,
    )


def parse_ingredient_predictions(parsed: Any) -> list[IngredientPrediction]:
    compact_ingredients = compact_ingredients_payload(parsed)
    if compact_ingredients is not None:
        values: list[IngredientPrediction] = []
        seen: set[str] = set()
        for item in compact_ingredients:
            if not isinstance(item, list) or len(item) < 4:
                continue
            ingredient_id = coerce_int(item[0])
            if ingredient_id is None:
                continue
            if len(item) >= 7:
                visible_pieces = coerce_float(item[1])
                cut_code = coerce_int(item[2])
                whole_count_vlm = coerce_count(item[3])
                count_confidence = coerce_float(item[4])
                many_instances = coerce_bool(item[5]) or count_value_suggests_many(item[1])
                portion_category = normalize_portion_category(item[6], default="") or None
                count = corrected_whole_count(
                    visible_pieces,
                    cut_code,
                    whole_count_vlm,
                    count_confidence,
                )
            elif len(item) == 6:
                visible_pieces = coerce_float(item[1])
                cut_code = coerce_int(item[2])
                whole_count_vlm = coerce_count(item[3])
                count_confidence = coerce_confidence_code(item[4])
                many_instances = False
                portion_category = normalize_portion_category(item[5], default="") or None
                count = corrected_whole_count(
                    visible_pieces,
                    cut_code,
                    whole_count_vlm,
                    count_confidence,
                )
            else:
                count = coerce_count(item[1])
                many_instances = coerce_bool(item[2]) or count_value_suggests_many(item[1])
                portion_category = normalize_portion_category(item[3], default="") or None
                visible_pieces = None
                cut_code = None
                whole_count_vlm = count
                count_confidence = None
            if (
                count is None
                and whole_count_vlm is None
                and (portion_category is None or portion_category == "none")
                and count_value_is_zero(item[1])
            ):
                continue
            key = f"id:{ingredient_id}"
            if key in seen:
                for idx, existing in enumerate(values):
                    if prediction_key(existing) != key:
                        continue
                    values[idx] = merge_ingredient_prediction(
                        existing,
                        raw_name="",
                        ingredient_id=ingredient_id,
                        count=count,
                        portion_category=portion_category,
                        many_instances=many_instances,
                        visible_pieces=visible_pieces,
                        cut_code=cut_code,
                        whole_count_vlm=whole_count_vlm,
                        count_confidence=count_confidence,
                    )
                    break
                continue
            values.append(
                IngredientPrediction(
                    raw_name="",
                    ingredient_id=ingredient_id,
                    count=count,
                    portion_category=portion_category,
                    many_instances=many_instances,
                    visible_pieces=visible_pieces,
                    cut_code=cut_code,
                    whole_count_vlm=whole_count_vlm,
                    count_confidence=count_confidence,
                )
            )
            seen.add(key)
            if len(values) >= MAX_INGREDIENT_PREDICTIONS:
                break
        return values

    if not isinstance(parsed, dict):
        return []
    ingredients = parsed.get("ingredients")
    if not isinstance(ingredients, list):
        return []
    counts = parse_count_map(parsed)
    portions = parse_portion_map(parsed)
    values: list[IngredientPrediction] = []
    seen: set[str] = set()
    for item in ingredients:
        ingredient_id: int | None = None
        count: float | None = None
        portion_category: str | None = None
        many_instances = False
        visible_pieces: float | None = None
        cut_code: int | None = None
        whole_count_vlm: float | None = None
        count_confidence: float | None = None
        used_cut_aware_fields = False
        if isinstance(item, dict):
            ingredient_id = coerce_int(item.get("id"))
            if ingredient_id is None:
                ingredient_id = coerce_int(item.get("ingredient_id"))
            name = str(
                item.get("name")
                or item.get("ingredient")
                or item.get("label")
                or item.get("ingredient_name")
                or ""
            ).strip()
            for count_field in ("count", "object_count", "instance_count", "quantity_count"):
                raw_count = item.get(count_field)
                many_instances = many_instances or count_value_suggests_many(raw_count)
                count = coerce_count(raw_count)
                if count is not None:
                    break
            visible_pieces = coerce_float(item.get("pieces"))
            if visible_pieces is None:
                visible_pieces = coerce_float(item.get("visible_pieces"))
            cut_code = coerce_int(item.get("cut"))
            if cut_code is None:
                cut_code = coerce_int(item.get("cut_code"))
            whole_count_vlm = coerce_count(item.get("whole_count"))
            if whole_count_vlm is None:
                whole_count_vlm = coerce_count(item.get("whole_object_count"))
            if whole_count_vlm is None:
                whole_count_vlm = coerce_count(item.get("whole_count_vlm"))
            count_confidence = coerce_float(item.get("conf"))
            if count_confidence is None:
                count_confidence = coerce_float(item.get("confidence"))
            if count_confidence is None:
                count_confidence = coerce_float(item.get("count_confidence"))
            used_cut_aware_fields = (
                visible_pieces is not None
                or cut_code is not None
                or whole_count_vlm is not None
                or count_confidence is not None
            )
            if used_cut_aware_fields:
                if whole_count_vlm is None:
                    whole_count_vlm = count
                count = corrected_whole_count(
                    visible_pieces,
                    cut_code,
                    whole_count_vlm,
                    count_confidence,
                )
            portion_category = normalize_portion_category(
                item.get("portion_category")
                or item.get("portion")
                or item.get("portion_size")
                or item.get("serving_size"),
                default="",
            ) or None
            many_instances = many_instances or coerce_bool(
                item.get("many_instances")
                or item.get("many")
                or item.get("too_many_to_count")
                or item.get("count_uncertain_many")
                or item.get("abundance_scene")
                or item.get("market_display")
                or item.get("display_case")
                or item.get("bulk_display")
            )
        else:
            name = str(item).strip()
        if count is None and not used_cut_aware_fields:
            count = lookup_count(ingredient_id, name, counts)
        if portion_category is None:
            portion_category = lookup_portion_category(ingredient_id, name, portions)
        key = f"id:{ingredient_id}" if ingredient_id is not None else normalize_name(name)
        if key and key not in seen:
            values.append(
                IngredientPrediction(
                    raw_name=name,
                    ingredient_id=ingredient_id,
                    count=count,
                    portion_category=portion_category,
                    many_instances=many_instances,
                    visible_pieces=visible_pieces,
                    cut_code=cut_code,
                    whole_count_vlm=whole_count_vlm,
                    count_confidence=count_confidence,
                )
            )
            seen.add(key)
            if len(values) >= MAX_INGREDIENT_PREDICTIONS:
                break
        elif key:
            for idx, existing in enumerate(values):
                if prediction_key(existing) != key:
                    continue
                values[idx] = merge_ingredient_prediction(
                    existing,
                    raw_name=name,
                    ingredient_id=ingredient_id,
                    count=count,
                    portion_category=portion_category,
                    many_instances=many_instances,
                    visible_pieces=visible_pieces,
                    cut_code=cut_code,
                    whole_count_vlm=whole_count_vlm,
                    count_confidence=count_confidence,
                )
                break
    return values


def parse_ingredient_list(parsed: Any) -> list[str]:
    labels: list[str] = []
    for prediction in parse_ingredient_predictions(parsed):
        if prediction.raw_name:
            labels.append(prediction.raw_name)
        elif prediction.ingredient_id is not None:
            labels.append(f"id:{prediction.ingredient_id}")
    return labels


def parse_percentages(parsed: Any) -> dict[str, float]:
    if not isinstance(parsed, dict):
        return {}
    raw = parsed.get("ingredient_percentages")
    percentages: dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            pct = coerce_float(value)
            if pct is not None:
                percentages[count_key(key)] = pct
    return percentages


def lookup_percentage(prediction: IngredientPrediction, percentages: dict[str, float]) -> float:
    if prediction.ingredient_id is not None:
        value = percentages.get(f"id:{prediction.ingredient_id}")
        if value is not None:
            return float(value)
    return float(percentages.get(normalize_name(prediction.raw_name), 0.0))


def normalize_percentages(
    ingredients: list[IngredientPrediction],
    percentages: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    values: dict[str, float] = {}
    for ingredient in ingredients:
        values[prediction_key(ingredient)] = lookup_percentage(ingredient, percentages)
    total = sum(values.values())
    if not ingredients:
        return values, ["no_ingredients_from_vlm"]
    if total <= 0:
        even = 100.0 / len(ingredients)
        warnings.append("percentage_sum_zero_used_equal_split")
        return {prediction_key(item): even for item in ingredients}, warnings
    if abs(total - 100.0) > 1.0:
        warnings.append(f"percentages_rescaled_from_{total:.1f}_to_100")
    return {key: value * 100.0 / total for key, value in values.items()}, warnings


def match_calorie_prediction(
    prediction: IngredientPrediction,
    calories_by_name: dict[str, CalorieTableRow],
    calories_by_id: dict[int, CalorieTableRow],
) -> tuple[CalorieTableRow | None, list[str]]:
    warnings: list[str] = []
    if prediction.ingredient_id is not None:
        row = calories_by_id.get(prediction.ingredient_id)
        if row is not None:
            if prediction.raw_name:
                predicted_name = normalize_name(prediction.raw_name)
                canonical_name = normalize_name(row.ingredient)
                alias_target = CALORIE_ALIASES.get(predicted_name)
                if predicted_name != canonical_name and normalize_name(alias_target or "") != canonical_name:
                    name_row, name_warning = match_calorie_row(prediction.raw_name, calories_by_name)
                    if name_row is not None and name_row.ingredient_id != row.ingredient_id:
                        warnings.append(
                            f"corrected_id_name_mismatch:{prediction.ingredient_id}:{prediction.raw_name}->{name_row.ingredient_id}:{name_row.ingredient}"
                        )
                        if name_warning:
                            warnings.append(name_warning)
                        return name_row, warnings
                    warnings.append(f"matched_id_name_mismatch:{prediction.ingredient_id}:{prediction.raw_name}->{row.ingredient}")
            return row, warnings
        warnings.append(f"unknown_ingredient_id:{prediction.ingredient_id}")
    if prediction.raw_name:
        row, warning = match_calorie_row(prediction.raw_name, calories_by_name)
        if warning:
            warnings.append(warning)
        return row, warnings
    warnings.append("missing_ingredient_name_and_id")
    return None, warnings


def fallback_mass_range(portion_size: str) -> list[float]:
    normalized = normalize_name(portion_size)
    if normalized == "small":
        return [220.0, 320.0]
    if normalized == "large":
        return [480.0, 700.0]
    return [330.0, 450.0]


def round_kcal(value: float) -> int:
    numeric = float(value)
    if numeric <= 0:
        return 0
    return max(1, int(numeric + 0.5))


def calories_from_grams(grams: float, calories_per_100g: float) -> int:
    return round_kcal(float(grams) * float(calories_per_100g) / 100.0)


def grams_from_single_object_calories(count: float, calories_per_single_object: float, calories_per_100g: float) -> float | None:
    if calories_per_100g <= 0:
        return None
    return float(count) * float(calories_per_single_object) * 100.0 / float(calories_per_100g)


def portion_reference(row: CalorieTableRow) -> tuple[float | None, float, str]:
    if row.average_portion_kcal is not None:
        return row.average_portion_g, row.average_portion_kcal, "average_portion"
    if row.calories_per_single_object is not None:
        grams = grams_from_single_object_calories(1.0, row.calories_per_single_object, row.calories_per_100g)
        return grams, row.calories_per_single_object, "single_object_as_normal_portion"
    return 100.0, row.calories_per_100g, "fallback_100g_portion"


def portion_calories(row: CalorieTableRow, portion_category: str) -> tuple[int, float | None, float, float, str]:
    normalized = normalize_portion_category(portion_category)
    factor = PORTION_FACTORS[normalized]
    portion_g, portion_kcal, source = portion_reference(row)
    kcal = round_kcal(factor * portion_kcal)
    grams = round(factor * portion_g, 1) if portion_g is not None else None
    return kcal, grams, factor, portion_kcal, source


def per_instance_reference(row: CalorieTableRow) -> tuple[float | None, float | None, str]:
    if row.calories_per_single_object is not None:
        grams = grams_from_single_object_calories(1.0, row.calories_per_single_object, row.calories_per_100g)
        return grams, row.calories_per_single_object, "single_object"
    if row.average_portion_kcal is not None:
        return row.average_portion_g, row.average_portion_kcal, "average_representative_item"
    return 100.0, row.calories_per_100g, "fallback_100g_representative_item"


def is_abundance_scene(prediction: IngredientPrediction, portion_category: str, scene_context: str) -> bool:
    return bool(
        scene_context == "abundance_display"
        or (prediction.many_instances and portion_category in ABUNDANCE_PORTION_CATEGORIES)
    )


def should_use_corrected_count(
    prediction: IngredientPrediction,
    countable: bool,
    scene_context: str,
) -> bool:
    if prediction.count is None or prediction.count <= 0:
        return False
    if not countable:
        return False
    if prediction.many_instances:
        return False
    if scene_context == "abundance_display":
        return False
    if prediction.cut_code in {CUT_SLICE_UNCLEAR, CUT_CHOPPED_PILE}:
        return False
    if prediction.count_confidence is not None and prediction.count_confidence < COUNT_CONFIDENCE_THRESHOLD:
        return False
    return True


def normalize_composition_calories(
    parsed: Any,
    calories_by_name: dict[str, CalorieTableRow],
    calories_by_id: dict[int, CalorieTableRow],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not isinstance(parsed, (dict, list)):
        return {
            "total_kcal": None,
            "ingredients": [],
            "notes": "Could not parse VLM composition JSON.",
            "composition": {},
        }, ["unsupported_json_payload"]

    raw_predictions = parse_ingredient_predictions(parsed)
    raw_percentages = parse_percentages(parsed)
    scene_context = parse_scene_context(parsed)
    if not raw_predictions:
        warnings.append("no_ingredient_predictions_parsed")

    ingredients: list[dict[str, Any]] = []
    for prediction in raw_predictions:
        calorie_row, match_warnings = match_calorie_prediction(prediction, calories_by_name, calories_by_id)
        warnings.extend(match_warnings)
        if calorie_row is None:
            continue
        countable = calorie_row.calories_per_single_object is not None
        raw_count = prediction.count if should_use_corrected_count(prediction, countable, scene_context) else None
        count = round_count_to_half_step(raw_count)
        portion_category = normalize_portion_category(
            prediction.portion_category,
            default="none" if count is not None and countable else "normal",
        )
        abundance_scene = is_abundance_scene(prediction, portion_category, scene_context)
        if (prediction.many_instances or scene_context == "abundance_display") and prediction.count is not None:
            warnings.append(f"ignored_count_for_many_instances:{calorie_row.ingredient}")
        if prediction.count is not None and prediction.count > 0 and not countable:
            warnings.append(f"ignored_count_for_non_countable:{calorie_row.ingredient}")
        if count is not None and countable:
            portion_category = "none"
            assert calorie_row.calories_per_single_object is not None
            kcal = round_kcal(float(count) * calorie_row.calories_per_single_object)
            grams = grams_from_single_object_calories(count, calorie_row.calories_per_single_object, calorie_row.calories_per_100g)
            calorie_method = "single_object_count"
            count_used = True
            portion_factor = None
            calories_per_portion = None
            portion_source = None
            per_instance_source = None
        else:
            if portion_category == "none" and not abundance_scene:
                portion_category = "small"
                warnings.append(f"none_portion_for_visible_ingredient_used_small:{calorie_row.ingredient}")
            if count is None and countable and not abundance_scene:
                warnings.append(f"count_missing_or_uncertain_used_portion_category:{calorie_row.ingredient}")
            if abundance_scene:
                grams, representative_kcal, per_instance_source = per_instance_reference(calorie_row)
                kcal = round_kcal(representative_kcal or 0)
                portion_factor = None
                calories_per_portion = representative_kcal
                portion_source = None
                calorie_method = "abundance_per_instance"
            else:
                kcal, grams, portion_factor, calories_per_portion, portion_source = portion_calories(calorie_row, portion_category)
                per_instance_source = None
                calorie_method = "portion_category"
            count_used = False
            if portion_source == "fallback_100g_portion":
                warnings.append(f"average_portion_missing_used_100g:{calorie_row.ingredient}")
        per_instance_kcal = kcal if abundance_scene else (calorie_row.calories_per_single_object if countable else None)
        serving_note = None
        if abundance_scene:
            serving_note = (
                "Full image total is not estimated because this looks like a market or display abundance scene."
            )
        ingredients.append(
            {
                "id": calorie_row.ingredient_id,
                "name": calorie_row.ingredient,
                "vlm_name": prediction.raw_name,
                "count": count if count_used else None,
                "visible_pieces": prediction.visible_pieces,
                "cut_code": prediction.cut_code,
                "cut_label": CUT_LABELS.get(prediction.cut_code) if prediction.cut_code is not None else None,
                "whole_count_vlm": prediction.whole_count_vlm,
                "count_confidence": prediction.count_confidence,
                "many_instances": prediction.many_instances,
                "abundance_scene": abundance_scene,
                "portion_category": portion_category,
                "portion_factor": portion_factor,
                "countable": countable,
                "count_used": count_used,
                "calorie_method": calorie_method,
                "calories_per_100g": calorie_row.calories_per_100g,
                "calories_per_single_object": calorie_row.calories_per_single_object,
                "calories_per_portion": calories_per_portion,
                "average_portion_g": calorie_row.average_portion_g,
                "average_portion_kcal": calorie_row.average_portion_kcal,
                "per_instance_kcal": per_instance_kcal,
                "per_instance_source": per_instance_source,
                "estimated_quantity_g": round(grams, 1) if grams is not None else None,
                "kcal": kcal,
                "serving_note": serving_note,
            }
        )

    all_abundance = bool(ingredients) and all(bool(item.get("abundance_scene")) for item in ingredients)
    total_kcal = sum(int(item["kcal"]) for item in ingredients) if ingredients else None
    if all_abundance and len(ingredients) > 1:
        total_kcal = None
    return {
        "total_kcal": total_kcal,
        "estimation_scope": "per_instance_not_full_image" if all_abundance else "visible_or_selected_portions",
        "ingredients": ingredients,
        "notes": (
            "Countable ingredients use visible whole-object-equivalent counts rounded to whole or half units when reliable. "
            "Ingredients without a reliable count use VLM portion categories and average-portion calories. "
            "Single-ingredient abundance scenes report representative per-item calories instead of a full-image total."
        ),
        "composition": {
            "scene_context": scene_context,
            "raw_ingredients": parse_ingredient_list(parsed),
            "raw_ingredient_predictions": [
                {
                    "id": prediction.ingredient_id,
                    "name": prediction.raw_name,
                    "count": prediction.count,
                    "visible_pieces": prediction.visible_pieces,
                    "cut_code": prediction.cut_code,
                    "cut_label": CUT_LABELS.get(prediction.cut_code) if prediction.cut_code is not None else None,
                    "whole_count_vlm": prediction.whole_count_vlm,
                    "count_confidence": prediction.count_confidence,
                    "many_instances": prediction.many_instances,
                    "portion_category": prediction.portion_category,
                }
                for prediction in raw_predictions
            ],
            "raw_percentages": raw_percentages,
            "raw_portions": parse_portion_map(parsed),
        },
    }, warnings


def kcal_phrase(kcal: int | None) -> str:
    if kcal is None:
        return "unavailable"
    return f"{kcal} kcal"


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_count(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_answer(calories: dict[str, Any]) -> str:
    total = calories.get("total_kcal") if isinstance(calories.get("total_kcal"), int) else None
    scope = str(calories.get("estimation_scope") or "")
    ingredients = calories.get("ingredients") if isinstance(calories.get("ingredients"), list) else []
    abundance_count = sum(1 for item in ingredients if isinstance(item, dict) and item.get("abundance_scene"))
    if scope == "per_instance_not_full_image":
        if abundance_count == 1 and total is not None:
            answer = f"Estimated calories for one representative item: {kcal_phrase(total)}."
        else:
            answer = "This looks like an abundance/display scene, so the full image total is not estimated."
    elif scope == "average_portion_not_full_image":
        answer = f"Estimated calories for an eating-portion assumption: {kcal_phrase(total)}."
    else:
        answer = f"Estimated total dish calories: {kcal_phrase(total)}."
    parts: list[str] = []
    serving_notes: list[str] = []
    for item in ingredients[:8]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        kcal = item.get("kcal") if isinstance(item.get("kcal"), int) else None
        count = item.get("count") if isinstance(item.get("count"), (int, float)) else None
        per_object = item.get("calories_per_single_object")
        per_instance = item.get("per_instance_kcal")
        portion_category = str(item.get("portion_category") or "").strip()
        serving_note = str(item.get("serving_note") or "").strip()
        if serving_note:
            serving_notes.append(serving_note)
        if name and kcal is not None and count is not None and isinstance(per_object, (int, float)):
            parts.append(f"{name} {kcal} kcal ({format_count(float(count))} x {format_number(float(per_object))} kcal)")
        elif name and kcal is not None and item.get("abundance_scene") and isinstance(per_instance, (int, float)):
            parts.append(f"{name} about {format_number(float(per_instance))} kcal each")
        elif name and kcal is not None and portion_category and portion_category != "none":
            parts.append(f"{name} {kcal} kcal ({portion_category} portion)")
        elif name and kcal is not None:
            parts.append(f"{name} {kcal} kcal")
        elif name:
            parts.append(name)
    if parts:
        answer += " Ingredient calories: " + "; ".join(parts) + "."
    if serving_notes and scope != "per_instance_not_full_image":
        answer += " " + " ".join(serving_notes[:2])
    return answer


def load_runtime(args: argparse.Namespace) -> CalorieRuntime:
    configure_hf_token(args)
    calories_by_name = read_calorie_table(args.calories_csv)
    calories_by_id = {row.ingredient_id: row for row in calories_by_name.values()}
    candidate_filter = None
    load_siglip2_sec = 0.0
    text_embeddings_sec = 0.0
    if (
        args.calorie_siglip_filter
        and not args.defer_calorie_siglip_filter
        and not args.mock_qwen_json
    ):
        candidate_filter, candidate_timings = CalorieCandidateFilter.from_args(args)
        load_siglip2_sec = float(candidate_timings.get("load_siglip2_sec") or 0.0)
        text_embeddings_sec = float(candidate_timings.get("text_embeddings_sec") or 0.0)
    estimator = None
    load_qwen_sec = 0.0
    if not args.mock_qwen_json and not args.defer_calorie_qwen:
        started = time.perf_counter()
        estimator = QwenCompositionEstimator(args)
        load_qwen_sec = time.perf_counter() - started
    return CalorieRuntime(
        args=args,
        estimator=estimator,
        calories_by_name=calories_by_name,
        calories_by_id=calories_by_id,
        candidate_filter=candidate_filter,
        model_load_timings={
            "load_siglip2_sec": load_siglip2_sec,
            "text_embeddings_sec": text_embeddings_sec,
            "load_qwen_sec": load_qwen_sec,
            "load_models_and_text_sec": load_siglip2_sec + text_embeddings_sec + load_qwen_sec,
        },
    )


def attach_task1_candidate_filter(runtime: CalorieRuntime, task1_runtime: Any, top_k: int | None = None) -> None:
    if not runtime.args.calorie_siglip_filter:
        return
    runtime.candidate_filter = CalorieCandidateFilter.from_task1_runtime(
        task1_runtime,
        top_k or int(runtime.args.calorie_candidate_top_k),
        candidate_list_mode=runtime.args.calorie_candidate_list_mode,
        dynamic_candidate_relative_delta=runtime.args.calorie_dynamic_candidate_relative_delta,
        dynamic_candidate_min_k=runtime.args.calorie_dynamic_candidate_min_k,
        dynamic_candidate_max_k=runtime.args.calorie_dynamic_candidate_max_k,
    )
    runtime.model_load_timings["load_siglip2_sec"] = 0.0
    runtime.model_load_timings["text_embeddings_sec"] = 0.0
    runtime.model_load_timings["load_models_and_text_sec"] = float(runtime.model_load_timings.get("load_qwen_sec") or 0.0)


def attach_task1_qwen_estimator(runtime: CalorieRuntime, task1_runtime: Any) -> None:
    qwen = getattr(task1_runtime, "qwen", None)
    if qwen is None:
        return
    runtime.estimator = SharedTask1QwenCompositionEstimator(qwen, runtime.args)
    runtime.model_load_timings["load_qwen_sec"] = 0.0
    runtime.model_load_timings["load_models_and_text_sec"] = (
        float(runtime.model_load_timings.get("load_siglip2_sec") or 0.0)
        + float(runtime.model_load_timings.get("text_embeddings_sec") or 0.0)
    )


def build_single_report(runtime: CalorieRuntime, image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    query_t0 = time.perf_counter()
    candidate_rows: list[CalorieTableRow] | None = None
    candidate_trace: list[dict[str, Any]] = []
    candidate_filter_timings: dict[str, float] = {}
    candidate_warnings: list[str] = []
    candidate_filter_used = False
    candidate_list_policy: dict[str, Any] | None = None
    candidate_count = 0
    if runtime.candidate_filter is not None:
        candidates, candidate_filter_timings, candidate_list_policy = runtime.candidate_filter.run(image_path)
        candidate_count = len(candidates)
        candidate_rows, candidate_trace, candidate_warnings = calorie_rows_from_candidates(candidates, runtime.calories_by_name)
        candidate_filter_used = bool(candidate_rows)
        if not candidate_rows:
            candidate_rows = None
            candidate_warnings.append("candidate_filter_no_calorie_matches_used_full_table")
    prompt = build_selected_composition_prompt(
        runtime.calories_by_name,
        candidate_rows,
        args.calorie_counting_logic,
    )
    started = time.perf_counter()
    if args.mock_qwen_json:
        raw_text = args.mock_qwen_json
    else:
        if runtime.estimator is None:
            raise RuntimeError("Qwen estimator was not initialized")
        raw_text = runtime.estimator.generate(image_path, prompt)
    qwen_sec = time.perf_counter() - started
    parsed, parse_warning = extract_json_object(raw_text)
    parse_warnings = [parse_warning] if parse_warning else []
    parse_warnings.extend(candidate_warnings)
    calories, normalize_warnings = normalize_composition_calories(parsed, runtime.calories_by_name, runtime.calories_by_id)
    parse_warnings.extend(normalize_warnings)
    total_image_sec = time.perf_counter() - query_t0
    answer = build_answer(calories)
    return {
        "schema_version": "dishcovery_calorie_composition_qwen_v13",
        "image": image_path.name,
        "image_path": str(image_path),
        "models": {
            "qwen_model": args.qwen_model,
            "vlm_model": args.qwen_model,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "calories_csv": str(args.calories_csv),
            "calorie_logic": (
                f"{getattr(runtime.candidate_filter, 'filter_name', 'siglip2')}_then_compact_qwen_counts_portions_and_abundance"
                if runtime.candidate_filter is not None
                else "full_table_then_compact_qwen_counts_portions_and_abundance"
            ),
            "calorie_candidate_filter": getattr(runtime.candidate_filter, "filter_name", "full_table"),
            "calorie_candidate_filter_used": candidate_filter_used,
            "calorie_candidate_top_k": getattr(runtime.candidate_filter, "top_k", None),
            "calorie_candidate_list_mode": getattr(runtime.candidate_filter, "candidate_list_mode", None),
            "calorie_dynamic_candidate_relative_delta": getattr(runtime.candidate_filter, "dynamic_candidate_relative_delta", None),
            "calorie_dynamic_candidate_min_k": getattr(runtime.candidate_filter, "dynamic_candidate_min_k", None),
            "calorie_dynamic_candidate_max_k": getattr(runtime.candidate_filter, "dynamic_candidate_max_k", None),
            "effective_calorie_candidate_k": candidate_count if runtime.candidate_filter is not None else None,
            "calorie_counting_logic": args.calorie_counting_logic,
            "qwen_min_pixels": args.qwen_min_pixels,
            "qwen_max_pixels": args.qwen_max_pixels,
            "qwen_max_new_tokens": args.qwen_max_new_tokens,
            "qwen_runtime_source": getattr(runtime.estimator, "source", "none") if runtime.estimator is not None else "none",
            "mock_qwen_json": bool(args.mock_qwen_json),
        },
        "calories": calories,
        "answer": answer,
        "qwen": {
            "prompt": prompt,
            "raw_text": raw_text,
            "parsed": parsed,
            "parse_warnings": parse_warnings,
            "candidate_filter": {
                "enabled": runtime.candidate_filter is not None,
                "used": candidate_filter_used,
                "source": getattr(runtime.candidate_filter, "source", "none") if runtime.candidate_filter else "none",
                "top_k": getattr(runtime.candidate_filter, "top_k", None),
                "candidate_list_mode": getattr(runtime.candidate_filter, "candidate_list_mode", None),
                "candidate_list_policy": candidate_list_policy,
                "effective_k": candidate_count if runtime.candidate_filter is not None else None,
                "matched_count": len(candidate_rows or []),
                "candidates": candidate_trace,
            },
        },
        "timings_sec": {
            **runtime.model_load_timings,
            **candidate_filter_timings,
            "qwen_sec": qwen_sec,
            "total_image_sec": total_image_sec,
        },
    }


def main() -> None:
    args = parse_args()
    if args.image is None:
        raise SystemExit("--image is required")
    runtime = load_runtime(args)
    report = build_single_report(runtime, args.image, args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
