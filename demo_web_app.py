#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from demo.command_parser import CommandParseError, DemoCommand, TaskName, parse_command
from demo.history import ResultHistory
from demo.nutrition import NutritionStore, parse_date
from demo.stt import DEFAULT_WHISPER_MODEL, SpeechToText
from demo.task_router import TaskRouter, TaskRouterConfig, split_extra_args


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "demo" / "web_static"
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024
DEFAULT_TASK1_SKIP_VLM_REL_GAP_THRESHOLD = 0.25
GOOD_SUBSET_TASK2_SIGLIP_KEEP_GAP = 2.0
DEFAULT_PIPER_MODEL = Path("models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
DEFAULT_PIPER_CONFIG = Path("models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
DEFAULT_SHOWCASE_IMAGE_DIR = ROOT / "external_assets/images/showcase"
DEFAULT_SHOWCASE_IMAGES_LIST = ROOT / "benchmark_inputs/demo_showcase/good_predictions.txt"
DEMO_SUBSET_IMAGE_DIR = DEFAULT_SHOWCASE_IMAGE_DIR
DEMO_SUBSET_LISTS = {
    "good": DEFAULT_SHOWCASE_IMAGES_LIST,
}
TASKS: set[str] = {"task1", "task2", "calories", "both"}
TASK_COMMAND_TEXT = {
    "task1": "find ingredients",
    "task2": "describe the dish",
    "calories": "estimate calories",
    "both": "execute both",
}


def default_script(name: str) -> Path:
    root_path = Path(name)
    code_path = Path("code") / name
    return root_path if root_path.exists() else code_path


def drop_options(command: list[str], names: set[str]) -> list[str]:
    """Remove CLI options and their single values from a generated command."""
    filtered: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in names:
            index += 2
            continue
        filtered.append(item)
        index += 1
    return filtered


def configure_quantized_backends(args: argparse.Namespace) -> None:
    """Bind the browser app to this repository's fixed quantized pipelines."""
    from scripts.repro_common import (
        ReproPaths,
        calories_command,
        task1_command,
        task2_command,
        validate_runtime,
    )

    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    for task in ("task1", "task2", "calories"):
        validate_runtime(paths, task)

    placeholder_image = ROOT / "web-demo-input.jpg"
    placeholder_output = ROOT / "demo_runs/web/placeholder.json"
    task1 = task1_command(paths, output_json=placeholder_output, image=placeholder_image)[2:]
    task2 = task2_command(paths, output_json=placeholder_output, image=placeholder_image)[2:]
    calories = calories_command(paths, output_json=placeholder_output, image=placeholder_image)[2:]
    ignored = {"--image", "--output-json", "--predictions-csv", "--final-score-mode", "--siglip-keep-gap"}
    args.task1_extra_args = shlex.join([*drop_options(task1, ignored), *split_extra_args(args.task1_extra_args)])
    args.task2_extra_args = shlex.join([*drop_options(task2, ignored), *split_extra_args(args.task2_extra_args)])
    args.calorie_extra_args = shlex.join([*drop_options(calories, ignored), *split_extra_args(args.calorie_extra_args)])


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc
    if length <= 0:
        return {}
    if length > MAX_IMAGE_BYTES * 2:
        raise ValueError("Request body is too large")
    body = handler.rfile.read(length)
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def extract_hf_env_value(line: str, key: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    name, value = stripped.split("=", 1)
    if name.strip() != key:
        return None
    return value.strip().strip("'").strip('"') or None


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        for raw_line in text.splitlines():
            value = extract_hf_env_value(raw_line, key)
            if value:
                return value
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line.strip().strip("'").strip('"') or None
    return None


def read_hf_token(args: argparse.Namespace) -> str | None:
    if args.hf_token.strip():
        return args.hf_token.strip()
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    if args.hf_token_file is not None:
        token = read_token_file(args.hf_token_file.expanduser())
        if token:
            return token
    for path in (Path(".env"), Path(".hf_token"), Path.home() / ".cache/huggingface/token"):
        token = read_token_file(path.expanduser())
        if token:
            return token
    return None


def resolve_preload_backends(value: str, backend_mode: str) -> list[str]:
    if backend_mode != "warm":
        return []
    raw_value = value.strip()
    normalized = raw_value.lower()
    if normalized in {"", "auto", "default"}:
        return ["task1", "task2_fast", "task2_full", "calories"]
    if normalized in {"none", "off", "false", "0"}:
        return []
    if normalized in {"all", "*"}:
        return ["task1", "task2_fast", "task2_full", "calories"]
    if normalized == "both":
        return ["task1", "task2_fast"]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def format_latency(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 2)


def human_list(text: str) -> str:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) <= 1:
        return text
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def spoken_answer(result: dict[str, Any]) -> str:
    task = result.get("task")
    mode = str(result.get("mode") or "fast")
    answer = str(result.get("answer") or "").strip()
    if task == "both":
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        task1 = details.get("task1") if isinstance(details.get("task1"), dict) else {}
        task2 = details.get("task2") if isinstance(details.get("task2"), dict) else {}
        ingredients = str(task1.get("answer") or "no ingredients selected").strip()
        caption = str(task2.get("answer") or "no caption selected").strip()
        return f"Detected {human_list(ingredients)}. Selected the caption: {caption}."
    if task == "task1" and mode != "compare":
        return f"Detected {human_list(answer)}."
    if task == "task2" and mode != "compare":
        return f"Selected the caption: {answer}."
    return answer


def top_caption_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    raw = details.get("raw") if isinstance(details.get("raw"), dict) else {}
    traces = raw.get("traces") if isinstance(raw.get("traces"), list) else []
    trace = traces[0] if traces and isinstance(traces[0], dict) else {}
    candidates = trace.get("ranked_candidates") if isinstance(trace.get("ranked_candidates"), list) else []
    rows: list[dict[str, Any]] = []
    for item in candidates[:5]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "caption": str(item.get("caption") or item.get("text") or "").strip(),
                "category": str(item.get("cat") or item.get("category") or "").strip(),
                "score": item.get("final_score", item.get("score")),
                "siglip_score": item.get("siglip_score"),
                "rerank_score": item.get("rerank_score"),
            }
        )
    return rows


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


TIMING_LABELS = {
    "load_models_and_text_sec": "Load models and text",
    "load_siglip2_sec": "Load SigLIP2",
    "load_qwen_sec": "Load VLM",
    "text_embeddings_sec": "Text embeddings",
    "siglip2_image_and_topk_sec": "Image embedding + top-k",
    "qwen_sec": "VLM generation",
    "fusion_and_selection_sec": "Fusion and selection",
    "total_image_sec": "Total image",
    "siglip_image_and_recall_sec": "SigLIP image + recall",
    "rerank_sec": "Reranker scoring",
    "final_selection_sec": "Final selection",
    "caption_embeddings_sec": "Caption embeddings",
    "load_reranker_sec": "Load reranker",
    "siglip2_candidate_filter_sec": "SigLIP2 calorie filter",
    "candidate_filter_sec": "Candidate filter",
    "calorie_filter_sec": "Calorie candidate filter",
    "browser_image_save_sec": "Read browser image",
}

HIDDEN_TIMING_KEYS = {
    "load_models_and_text_sec",
    "load_siglip2_sec",
    "load_qwen_sec",
    "load_reranker_sec",
    "text_embeddings_sec",
    "caption_embeddings_sec",
    "final_selection_sec",
}


def timing_label(key: str) -> str:
    if key in TIMING_LABELS:
        return TIMING_LABELS[key]
    text = key.removesuffix("_sec").replace("_", " ")
    return text[:1].upper() + text[1:]


def timing_rows_from_dict(timings: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if not isinstance(timings, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key, value in timings.items():
        if str(key) in HIDDEN_TIMING_KEYS:
            continue
        number = numeric_value(value)
        if number is None:
            continue
        label = timing_label(str(key))
        if prefix:
            label = f"{prefix}: {label}"
        rows.append({"key": str(key), "label": label, "sec": number})
    return rows


def result_timing_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    raw = details.get("raw") if isinstance(details.get("raw"), dict) else {}
    task = str(result.get("task") or "")
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    rows = timing_rows_from_dict(diagnostics.get("web_timings_sec"), prefix="Input")
    rows.extend(timing_rows_from_dict(raw.get("timings_sec"), prefix="Setup" if task == "task2" else ""))
    if task == "task2":
        traces = raw.get("traces") if isinstance(raw.get("traces"), list) else []
        trace = traces[0] if traces and isinstance(traces[0], dict) else {}
        rows.extend(timing_rows_from_dict(trace.get("timings_sec")))
    return rows


def diagnostics_summary(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    power = diagnostics.get("power") if isinstance(diagnostics.get("power"), dict) else {}
    resources = diagnostics.get("resources") if isinstance(diagnostics.get("resources"), dict) else {}
    system = diagnostics.get("system") if isinstance(diagnostics.get("system"), dict) else {}
    return {
        "timings": result_timing_rows(result),
        "power": power,
        "resources": resources,
        "system": system,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    task = str(result.get("task") or "")
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    summary: dict[str, Any] = {
        "task": task,
        "mode": result.get("mode"),
        "answer": str(result.get("answer") or "").strip(),
        "latency_sec": format_latency(result.get("latency_sec")),
        "rerank_or_vlm": str(result.get("rerank_or_vlm") or "").strip(),
        "spoken_answer": spoken_answer(result),
        "backend_json": "",
        "diagnostics": diagnostics_summary(result),
        "sections": [],
    }
    if task == "both":
        task1 = details.get("task1") if isinstance(details.get("task1"), dict) else {}
        task2 = details.get("task2") if isinstance(details.get("task2"), dict) else {}
        summary["sections"] = [summarize_result(task1), summarize_result(task2)]
        return summary
    if task == "task1":
        labels = details.get("selected_labels") if isinstance(details.get("selected_labels"), list) else []
        summary["selected_labels"] = [str(item) for item in labels if str(item).strip()]
    elif task == "task2":
        summary["prediction_caption"] = details.get("prediction_caption", summary["answer"])
        summary["prediction_category"] = str(details.get("prediction_cat") or "").strip()
        summary["final_score_mode"] = str(details.get("final_score_mode") or "").strip()
        if "rerank_applied" in details:
            summary["rerank_applied"] = bool(details.get("rerank_applied"))
        if isinstance(details.get("siglip_top_gap"), (int, float)):
            summary["siglip_top_gap"] = details.get("siglip_top_gap")
        summary["top_candidates"] = top_caption_candidates(result)
    elif task == "calories":
        calories = details.get("calories") if isinstance(details.get("calories"), dict) else {}
        summary["calories"] = calories
    if isinstance(details.get("backend_json"), str):
        summary["backend_json"] = details["backend_json"]
    return summary


def mime_to_suffix(mime: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(mime.lower(), ".jpg")


def save_data_url_image(image_data: str, run_dir: Path, image_name: str | None = None) -> Path:
    if not image_data or not isinstance(image_data, str):
        raise ValueError("image_data is required")
    mime = "image/jpeg"
    payload = image_data
    if image_data.startswith("data:"):
        header, sep, payload = image_data.partition(",")
        if not sep:
            raise ValueError("Invalid image data URL")
        mime = header[5:].split(";", 1)[0] or mime
    raw = base64.b64decode(payload, validate=True)
    if not raw:
        raise ValueError("Decoded image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Decoded image is too large")
    suffix = mime_to_suffix(mime)
    if image_name:
        candidate = Path(image_name).name
        file_suffix = Path(candidate).suffix.lower()
        if file_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg" if file_suffix == ".jpeg" else file_suffix
    path = run_dir / f"browser_image{suffix}"
    path.write_bytes(raw)
    return path


def decode_data_url(value: str, max_bytes: int, default_mime: str) -> tuple[bytes, str]:
    if not value or not isinstance(value, str):
        raise ValueError("Data URL is required")
    mime = default_mime
    payload = value
    if value.startswith("data:"):
        header, sep, payload = value.partition(",")
        if not sep:
            raise ValueError("Invalid data URL")
        mime = header[5:].split(";", 1)[0] or default_mime
    raw = base64.b64decode(payload, validate=True)
    if not raw:
        raise ValueError("Decoded payload is empty")
    if len(raw) > max_bytes:
        raise ValueError("Decoded payload is too large")
    return raw, mime


def save_data_url_audio(audio_data: str, output_dir: Path) -> Path:
    raw, mime = decode_data_url(audio_data, MAX_AUDIO_BYTES, "audio/wav")
    if mime not in {"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"}:
        raise ValueError(f"Unsupported audio type: {mime}; expected WAV")
    path = output_dir / "voice_command.wav"
    path.write_bytes(raw)
    return path


def normalize_tts_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())[:1200]


def stage_for_message(task: str, message: str) -> str:
    text = message.lower()
    if "queue" in text or "waiting" in text:
        return "queue"
    if "browser image" in text or "captur" in text or "saved" in text or text == "image ready":
        return "input"
    if "load" in text or "model" in text or "backend ready" in text:
        return "load"
    if task == "task1":
        if "vlm" in text or "qwen" in text or "selector" in text:
            return "vlm"
        if "candidate" in text or "rank" in text or "visual" in text or "embedding" in text or "encoding image" in text:
            return "retrieve"
    elif task == "task2":
        if "caption" in text or "recall" in text:
            return "caption"
        if "rerank" in text or "score" in text:
            return "rerank"
    elif task == "calories":
        if "composition" in text or "vlm" in text:
            return "composition"
        if "calorie" in text or "portion" in text or "grams" in text:
            return "math"
    elif task == "both":
        if "task 1" in text:
            return "task1"
        if "task 2" in text:
            return "task2"
    if "format" in text or "result" in text or "output" in text:
        return "output"
    return "run"


def sample_images(image_dir: Path, image_list: Path | None, max_images: int) -> list[Path]:
    paths: list[Path] = []
    if image_list is not None and image_list.exists():
        for raw_line in image_list.read_text(encoding="utf-8").splitlines():
            name = raw_line.strip()
            if not name:
                continue
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = image_dir / candidate
            if candidate.exists() and candidate.is_file():
                paths.append(candidate)
            if len(paths) >= max_images:
                break
    if not paths and image_dir.exists():
        allowed = {".jpg", ".jpeg", ".png", ".webp"}
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed:
                paths.append(path)
            if len(paths) >= max_images:
                break
    return paths


def apply_demo_subset(args: argparse.Namespace) -> None:
    if args.demo_subset == "default":
        return
    subset_list = DEMO_SUBSET_LISTS[args.demo_subset]
    if not subset_list.exists():
        raise FileNotFoundError(f"Curated demo subset list not found: {subset_list}")
    args.sample_image_dir = DEMO_SUBSET_IMAGE_DIR
    args.sample_images_list = subset_list


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def using_good_demo_subset(args: argparse.Namespace) -> bool:
    if args.demo_subset == "good":
        return True
    sample_list = args.sample_images_list
    if not sample_list.is_absolute():
        sample_list = ROOT / sample_list
    good_list = DEMO_SUBSET_LISTS["good"]
    if not good_list.is_absolute():
        good_list = ROOT / good_list
    return same_path(sample_list, good_list)


def apply_good_subset_task2_defaults(args: argparse.Namespace) -> None:
    if not using_good_demo_subset(args):
        return
    if args.task2_fast_final_score_mode != "siglip_guarded":
        return
    if "--siglip-keep-gap" in str(args.task2_extra_args):
        return
    gap_arg = f"--siglip-keep-gap {GOOD_SUBSET_TASK2_SIGLIP_KEEP_GAP:.1f}"
    args.task2_extra_args = " ".join(part for part in [str(args.task2_extra_args).strip(), gap_arg] if part)


@dataclass
class JobEvent:
    stage: str
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class Job:
    id: str
    command: dict[str, Any]
    status: str = "queued"
    task: str = ""
    mode: str = "fast"
    transcript: str = ""
    run_dir: str = ""
    image_path: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    traceback_text: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[JobEvent] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add_event(self, stage: str, message: str) -> None:
        with self.lock:
            self.events.append(JobEvent(stage=stage, message=message))
            self.updated_at = time.time()

    def set_status(self, status: str, *, error: str = "", traceback_text: str = "") -> None:
        with self.lock:
            self.status = status
            self.error = error
            self.traceback_text = traceback_text
            self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "task": self.task,
                "mode": self.mode,
                "transcript": self.transcript,
                "command": self.command,
                "run_dir": self.run_dir,
                "image_path": self.image_path,
                "result": self.result,
                "error": self.error,
                "traceback": self.traceback_text,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "events": [event.to_dict() for event in self.events],
            }


class DemoController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.history = ResultHistory(args.history_root)
        self.nutrition = NutritionStore(args.nutrition_root)
        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.RLock()
        self.backend_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.stt_lock = threading.Lock()
        self.preload_lock = threading.RLock()
        self.preload_status = "skipped" if args.mock_backend else "idle"
        self.preload_events: list[JobEvent] = []
        self.preload_targets = resolve_preload_backends(args.preload_backends, args.backend_mode)
        self.sample_paths = sample_images(args.sample_image_dir, args.sample_images_list, args.sample_image_count)
        self.mock_backend = bool(args.mock_backend)
        self.router: TaskRouter | None = None
        self._stt: SpeechToText | None = None
        if not self.mock_backend:
            self.router = TaskRouter(
                TaskRouterConfig(
                    task1_script=args.task1_script,
                    task2_script=args.task2_script,
                    calorie_script=args.calorie_script,
                    python_executable=args.backend_python or sys.executable,
                    task1_fast_skip_vlm_rel_gap_threshold=args.task1_fast_skip_vlm_rel_gap_threshold,
                    task2_fast_final_score_mode=args.task2_fast_final_score_mode,
                    task2_full_final_score_mode=args.task2_full_final_score_mode,
                    task2_reranker_size=args.task2_reranker_size or None,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                    hf_token=read_hf_token(args),
                    execution_mode=args.backend_mode,
                    task1_extra_args=split_extra_args(args.task1_extra_args),
                    task2_extra_args=split_extra_args(args.task2_extra_args),
                    calorie_extra_args=split_extra_args(args.calorie_extra_args),
                    diagnostics_power=args.diagnostics_power,
                    diagnostics_power_interval_ms=args.diagnostics_power_interval_ms,
                )
            )

    def start_preload(self) -> None:
        targets = self.preload_targets
        if self.mock_backend or not targets:
            with self.preload_lock:
                self.preload_status = "skipped"
            return
        thread = threading.Thread(target=self._preload_worker, args=(targets,), name="web-demo-preload", daemon=True)
        thread.start()

    def _preload_worker(self, targets: list[str]) -> None:
        assert self.router is not None
        with self.preload_lock:
            self.preload_status = "running"
            self.preload_events.append(JobEvent("load", f"Preloading: {', '.join(targets)}"))
        try:
            with self.backend_lock:
                self.router.preload(targets, progress=lambda message: self._add_preload_event("load", message))
            with self.preload_lock:
                self.preload_status = "ready"
                self.preload_events.append(JobEvent("load", "Warm backends ready"))
        except Exception as exc:
            with self.preload_lock:
                self.preload_status = "error"
                self.preload_events.append(JobEvent("error", f"{type(exc).__name__}: {exc}"))

    def _add_preload_event(self, stage: str, message: str) -> None:
        with self.preload_lock:
            self.preload_events.append(JobEvent(stage, message))

    def status(self) -> dict[str, Any]:
        with self.jobs_lock:
            active = sum(1 for job in self.jobs.values() if job.status in {"queued", "running"})
        with self.preload_lock:
            preload = {
                "status": self.preload_status,
                "targets": list(self.preload_targets),
                "events": [event.to_dict() for event in self.preload_events[-20:]],
            }
        return {
            "ok": True,
            "mock_backend": self.mock_backend,
            "backend_mode": self.args.backend_mode,
            "task2": {
                "fast_final_score_mode": self.args.task2_fast_final_score_mode,
                "reranker_size": self.args.task2_reranker_size,
                "extra_args": self.args.task2_extra_args,
            },
            "diagnostics": {
                "power": self.args.diagnostics_power,
                "power_interval_ms": self.args.diagnostics_power_interval_ms,
            },
            "preload": preload,
            "active_jobs": active,
            "sample_images": len(self.sample_paths),
            "tts": {
                "available": self.tts_available(),
                "engine": "piper",
            },
            "voice": {
                "engine": "server_whisper_wav",
                "model": self.args.web_stt_model,
            },
        }

    def tts_available(self) -> bool:
        return (
            self.args.web_tts_enabled
            and self.args.piper_model is not None
            and self.args.piper_model.exists()
            and shutil.which(self.args.piper_command) is not None
        )

    def synthesize_tts(self, text: str) -> bytes:
        clean = normalize_tts_text(text)
        if not clean:
            raise ValueError("TTS text is empty")
        if not self.tts_available():
            raise RuntimeError(
                f"Local Piper TTS is unavailable: command={self.args.piper_command!r}, model={self.args.piper_model}"
            )
        with self.tts_lock:
            with tempfile.NamedTemporaryFile(prefix="dishcovery_web_tts_", suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
            try:
                cmd = [self.args.piper_command, "--model", str(self.args.piper_model), "--output_file", str(wav_path)]
                if self.args.piper_config is not None and self.args.piper_config.exists():
                    cmd.extend(["--config", str(self.args.piper_config)])
                subprocess.run(cmd, input=clean + "\n", text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return wav_path.read_bytes()
            except subprocess.CalledProcessError as exc:
                message = (exc.stderr or exc.stdout or "").strip() or "no Piper output"
                raise RuntimeError(f"Piper TTS failed: {message}") from exc
            finally:
                try:
                    wav_path.unlink()
                except OSError:
                    pass

    def _load_stt(self) -> SpeechToText:
        if self._stt is None:
            self._stt = SpeechToText(
                backend="whisper",
                model=self.args.web_stt_model,
                command_template=self.args.web_stt_command,
                listen_seconds=float(self.args.web_voice_seconds),
                sample_rate=int(self.args.web_stt_sample_rate),
                device=self.args.web_stt_device,
                vad_enabled=False,
            )
        return self._stt

    def transcribe_voice_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.history.create_run_dir()
        audio_path = save_data_url_audio(str(payload.get("audio_data") or ""), run_dir)
        try:
            with self.stt_lock:
                transcript = self._load_stt()._transcribe_whisper_wav(audio_path)
        except Exception:
            raise
        transcript = " ".join(str(transcript or "").strip().split())
        if not transcript:
            raise CommandParseError("No voice command was transcribed")
        command = parse_command(transcript)
        if command.action != "run" or command.task is None:
            raise CommandParseError("Voice command did not map to a runnable task")
        return {
            "transcript": transcript,
            "task": command.task,
            "mode": command.mode or "fast",
            "command": command.to_dict(),
            "audio_path": str(audio_path),
            "run_dir": str(run_dir),
        }

    def create_job(self, payload: dict[str, Any]) -> Job:
        command, transcript = self._command_from_payload(payload)
        image_data = str(payload.get("image_data") or "")
        image_name = str(payload.get("image_name") or "browser-image.jpg")
        if not image_data:
            raise ValueError("image_data is required")
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            task=str(command.task or ""),
            mode=str(command.mode or "fast"),
            transcript=transcript,
            command=command.to_dict(),
        )
        with self.jobs_lock:
            self.jobs[job_id] = job
            self._trim_jobs_locked()
        thread = threading.Thread(
            target=self._run_job,
            args=(job, command, image_data, image_name),
            name=f"web-demo-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def random_sample_path(self) -> Path | None:
        if not self.sample_paths:
            return None
        return random.choice(self.sample_paths)

    def _trim_jobs_locked(self) -> None:
        if len(self.jobs) <= self.args.max_jobs:
            return
        ordered = sorted(self.jobs.values(), key=lambda job: job.created_at)
        for job in ordered[: max(0, len(ordered) - self.args.max_jobs)]:
            self.jobs.pop(job.id, None)

    def _command_from_payload(self, payload: dict[str, Any]) -> tuple[DemoCommand, str]:
        transcript = str(payload.get("transcript") or "").strip()
        if transcript:
            command = parse_command(transcript)
            if command.action != "run":
                raise CommandParseError("Only run commands are supported in the web demo")
            return command, transcript
        task = str(payload.get("task") or "").strip()
        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(sorted(TASKS))}")
        return DemoCommand(action="run", task=task, mode="fast", text=TASK_COMMAND_TEXT[task]), ""

    def _run_job(self, job: Job, command: DemoCommand, image_data: str, image_name: str) -> None:
        job.set_status("running")
        try:
            run_dir = self.history.create_run_dir()
            job.run_dir = str(run_dir)
            job.add_event("input", "Saving browser image")
            image_save_started = time.perf_counter()
            image_path = save_data_url_image(image_data, run_dir, image_name)
            image_save_sec = time.perf_counter() - image_save_started
            job.image_path = str(image_path)
            job.add_event("input", "Image ready")

            acquired = self.backend_lock.acquire(blocking=False)
            if not acquired:
                job.add_event("queue", "Waiting for active pipeline")
                self.backend_lock.acquire()
            try:
                if self.mock_backend:
                    result = self._mock_result(job, command)
                else:
                    if self.router is None:
                        raise RuntimeError("Router is not initialized")
                    task = str(command.task or "")
                    job.add_event(stage_for_message(task, "Starting pipeline"), "Starting pipeline")
                    result = self.router.run(
                        image_path,
                        command,
                        run_dir / "backend",
                        progress=lambda message: job.add_event(stage_for_message(task, message), message),
                    )
            finally:
                self.backend_lock.release()

            self._attach_web_diagnostics(result, {"browser_image_save_sec": image_save_sec})
            job.add_event("output", "Rendering output")
            presentation = summarize_result(result)
            speech = str(presentation.get("spoken_answer") or result.get("answer") or "")
            self.history.save(run_dir, command.text, command.to_dict(), image_path, result, speech)
            job.result = presentation
            job.add_event("done", "Result ready")
            job.set_status("completed")
        except Exception as exc:
            trace = traceback.format_exc()
            job.add_event("error", f"{type(exc).__name__}: {exc}")
            job.set_status("error", error=f"{type(exc).__name__}: {exc}", traceback_text=trace)

    def _attach_web_diagnostics(self, result: dict[str, Any], timings: dict[str, float]) -> None:
        diagnostics = result.setdefault("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            result["diagnostics"] = diagnostics
        web_timings = diagnostics.setdefault("web_timings_sec", {})
        if not isinstance(web_timings, dict):
            web_timings = {}
            diagnostics["web_timings_sec"] = web_timings
        for key, value in timings.items():
            web_timings[key] = float(value)

    def _mock_diagnostics(self, latency_sec: float) -> dict[str, Any]:
        return {
            "wall_sec": latency_sec + 0.08,
            "latency_sec": latency_sec,
            "power": {
                "enabled": True,
                "available": True,
                "sample_count": 8,
                "avg_power_w": 11.4,
                "median_power_w": 11.2,
                "min_power_w": 8.8,
                "max_power_w": 13.1,
                "energy_j": 11.4 * (latency_sec + 0.08),
                "selected_power_source": "sum:VDD_GPU_SOC+VDD_CPU_CV",
                "rail_avg_power_w": {"VDD_GPU_SOC": 10.5, "VDD_CPU_CV": 0.9},
            },
            "resources": {
                "sample_count": 8,
                "stats": {
                    "cpu_all_core_avg_util_pct": {"avg": 28.0, "max": 48.0, "p95": 45.0},
                    "cpu_all_core_max_util_pct": {"avg": 42.0, "max": 72.0, "p95": 68.0},
                    "gpu_gr3d_util_pct": {"avg": 89.0, "max": 99.0, "p95": 98.0},
                    "ram_used_mb": {"avg": 32800.0, "max": 33400.0, "p95": 33340.0},
                    "ram_total_mb": {"avg": 62828.0, "max": 62828.0},
                },
            },
            "system": {
                "device_model": "NVIDIA Jetson AGX Orin",
                "ram_total_label": "64 GB",
                "platform": "Linux / Jetson",
            },
        }

    def _mock_result(self, job: Job, command: DemoCommand) -> dict[str, Any]:
        task = str(command.task or "task1")
        sequences = {
            "task1": [
                ("load", "Loading Task 1 models and text embeddings"),
                ("retrieve", "Encoding image and recalling ingredient candidates"),
                ("vlm", "Running VLM selector"),
                ("output", "Formatting ingredient output"),
            ],
            "task2": [
                ("load", "Loading Task 2 fast SigLIP caption pipeline"),
                ("caption", "Recalling candidate captions"),
                ("rerank", "Scoring caption candidates"),
                ("output", "Formatting caption output"),
            ],
            "calories": [
                ("load", "Loading calorie-composition Qwen-VL backend"),
                ("composition", "Estimating dish composition"),
                ("math", "Computing grams and calories"),
                ("output", "Formatting calorie output"),
            ],
            "both": [
                ("task1", "Running Task 1 ingredient pipeline"),
                ("task2", "Running Task 2 caption pipeline"),
                ("output", "Combining outputs"),
            ],
        }
        for stage, message in sequences.get(task, sequences["task1"]):
            time.sleep(max(0.0, float(self.args.mock_delay)))
            job.add_event(stage, message)
        if task == "task1":
            return {
                "task": "task1",
                "mode": "fast",
                "answer": "tomato, mozzarella, basil, olive oil",
                "latency_sec": 2.36,
                "rerank_or_vlm": "VLM used",
                "diagnostics": self._mock_diagnostics(2.36),
                "details": {
                    "selected_labels": ["tomato", "mozzarella", "basil", "olive oil"],
                    "backend_json": str(Path(job.run_dir) / "backend" / "task1_fast.json"),
                    "raw": {
                        "timings_sec": {
                            "siglip2_image_and_topk_sec": 0.62,
                            "qwen_sec": 1.58,
                            "fusion_and_selection_sec": 0.03,
                            "total_image_sec": 2.36,
                        }
                    },
                },
            }
        if task == "task2":
            return {
                "task": "task2",
                "mode": "fast",
                "answer": "A plate of caprese salad with tomato, mozzarella, and basil.",
                "latency_sec": 1.42,
                "rerank_or_vlm": "Re-rank skipped",
                "diagnostics": self._mock_diagnostics(1.42),
                "details": {
                    "prediction_caption": "A plate of caprese salad with tomato, mozzarella, and basil.",
                    "prediction_cat": "salad",
                    "final_score_mode": "siglip",
                    "rerank_applied": False,
                    "backend_json": str(Path(job.run_dir) / "backend" / "task2_fast.json"),
                    "raw": {
                        "traces": [
                            {
                                "timings_sec": {
                                    "siglip_image_and_recall_sec": 0.77,
                                    "rerank_sec": 0.0,
                                    "final_selection_sec": 0.02,
                                    "total_image_sec": 1.42,
                                },
                                "ranked_candidates": [
                                    {
                                        "caption": "A plate of caprese salad with tomato, mozzarella, and basil.",
                                        "cat": "salad",
                                        "final_score": 0.91,
                                    },
                                    {"caption": "A fresh tomato and cheese appetizer.", "cat": "appetizer", "final_score": 0.84},
                                ]
                            }
                        ]
                    },
                },
            }
        if task == "calories":
            return {
                "task": "calories",
                "mode": "fast",
                "answer": "Estimated calories for one representative item: 105 kcal. Ingredient calories: banana about 105 kcal each.",
                "latency_sec": 3.18,
                "rerank_or_vlm": "Composition VLM used",
                "diagnostics": self._mock_diagnostics(3.18),
                "details": {
                    "backend_json": str(Path(job.run_dir) / "backend" / "calories_fast.json"),
                    "raw": {
                        "timings_sec": {
                            "candidate_filter_sec": 0.54,
                            "qwen_sec": 2.42,
                            "total_image_sec": 3.18,
                        }
                    },
                    "calories": {
                        "total_kcal": 105,
                        "estimation_scope": "per_instance_not_full_image",
                        "ingredients": [
                            {
                                "id": 10,
                                "name": "banana",
                                "count": None,
                                "many_instances": True,
                                "abundance_scene": True,
                                "portion_category": "double",
                                "per_instance_kcal": 105,
                                "kcal": 105,
                            },
                        ],
                    },
                },
            }
        task1 = self._mock_result(job, DemoCommand(action="run", task="task1", mode="fast", text="find ingredients"))
        task2 = self._mock_result(job, DemoCommand(action="run", task="task2", mode="fast", text="describe the dish"))
        return {
            "task": "both",
            "mode": "fast",
            "answer": f"Ingredients: {task1['answer']}. Caption: {task2['answer']}.",
            "latency_sec": float(task1["latency_sec"]) + float(task2["latency_sec"]),
            "rerank_or_vlm": "task1=VLM used; task2=Re-rank skipped",
            "details": {"task1": task1, "task2": task2},
        }


class WebHandler(BaseHTTPRequestHandler):
    server_version = "DishcoveryWebDemo/1.0"

    @property
    def controller(self) -> DemoController:
        return self.server.controller  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if not getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._json_response(self.controller.status())
            return
        if path == "/api/nutrition/profile":
            self._json_response(self.controller.nutrition.profile_response())
            return
        if path == "/api/nutrition/ingredients":
            self._json_response(self.controller.nutrition.ingredient_options())
            return
        if path == "/api/nutrition/history":
            try:
                query = parse_qs(parsed.query)
                today = date.today()
                start = parse_date(query.get("from", [""])[0], today - timedelta(days=29)) if "from" in query else None
                end = parse_date(query.get("to", [""])[0], today) if "to" in query else None
                self._json_response(self.controller.nutrition.history(start, end))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/nutrition/summary":
            try:
                query = parse_qs(parsed.query)
                today = date.today()
                start = parse_date(query.get("from", [""])[0], today - timedelta(days=29))
                end = parse_date(query.get("to", [""])[0], today)
                self._json_response(self.controller.nutrition.summary(start, end))
            except ValueError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/config":
            self._json_response(
                {
                    "tasks": [
                        {"id": "task1", "label": "Ingredients", "command": TASK_COMMAND_TEXT["task1"]},
                        {"id": "task2", "label": "Dish description", "command": TASK_COMMAND_TEXT["task2"]},
                        {"id": "calories", "label": "Calories", "command": TASK_COMMAND_TEXT["calories"]},
                    ],
                    "status": self.controller.status(),
                }
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = self.controller.get_job(job_id)
            if job is None:
                self._json_response({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._json_response(job.to_dict())
            return
        if path == "/api/sample-image":
            self._sample_image_response(parse_qs(parsed.query))
            return
        self._static_response(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/jobs":
                job = self.controller.create_job(payload)
                self._json_response(job.to_dict(), status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/nutrition/history":
                self._json_response(self.controller.nutrition.add_history_entry(payload), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice-command":
                result = self.controller.transcribe_voice_command(payload)
                self._json_response(result)
                return
            if parsed.path == "/api/tts":
                audio = self.controller.synthesize_tts(str(payload.get("text") or ""))
                self._binary_response(audio, "audio/wav")
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, CommandParseError) as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/nutrition/profile":
                self._json_response(self.controller.nutrition.save_profile(payload))
                return
            if parsed.path.startswith("/api/nutrition/history/"):
                entry_id = parsed.path.rsplit("/", 1)[-1]
                self._json_response(self.controller.nutrition.update_history_entry(entry_id, payload))
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/nutrition/history/"):
                entry_id = parsed.path.rsplit("/", 1)[-1]
                self._json_response(self.controller.nutrition.delete_history_entry(entry_id))
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json_response({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _binary_response(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _static_response(self, path: str) -> None:
        if path in {"", "/"}:
            target = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            target = STATIC_DIR / path.removeprefix("/static/")
        else:
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            resolved = target.resolve()
            if STATIC_DIR.resolve() not in resolved.parents and resolved != STATIC_DIR.resolve():
                raise FileNotFoundError
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError
            content = resolved.read_bytes()
        except FileNotFoundError:
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _sample_image_response(self, query: dict[str, list[str]]) -> None:
        path: Path | None = None
        index = 0
        sample_paths = self.controller.sample_paths
        if "index" in query and sample_paths:
            try:
                index = int(query.get("index", ["0"])[0])
            except ValueError:
                self._json_response({"error": "Invalid sample image index"}, status=HTTPStatus.BAD_REQUEST)
                return
            index = index % len(sample_paths)
            path = sample_paths[index]
        elif query.get("random", ["1"])[0] in {"0", "false", "no"} and sample_paths:
            path = sample_paths[0]
            index = 0
        else:
            path = self.controller.random_sample_path()
            if path is not None and sample_paths:
                try:
                    index = sample_paths.index(path)
                except ValueError:
                    index = 0
        if path is None or not path.exists():
            self._json_response({"error": "No sample image available"}, status=HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Dishcovery-Image-Index", str(index))
        self.send_header("X-Dishcovery-Image-Count", str(len(sample_paths)))
        self.send_header("X-Dishcovery-Image-Name", path.name)
        self.end_headers()
        self.wfile.write(content)


class DemoHTTPServer(ThreadingHTTPServer):
    controller: DemoController
    quiet: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for the Dishcovery demo pipelines.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--assets", type=Path, default=ROOT / "external_assets")
    parser.add_argument("--edge-llm", type=Path, default=ROOT.parent / "TensorRT-Edge-LLM")
    parser.add_argument("--task1-script", type=Path, default=ROOT / "code/orin_task1_pipeline.py")
    parser.add_argument("--task2-script", type=Path, default=ROOT / "code/orin_task2_pipeline.py")
    parser.add_argument("--calorie-script", type=Path, default=ROOT / "code/orin_calorie_demo_edgellm.py")
    parser.add_argument("--backend-python", default="")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument(
        "--task1-fast-skip-vlm-rel-gap-threshold",
        type=float,
        default=DEFAULT_TASK1_SKIP_VLM_REL_GAP_THRESHOLD,
        help=(
            "Task 1 fast-mode skip-close-choice threshold. Default matches the measured "
            "skip policy used in reports: 0.25."
        ),
    )
    parser.add_argument(
        "--no-task1-fast-skip-vlm",
        action="store_const",
        const=None,
        dest="task1_fast_skip_vlm_rel_gap_threshold",
        help="Disable Task 1 fast-mode VLM skipping and always run the VLM selector.",
    )
    parser.add_argument("--task1-extra-args", default="")
    parser.add_argument("--task2-fast-final-score-mode", choices=("siglip", "siglip_guarded"), default="siglip_guarded")
    parser.add_argument("--task2-full-final-score-mode", choices=("rerank", "blend_z", "siglip_guarded"), default="rerank")
    parser.add_argument("--task2-reranker-size", default="2B")
    parser.add_argument("--task2-extra-args", default="")
    parser.add_argument("--calorie-extra-args", default="")
    parser.add_argument("--backend-mode", choices=("subprocess", "warm"), default="warm")
    parser.add_argument("--no-diagnostics-power", dest="diagnostics_power", action="store_false")
    parser.add_argument("--diagnostics-power-interval-ms", type=int, default=200)
    parser.set_defaults(diagnostics_power=True)
    parser.add_argument(
        "--preload-backends",
        default="task1,task2_fast,calories",
        help=(
            "auto, all, none, task1, task2_fast, task2_full, calories, both, or a comma-separated list. "
            "auto preloads every warm runtime used by the demo."
        ),
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--hf-token-file", type=Path, default=None)
    parser.add_argument("--history-root", type=Path, default=Path("demo_runs/web"))
    parser.add_argument("--nutrition-root", type=Path, default=None)
    parser.add_argument("--sample-image-dir", type=Path, default=DEFAULT_SHOWCASE_IMAGE_DIR)
    parser.add_argument("--sample-images-list", type=Path, default=DEFAULT_SHOWCASE_IMAGES_LIST)
    parser.add_argument("--sample-image-count", type=int, default=200)
    parser.add_argument(
        "--demo-subset",
        choices=("default", "good"),
        default="default",
        help="Load a curated presentation subset instead of the default sample image list.",
    )
    parser.add_argument("--max-jobs", type=int, default=30)
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-config", type=Path, default=DEFAULT_PIPER_CONFIG)
    parser.add_argument("--piper-command", default="piper")
    parser.add_argument("--no-web-tts", dest="web_tts_enabled", action="store_false")
    parser.set_defaults(web_tts_enabled=True)
    parser.add_argument("--web-voice-seconds", type=float, default=6.0)
    parser.add_argument("--web-stt-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--web-stt-command", default="")
    parser.add_argument("--web-stt-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--web-stt-sample-rate", type=int, default=16_000)
    parser.add_argument("--mock-backend", action="store_true", help="Use deterministic fake outputs for UI testing.")
    parser.add_argument("--mock-delay", type=float, default=0.35)
    args = parser.parse_args()
    if args.nutrition_root is None:
        args.nutrition_root = args.history_root / "nutrition"
    return args


def main() -> None:
    args = parse_args()
    args.assets = args.assets.expanduser().resolve()
    args.edge_llm = args.edge_llm.expanduser().resolve()
    if not args.mock_backend:
        configure_quantized_backends(args)
    apply_demo_subset(args)
    apply_good_subset_task2_defaults(args)
    if args.backend_python:
        resolved = shutil.which(args.backend_python) or args.backend_python
        if not Path(resolved).exists() and shutil.which(args.backend_python) is None:
            print(f"Warning: backend Python executable was not found: {args.backend_python}", file=sys.stderr)
    controller = DemoController(args)
    server = DemoHTTPServer((args.host, args.port), WebHandler)
    server.controller = controller
    server.quiet = bool(args.quiet)
    controller.start_preload()
    url = f"http://{args.host}:{args.port}"
    print(f"Dishcovery web demo ready at {url}", flush=True)
    if args.mock_backend:
        print("Mock backend enabled; buttons return simulated pipeline outputs.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
