from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReproPaths:
    root: Path
    assets: Path
    edge_llm: Path
    siglip_engine: Path
    task1_images: Path
    task1_llm: Path
    task1_visual: Path
    task2_images: Path
    task2_llm: Path
    task2_visual: Path

    @classmethod
    def from_args(cls, assets: Path, edge_llm: Path) -> "ReproPaths":
        assets = assets.expanduser().resolve()
        edge_llm = edge_llm.expanduser().resolve()
        models = assets / "models"
        return cls(
            root=ROOT,
            assets=assets,
            edge_llm=edge_llm,
            siglip_engine=models / "shared_SigLIP2_FP16_TensorRT/visual_fp16.engine",
            task1_images=assets / "images/task1_350",
            task1_llm=models / "task1_Qwen3-VL-4B-AWQ-INT4/engines/llm_in4096_kv8192",
            task1_visual=models / "task1_Qwen3-VL-4B-AWQ-INT4/engines/visual",
            task2_images=assets / "images/task2_350",
            task2_llm=models / "task2_Qwen3-VL-Reranker-2B-INT4/engine_in1024/llm",
            task2_visual=models / "task2_Qwen3-VL-Reranker-2B-INT4/engine/visual",
        )


def default_assets() -> Path:
    return Path(os.environ.get("DISHCOVERY_ASSETS", ROOT / "external_assets"))


def default_edge_llm() -> Path:
    return Path(os.environ.get("EDGE_LLM_PATH", ROOT.parent / "TensorRT-Edge-LLM"))


def required_runtime_paths(paths: ReproPaths, task: str) -> dict[str, Path]:
    common = {
        "SigLIP2 engine": paths.siglip_engine,
        "EdgeLLM persistent server": paths.edge_llm / "build/examples/llm/llm_persistent_server",
        "EdgeLLM plugin": paths.edge_llm / "build/libNvInfer_edgellm_plugin.so",
    }
    if task in {"task1", "calories"}:
        return {
            **common,
            "Task 1 images": paths.task1_images,
            "Task 1 LLM engine": paths.task1_llm / "llm.engine",
            "Task 1 visual engine": paths.task1_visual / "visual/visual.engine",
        }
    if task == "task2":
        return {
            **common,
            "Task 2 images": paths.task2_images,
            "Task 2 LLM engine": paths.task2_llm / "llm.engine",
            "Task 2 visual engine": paths.task2_visual / "visual/visual.engine",
        }
    raise ValueError(task)


def validate_runtime(paths: ReproPaths, task: str) -> None:
    missing = [
        f"{label}: {path}"
        for label, path in required_runtime_paths(paths, task).items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))


def task1_command(
    paths: ReproPaths,
    *,
    output_json: Path,
    predictions_csv: Path | None = None,
    image: Path | None = None,
    image_dir: Path | None = None,
    images_list: Path | None = None,
    eval_samples: int = 350,
) -> list[str]:
    image_dir = image_dir or paths.task1_images
    images_list = images_list or ROOT / "benchmark_inputs/task1/images.txt"
    command = [
        sys.executable,
        str(ROOT / "code/orin_task1_pipeline.py"),
        "--image-dir",
        str(image_dir),
        "--images-list",
        str(images_list),
        "--cleaned-json",
        str(ROOT / "benchmark_inputs/task1/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json"),
        "--image-ground-truth-map",
        str(ROOT / "benchmark_inputs/task1/image_ground_truth_rows.csv"),
        "--task1-logic",
        "legacy_siglip_qwen",
        "--legacy-keep-candidate-list",
        "--candidate-list-mode",
        "fixed_topk",
        "--top-k",
        "20",
        "--skip-vlm-rel-gap-threshold",
        "0.25",
        "--skip-vlm-visual-selector",
        "top_ratio",
        "--skip-vlm-visual-selector-max-labels",
        "3",
        "--siglip-backend",
        "tensorrt_engine",
        "--siglip-trt-engine",
        str(paths.siglip_engine),
        "--text-cache",
        str(ROOT / "benchmark_inputs/caches/task1_350_siglip2_pytorch_fp16_text_cache.npz"),
        "--text-cache-require-hit",
        "--vlm-backend",
        "edgellm",
        "--edgellm-root",
        str(paths.edge_llm),
        "--edgellm-model-name",
        "Qwen3-VL-4B-Instruct",
        "--edgellm-llm-engine-profile",
        "max-input-4096",
        "--edgellm-llm-engine-dir",
        str(paths.task1_llm),
        "--edgellm-visual-engine-dir",
        str(paths.task1_visual),
        "--edgellm-runtime-mode",
        "persistent",
        "--checkpoint-every",
        "0",
        "--measure",
        "--output-json",
        str(output_json),
    ]
    if image is None:
        command.extend(["--eval-samples", str(eval_samples), "--eval-first", "--seed", "7"])
        if predictions_csv is not None:
            command.extend(["--predictions-csv", str(predictions_csv)])
    else:
        command.extend(["--image", str(image)])
    return command


def task2_command(
    paths: ReproPaths,
    *,
    output_json: Path,
    predictions_csv: Path | None = None,
    image: Path | None = None,
    image_dir: Path | None = None,
    images_list: Path | None = None,
) -> list[str]:
    image_dir = image_dir or paths.task2_images
    images_list = images_list or ROOT / "benchmark_inputs/task2/images.txt"
    command = [
        sys.executable,
        str(ROOT / "code/orin_task2_pipeline.py"),
        "--image-dir",
        str(image_dir),
        "--manifest",
        str(ROOT / "benchmark_inputs/task2/manifest.csv"),
        "--images-list",
        str(images_list),
        "--evaluation-json",
        str(ROOT / "benchmark_inputs/task2/evaluation_data.json"),
        "--caption-bank-json",
        str(ROOT / "benchmark_inputs/task2/evaluation_data.json"),
        "--caption-text-mode",
        "class_caption",
        "--siglip-backend",
        "tensorrt_engine",
        "--siglip-trt-engine",
        str(paths.siglip_engine),
        "--caption-cache",
        str(ROOT / "benchmark_inputs/caches/food500_full_bank4940_siglip2_fp16_caption_cache.npz"),
        "--caption-cache-allow-mismatch",
        "--caption-cache-require-hit",
        "--candidate-list-mode",
        "fixed_topk",
        "--final-score-mode",
        "siglip_guarded",
        "--siglip-keep-gap",
        "3.0",
        "--siglip-top-k",
        "5",
        "--rerank-top-k",
        "5",
        "--reranker-size",
        "2B",
        "--reranker-backend",
        "edgellm",
        "--reranker-edgellm-score-mode",
        "logit_score",
        "--edgellm-root",
        str(paths.edge_llm),
        "--edgellm-model-name",
        "Qwen3-VL-Reranker-2B",
        "--edgellm-llm-engine-dir",
        str(paths.task2_llm),
        "--edgellm-visual-engine-dir",
        str(paths.task2_visual),
        "--edgellm-runtime-mode",
        "persistent",
        "--measure",
        "--output-json",
        str(output_json),
    ]
    if image is None:
        command.extend(["--eval-all", "--seed", "42", "--measure-ram", "--ram-sample-interval-ms", "500"])
    else:
        command.extend(["--image", str(image), "--inference-only"])
    if predictions_csv is not None:
        command.extend(["--predictions-csv", str(predictions_csv)])
    return command


def calories_command(paths: ReproPaths, *, output_json: Path, image: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "code/orin_calorie_demo_edgellm.py"),
        "--image",
        str(image),
        "--output-json",
        str(output_json),
        "--calories-csv",
        str(ROOT / "benchmark_inputs/calories/calories_realistic_average_portions.csv"),
        "--cleaned-json",
        str(ROOT / "benchmark_inputs/task1/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json"),
        "--captions",
        str(ROOT / "benchmark_inputs/calories/captions_cleaned.txt"),
        "--siglip-backend",
        "tensorrt_engine",
        "--siglip-trt-engine",
        str(paths.siglip_engine),
        "--text-cache",
        str(ROOT / "benchmark_inputs/caches/calorie_siglip2_text_cache.npz"),
        "--text-cache-require-hit",
        "--calorie-candidate-list-mode",
        "fixed_topk",
        "--calorie-candidate-top-k",
        "20",
        "--calorie-counting-logic",
        "cut-aware",
        "--edgellm-calorie-logic",
        "compact",
        "--vlm-backend",
        "edgellm",
        "--edgellm-root",
        str(paths.edge_llm),
        "--edgellm-model-name",
        "Qwen3-VL-4B-Instruct",
        "--edgellm-llm-engine-profile",
        "max-input-4096",
        "--edgellm-llm-engine-dir",
        str(paths.task1_llm),
        "--edgellm-visual-engine-dir",
        str(paths.task1_visual),
        "--edgellm-runtime-mode",
        "persistent",
    ]
