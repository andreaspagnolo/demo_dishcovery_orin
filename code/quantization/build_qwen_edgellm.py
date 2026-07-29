#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_EDGE_LLM_REPO = Path("TensorRT-Edge-LLM")
DEFAULT_WORKSPACE_DIR = Path("model_build")
DEFAULT_RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_RERANKER_NAME = "Qwen3-VL-Reranker-2B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize, export, and build a Qwen VLM with TensorRT EdgeLLM."
    )
    parser.add_argument("--edge-llm-repo", type=Path, default=Path(os.environ.get("EDGELLM_REPO", DEFAULT_EDGE_LLM_REPO)))
    parser.add_argument("--workspace-dir", type=Path, default=Path(os.environ.get("WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR)))
    parser.add_argument("--model-id", default=os.environ.get("RERANKER_MODEL_ID", DEFAULT_RERANKER_MODEL_ID))
    parser.add_argument("--model-name", default=os.environ.get("RERANKER_NAME", DEFAULT_RERANKER_NAME))
    parser.add_argument("--model-dir", type=Path, default=None, help="Local HF checkpoint directory. Defaults to HF cache or download.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--copy-hf", action="store_true", help="Copy the checkpoint into the EdgeLLM workspace instead of symlinking.")
    parser.add_argument("--skip-download", action="store_true", help="Fail instead of downloading if the model is not cached.")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=("fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"),
        default="int4_awq",
    )
    parser.add_argument(
        "--quantized-dir-name",
        default=None,
        help="Output directory name for the quantized checkpoint. Defaults to quantized_<quantization>.",
    )
    parser.add_argument(
        "--engine-dir-name",
        default="engine",
        help="Output engine directory name under the model workspace.",
    )
    parser.add_argument(
        "--llm-engine-dir",
        type=Path,
        default=None,
        help="Exact LLM engine output directory. Overrides <engine-dir-name>/llm.",
    )
    parser.add_argument(
        "--visual-engine-dir",
        type=Path,
        default=None,
        help="Exact visual engine output directory. Overrides <engine-dir-name>/visual.",
    )
    parser.add_argument(
        "--onnx-dir-name",
        default="onnx",
        help="Output ONNX directory name under the model workspace.",
    )
    parser.add_argument("--skip-llm-build", action="store_true")
    parser.add_argument("--skip-visual-build", action="store_true")
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-kv-cache-capacity", type=int, default=512)
    parser.add_argument("--min-image-tokens", type=int, default=4)
    parser.add_argument("--max-image-tokens", type=int, default=1024)
    parser.add_argument("--max-image-tokens-per-image", type=int, default=512)
    return parser.parse_args()


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache/huggingface/hub"


def cached_snapshot(repo_id: str) -> Path | None:
    cache = hf_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots = cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates: list[Path] = []
    ref = cache / "refs/main"
    if ref.exists():
        text = ref.read_text(encoding="utf-8").strip()
        if text:
            candidates.append(snapshots / text)
    candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if (path / "config.json").exists():
            return path
    return None


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def prepare_hf_dir(args: argparse.Namespace, reranker_dir: Path, summary: dict[str, Any]) -> Path:
    hf_dir = reranker_dir / "hf"
    if hf_dir.exists():
        summary["hf_dir_status"] = "existing"
        return hf_dir

    source = args.model_dir.expanduser().resolve() if args.model_dir else cached_snapshot(args.model_id)
    if source is None:
        if args.skip_download:
            raise FileNotFoundError(f"{args.model_id} is not cached and --skip-download was provided.")
        hf_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.python),
            "-m",
            "huggingface_hub.commands.huggingface_cli",
            "download",
            args.model_id,
            "--local-dir",
            str(hf_dir),
        ]
        summary["hf_download_returncode"] = run_logged(cmd, reranker_dir / "hf_download.log")
        return hf_dir

    if args.copy_hf:
        shutil.copytree(source, hf_dir, symlinks=False)
        summary["hf_dir_status"] = f"copied_from:{source}"
    else:
        hf_dir.symlink_to(source, target_is_directory=True)
        summary["hf_dir_status"] = f"symlinked_to:{source}"
    return hf_dir


def command_or_module(command: str, module: str, args: argparse.Namespace) -> list[str]:
    executable = shutil.which(command)
    if executable:
        return [executable]
    return [str(args.python), "-m", module]


def edge_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.edge_llm_repo) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["EDGELLM_REPO"] = str(args.edge_llm_repo)
    env["EDGELLM_PLUGIN_PATH"] = str(args.edge_llm_repo / "build/libNvInfer_edgellm_plugin.so")
    env["LD_LIBRARY_PATH"] = str(args.edge_llm_repo / "build") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def main() -> None:
    args = parse_args()
    args.edge_llm_repo = args.edge_llm_repo.expanduser().resolve()
    args.workspace_dir = args.workspace_dir.expanduser().resolve()
    reranker_dir = args.workspace_dir / args.model_name
    reranker_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "model_name": args.model_name,
        "reranker_dir": str(reranker_dir),
        "edge_llm_repo": str(args.edge_llm_repo),
        "quantization": args.quantization,
        "quantized_dir_name": args.quantized_dir_name or f"quantized_{args.quantization}",
        "onnx_dir_name": args.onnx_dir_name,
        "engine_dir_name": args.engine_dir_name,
        "max_input_len": args.max_input_len,
        "max_kv_cache_capacity": args.max_kv_cache_capacity,
        "min_image_tokens": args.min_image_tokens,
        "max_image_tokens": args.max_image_tokens,
        "max_image_tokens_per_image": args.max_image_tokens_per_image,
        "fallback_if_failed": "keep Qwen3-VL-Reranker-2B PyTorch/Transformers FP16",
    }

    hf_dir = prepare_hf_dir(args, reranker_dir, summary)
    summary["hf_dir"] = str(hf_dir)

    env = edge_env(args)
    quantized_dir = reranker_dir / (args.quantized_dir_name or f"quantized_{args.quantization}")
    onnx_dir = reranker_dir / args.onnx_dir_name
    engine_dir = reranker_dir / args.engine_dir_name

    if not args.skip_quantize:
        quant_cmd = command_or_module("tensorrt-edgellm-quantize", "tensorrt_edgellm.scripts.quantize", args)
        quant_cmd.extend(
            [
                "llm",
                "--model_dir",
                str(hf_dir),
                "--output_dir",
                str(quantized_dir),
                "--quantization",
                args.quantization,
            ]
        )
        summary["quantize_returncode"] = run_logged(quant_cmd, reranker_dir / f"quantize_{args.quantization}.log", env=env)

    if not args.skip_export and quantized_dir.exists():
        export_cmd = command_or_module("tensorrt-edgellm-export", "tensorrt_edgellm.scripts.export", args)
        export_cmd.extend([str(quantized_dir), str(onnx_dir)])
        if args.quantization == "int4_awq":
            export_cmd.extend(["--externalize-weights", "int4_ffn"])
        summary["export_returncode"] = run_logged(export_cmd, reranker_dir / "export_onnx.log", env=env)
    elif not args.skip_export:
        summary["export_skipped_reason"] = f"Quantized directory not found: {quantized_dir}"

    if not args.skip_build:
        llm_build = args.edge_llm_repo / "build/examples/llm/llm_build"
        visual_build = args.edge_llm_repo / "build/examples/multimodal/visual_build"
        if args.skip_llm_build:
            summary["build_llm_skipped_reason"] = "--skip-llm-build"
        elif (onnx_dir / "llm").is_dir() and llm_build.exists():
            llm_engine_dir = (
                args.llm_engine_dir.expanduser().resolve()
                if args.llm_engine_dir is not None
                else engine_dir / "llm"
            )
            llm_engine_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(llm_build),
                "--onnxDir",
                str(onnx_dir / "llm"),
                "--engineDir",
                str(llm_engine_dir),
                "--maxBatchSize",
                "1",
                "--maxInputLen",
                str(args.max_input_len),
                "--maxKVCacheCapacity",
                str(args.max_kv_cache_capacity),
            ]
            summary["build_llm_returncode"] = run_logged(cmd, reranker_dir / "build_llm_engine.log", env=env)
        else:
            summary["build_llm_skipped_reason"] = "Missing ONNX llm folder or llm_build binary."

        if args.skip_visual_build:
            summary["build_visual_skipped_reason"] = "--skip-visual-build"
        elif (onnx_dir / "visual").is_dir() and visual_build.exists():
            visual_engine_dir = (
                args.visual_engine_dir.expanduser().resolve()
                if args.visual_engine_dir is not None
                else engine_dir / "visual"
            )
            visual_engine_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(visual_build),
                "--onnxDir",
                str(onnx_dir / "visual"),
                "--engineDir",
                str(visual_engine_dir),
                "--minImageTokens",
                str(args.min_image_tokens),
                "--maxImageTokens",
                str(args.max_image_tokens),
                "--maxImageTokensPerImage",
                str(args.max_image_tokens_per_image),
            ]
            summary["build_visual_returncode"] = run_logged(cmd, reranker_dir / "build_visual_engine.log", env=env)
        else:
            summary["build_visual_skipped_reason"] = "Missing ONNX visual folder or visual_build binary."

    summary["quantized_dir_exists"] = quantized_dir.exists()
    summary["onnx_llm_exists"] = (onnx_dir / "llm").is_dir()
    summary["onnx_visual_exists"] = (onnx_dir / "visual").is_dir()
    llm_engine_dir = (
        args.llm_engine_dir.expanduser().resolve()
        if args.llm_engine_dir is not None
        else engine_dir / "llm"
    )
    visual_engine_dir = (
        args.visual_engine_dir.expanduser().resolve()
        if args.visual_engine_dir is not None
        else engine_dir / "visual"
    )
    llm_engine = llm_engine_dir / "llm.engine"
    visual_engine = visual_engine_dir / "visual/visual.engine"
    summary["llm_engine"] = str(llm_engine) if llm_engine.exists() else None
    summary["visual_engine"] = str(visual_engine) if visual_engine.exists() else None
    summary["llm_engine_exists"] = llm_engine.exists()
    summary["visual_engine_exists"] = visual_engine.exists()
    summary["visual_engine_dir_exists"] = visual_engine_dir.is_dir()

    summary_path = reranker_dir / "edgellm_quantization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {summary_path}")
    if not summary["llm_engine_exists"]:
        print("TensorRT EdgeLLM reranker engine was not built. Keep the PyTorch FP16 reranker fallback.")


if __name__ == "__main__":
    main()
