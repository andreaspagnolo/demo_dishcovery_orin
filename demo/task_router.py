from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import importlib.util
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .command_parser import DemoCommand, ModeName, TaskName


@dataclass
class TaskRouterConfig:
    task1_script: Path
    task2_script: Path
    calorie_script: Path
    python_executable: str = sys.executable
    task1_fast_skip_vlm_rel_gap_threshold: float | None = 0.25
    task2_fast_final_score_mode: str = "siglip_guarded"
    task2_full_final_score_mode: str = "rerank"
    task2_reranker_size: str | None = None
    device: str | None = None
    torch_dtype: str | None = None
    hf_token: str | None = None
    execution_mode: str = "subprocess"
    task1_extra_args: list[str] = field(default_factory=list)
    task2_extra_args: list[str] = field(default_factory=list)
    calorie_extra_args: list[str] = field(default_factory=list)
    diagnostics_power: bool = False
    diagnostics_power_interval_ms: int = 200


class TaskRouter:
    def __init__(self, config: TaskRouterConfig) -> None:
        self.config = config
        self._task1_module: Any | None = None
        self._task1_runtime: Any | None = None
        self._task1_base_args: argparse.Namespace | None = None
        self._task2_module: Any | None = None
        self._task2_runtimes: dict[str, "WarmTask2Runtime"] = {}
        self._calorie_module: Any | None = None
        self._calorie_runtime: Any | None = None
        self._calorie_base_args: argparse.Namespace | None = None
        self._measurement_module: Any | None = None
        self._system_info: dict[str, Any] | None = None
        self._current_image_cache_key: str | None = None
        self._task_result_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _image_cache_key(self, image_path: Path) -> str:
        digest = hashlib.sha256()
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_image_cache(self, image_path: Path) -> None:
        cache_key = self._image_cache_key(image_path)
        if cache_key != self._current_image_cache_key:
            self._current_image_cache_key = cache_key
            self._task_result_cache.clear()

    @staticmethod
    def _cacheable_task(task: TaskName, mode: ModeName) -> bool:
        return str(task) in {"task1", "calories"} and str(mode) != "compare"

    def _cached_task_result(self, task: TaskName, mode: ModeName) -> dict[str, Any] | None:
        cached = self._task_result_cache.get((str(task), str(mode)))
        if cached is None:
            return None
        result = copy.deepcopy(cached)
        original_latency = result.get("latency_sec")
        result["latency_sec"] = 0.0
        result["cache_hit"] = True
        details = result.setdefault("details", {})
        if isinstance(details, dict):
            details["cache_hit"] = True
            details["cached_latency_sec"] = original_latency
        rerank_or_vlm = str(result.get("rerank_or_vlm") or "").strip()
        result["rerank_or_vlm"] = f"{rerank_or_vlm}; cache hit" if rerank_or_vlm else "cache hit"
        return result

    def _store_task_result(self, task: TaskName, mode: ModeName, result: dict[str, Any]) -> None:
        if self._cacheable_task(task, mode):
            self._task_result_cache[(str(task), str(mode))] = copy.deepcopy(result)

    def _cached_task1_raw_report(self, mode: ModeName = "fast") -> dict[str, Any] | None:
        cached = self._task_result_cache.get(("task1", str(mode)))
        if not isinstance(cached, dict):
            return None
        details = cached.get("details") if isinstance(cached.get("details"), dict) else {}
        raw = details.get("raw") if isinstance(details, dict) else None
        return copy.deepcopy(raw) if isinstance(raw, dict) else None

    def preload(self, tasks: list[str], progress: Callable[[str], None] | None = None) -> None:
        def notify(message: str) -> None:
            if progress is not None:
                progress(message)

        for task in tasks:
            normalized = task.strip().lower()
            if normalized == "task1":
                notify("Loading FP16 SigLIP2 visual encoder and INT4 Qwen-VL-Instruct-4B")
                self._load_task1_runtime()
                notify("FP16 SigLIP2 and INT4 Qwen-VL-Instruct-4B ready")
            elif normalized == "task2_fast":
                notify("Loading guarded FP16 SigLIP2 and INT4 Qwen-VL-Reranker-2B")
                self._load_task2_runtime("fast")
                notify("Guarded FP16 SigLIP2 and INT4 reranker ready")
            elif normalized == "task2_full":
                notify("Loading INT4 Qwen-VL-Reranker-2B caption reranker")
                self._load_task2_runtime("full")
                notify("INT4 Qwen-VL-Reranker-2B ready")
            elif normalized == "task2":
                notify("Preparing SigLIP2 caption recall")
                self._load_task2_runtime("fast")
                notify("SigLIP2 caption recall ready")
            elif normalized in {"calorie", "calories"}:
                notify("Preparing calorie backend with SigLIP2 and Qwen-VL-Instruct-4B")
                self._load_calorie_runtime()
                notify("Calorie backend ready")
            elif normalized == "both":
                notify("Loading FP16 SigLIP2 visual encoder and INT4 Qwen-VL-Instruct-4B")
                self._load_task1_runtime()
                notify("FP16 SigLIP2 and INT4 Qwen-VL-Instruct-4B ready")
                notify("Preparing SigLIP2 caption recall")
                self._load_task2_runtime("fast")
                notify("SigLIP2 caption recall ready")
            elif normalized in {"", "none"}:
                continue
            else:
                raise ValueError(f"Unknown warm backend preload target: {task}")

    def run(
        self,
        image_path: Path,
        command: DemoCommand,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if command.action != "run":
            raise ValueError(f"TaskRouter only handles run commands, got {command.action}")
        if command.task is None or command.mode is None:
            raise ValueError(f"Run command needs task and mode: {command}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if command.task == "both":
            return self._run_both(image_path, output_dir, progress=progress)
        if command.mode == "compare":
            return self._run_compare(image_path, command.task, output_dir, progress=progress)
        return self._run_single(image_path, command.task, command.mode, output_dir, progress=progress)

    def _run_both(
        self,
        image_path: Path,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if progress is not None:
            progress("Running Task 1 ingredient pipeline")
        task1 = self._run_single(image_path, "task1", "fast", output_dir / "task1", progress=progress)
        if progress is not None:
            progress("Running Task 2 caption pipeline")
        task2 = self._run_single(image_path, "task2", "fast", output_dir / "task2", progress=progress)
        latency_sec = float(task1.get("latency_sec") or 0.0) + float(task2.get("latency_sec") or 0.0)
        if progress is not None:
            progress("Combining Task 1 and Task 2 outputs")
        return {
            "task": "both",
            "mode": "fast",
            "answer": f"Ingredients: {task1['answer']}. Caption: {task2['answer']}.",
            "latency_sec": latency_sec,
            "details": {
                "task1": task1,
                "task2": task2,
                "same_captured_frame": str(image_path),
            },
            "rerank_or_vlm": f"task1={task1.get('rerank_or_vlm', '')}; task2={task2.get('rerank_or_vlm', '')}",
        }

    def _run_compare(
        self,
        image_path: Path,
        task: TaskName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        fast = self._run_single(image_path, task, "fast", output_dir / "fast", progress=progress)
        full = self._run_single(image_path, task, "full", output_dir / "full", progress=progress)
        answer = f"Fast: {fast['answer']}. Full: {full['answer']}."
        latency_sec = float(fast.get("latency_sec") or 0.0) + float(full.get("latency_sec") or 0.0)
        return {
            "task": task,
            "mode": "compare",
            "answer": answer,
            "latency_sec": latency_sec,
            "details": {
                "fast": fast,
                "full": full,
                "same_captured_frame": str(image_path),
            },
            "rerank_or_vlm": f"fast={fast.get('rerank_or_vlm', '')}; full={full.get('rerank_or_vlm', '')}",
        }

    def _run_single(
        self,
        image_path: Path,
        task: TaskName,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        self._ensure_image_cache(image_path)
        monitor = self._new_power_monitor()
        monitor.start()
        started = time.perf_counter()
        try:
            cached = self._cached_task_result(task, mode) if self._cacheable_task(task, mode) else None
            if cached is not None:
                if progress is not None:
                    progress(f"Reusing cached {task} result for current image")
                result = cached
            else:
                result = self._run_single_core(image_path, task, mode, output_dir, progress=progress)
                self._store_task_result(task, mode, result)
        except Exception:
            monitor.stop(wall_sec=time.perf_counter() - started)
            raise
        diagnostics_wall_sec = time.perf_counter() - started
        power_summary = monitor.stop(wall_sec=diagnostics_wall_sec)
        result["diagnostics"] = self._build_diagnostics(result, power_summary, diagnostics_wall_sec)
        return result

    def _run_single_core(
        self,
        image_path: Path,
        task: TaskName,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if task == "calories":
            return self._run_calorie_estimator(image_path, mode, output_dir, progress=progress)
        if self.config.execution_mode == "warm":
            if task == "task1":
                return self._run_task1_warm(image_path, mode, output_dir, progress=progress)
            if task == "task2":
                return self._run_task2_warm(image_path, mode, output_dir, progress=progress)
        if task == "task1":
            output_json = output_dir / f"task1_{mode}.json"
            argv = self._task1_argv(image_path, mode, output_json)
            if progress is not None:
                progress("Running Task 1 backend subprocess")
            started = time.perf_counter()
            completed = self._run_subprocess(argv)
            wall_sec = time.perf_counter() - started
            payload = self._load_json(output_json)
            if progress is not None:
                progress("Formatting Task 1 ingredient output")
            return self._normalize_task1(mode, payload, output_json, argv, completed, wall_sec)
        if task == "task2":
            output_json = output_dir / f"task2_{mode}.json"
            argv = self._task2_argv(image_path, mode, output_json, output_dir / f"task2_{mode}.csv")
            if progress is not None:
                progress("Running Task 2 backend subprocess")
            started = time.perf_counter()
            completed = self._run_subprocess(argv)
            wall_sec = time.perf_counter() - started
            payload = self._load_json(output_json)
            if progress is not None:
                progress("Formatting Task 2 caption output")
            return self._normalize_task2(mode, payload, output_json, argv, completed, wall_sec)
        raise ValueError(f"Unsupported task: {task}")

    def _load_measurement_module(self) -> Any:
        if self._measurement_module is not None:
            return self._measurement_module
        path = Path(__file__).resolve().parents[1] / "code" / "measurement_utils.py"
        spec = importlib.util.spec_from_file_location("dishcovery_measurement_utils", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load measurement utilities from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._measurement_module = module
        return module

    def _new_power_monitor(self) -> Any:
        module = self._load_measurement_module()
        return module.PowerMonitor(
            enabled=bool(self.config.diagnostics_power),
            interval_ms=int(self.config.diagnostics_power_interval_ms),
        )

    def _read_first_existing_text(self, paths: list[Path]) -> str:
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip("\x00 \n\t")
            except OSError:
                continue
            if text:
                return text
        return ""

    def _get_system_info(self) -> dict[str, Any]:
        if self._system_info is not None:
            return self._system_info
        mem_total_mb = None
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    mem_total_mb = round(float(line.split()[1]) / 1024.0)
                    break
        except OSError:
            pass
        model = self._read_first_existing_text(
            [
                Path("/proc/device-tree/model"),
                Path("/sys/firmware/devicetree/base/model"),
            ]
        )
        self._system_info = {
            "device_model": model or "Jetson Orin",
            "ram_total_mb": mem_total_mb,
            "ram_total_label": f"{mem_total_mb / 1024.0:.0f} GB" if mem_total_mb else "64 GB",
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        return self._system_info

    def _build_diagnostics(self, result: dict[str, Any], power_summary: dict[str, Any], wall_sec: float) -> dict[str, Any]:
        avg_power_w = power_summary.get("avg_power_w")
        energy_j = float(avg_power_w) * float(wall_sec) if isinstance(avg_power_w, (int, float)) else None
        rail_avg_power_w = power_summary.get("rail_avg_power_w", {})
        selected_rail_avg_power_w = {
            rail: watts
            for rail, watts in rail_avg_power_w.items()
            if str(rail).upper() in {"VDD_GPU_SOC", "VDD_CPU_CV"}
        } if isinstance(rail_avg_power_w, dict) else {}
        return {
            "wall_sec": float(wall_sec),
            "latency_sec": result.get("latency_sec"),
            "power": {
                "enabled": power_summary.get("enabled"),
                "available": power_summary.get("available"),
                "error": power_summary.get("error"),
                "sample_count": power_summary.get("sample_count"),
                "avg_power_w": avg_power_w,
                "median_power_w": power_summary.get("median_power_w"),
                "min_power_w": power_summary.get("min_power_w"),
                "max_power_w": power_summary.get("max_power_w"),
                "energy_j": energy_j,
                "selected_power_source": power_summary.get("selected_power_source"),
                "rail_avg_power_w": selected_rail_avg_power_w,
            },
            "resources": power_summary.get("resources", {}),
            "system": self._get_system_info(),
        }

    def _run_calorie_estimator(
        self,
        image_path: Path,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress("Estimating dish composition and calories")
        if self.config.execution_mode == "warm":
            return self._run_calories_warm(image_path, mode, output_dir, progress=progress)
        output_json = output_dir / f"calories_{mode}.json"
        argv = self._calorie_argv(image_path, mode, output_json)
        if progress is not None:
            progress("Running calorie backend subprocess")
        started = time.perf_counter()
        completed = self._run_subprocess(argv)
        wall_sec = time.perf_counter() - started
        payload = self._load_json(output_json)
        if progress is not None:
            progress("Formatting calorie output")
        return self._normalize_calories(mode, payload, output_json, argv, completed, wall_sec)

    def _run_task1_warm(
        self,
        image_path: Path,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        output_json = output_dir / f"task1_{mode}.json"
        needs_load = self._task1_runtime is None
        if needs_load and progress is not None:
            progress("Loading Task 1 models and text embeddings")
        runtime = self._load_task1_runtime()
        if needs_load and progress is not None:
            progress("Task 1 models ready")
        assert self._task1_base_args is not None
        args = argparse.Namespace(**vars(self._task1_base_args))
        args.image = image_path
        args.output_json = output_json
        if mode == "fast" and self.config.task1_fast_skip_vlm_rel_gap_threshold is not None:
            args.skip_vlm_rel_gap_threshold = self.config.task1_fast_skip_vlm_rel_gap_threshold
        else:
            args.skip_vlm_rel_gap_threshold = None
        if progress is not None:
            progress("Task 1 encoding image, recalling candidates, and running VLM selector")
        started = time.perf_counter()
        report = self._task1_module.build_single_report(runtime, image_path, args)  # type: ignore[union-attr]
        wall_sec = time.perf_counter() - started
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        process = subprocess.CompletedProcess(["warm-task1"], 0, "", "")
        if progress is not None:
            progress("Formatting Task 1 ingredient output")
        return self._normalize_task1(mode, report, output_json, ["warm-task1"], process, wall_sec)

    def _run_task2_warm(
        self,
        image_path: Path,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        output_json = output_dir / f"task2_{mode}.json"
        needs_load = str(mode) not in self._task2_runtimes
        if needs_load and progress is not None:
            label = "fast guarded reranker pipeline" if mode == "fast" else "full reranker pipeline"
            progress(f"Loading Task 2 {label}")
        runtime = self._load_task2_runtime(mode)
        if needs_load and progress is not None:
            label = "fast pipeline" if mode == "fast" else "full pipeline"
            progress(f"Task 2 {label} ready")
        if progress is not None:
            progress("Task 2 encoding image, recalling captions, and scoring candidates")
        started = time.perf_counter()
        report = runtime.run_one(image_path)
        wall_sec = time.perf_counter() - started
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        process = subprocess.CompletedProcess(["warm-task2"], 0, "", "")
        if progress is not None:
            progress("Formatting Task 2 caption output")
        return self._normalize_task2(mode, report, output_json, ["warm-task2"], process, wall_sec)

    def _run_calories_warm(
        self,
        image_path: Path,
        mode: ModeName,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        output_json = output_dir / f"calories_{mode}.json"
        needs_load = self._calorie_runtime is None
        if needs_load and progress is not None:
            progress("Loading calorie-composition Qwen-VL backend")
        runtime = self._load_calorie_runtime()
        if needs_load and progress is not None:
            progress("Calorie-composition backend ready")
        assert self._calorie_base_args is not None
        args = argparse.Namespace(**vars(self._calorie_base_args))
        args.image = image_path
        args.output_json = output_json
        cached_task1_report = self._cached_task1_raw_report("fast")
        if cached_task1_report is not None and hasattr(self._calorie_module, "attach_task1_cached_report"):
            self._calorie_module.attach_task1_cached_report(runtime, cached_task1_report)  # type: ignore[union-attr]
        elif hasattr(runtime, "task1_cached_report"):
            runtime.task1_cached_report = None
        if progress is not None:
            progress("Calorie pipeline running composition VLM and calorie table math")
        started = time.perf_counter()
        report = self._calorie_module.build_single_report(runtime, image_path, args)  # type: ignore[union-attr]
        wall_sec = time.perf_counter() - started
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        process = subprocess.CompletedProcess(["warm-calories"], 0, "", "")
        if progress is not None:
            progress("Formatting calorie output")
        normalized = self._normalize_calories(mode, report, output_json, ["warm-calories"], process, wall_sec)
        self._store_task1_report_from_calorie_report(report, output_dir / "task1_from_calories.json")
        return normalized

    def _store_task1_report_from_calorie_report(self, calorie_report: dict[str, Any], output_json: Path) -> None:
        qwen = calorie_report.get("qwen") if isinstance(calorie_report.get("qwen"), dict) else {}
        ingredient_step = qwen.get("ingredient_step") if isinstance(qwen.get("ingredient_step"), dict) else {}
        task1_report = ingredient_step.get("task1_report") if isinstance(ingredient_step.get("task1_report"), dict) else None
        if task1_report is None:
            return
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(task1_report, indent=2, sort_keys=True), encoding="utf-8")
        timings = task1_report.get("timings_sec") if isinstance(task1_report.get("timings_sec"), dict) else {}
        wall_sec = float(timings.get("total_image_sec") or 0.0)
        process = subprocess.CompletedProcess(["warm-task1-from-calories"], 0, "", "")
        normalized = self._normalize_task1("fast", task1_report, output_json, ["warm-task1-from-calories"], process, wall_sec)
        self._store_task_result("task1", "fast", normalized)

    def _load_task1_runtime(self) -> Any:
        if self._task1_runtime is not None:
            return self._task1_runtime
        module = self._load_module("demo_backend_task1", self.config.task1_script)
        argv = self._task1_base_argv()
        args = self._parse_module_args(module, self.config.task1_script, argv)
        self._apply_hf_env_to_current_process()
        self._task1_module = module
        self._task1_base_args = args
        self._task1_runtime = module.load_runtime(args)
        return self._task1_runtime

    def _load_task2_runtime(self, mode: ModeName | str) -> "WarmTask2Runtime":
        key = str(mode)
        if key in self._task2_runtimes:
            return self._task2_runtimes[key]
        module = self._load_module("demo_backend_task2", self.config.task2_script)
        argv = self._task2_base_argv(key)
        args = self._parse_module_args(module, self.config.task2_script, argv)
        self._apply_hf_env_to_current_process()
        runtime = WarmTask2Runtime(
            module,
            args,
            shared_embedder=self._shared_task1_siglip_for_task2(args),
            shared_reranker=self._shared_task2_reranker(args),
        )
        self._task2_module = module
        self._task2_runtimes[key] = runtime
        return runtime

    def _shared_task1_siglip_for_task2(self, task2_args: argparse.Namespace) -> "Task1Siglip2Adapter | None":
        if self._task1_runtime is None or self._task1_base_args is None or self._task1_module is None:
            return None
        task1_args = self._task1_base_args
        comparable_fields = (
            "siglip_model",
            "siglip_pretrained",
            "siglip_backend",
            "siglip_trt_engine",
            "device",
            "torch_dtype",
        )
        if any(getattr(task1_args, field, None) != getattr(task2_args, field, None) for field in comparable_fields):
            return None
        if int(getattr(task2_args, "siglip_image_size", 384)) != 384:
            return None
        return Task1Siglip2Adapter(self._task1_module, self._task1_runtime.embedder)

    def _shared_task2_reranker(self, task2_args: argparse.Namespace) -> Any | None:
        comparable_fields = (
            "reranker_backend",
            "reranker_model",
            "reranker_attn_implementation",
            "device",
            "torch_dtype",
            "reranker_edgellm_score_mode",
            "reranker_edgellm_true_token_id",
            "reranker_edgellm_false_token_id",
            "reranker_edgellm_doc_max_chars",
            "reranker_edgellm_max_new_tokens",
            "reranker_edgellm_ambiguous_score",
            "edgellm_root",
            "edgellm_workspace",
            "edgellm_model_name",
            "edgellm_llm_engine_dir",
            "edgellm_visual_engine_dir",
            "edgellm_persistent_binary",
            "edgellm_plugin_path",
            "edgellm_runtime_mode",
        )
        for runtime in self._task2_runtimes.values():
            if runtime.reranker is None:
                continue
            if not any(
                getattr(runtime.args, field, None) != getattr(task2_args, field, None)
                for field in comparable_fields
            ):
                return runtime.reranker
        return None

    def _load_calorie_runtime(self) -> Any:
        if self._calorie_runtime is not None:
            return self._calorie_runtime
        module = self._load_module("demo_backend_calories", self.config.calorie_script)
        argv = self._calorie_base_argv()
        args = self._parse_module_args(module, self.config.calorie_script, argv)
        reuse_task1_siglip = self._task1_runtime is not None and bool(getattr(args, "calorie_siglip_filter", False))
        reuse_task1_qwen = self._can_reuse_task1_qwen_for_calories(args)
        reparse = False
        if reuse_task1_siglip and not bool(getattr(args, "defer_calorie_siglip_filter", False)):
            argv.append("--defer-calorie-siglip-filter")
            reparse = True
        if reuse_task1_qwen and not bool(getattr(args, "defer_calorie_qwen", False)):
            argv.append("--defer-calorie-qwen")
            reparse = True
        if reparse:
            args = self._parse_module_args(module, self.config.calorie_script, argv)
        self._apply_hf_env_to_current_process()
        self._calorie_module = module
        self._calorie_base_args = args
        self._calorie_runtime = module.load_runtime(args)
        if reuse_task1_siglip and hasattr(module, "attach_task1_candidate_filter"):
            module.attach_task1_candidate_filter(self._calorie_runtime, self._task1_runtime)
        if reuse_task1_qwen and hasattr(module, "attach_task1_qwen_estimator"):
            module.attach_task1_qwen_estimator(self._calorie_runtime, self._task1_runtime)
        return self._calorie_runtime

    def _can_reuse_task1_qwen_for_calories(self, calorie_args: argparse.Namespace) -> bool:
        if self._task1_runtime is None or self._task1_base_args is None:
            return False
        if bool(getattr(calorie_args, "mock_qwen_json", "")):
            return False
        qwen = getattr(self._task1_runtime, "qwen", None)
        if qwen is None or getattr(qwen, "family", "") != "qwen":
            return False
        task1_args = self._task1_base_args
        comparable_fields = (
            "qwen_model",
            "device",
            "torch_dtype",
            "edgellm_model_name",
            "edgellm_llm_engine_profile",
            "edgellm_llm_engine_dir",
            "edgellm_visual_engine_dir",
            "edgellm_workspace",
        )
        return not any(getattr(task1_args, field, None) != getattr(calorie_args, field, None) for field in comparable_fields)

    def _task1_base_argv(self) -> list[str]:
        argv: list[str] = []
        self._append_common_args(argv)
        argv.extend(self.config.task1_extra_args)
        return argv

    def _task2_base_argv(self, mode: str) -> list[str]:
        final_score_mode = (
            self.config.task2_full_final_score_mode if mode == "full" else self.config.task2_fast_final_score_mode
        )
        argv = ["--measure", "--inference-only", "--final-score-mode", str(final_score_mode)]
        if self.config.task2_reranker_size:
            argv.extend(["--reranker-size", self.config.task2_reranker_size])
        self._append_common_args(argv)
        argv.extend(self.config.task2_extra_args)
        return argv

    def _calorie_base_argv(self) -> list[str]:
        argv: list[str] = []
        self._append_common_args(argv)
        argv.extend(self.config.calorie_extra_args)
        return argv

    def _load_module(self, module_name: str, script: Path) -> Any:
        script = script.resolve()
        script_dir = str(script.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import backend script: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _parse_module_args(self, module: Any, script: Path, argv: list[str]) -> argparse.Namespace:
        old_argv = sys.argv[:]
        try:
            sys.argv = [str(script), *argv]
            return module.parse_args()
        finally:
            sys.argv = old_argv

    def _apply_hf_env_to_current_process(self) -> None:
        if self.config.hf_token:
            token = self.config.hf_token.strip()
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    def _task1_argv(self, image_path: Path, mode: ModeName, output_json: Path) -> list[str]:
        argv = [
            self.config.python_executable,
            str(self.config.task1_script),
            "--image",
            str(image_path),
            "--output-json",
            str(output_json),
        ]
        self._append_common_args(argv)
        if mode == "fast" and self.config.task1_fast_skip_vlm_rel_gap_threshold is not None:
            argv.extend(
                [
                    "--skip-vlm-rel-gap-threshold",
                    str(self.config.task1_fast_skip_vlm_rel_gap_threshold),
                ]
            )
        argv.extend(self.config.task1_extra_args)
        return argv

    def _task2_argv(self, image_path: Path, mode: ModeName, output_json: Path, predictions_csv: Path) -> list[str]:
        final_score_mode = (
            self.config.task2_full_final_score_mode if mode == "full" else self.config.task2_fast_final_score_mode
        )
        argv = [
            self.config.python_executable,
            str(self.config.task2_script),
            "--image",
            str(image_path),
            "--output-json",
            str(output_json),
            "--predictions-csv",
            str(predictions_csv),
            "--measure",
            "--inference-only",
            "--final-score-mode",
            str(final_score_mode),
        ]
        if self.config.task2_reranker_size:
            argv.extend(["--reranker-size", self.config.task2_reranker_size])
        self._append_common_args(argv)
        argv.extend(self.config.task2_extra_args)
        return argv

    def _calorie_argv(self, image_path: Path, mode: ModeName, output_json: Path) -> list[str]:
        argv = [
            self.config.python_executable,
            str(self.config.calorie_script),
            "--image",
            str(image_path),
            "--output-json",
            str(output_json),
        ]
        self._append_common_args(argv)
        argv.extend(self.config.calorie_extra_args)
        return argv

    def _append_common_args(self, argv: list[str]) -> None:
        if self.config.device:
            argv.extend(["--device", self.config.device])
        if self.config.torch_dtype:
            argv.extend(["--torch-dtype", self.config.torch_dtype])

    def _run_subprocess(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                argv,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._backend_env(),
            )
        except subprocess.CalledProcessError as exc:
            output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
            tail = output.strip()[-2000:] or "no backend output captured"
            raise RuntimeError(f"Backend command failed with exit code {exc.returncode}: {tail}") from exc

    def _backend_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.hf_token:
            token = self.config.hf_token.strip()
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        return env

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Backend did not produce expected JSON: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        return payload

    @staticmethod
    def _normalize_task1(
        mode: ModeName,
        payload: dict[str, Any],
        output_json: Path,
        argv: list[str],
        process: subprocess.CompletedProcess[str],
        wall_sec: float,
    ) -> dict[str, Any]:
        selected = [str(item) for item in payload.get("selected_labels", []) if str(item).strip()]
        answer = ", ".join(selected) if selected else "no ingredients selected"
        timings = payload.get("timings_sec") if isinstance(payload.get("timings_sec"), dict) else {}
        latency = float(timings.get("total_image_sec") or wall_sec)
        skip_vlm = payload.get("skip_vlm") if isinstance(payload.get("skip_vlm"), dict) else {}
        qwen = payload.get("qwen") if isinstance(payload.get("qwen"), dict) else {}
        skipped = bool(skip_vlm.get("skipped") or qwen.get("skipped"))
        rerank_or_vlm = "VLM skipped" if skipped else "VLM used"
        return {
            "task": "task1",
            "mode": mode,
            "answer": answer,
            "latency_sec": latency,
            "rerank_or_vlm": rerank_or_vlm,
            "details": {
                "selected_labels": selected,
                "backend_json": str(output_json),
                "subprocess_wall_sec": wall_sec,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "argv": argv,
                "raw": payload,
            },
        }

    @staticmethod
    def _normalize_task2(
        mode: ModeName,
        payload: dict[str, Any],
        output_json: Path,
        argv: list[str],
        process: subprocess.CompletedProcess[str],
        wall_sec: float,
    ) -> dict[str, Any]:
        traces = payload.get("traces")
        trace = traces[0] if isinstance(traces, list) and traces and isinstance(traces[0], dict) else {}
        answer = str(trace.get("prediction_caption") or "").strip() or "no caption selected"
        timings = trace.get("timings_sec") if isinstance(trace.get("timings_sec"), dict) else {}
        latency = float(timings.get("total_image_sec") or wall_sec)
        rerank_applied = bool(trace.get("rerank_applied"))
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        final_score_mode = settings.get("final_score_mode", "")
        if final_score_mode == "siglip" or not rerank_applied:
            rerank_or_vlm = "Re-rank skipped"
        else:
            rerank_or_vlm = "Re-rank used"
        return {
            "task": "task2",
            "mode": mode,
            "answer": answer,
            "latency_sec": latency,
            "rerank_or_vlm": rerank_or_vlm,
            "details": {
                "prediction_caption": answer,
                "prediction_cat": trace.get("prediction_cat", ""),
                "final_score_mode": final_score_mode,
                "rerank_applied": rerank_applied,
                "siglip_top_gap": trace.get("siglip_top_gap"),
                "reranker_mode": models.get("reranker_mode", ""),
                "backend_json": str(output_json),
                "subprocess_wall_sec": wall_sec,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "argv": argv,
                "raw": payload,
            },
        }

    @staticmethod
    def _format_calorie_answer(calories: dict[str, Any]) -> str:
        total = calories.get("total_kcal")
        scope = str(calories.get("estimation_scope") or "")
        ingredients = calories.get("ingredients") if isinstance(calories.get("ingredients"), list) else []
        abundance_count = sum(1 for item in ingredients if isinstance(item, dict) and item.get("abundance_scene"))
        if scope == "per_instance_not_full_image":
            if abundance_count == 1 and isinstance(total, (int, float)):
                total_text = f"Estimated calories for one representative item: {int(total)} kcal"
            else:
                total_text = "This looks like an abundance/display scene, so the full image total is not estimated"
        elif scope == "average_portion_not_full_image":
            total_text = f"Estimated calories for an eating-portion assumption: {int(total)} kcal" if isinstance(total, (int, float)) else "Estimated calories for an eating-portion assumption: unavailable"
        else:
            total_text = f"Estimated total dish calories: {int(total)} kcal" if isinstance(total, (int, float)) else "Estimated total dish calories: unavailable"
        parts: list[str] = []
        for item in ingredients[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            kcal = item.get("kcal")
            count = item.get("count")
            per_object = item.get("calories_per_single_object")
            per_instance = item.get("per_instance_kcal")
            portion_category = str(item.get("portion_category") or "").strip()
            if not name:
                continue
            if isinstance(kcal, (int, float)) and isinstance(count, (int, float)) and isinstance(per_object, (int, float)):
                parts.append(f"{name}: {int(kcal)} kcal ({float(count):g} x {float(per_object):g} kcal)")
            elif isinstance(kcal, (int, float)) and item.get("abundance_scene") and isinstance(per_instance, (int, float)):
                parts.append(f"{name}: about {float(per_instance):g} kcal each")
            elif isinstance(kcal, (int, float)) and portion_category and portion_category != "none":
                parts.append(f"{name}: {int(kcal)} kcal ({portion_category} portion)")
            elif isinstance(kcal, (int, float)):
                parts.append(f"{name}: {int(kcal)} kcal")
            else:
                parts.append(name)
        if parts:
            return f"{total_text}. Ingredient calories: {'; '.join(parts)}"
        return total_text

    @classmethod
    def _normalize_calories(
        cls,
        mode: ModeName,
        payload: dict[str, Any],
        output_json: Path,
        argv: list[str],
        process: subprocess.CompletedProcess[str],
        wall_sec: float,
    ) -> dict[str, Any]:
        calories = payload.get("calories") if isinstance(payload.get("calories"), dict) else {}
        answer = str(payload.get("answer") or "").strip() or cls._format_calorie_answer(calories)
        timings = payload.get("timings_sec") if isinstance(payload.get("timings_sec"), dict) else {}
        latency = float(timings.get("total_image_sec") or wall_sec)
        qwen = payload.get("qwen") if isinstance(payload.get("qwen"), dict) else {}
        quantity_vlm_used = bool(str(qwen.get("raw_text") or "").strip())
        rerank_or_vlm = "Composition VLM used" if quantity_vlm_used else "Composition VLM skipped"
        return {
            "task": "calories",
            "mode": mode,
            "answer": answer,
            "latency_sec": latency,
            "rerank_or_vlm": rerank_or_vlm,
            "details": {
                "calories": calories,
                "backend_json": str(output_json),
                "subprocess_wall_sec": wall_sec,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "argv": argv,
                "raw": payload,
            },
        }


def split_extra_args(value: str) -> list[str]:
    return shlex.split(value) if value.strip() else []


class Task1Siglip2Adapter:
    def __init__(self, task1_module: Any, embedder: Any) -> None:
        self.task1_module = task1_module
        self.embedder = embedder
        self.backend = getattr(embedder, "backend", "pytorch")
        self.onnx_provider = getattr(embedder, "onnx_provider", None)
        self.onnx_dir = getattr(embedder, "onnx_dir", None)
        self.engine_path = getattr(embedder, "engine_path", None)
        self.session_providers = getattr(embedder, "session_providers", [])

    def encode_texts(self, texts: list[str], batch_size: int) -> Any:
        return self.embedder.encode_texts(texts, batch_size)

    def encode_image(self, image_path: Path) -> Any:
        with self.task1_module.Image.open(image_path) as image:
            return self.embedder.encode_image(image.convert("RGB"))


class WarmTask2Runtime:
    def __init__(
        self,
        module: Any,
        args: argparse.Namespace,
        shared_embedder: Task1Siglip2Adapter | None = None,
        shared_reranker: Any | None = None,
    ) -> None:
        self.m = module
        self.args = args
        self.hf_token = module.configure_hf_token(args)
        caption_bank_json = args.caption_bank_json or args.evaluation_json
        self.caption_bank = module.read_caption_bank_items(caption_bank_json, args.caption_text_mode)
        print(f"Candidate captions: {len(self.caption_bank)}")
        module.preflight_reranker_requirements(args)
        self.siglip_runtime_source = "task2_runtime"
        if shared_embedder is not None:
            self.embedder = shared_embedder
            self.load_siglip2_sec = 0.0
            self.siglip_runtime_source = "reused_task1_runtime"
        else:
            started = time.perf_counter()
            if hasattr(module, "build_siglip2_embedder"):
                self.embedder = module.build_siglip2_embedder(args)
            else:
                self.embedder = module.OpenCLIPSigLIP2Embedder(
                    args.siglip_model,
                    args.siglip_pretrained,
                    args.device,
                    args.torch_dtype,
                    args.siglip_image_size,
                )
            self.load_siglip2_sec = time.perf_counter() - started
        started = time.perf_counter()
        self.caption_embeddings, self.caption_cache = module.load_or_build_caption_embeddings(self.embedder, self.caption_bank, args)
        self.caption_embeddings_sec = time.perf_counter() - started
        self.reranker = None
        started = time.perf_counter()
        if args.mock_reranker or args.final_score_mode == "siglip":
            self.reranker_mode = "mock_siglip_scores" if args.mock_reranker else "skipped_siglip_final_score"
        elif shared_reranker is not None:
            self.reranker = shared_reranker
            self.reranker_mode = module.caption_reranker_mode(args)
        else:
            if hasattr(module, "build_caption_reranker"):
                self.reranker = module.build_caption_reranker(args, self.hf_token)
                self.reranker_mode = module.caption_reranker_mode(args)
            elif getattr(args, "reranker_backend", "sentence_transformers") == "edgellm":
                self.reranker = module.EdgeLLMCaptionReranker(args)
                prefix = "edgellm_" + str(args.reranker_edgellm_score_mode)
                self.reranker_mode = (
                    prefix + ":" + str(args.edgellm_model_name)
                    if args.final_score_mode != "siglip_guarded"
                    else "siglip_guarded:" + prefix + ":" + str(args.edgellm_model_name)
                )
            else:
                self.reranker = module.QwenVLCaptionReranker(args, self.hf_token)
                self.reranker_mode = (
                    args.reranker_model
                    if args.final_score_mode != "siglip_guarded"
                    else f"siglip_guarded:{args.reranker_model}"
                )
        self.load_reranker_sec = time.perf_counter() - started

    def run_one(self, image_path: Path) -> dict[str, Any]:
        args = self.args
        query_t0 = time.perf_counter()
        started = time.perf_counter()
        image_embedding = self.embedder.encode_image(image_path)
        scores = 100.0 * (image_embedding @ self.caption_embeddings.T)
        candidates = self.m.build_caption_recall(scores, self.caption_bank, args)
        siglip_image_and_recall_sec = time.perf_counter() - started

        started = time.perf_counter()
        rerank_applied = False
        if args.mock_reranker or args.final_score_mode == "siglip":
            for candidate in candidates:
                candidate.rerank_score = candidate.siglip_score
        elif args.final_score_mode == "siglip_guarded" and self.m.siglip_is_confident(candidates, args):
            pass
        else:
            if self.reranker is None:
                raise RuntimeError("Reranker was not initialized")
            pair_paths = [image_path for _ in candidates]
            pair_texts = [candidate.text for candidate in candidates]
            rerank_applied = True
            rerank_scores = self.reranker.score_pairs(pair_paths, pair_texts, args.reranker_batch_size)
            for candidate, score in zip(candidates, rerank_scores):
                candidate.rerank_score = score
        rerank_sec = time.perf_counter() - started

        started = time.perf_counter()
        final_scores = self.m.final_candidate_scores(candidates, args)
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
        final_selection_sec = time.perf_counter() - started
        total_image_sec = time.perf_counter() - query_t0

        trace = {
            "image": image_path.name,
            "image_path": str(image_path),
            "truth_cat": "",
            "truth_caption": "",
            "prediction_filename": pred.filename if pred else "",
            "prediction_cat": pred.cat if pred else "",
            "prediction_caption": pred.caption if pred else "",
            "caption_correct": None,
            "class_correct": None,
            "siglip_top1_filename": siglip_pred.filename if siglip_pred else "",
            "siglip_top1_caption": siglip_pred.caption if siglip_pred else "",
            "siglip_top1_caption_correct": None,
            "truth_caption_in_candidates": None,
            "siglip_top_gap": self.m.siglip_top_gap(candidates),
            "rerank_applied": rerank_applied,
            "timings_sec": {
                "siglip_image_and_recall_sec": siglip_image_and_recall_sec,
                "rerank_sec": rerank_sec,
                "final_selection_sec": final_selection_sec,
                "total_image_sec": total_image_sec,
            },
            "ranked_candidates": [self.m.candidate_to_dict(candidate, ranked_scores[id(candidate)]) for candidate in ranked_candidates],
        }
        return {
            "schema_version": "dishcovery_demo_warm_task2_v1",
            "models": {
                "siglip_model": args.siglip_model,
                "siglip_pretrained": args.siglip_pretrained,
                "siglip_backend": getattr(self.embedder, "backend", "pytorch"),
                "siglip_onnx_dir": str(getattr(self.embedder, "onnx_dir", "") or "") or None,
                "siglip_trt_engine": str(getattr(self.embedder, "engine_path", "")) or None,
                "siglip_onnx_provider": getattr(self.embedder, "onnx_provider", None),
                "siglip_session_providers": getattr(self.embedder, "session_providers", []),
                "reranker_size": args.reranker_size,
                "reranker_model": args.reranker_model,
                "reranker_mode": self.reranker_mode,
                "reranker_backend": getattr(args, "reranker_backend", "sentence_transformers"),
                "reranker_edgellm_score_mode": getattr(args, "reranker_edgellm_score_mode", None),
                "edgellm_model_name": getattr(args, "edgellm_model_name", None),
                "siglip_runtime_source": self.siglip_runtime_source,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
            },
            "settings": {
                "caption_text_mode": args.caption_text_mode,
                "siglip_image_size": args.siglip_image_size,
                "siglip_top_k": args.siglip_top_k,
                "rerank_top_k": args.rerank_top_k,
                "final_score_mode": args.final_score_mode,
                "rerank_weight": args.rerank_weight,
                "siglip_keep_gap": args.siglip_keep_gap,
                "inference_only": True,
                "warm_runtime": True,
            },
            "caption_bank_count": len(self.caption_bank),
            "caption_cache": self.caption_cache,
            "timings_sec": {
                "load_siglip2_sec": self.load_siglip2_sec,
                "caption_embeddings_sec": self.caption_embeddings_sec,
                "load_reranker_sec": self.load_reranker_sec,
            },
            "metrics": {
                "rows": 1,
                "metric_available": False,
                "rerank_pair_count": len(candidates) if rerank_applied else 0,
                "rerank_skipped_by_siglip_guard": int(args.final_score_mode == "siglip_guarded" and not rerank_applied),
            },
            "traces": [trace],
        }
