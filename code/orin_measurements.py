#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK1_PIPELINE = "code/orin_task1_pipeline.py"
TASK2_PIPELINE = "code/orin_task2_pipeline.py"
DEFAULT_TASK1_IMAGES = ROOT / "dataset/MM-Food-100K/images_significant_subset_qwen3_350.txt"
DEFAULT_TASK2_IMAGES = ROOT / "dataset/food500_subset/images.txt"
DEFAULT_SUITE_OUTPUT_DIR = ROOT / "outputs/orin_measurements"
DEFAULT_RESOURCE_OUTPUT_DIR = ROOT / "outputs/orin_resource_measurements"
DEFAULT_ORIN_PYTHON = Path.home() / "miniconda3/envs/dishcovery_andrea/bin/python"

PREFERRED_POWER_RAILS = (
    "VDD_IN",
    "VDD_CPU_GPU_CV",
    "VDD_GPU_SOC",
    "VDD_CPU_CV",
    "VDD_SOC",
    "VIN_SYS_5V0",
)
POWER_RE = re.compile(r"\b([A-Za-z0-9_]+)\s+([0-9]+(?:\.[0-9]+)?)mW(?:/([0-9]+(?:\.[0-9]+)?)mW)?")
RAM_RE = re.compile(r"\bRAM\s+([0-9]+)\/([0-9]+)MB\b")
SWAP_RE = re.compile(r"\bSWAP\s+([0-9]+)\/([0-9]+)MB\b")
CPU_RE = re.compile(r"\bCPU\s+\[([^\]]*)\]")
CPU_CORE_RE = re.compile(r"^([0-9]+)%@([0-9]+)")
GR3D_RE = re.compile(r"\bGR3D_FREQ\s+([0-9]+)%")


@dataclass(frozen=True)
class PowerSample:
    timestamp_sec: float
    power_w: float
    rail: str
    raw: str


def parse_tegrastats_rails(line: str) -> dict[str, float]:
    rails: dict[str, float] = {}
    for match in POWER_RE.finditer(line):
        rails[match.group(1)] = float(match.group(2)) / 1000.0
    return rails


def select_power_from_rails(rails: dict[str, float]) -> tuple[float, str] | None:
    if not rails:
        return None

    by_upper = {rail.upper(): (rail, watts) for rail, watts in rails.items()}
    if "VDD_GPU_SOC" in by_upper and "VDD_CPU_CV" in by_upper:
        gpu_rail, gpu_watts = by_upper["VDD_GPU_SOC"]
        cpu_rail, cpu_watts = by_upper["VDD_CPU_CV"]
        return gpu_watts + cpu_watts, f"sum:{gpu_rail}+{cpu_rail}"
    if "VDD_IN" in by_upper:
        rail, watts = by_upper["VDD_IN"]
        return watts, rail
    for preferred in PREFERRED_POWER_RAILS:
        item = by_upper.get(preferred)
        if item is not None:
            return item[1], item[0]

    if len(rails) == 1:
        rail, watts = next(iter(rails.items()))
        return watts, rail

    total_watts = sum(rails.values())
    return total_watts, "sum:" + "+".join(rails)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def parse_tegrastats_power(line: str) -> tuple[float, str] | None:
    return select_power_from_rails(parse_tegrastats_rails(line))


def parse_tegrastats_resources(line: str) -> dict[str, float]:
    resources: dict[str, float] = {}

    ram_match = RAM_RE.search(line)
    if ram_match:
        used_mb = float(ram_match.group(1))
        total_mb = float(ram_match.group(2))
        resources["ram_used_mb"] = used_mb
        resources["ram_total_mb"] = total_mb
        resources["ram_used_pct"] = 100.0 * used_mb / total_mb if total_mb > 0.0 else 0.0

    swap_match = SWAP_RE.search(line)
    if swap_match:
        used_mb = float(swap_match.group(1))
        total_mb = float(swap_match.group(2))
        resources["swap_used_mb"] = used_mb
        resources["swap_total_mb"] = total_mb
        resources["swap_used_pct"] = 100.0 * used_mb / total_mb if total_mb > 0.0 else 0.0

    cpu_match = CPU_RE.search(line)
    if cpu_match:
        all_utils: list[float] = []
        online_utils: list[float] = []
        online_freqs: list[float] = []
        for item in cpu_match.group(1).split(","):
            text = item.strip()
            if not text or text == "off":
                all_utils.append(0.0)
                continue
            core_match = CPU_CORE_RE.match(text)
            if core_match is None:
                continue
            util = float(core_match.group(1))
            freq = float(core_match.group(2))
            all_utils.append(util)
            online_utils.append(util)
            online_freqs.append(freq)
        if all_utils:
            resources["cpu_core_count"] = float(len(all_utils))
            resources["cpu_all_core_avg_util_pct"] = sum(all_utils) / len(all_utils)
            resources["cpu_all_core_max_util_pct"] = max(all_utils)
        if online_utils:
            resources["cpu_online_core_count"] = float(len(online_utils))
            resources["cpu_online_avg_util_pct"] = sum(online_utils) / len(online_utils)
            resources["cpu_online_max_util_pct"] = max(online_utils)
        else:
            resources["cpu_online_core_count"] = 0.0
            resources["cpu_online_avg_util_pct"] = 0.0
            resources["cpu_online_max_util_pct"] = 0.0
        if online_freqs:
            resources["cpu_online_avg_freq_mhz"] = sum(online_freqs) / len(online_freqs)
            resources["cpu_online_max_freq_mhz"] = max(online_freqs)

    gr3d_match = GR3D_RE.search(line)
    if gr3d_match:
        resources["gpu_gr3d_util_pct"] = float(gr3d_match.group(1))

    return resources


def summarize_numeric(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "avg": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95.0),
    }


class PowerMonitor:
    def __init__(self, enabled: bool, interval_ms: int = 200) -> None:
        self.enabled = bool(enabled)
        self.interval_ms = max(20, int(interval_ms))
        self.samples: list[PowerSample] = []
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._error: str | None = None
        self._rail_sums: dict[str, float] = {}
        self._rail_counts: dict[str, int] = {}
        self._rail_mins: dict[str, float] = {}
        self._rail_maxs: dict[str, float] = {}
        self._resource_samples: list[dict[str, float]] = []

    @property
    def available(self) -> bool:
        return shutil.which("tegrastats") is not None

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._ended_at = None
        if not self.enabled:
            return
        if not self.available:
            self._error = "tegrastats_not_found"
            return
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._proc = None
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                if self._stop_requested.is_set():
                    break
                rails = parse_tegrastats_rails(raw_line)
                resources = parse_tegrastats_resources(raw_line)
                if resources:
                    self._resource_samples.append(resources)
                if not rails and not resources:
                    continue
                for rail, watts in rails.items():
                    self._rail_sums[rail] = self._rail_sums.get(rail, 0.0) + float(watts)
                    self._rail_counts[rail] = self._rail_counts.get(rail, 0) + 1
                    self._rail_mins[rail] = min(self._rail_mins.get(rail, float("inf")), float(watts))
                    self._rail_maxs[rail] = max(self._rail_maxs.get(rail, float("-inf")), float(watts))
                parsed = select_power_from_rails(rails)
                if parsed is None:
                    continue
                power_w, rail = parsed
                self.samples.append(
                    PowerSample(
                        timestamp_sec=time.perf_counter(),
                        power_w=float(power_w),
                        rail=rail,
                        raw=raw_line.strip(),
                    )
                )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def stop(self, wall_sec: float | None = None) -> dict[str, Any]:
        self._ended_at = time.perf_counter()
        self._stop_requested.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.summary(wall_sec=wall_sec)

    def summary(self, wall_sec: float | None = None) -> dict[str, Any]:
        powers = [sample.power_w for sample in self.samples]
        rail_counts: dict[str, int] = {}
        for sample in self.samples:
            rail_counts[sample.rail] = rail_counts.get(sample.rail, 0) + 1
        if wall_sec is None and self._started_at is not None:
            end = self._ended_at if self._ended_at is not None else time.perf_counter()
            wall_sec = max(0.0, end - self._started_at)
        resource_keys = sorted({key for sample in self._resource_samples for key in sample})
        resource_stats = {
            key: summarize_numeric([sample[key] for sample in self._resource_samples if key in sample])
            for key in resource_keys
        }
        return {
            "enabled": self.enabled,
            "available": self.available,
            "error": self._error,
            "sample_count": len(powers),
            "rail_counts": rail_counts,
            "rail_sample_counts": dict(sorted(self._rail_counts.items())),
            "rail_avg_power_w": {
                rail: self._rail_sums[rail] / count
                for rail, count in sorted(self._rail_counts.items())
                if count
            },
            "rail_min_power_w": dict(sorted(self._rail_mins.items())),
            "rail_max_power_w": dict(sorted(self._rail_maxs.items())),
            "selected_power_source": max(rail_counts, key=rail_counts.get) if rail_counts else "",
            "avg_power_w": (sum(powers) / len(powers)) if powers else None,
            "median_power_w": statistics.median(powers) if powers else None,
            "min_power_w": min(powers) if powers else None,
            "max_power_w": max(powers) if powers else None,
            "wall_sec": wall_sec,
            "last_raw": self.samples[-1].raw if self.samples else "",
            "resources": {
                "sample_count": len(self._resource_samples),
                "stats": resource_stats,
            },
        }


def query_nvpmodel() -> dict[str, Any]:
    if shutil.which("nvpmodel") is None:
        return {"available": False, "error": "nvpmodel_not_found"}
    try:
        completed = subprocess.run(
            ["nvpmodel", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_benchmark_summary(
    *,
    task_name: str,
    query_count: int,
    latencies_sec: list[float],
    measured_wall_sec: float,
    task_metric_name: str,
    task_metric_value: float | None,
    power_summary: dict[str, Any],
    w_config: str = "",
    nvpmodel: dict[str, Any] | None = None,
    extra_task_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latencies = [float(value) for value in latencies_sec]
    rows = int(query_count)
    avg_power_w = power_summary.get("avg_power_w")
    energy_total_j = None
    energy_per_query_j = None
    if avg_power_w is not None and rows > 0:
        energy_total_j = float(avg_power_w) * float(measured_wall_sec)
        energy_per_query_j = energy_total_j / rows
    return {
        "schema_version": "orin_latency_power_measurement_v1",
        "task": task_name,
        "w_config": w_config,
        "query_count": rows,
        "measured_wall_sec": float(measured_wall_sec),
        "avg_latency_per_query_sec": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "median_latency_per_query_sec": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_per_query_sec": percentile(latencies, 95.0),
        "min_latency_per_query_sec": min(latencies) if latencies else 0.0,
        "max_latency_per_query_sec": max(latencies) if latencies else 0.0,
        "throughput_images_per_second": (rows / measured_wall_sec) if measured_wall_sec > 0.0 else 0.0,
        "avg_power_w": avg_power_w,
        "energy_total_j": energy_total_j,
        "energy_per_query_j": energy_per_query_j,
        "task_metric": {
            "name": task_metric_name,
            "value": None if task_metric_value is None else float(task_metric_value),
        },
        "extra_task_metrics": extra_task_metrics or {},
        "power": power_summary,
        "nvpmodel": nvpmodel or {},
        "notes": "Latency and energy exclude model/text-bank loading when the caller starts measurement after setup.",
    }


def print_benchmark_summary(summary: dict[str, Any]) -> None:
    print("Benchmark summary:")
    print(
        "- latency/query avg={avg:.3f}s median={median:.3f}s p95={p95:.3f}s".format(
            avg=float(summary.get("avg_latency_per_query_sec", 0.0)),
            median=float(summary.get("median_latency_per_query_sec", 0.0)),
            p95=float(summary.get("p95_latency_per_query_sec", 0.0)),
        )
    )
    avg_power_w = summary.get("avg_power_w")
    energy_per_query_j = summary.get("energy_per_query_j")
    power_text = "unavailable" if avg_power_w is None else f"{float(avg_power_w):.3f}W"
    energy_text = "unavailable" if energy_per_query_j is None else f"{float(energy_per_query_j):.3f}J"
    print(
        "- power avg={power} energy/query={energy} throughput={throughput:.3f} img/s".format(
            power=power_text,
            energy=energy_text,
            throughput=float(summary.get("throughput_images_per_second", 0.0)),
        )
    )
    metric = summary.get("task_metric", {})
    metric_value = metric.get("value")
    if metric_value is None:
        print(f"- task metric {metric.get('name', 'metric')}=unavailable")
    else:
        print(f"- task metric {metric.get('name', 'metric')}={float(metric_value):.4f}")
    resources = summary.get("power", {}).get("resources", {}).get("stats", {})
    ram = resources.get("ram_used_mb") or {}
    cpu = resources.get("cpu_all_core_avg_util_pct") or {}
    gpu = resources.get("gpu_gr3d_util_pct") or {}
    if ram or cpu or gpu:
        print(
            "- resources RAM avg={ram_avg:.0f}MB max={ram_max:.0f}MB "
            "CPU avg={cpu_avg:.1f}% max={cpu_max:.1f}% "
            "GPU avg={gpu_avg:.1f}% max={gpu_max:.1f}%".format(
                ram_avg=float(ram.get("avg", 0.0)),
                ram_max=float(ram.get("max", 0.0)),
                cpu_avg=float(cpu.get("avg", 0.0)),
                cpu_max=float(cpu.get("max", 0.0)),
                gpu_avg=float(gpu.get("avg", 0.0)),
                gpu_max=float(gpu.get("max", 0.0)),
            )
        )


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_power_mode_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in split_csv(value):
        if ":" not in item:
            raise ValueError(f"Invalid --nvpmodel-modes item: {item}. Expected label:mode.")
        label, mode = item.split(":", 1)
        mapping[label.strip()] = mode.strip()
    return mapping


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def maybe_set_nvpmodel(label: str, mode_by_label: dict[str, str], *, enabled: bool, dry_run: bool) -> None:
    if not enabled:
        return
    mode = mode_by_label.get(label)
    if mode is None:
        raise ValueError(f"No nvpmodel mode provided for {label}")
    run_command(["sudo", "nvpmodel", "-m", mode], dry_run=dry_run)
    if not dry_run:
        time.sleep(3.0)


def task1_measurement_command(
    *,
    python: str,
    images_list: Path,
    output_json: Path,
    predictions_csv: Path,
    power_sample_interval_ms: int,
    w_config: str,
    samples: int,
    seed: int | None = None,
    eval_first: bool = False,
) -> list[str]:
    cmd = [
        python,
        TASK1_PIPELINE,
        "--eval-samples",
        str(samples),
    ]
    if eval_first:
        cmd.append("--eval-first")
    elif seed is not None:
        cmd.extend(["--seed", str(seed)])
    cmd.extend(
        [
            "--images-list",
            str(images_list),
            "--measure",
            "--measure-power",
            "--power-sample-interval-ms",
            str(power_sample_interval_ms),
            "--w-config",
            w_config,
            "--output-json",
            str(output_json),
            "--predictions-csv",
            str(predictions_csv),
        ]
    )
    return cmd


def task2_measurement_command(
    *,
    python: str,
    images_list: Path,
    output_json: Path,
    predictions_csv: Path,
    power_sample_interval_ms: int,
    w_config: str,
    eval_all: bool,
    samples: int,
    seed: int,
    reranker_size: str | None = None,
    final_score_mode: str | None = None,
    siglip_keep_gap: float | None = None,
    siglip_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> list[str]:
    cmd = [python, TASK2_PIPELINE]
    if eval_all:
        cmd.append("--eval-all")
    else:
        cmd.extend(["--eval-samples", str(samples), "--seed", str(seed)])
    cmd.extend(
        [
            "--images-list",
            str(images_list),
            "--measure",
            "--measure-power",
            "--power-sample-interval-ms",
            str(power_sample_interval_ms),
            "--w-config",
            w_config,
            "--output-json",
            str(output_json),
            "--predictions-csv",
            str(predictions_csv),
        ]
    )
    if reranker_size:
        cmd.extend(["--reranker-size", reranker_size])
    if final_score_mode:
        cmd.extend(["--final-score-mode", final_score_mode])
    if siglip_top_k is not None:
        cmd.extend(["--siglip-top-k", str(siglip_top_k)])
    if rerank_top_k is not None:
        cmd.extend(["--rerank-top-k", str(rerank_top_k)])
    if siglip_keep_gap is not None:
        cmd.extend(["--siglip-keep-gap", str(siglip_keep_gap)])
    return cmd


def benchmark_summary_row(path: Path, *, task: str, w_config: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload.get("benchmark") or {}
    metric = benchmark.get("task_metric") or {}
    return {
        "task": task,
        "w_config": w_config,
        "query_count": benchmark.get("query_count"),
        "avg_latency_per_query_sec": benchmark.get("avg_latency_per_query_sec"),
        "median_latency_per_query_sec": benchmark.get("median_latency_per_query_sec"),
        "p95_latency_per_query_sec": benchmark.get("p95_latency_per_query_sec"),
        "avg_power_w": benchmark.get("avg_power_w"),
        "energy_per_query_j": benchmark.get("energy_per_query_j"),
        "throughput_images_per_second": benchmark.get("throughput_images_per_second"),
        "task_metric_name": metric.get("name"),
        "task_metric_value": metric.get("value"),
        "json": str(path),
    }


def nested_stat(payload: dict[str, Any], key: str, field: str) -> Any:
    stats = payload.get("benchmark", {}).get("power", {}).get("resources", {}).get("stats", {})
    return (stats.get(key) or {}).get(field)


def resource_summary_row(path: Path, *, config_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload.get("benchmark", {})
    metrics = payload.get("metrics", {})
    models = payload.get("models", {})
    settings = payload.get("settings", {})
    return {
        "config": config_name,
        "w_config": benchmark.get("w_config"),
        "rows": metrics.get("rows"),
        "reranker_size": models.get("reranker_size", ""),
        "final_score_mode": settings.get("final_score_mode", ""),
        "avg_latency_sec": benchmark.get("avg_latency_per_query_sec"),
        "p95_latency_sec": benchmark.get("p95_latency_per_query_sec"),
        "throughput_images_per_second": benchmark.get("throughput_images_per_second"),
        "avg_power_w": benchmark.get("avg_power_w"),
        "ram_used_avg_mb": nested_stat(payload, "ram_used_mb", "avg"),
        "ram_used_max_mb": nested_stat(payload, "ram_used_mb", "max"),
        "ram_used_avg_pct": nested_stat(payload, "ram_used_pct", "avg"),
        "swap_used_max_mb": nested_stat(payload, "swap_used_mb", "max"),
        "cpu_all_core_avg_util_pct": nested_stat(payload, "cpu_all_core_avg_util_pct", "avg"),
        "cpu_all_core_max_util_pct": nested_stat(payload, "cpu_all_core_max_util_pct", "max"),
        "cpu_online_avg_util_pct": nested_stat(payload, "cpu_online_avg_util_pct", "avg"),
        "cpu_online_max_util_pct": nested_stat(payload, "cpu_online_max_util_pct", "max"),
        "cpu_online_avg_freq_mhz": nested_stat(payload, "cpu_online_avg_freq_mhz", "avg"),
        "cpu_online_max_freq_mhz": nested_stat(payload, "cpu_online_max_freq_mhz", "max"),
        "gpu_gr3d_avg_util_pct": nested_stat(payload, "gpu_gr3d_util_pct", "avg"),
        "gpu_gr3d_max_util_pct": nested_stat(payload, "gpu_gr3d_util_pct", "max"),
        "json": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_shared_runner_args(parser: argparse.ArgumentParser, *, default_output_dir: Path, default_python: str) -> None:
    parser.add_argument("--task1-images-list", type=Path, default=DEFAULT_TASK1_IMAGES)
    parser.add_argument("--task2-images-list", type=Path, default=DEFAULT_TASK2_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--python", default=default_python)
    parser.add_argument("--power-sample-interval-ms", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")


def add_suite_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "suite",
        help="Run Task 1 and/or Task 2 measurements across power-budget labels.",
    )
    parser.add_argument("--tasks", default="task1,task2", help="Comma-separated tasks: task1, task2, or both.")
    parser.add_argument("--w-configs", default="15W,30W,50W", help="Comma-separated power labels to record.")
    parser.add_argument("--nvpmodel-modes", default="", help="Comma-separated label:mode map, for example 15W:1,30W:2,50W:0.")
    parser.add_argument("--set-nvpmodel", action="store_true", help="Run sudo nvpmodel -m MODE before each configured run.")
    add_shared_runner_args(parser, default_output_dir=DEFAULT_SUITE_OUTPUT_DIR, default_python=sys.executable)
    parser.set_defaults(command="suite")


def add_resources_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "resources",
        help="Run short resource measurements for Task 1 and Task 2 reranker configurations.",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-config", default="current")
    parser.add_argument("--siglip-keep-gap", type=float, default=3.0)
    parser.add_argument("--siglip-top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    default_python = str(DEFAULT_ORIN_PYTHON if DEFAULT_ORIN_PYTHON.exists() else Path(sys.executable))
    add_shared_runner_args(parser, default_output_dir=DEFAULT_RESOURCE_OUTPUT_DIR, default_python=default_python)
    parser.set_defaults(command="resources")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson Orin measurement helpers and batch runners.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_suite_parser(subparsers)
    add_resources_parser(subparsers)
    return parser.parse_args()


def run_suite(args: argparse.Namespace) -> None:
    tasks = split_csv(args.tasks)
    w_configs = split_csv(args.w_configs)
    mode_by_label = parse_power_mode_map(args.nvpmodel_modes)
    unknown_tasks = sorted(set(tasks) - {"task1", "task2", "both"})
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if "both" in tasks:
        tasks = ["task1", "task2"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for label in w_configs:
        maybe_set_nvpmodel(label, mode_by_label, enabled=args.set_nvpmodel, dry_run=args.dry_run)
        for task in tasks:
            task_dir = args.output_dir / task
            out_json = task_dir / f"{task}_{label}.json"
            out_csv = task_dir / f"{task}_{label}_predictions.csv"
            if task == "task1":
                cmd = task1_measurement_command(
                    python=args.python,
                    images_list=args.task1_images_list,
                    output_json=out_json,
                    predictions_csv=out_csv,
                    power_sample_interval_ms=args.power_sample_interval_ms,
                    w_config=label,
                    samples=350,
                    eval_first=True,
                )
            else:
                cmd = task2_measurement_command(
                    python=args.python,
                    images_list=args.task2_images_list,
                    output_json=out_json,
                    predictions_csv=out_csv,
                    power_sample_interval_ms=args.power_sample_interval_ms,
                    w_config=label,
                    eval_all=True,
                    samples=0,
                    seed=0,
                )
            run_command(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                rows.append(benchmark_summary_row(out_json, task=task, w_config=label))

    if not args.dry_run:
        summary_csv = args.output_dir / "summary.csv"
        write_csv(summary_csv, rows)
        print(f"Summary CSV: {summary_csv}")


RESOURCE_CONFIGS = {
    "task1": {},
    "task2_reranker2b_guarded": {"reranker_size": "2B", "final_score_mode": "siglip_guarded"},
    "task2_reranker8b_full": {"reranker_size": "8B", "final_score_mode": "rerank"},
}


def run_resources(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config_name, config in RESOURCE_CONFIGS.items():
        out_json = args.output_dir / f"{config_name}_{args.w_config}_{args.samples}.json"
        out_csv = args.output_dir / f"{config_name}_{args.w_config}_{args.samples}_predictions.csv"
        if config_name == "task1":
            cmd = task1_measurement_command(
                python=args.python,
                images_list=args.task1_images_list,
                output_json=out_json,
                predictions_csv=out_csv,
                power_sample_interval_ms=args.power_sample_interval_ms,
                w_config=args.w_config,
                samples=args.samples,
                seed=args.seed,
            )
        else:
            cmd = task2_measurement_command(
                python=args.python,
                images_list=args.task2_images_list,
                output_json=out_json,
                predictions_csv=out_csv,
                power_sample_interval_ms=args.power_sample_interval_ms,
                w_config=args.w_config,
                eval_all=False,
                samples=args.samples,
                seed=args.seed,
                siglip_keep_gap=args.siglip_keep_gap if config["final_score_mode"] == "siglip_guarded" else None,
                siglip_top_k=args.siglip_top_k,
                rerank_top_k=args.rerank_top_k,
                **config,
            )
        run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            rows.append(resource_summary_row(out_json, config_name=config_name))

    if not args.dry_run:
        summary_csv = args.output_dir / f"summary_{args.w_config}_{args.samples}.csv"
        write_csv(summary_csv, rows)
        print(f"Resource summary CSV: {summary_csv}")


def main() -> None:
    args = parse_args()
    if args.command == "suite":
        run_suite(args)
    elif args.command == "resources":
        run_resources(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
