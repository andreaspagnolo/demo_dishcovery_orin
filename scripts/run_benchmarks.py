#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from repro_common import (
    ReproPaths,
    default_assets,
    default_edge_llm,
    task1_command,
    task2_command,
    validate_runtime,
)


EXPECTED = {
    "task1": {
        "f1": 0.7578772802653398,
        "precision": 0.7879310344827586,
        "recall": 0.7300319488817891,
    },
    "task2": {
        "caption_accuracy": 0.6714285714285714,
        "class_accuracy": 0.8742857142857143,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed quantized 350-image benchmarks.")
    parser.add_argument("task", choices=("task1", "task2", "all"))
    parser.add_argument("--assets", type=Path, default=default_assets())
    parser.add_argument("--edge-llm", type=Path, default=default_edge_llm())
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs"))
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def verify_metrics(task: str, output_json: Path) -> None:
    report = json.loads(output_json.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    failures = []
    for name, expected in EXPECTED[task].items():
        actual = float(metrics[name])
        if abs(actual - expected) > 1e-9:
            failures.append(f"{name}: expected {expected:.12f}, got {actual:.12f}")
    if failures:
        raise RuntimeError("Metric verification failed:\n" + "\n".join(failures))
    print(f"{task}: metrics match the archived reference.")


def run_one(task: str, args: argparse.Namespace, paths: ReproPaths) -> None:
    validate_runtime(paths, task)
    task_dir = args.output_dir.resolve() / task
    output_json = task_dir / "result.json"
    predictions_csv = task_dir / "predictions.csv"
    task_dir.mkdir(parents=True, exist_ok=True)
    command = (
        task1_command(paths, output_json=output_json, predictions_csv=predictions_csv)
        if task == "task1"
        else task2_command(paths, output_json=output_json, predictions_csv=predictions_csv)
    )
    print("+", " ".join(command), flush=True)
    if args.print_only:
        return
    subprocess.run(command, cwd=paths.root, check=True)
    verify_metrics(task, output_json)


def main() -> None:
    args = parse_args()
    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    tasks = ("task1", "task2") if args.task == "all" else (args.task,)
    for task in tasks:
        run_one(task, args, paths)


if __name__ == "__main__":
    main()
