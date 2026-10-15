#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
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
    parser.add_argument("--task2-top-k", type=int, choices=(5, 7), default=None,
                        help="Task 2 candidate pool (default: 7; archived reference: 5).")
    parser.add_argument("--archived-reference", action="store_true",
                        help="Replay and verify the archived 350-image top-20/top-5 results.")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.archived_reference and args.task2_top_k not in (None, 5):
        parser.error("--archived-reference requires Task 2 top-5")
    if args.task2_top_k is None:
        args.task2_top_k = 5 if args.archived_reference else 7
    return args


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
    profile = "archived_reference" if args.archived_reference else "stage4e"
    task_dir = args.output_dir.resolve() / profile / task
    output_json = task_dir / "result.json"
    predictions_csv = task_dir / "predictions.csv"
    command = (
        task1_command(paths, output_json=output_json, predictions_csv=predictions_csv)
        if task == "task1"
        else task2_command(paths, output_json=output_json, predictions_csv=predictions_csv,
                           top_k=args.task2_top_k)
    )
    print("+", shlex.join(command), flush=True)
    if args.print_only:
        return
    validate_runtime(paths, task)
    if output_json.exists() or predictions_csv.exists():
        raise FileExistsError(f"Existing results in {task_dir}; choose a new --output-dir.")
    task_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=paths.root, check=True)
    if args.archived_reference:
        verify_metrics(task, output_json)


def main() -> None:
    args = parse_args()
    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    tasks = ("task1", "task2") if args.task == "all" else (args.task,)
    for task in tasks:
        run_one(task, args, paths)


if __name__ == "__main__":
    main()
