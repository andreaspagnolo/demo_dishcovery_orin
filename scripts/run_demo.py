#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one image through a quantized Dishcovery pipeline.")
    parser.add_argument("task", choices=("task1", "task2"))
    parser.add_argument("image", type=Path)
    parser.add_argument("--assets", type=Path, default=default_assets())
    parser.add_argument("--edge-llm", type=Path, default=default_edge_llm())
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    validate_runtime(paths, args.task)
    output_json = args.output_dir.resolve() / f"{args.task}_{image.stem}.json"
    predictions_csv = args.output_dir.resolve() / f"{args.task}_{image.stem}.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = (
        task1_command(paths, output_json=output_json, image=image)
        if args.task == "task1"
        else task2_command(
            paths,
            output_json=output_json,
            predictions_csv=predictions_csv,
            image=image,
        )
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=paths.root, check=True)
    print(f"Trace: {output_json}")


if __name__ == "__main__":
    main()
