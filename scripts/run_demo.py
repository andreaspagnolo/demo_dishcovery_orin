#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from repro_common import (
    ROOT,
    ReproPaths,
    default_assets,
    default_edge_llm,
    task1_command,
    task2_command,
    validate_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quantized Dishcovery pipeline on the default good-predictions subset or one image."
    )
    parser.add_argument("task", choices=("task1", "task2"))
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Optional image path. Omit it to run the task's curated good-predictions subset.",
    )
    parser.add_argument("--assets", type=Path, default=default_assets())
    parser.add_argument("--edge-llm", type=Path, default=default_edge_llm())
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    return parser.parse_args()


def good_prediction_paths(task: str) -> list[str]:
    source = ROOT / "benchmark_inputs/demo_showcase/good_predictions.txt"
    prefix = "MM-Food-100K-images-filtered/" if task == "task1" else "food500_subset/images/"
    paths = [line.strip() for line in source.read_text().splitlines() if line.strip().startswith(prefix)]
    if not paths:
        raise ValueError(f"No {task} demo images found in {source}")
    return paths


def main() -> None:
    args = parse_args()
    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    validate_runtime(paths, args.task)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image = args.image.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        output_json = output_dir / f"{args.task}_{image.stem}.json"
        predictions_csv = output_dir / f"{args.task}_{image.stem}.csv"
        command = (
            task1_command(paths, output_json=output_json, image=image)
            if args.task == "task1"
            else task2_command(paths, output_json=output_json, predictions_csv=predictions_csv, image=image)
        )
    else:
        image_paths = good_prediction_paths(args.task)
        showcase_root = paths.assets / "images/showcase"
        missing = [showcase_root / path for path in image_paths if not (showcase_root / path).is_file()]
        if missing:
            raise FileNotFoundError("Missing demo images:\n" + "\n".join(map(str, missing)))
        selected_list = output_dir / f"{args.task}_good_predictions_images.txt"
        selected_list.write_text("\n".join(image_paths) + "\n")
        output_json = output_dir / f"{args.task}_good_predictions.json"
        predictions_csv = output_dir / f"{args.task}_good_predictions.csv"
        command = (
            task1_command(
                paths,
                output_json=output_json,
                predictions_csv=predictions_csv,
                image_dir=showcase_root,
                images_list=selected_list,
                eval_samples=len(image_paths),
            )
            if args.task == "task1"
            else task2_command(
                paths,
                output_json=output_json,
                predictions_csv=predictions_csv,
                image_dir=showcase_root,
                images_list=selected_list,
            )
        )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=paths.root, check=True)
    print(f"Trace: {output_json}")


if __name__ == "__main__":
    main()
