#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Standard-library readers copied from evaluate_food100k_cleaned_visual_task1.
# Sampling and shortage allocation preserve the original construction algorithm.
LABEL_PREFIX = "A plate of food containing "


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())

def strip_caption(value: str) -> str:
    text = value.strip()
    if text.lower().startswith(LABEL_PREFIX.lower()):
        text = text[len(LABEL_PREFIX) :]
    return normalize_text(text)

def image_index(name: str) -> int:
    return int(Path(name).stem.split("_")[1])

def image_sort_key(name: str) -> tuple[int, int | str, str]:
    try:
        return (0, image_index(name), name)
    except Exception:
        return (1, name, name)

def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def load_labels(path: Path) -> tuple[list[str], dict[str, int]]:
    labels = [strip_caption(line) for line in read_lines(path)]
    if len(labels) != len(set(labels)):
        duplicates = [label for label, count in Counter(labels).items() if count > 1]
        raise ValueError(f"Duplicate cleaned labels: {duplicates}")
    return labels, {label: idx for idx, label in enumerate(labels)}

def load_ground_truth(path: Path, label_to_id: dict[str, int]) -> list[list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list, got {type(raw).__name__}")
    rows: list[list[int]] = []
    unmatched: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"Row {idx} in {path} is not an object")
        ingredients = row.get("ingredients")
        if not isinstance(ingredients, list):
            raise TypeError(f"Row {idx} in {path} does not contain an ingredient list")
        ids: list[int] = []
        seen: set[int] = set()
        for ingredient in ingredients:
            label = normalize_text(str(ingredient))
            label_id = label_to_id.get(label)
            if label_id is None:
                unmatched.append({"row": idx, "ingredient": ingredient})
                continue
            if label_id not in seen:
                ids.append(label_id)
                seen.add(label_id)
        rows.append(ids)
    if unmatched:
        raise ValueError(f"Found {len(unmatched)} unmatched ingredients; examples: {unmatched[:20]}")
    return rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_COUNTS = {
    "1": 4046,
    "2": 1589,
    "3": 607,
    "4": 379,
    "5": 46,
    "6_plus": 89,
}
BUCKET_ORDER = ["1", "2", "3", "4", "5", "6_plus"]
BUCKET_REPRESENTATIVE_COUNT = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6_plus": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic Food100K image-list subset whose ground-truth "
            "ingredient-count distribution matches the current Dishcovery Test1 "
            "prediction-count distribution."
        )
    )
    parser.add_argument("--cleaned-json", type=Path, default=ROOT / "benchmark_inputs/task1/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json")
    parser.add_argument("--captions", type=Path, default=ROOT / "dataset/captions_cleaned.txt")
    parser.add_argument("--source-images-list", type=Path, default=ROOT / "dataset/images.txt")
    parser.add_argument("--source-ground-truth-map", type=Path, default=ROOT / "labels/image_ground_truth_rows.csv")
    parser.add_argument("--output-images-list", type=Path, default=ROOT / "dataset/images_dishcovery_aligned_qwen3_counts.txt")
    parser.add_argument("--output-ground-truth-map", type=Path, default=ROOT / "dataset/image_ground_truth_rows_dishcovery_aligned_qwen3_counts.csv")
    parser.add_argument("--summary-json", type=Path, default=ROOT / "run_outputs/subset/food100k_dishcovery_aligned_subset_summary.json")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "run_outputs/subset/food100k_dishcovery_aligned_subset_summary.md")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--verify-only", action="store_true", help="Compare the reconstructed list/map with the published files without writing outputs.")
    return parser.parse_args()


def count_bin(label_count: int) -> str:
    return "6_plus" if label_count >= 6 else str(label_count)


def distance_bucket(source: str, target: str) -> int:
    return abs(BUCKET_REPRESENTATIVE_COUNT[source] - BUCKET_REPRESENTATIVE_COUNT[target])


def read_ground_truth_map(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = str(row.get("image") or "").strip()
            gt_row = str(row.get("ground_truth_row") or "").strip()
            if image and gt_row:
                mapping[image] = int(gt_row)
    return mapping


def write_ground_truth_map(path: Path, image_names: list[str], gt_map: dict[str, int]) -> None:
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "ground_truth_row", "food100k_idx"])
        writer.writeheader()
        for name in image_names:
            writer.writerow(
                {
                    "image": name,
                    "ground_truth_row": gt_map[name],
                    "food100k_idx": image_index(name),
                }
            )
        write_artifact(path, handle.getvalue().encode("utf-8"))


def write_artifact(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace {path}; choose a new output path.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    args = parse_args()
    labels, label_to_id = load_labels(args.captions)
    gt_rows = load_ground_truth(args.cleaned_json, label_to_id)
    image_names = read_lines(args.source_images_list)
    gt_map = read_ground_truth_map(args.source_ground_truth_map)
    missing_map = [name for name in image_names if name not in gt_map]
    if missing_map:
        raise ValueError(f"{args.source_ground_truth_map} is missing {len(missing_map)} source images; examples: {missing_map[:10]}")

    buckets: dict[str, list[str]] = defaultdict(list)
    exact_count_distribution: Counter[int] = Counter()
    for name in image_names:
        label_count = len(gt_rows[gt_map[name]])
        exact_count_distribution[label_count] += 1
        bucket = count_bin(label_count)
        if bucket in DEFAULT_TARGET_COUNTS:
            buckets[bucket].append(name)

    rng = random.Random(args.seed)
    selected_by_bucket: dict[str, list[str]] = {}
    remaining_by_bucket: dict[str, list[str]] = {}
    selected_count_by_bucket: dict[str, int] = {}
    shortages: dict[str, int] = {}
    for bucket, target_count in DEFAULT_TARGET_COUNTS.items():
        available = list(buckets.get(bucket, []))
        rng.shuffle(available)
        selected_count = min(len(available), target_count)
        selected_count_by_bucket[bucket] = selected_count
        selected_by_bucket[bucket] = available[:selected_count]
        remaining_by_bucket[bucket] = available[selected_count:]
        if selected_count < target_count:
            shortages[bucket] = target_count - selected_count

    fill_allocation_counts: Counter[tuple[str, str, int]] = Counter()
    for missing_bucket, deficit in shortages.items():
        while deficit > 0:
            donors_by_distance: dict[int, list[str]] = defaultdict(list)
            for donor in BUCKET_ORDER:
                if donor == missing_bucket or not remaining_by_bucket.get(donor):
                    continue
                donors_by_distance[distance_bucket(donor, missing_bucket)].append(donor)
            if not donors_by_distance:
                raise ValueError(f"Could not fill remaining deficit of {deficit} rows for bucket {missing_bucket}")
            nearest_distance = min(donors_by_distance)
            donors = sorted(donors_by_distance[nearest_distance], key=lambda bucket: BUCKET_REPRESENTATIVE_COUNT[bucket])
            while deficit > 0:
                added_this_round = 0
                for donor in donors:
                    if deficit <= 0:
                        break
                    if not remaining_by_bucket[donor]:
                        continue
                    selected_by_bucket[donor].append(remaining_by_bucket[donor].pop(0))
                    fill_allocation_counts[(missing_bucket, donor, nearest_distance)] += 1
                    deficit -= 1
                    added_this_round += 1
                if added_this_round == 0:
                    break

    fill_allocations = [
        {
            "missing_bucket": missing_bucket,
            "donor_bucket": donor,
            "rows": rows,
            "bucket_distance": distance,
        }
        for (missing_bucket, donor, distance), rows in sorted(fill_allocation_counts.items())
    ]

    selected: list[str] = []
    for bucket in BUCKET_ORDER:
        selected.extend(selected_by_bucket[bucket])
    selected = sorted(selected, key=image_sort_key)

    selected_distribution: Counter[str] = Counter()
    selected_exact_distribution: Counter[int] = Counter()
    for name in selected:
        label_count = len(gt_rows[gt_map[name]])
        selected_distribution[count_bin(label_count)] += 1
        selected_exact_distribution[label_count] += 1

    if args.verify_only:
        expected_names = read_lines(args.output_images_list)
        expected_map = read_ground_truth_map(args.output_ground_truth_map)
        if selected != expected_names or expected_map != {name: gt_map[name] for name in selected}:
            raise ValueError("Reconstructed subset differs from the published image list or ground-truth map")
        print(f"Verified {len(selected)} images, {len(labels)} labels, seed {args.seed}")
        return

    write_artifact(args.output_images_list, ("\n".join(selected) + "\n").encode("utf-8"))
    write_ground_truth_map(args.output_ground_truth_map, selected, gt_map)

    summary: dict[str, Any] = {
        "schema_version": "food100k_dishcovery_aligned_subset_v1",
        "seed": args.seed,
        "source_images_list": str(args.source_images_list),
        "source_ground_truth_map": str(args.source_ground_truth_map),
        "output_images_list": str(args.output_images_list),
        "output_ground_truth_map": str(args.output_ground_truth_map),
        "rows": len(selected),
        "label_count": len(labels),
        "target_distribution": DEFAULT_TARGET_COUNTS,
        "selected_distribution": dict(sorted(selected_distribution.items())),
        "selected_exact_distribution": {str(k): v for k, v in sorted(selected_exact_distribution.items())},
        "source_exact_distribution": {str(k): v for k, v in sorted(exact_count_distribution.items())},
        "available_bucket_distribution": {bucket: len(buckets.get(bucket, [])) for bucket in DEFAULT_TARGET_COUNTS},
        "shortages": shortages,
        "fill_allocations": fill_allocations,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.summary_md.write_text(
        "\n".join(
            [
                "# Food100K Dishcovery-Aligned Subset",
                "",
                f"- Rows: `{len(selected)}`",
                f"- Labels: `{len(labels)}`",
                f"- Seed: `{args.seed}`",
                f"- Image list: `{args.output_images_list}`",
                f"- Ground-truth map: `{args.output_ground_truth_map}`",
                "",
                "## Selected Distribution",
                "",
                *[f"- {key}: `{value}`" for key, value in summary["selected_distribution"].items()],
                "",
                "## Shortage Fill",
                "",
                *[
                    f"- {row['missing_bucket']} filled from {row['donor_bucket']}: `{row['rows']}` rows"
                    for row in fill_allocations
                ],
                "",
                "## Selected Exact Distribution",
                "",
                *[f"- {key}: `{value}`" for key, value in summary["selected_exact_distribution"].items()],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
