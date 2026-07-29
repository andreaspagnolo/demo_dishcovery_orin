#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from repro_common import ReproPaths, ROOT, default_assets, default_edge_llm


EXPECTED_HASHES = {
    "benchmark_inputs/task1/images.txt": "ebbdde7f4b54b16c3d1d2cacef7eb06642617a4f46fffa26925a5e2ec2342f47",
    "benchmark_inputs/task1/image_ground_truth_rows.csv": "1b1648c617da2aa820387d745d02547347d67a7ccdfedbd3b38f8c2b7e2a4910",
    "benchmark_inputs/task1/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json": "1d7c5311db3ec3b6a954e85dc281a069f2e5f094bde0560d6b7684c54cbff79e",
    "benchmark_inputs/task1/images.txt": "ebbdde7f4b54b16c3d1d2cacef7eb06642617a4f46fffa26925a5e2ec2342f47",
    "benchmark_inputs/task2/images.txt": "766216ee10334898a8fd6a2af1635eb8de37efc53b09a8b6784d263eee98ca3d",
    "benchmark_inputs/task2/manifest.csv": "c26c5e7a5e771285eed02773866d0fa5815fd09bb6baea605e8ca50970d91464",
    "benchmark_inputs/task2/evaluation_data.json": "401b615984ec06c2da431d12c44e1ad5f9efd220f92f7fe4e2b19ca3f6e02e63",
    "benchmark_inputs/caches/task1_350_siglip2_pytorch_fp16_text_cache.npz": "0a97391f90b5783287f77b7620595ba2b67e70824f0f49632892569687a24128",
    "benchmark_inputs/caches/food500_full_bank4940_siglip2_fp16_caption_cache.npz": "f7faf176606e228546caeff3aca1724b4c43faca2fedc7d989d9d834f9681fe5",
}

ENGINE_HASHES = {
    "SigLIP2 FP16": (
        "siglip_engine",
        "ccef0a77ae8574f021a42aeac370775d7f530315b84b59096775bfc1e37c0a07",
    ),
    "Task 1 LLM INT4": (
        "task1_llm",
        "74ef16dc90647f17dd235c27b93515ace5c9384314738b494f019ba00cc4177b",
    ),
    "Task 1 visual FP16": (
        "task1_visual",
        "6b978e40f076fb6495ae6514d5f59275d90454e19ddf88968f11ea5d98dff065",
    ),
    "Task 2 LLM INT4": (
        "task2_llm",
        "b5e1d21324a60bf749e6d2f89ba9066aa997804928ecad0c23012a3439dc9efe",
    ),
    "Task 2 visual FP16": (
        "task2_visual",
        "8e0dc44e9f5ea3a7d86e5755398aa8a69b272a988585c680419dbe69922d20b1",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_hash(label: str, path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"MISSING  {label}: {path}")
        return False
    actual = digest(path)
    if actual != expected:
        print(f"MISMATCH {label}: expected {expected}, got {actual}")
        return False
    print(f"OK       {label}")
    return True


def check_images(label: str, root: Path, manifest: Path) -> bool:
    names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [name for name in names if not (root / name).is_file()]
    if len(names) != 350 or missing:
        print(f"MISMATCH {label}: rows={len(names)}, missing={len(missing)}")
        for name in missing[:5]:
            print(f"         missing {root / name}")
        return False
    print(f"OK       {label}: 350/350 images")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=default_assets())
    parser.add_argument("--edge-llm", type=Path, default=default_edge_llm())
    parser.add_argument("--skip-engine-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ReproPaths.from_args(args.assets, args.edge_llm)
    ok = True

    for relative, expected in EXPECTED_HASHES.items():
        ok = check_hash(relative, ROOT / relative, expected) and ok

    ok = check_images(
        "Task 1 subset",
        paths.task1_images,
        ROOT / "benchmark_inputs/task1/images.txt",
    ) and ok
    ok = check_images(
        "Task 2 subset",
        paths.task2_images,
        ROOT / "benchmark_inputs/task2/images.txt",
    ) and ok

    required_edge = {
        "EdgeLLM plugin": paths.edge_llm / "build/libNvInfer_edgellm_plugin.so",
        "persistent server": paths.edge_llm / "build/examples/llm/llm_persistent_server",
    }
    for label, path in required_edge.items():
        if path.is_file():
            print(f"OK       {label}")
        else:
            print(f"MISSING  {label}: {path}")
            ok = False

    if not args.skip_engine_hashes:
        for label, (attribute, expected) in ENGINE_HASHES.items():
            base = getattr(paths, attribute)
            path = (
                base
                if attribute == "siglip_engine"
                else base / ("visual/visual.engine" if "visual" in attribute else "llm.engine")
            )
            ok = check_hash(label, path, expected) and ok

    if not ok:
        raise SystemExit(1)
    print("Setup is complete and matches the archived inputs.")


if __name__ == "__main__":
    main()
