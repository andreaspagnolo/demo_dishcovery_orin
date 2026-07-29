#!/usr/bin/env python3
"""Build the exact FP16 TensorRT SigLIP2 visual engine used by the benchmark."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trtexec",
        type=Path,
        default=Path("/usr/src/tensorrt/bin/trtexec"),
    )
    parser.add_argument("--workspace-mib", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not args.trtexec.is_file():
        raise FileNotFoundError(args.trtexec)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.trtexec),
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        "--fp16",
        "--shapes=pixel_values:1x3x384x384",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Built: {args.output}")


if __name__ == "__main__":
    main()
