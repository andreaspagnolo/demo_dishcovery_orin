#!/usr/bin/env python3
"""Export the SigLIP2 visual tower used by the Orin benchmarks to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


MODEL_ID = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"


class VisualEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("Install open_clip_torch before exporting SigLIP2.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained="")
    wrapper = VisualEncoder(model.eval()).cpu()
    example = torch.zeros((1, 3, 384, 384), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (example,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
    )
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
