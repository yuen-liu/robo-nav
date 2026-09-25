"""
Independent depth/scale cross-check using MoGe-3 (github.com/microsoft/MoGe).

localize.py's Stage 2 (windowed) and Stage 3 (streaming) reconstructions of the
same map anchor frame disagree on median depth by a consistent ~1.35-1.46x
factor (see the `anchor_map_depth_median` / `anchor_sub_depth_median` values
localize.py now prints per hypothesis). MoGe-3 is a separate model with its own
metric-scale training, so running it on the same frame(s) tells us whether
Stage 2 or Stage 3 (or neither) is the one that's off.

Usage:
    python compare_moge_depth.py tm4_data/frames_test/frame_0095.png \
        --checkpoint /path/to/moge-3-vitl.pt
"""

import argparse

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Print MoGe-3 median depth for one or more images")
    parser.add_argument("image_paths", nargs="+", help="Path(s) to image(s) to run MoGe-3 on")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MoGe-3 checkpoint (.pt)")
    args = parser.parse_args()

    from moge.model.v3 import MoGeModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained(args.checkpoint).to(device).eval()

    for path in args.image_paths:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1).to(device)

        with torch.no_grad():
            output = model.infer(image_tensor)

        depth = output["depth"].detach().cpu().numpy()
        valid = np.isfinite(depth)
        print(f"{path}: median_depth={np.median(depth[valid]):.4f}  "
              f"mean_depth={depth[valid].mean():.4f}  "
              f"min={depth[valid].min():.4f}  max={depth[valid].max():.4f}")


if __name__ == "__main__":
    main()
