"""Run under EITHER branch's active mamba_ssm install:

  python run_side.py --tag nano
  python run_side.py --tag rtx

Loads the shared weights + 10 input patterns from shared_data.pt, runs both
the block-extraction probe and the real forward() for every pattern, and
saves everything (moved to CPU/float32) to results_<tag>.pt for compare.py.
"""

import argparse
import os
import torch
from mamba_ssm import Mamba3

from probe import extract_blocks

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["nano", "rtx"])
    ap.add_argument("--data", default=os.path.join(HERE, "shared_data.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Ctrl+F "config" and "input" in gen_shared_data.py to know the content of them:
    blob = torch.load(args.data, map_location="cpu")
    config = blob["config"]
    x_all = blob["input"].to("cuda")

    model = Mamba3(**config).to("cuda")
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=True)

    # Switch the model to the inference mode
    model.eval()

    results = []
    with torch.no_grad():
        for p in range(x_all.shape[0]):
            u = x_all[p]
            blocks = extract_blocks(model, u)
            out = model(u)
            entry = {k: v.detach().float().cpu() for k, v in blocks.items()}
            entry["out"] = out.detach().float().cpu()
            results.append(entry)
            print(f"[{args.tag}] pattern {p} done, out.shape={tuple(out.shape)}")

    out_path = args.out or os.path.join(HERE, f"results_{args.tag}.pt")
    torch.save(results, out_path)
    print(f"[{args.tag}] saved {out_path} ({len(results)} patterns)")


if __name__ == "__main__":
    main()
