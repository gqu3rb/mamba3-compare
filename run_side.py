"""Run under EITHER branch's active mamba_ssm install:

  python run_side.py --tag nano
  python run_side.py --tag rtx

Loads the shared weights + 10 input patterns from shared_data.pt, runs the real
forward() once per pattern with tap capture enabled, and saves every recorded
intermediate (already CPU/float32) to results_<tag>.pt for compare.py.
"""

import argparse
import os
import torch
from mamba_ssm import Mamba3
from mamba_ssm.utils import tap

from probe import extract_blocks

HERE = os.path.dirname(os.path.abspath(__file__))

# Both implementations live in the same git repo on two branches, and each branch is
# pinned to one machine. Without this mapping a stray checkout (or a PYTHONPATH shim that
# silently failed) would compare an implementation against itself and report near-perfect
# agreement -- a false pass. tap.IMPL travels inside the library file that actually got
# imported, so it reports what Python really loaded.
TAG_TO_IMPL = {"nano": "nano", "rtx": "rtx6000-adapt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["nano", "rtx"])
    ap.add_argument("--data", default=os.path.join(HERE, "shared_data.pt"))
    ap.add_argument("--out", default=None)
    # The rtx6000-adapt side is a Python loop over every timestep, so a full 10-pattern run
    # is slow. --patterns 1 gives a quick smoke test that the taps fire and the key names
    # match before committing to the whole thing.
    ap.add_argument("--patterns", type=int, default=None,
                    help="only run the first N input patterns (default: all)")
    args = ap.parse_args()

    # Fail before doing any work if the wrong implementation got imported.
    tap.expect_impl(TAG_TO_IMPL[args.tag])

    # Ctrl+F "config" and "input" in gen_shared_data.py to know the content of them:
    blob = torch.load(args.data, map_location="cpu")
    config = blob["config"]
    x_all = blob["input"].to("cuda")

    model = Mamba3(**config).to("cuda")
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=True)

    # Switch the model to the inference mode
    model.eval()

    n_patterns = x_all.shape[0] if args.patterns is None else min(args.patterns, x_all.shape[0])

    results = []
    with torch.no_grad():
        for p in range(n_patterns):
            # One forward pass now does both jobs: the taps inside forward() record every
            # intermediate (including "out"), already detached/float32/CPU. This used to be
            # two passes -- extract_blocks() re-derived the math, then model(u) ran it again.
            entry = extract_blocks(model, x_all[p])
            results.append(entry)
            print(f"[{args.tag}] pattern {p} done, out.shape={tuple(entry['out'].shape)}")

    out_path = args.out or os.path.join(HERE, f"results_{args.tag}.pt")
    torch.save(results, out_path)
    print(f"[{args.tag}] saved {out_path} ({len(results)} patterns)")


if __name__ == "__main__":
    main()
