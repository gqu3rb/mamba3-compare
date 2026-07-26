"""Run after both results_nano.pt and results_rtx.pt exist. Only needs
torch (no mamba_ssm import), so it can run in either env.

python compare.py [--tol 1e-2]
"""

import argparse
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

def stats(a, b):
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    # (diff / denom).max().item() calculates the maximum relative error
    # relative error means how large the error of a value is comparing to the value itself
    return diff.max().item(), diff.mean().item(), (diff / denom).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano", default=os.path.join(HERE, "results_nano.pt"))
    ap.add_argument("--rtx", default=os.path.join(HERE, "results_rtx.pt"))
    ap.add_argument("--tol", type=float, default=1e-2) # tolerance for absolute error
    args = ap.parse_args()

    nano = torch.load(args.nano)
    rtx = torch.load(args.rtx)
    # print out:
    """
    type(nano) = <class 'list'>
    type(nano[0]) = <class 'dict'>
    """
    #print(f"type(nano) = {type(nano)}")
    #print(f"type(nano[0]) = {type(nano[0])}")

    assert len(nano) == len(rtx), f"pattern count mismatch: {len(nano)} vs {len(rtx)}"

    # Ctrl+F "results" in run_side.py to know the content of `keys`
    keys = list(nano[0].keys())
    # print the title of each column to the terminal
    # 'block':<14 prints out left-aligned "block" in a 14-character-wide column
    # 'pattern':>8 prints out right-aligned "pattern" in a 8-character-wide column
    header = f"{'block':<14}{'pattern':>8}{'max_abs':>14}{'mean_abs':>14}{'max_rel':>14}"
    print(header)
    print("-" * len(header))

    worst = {}
    # `nano` and `rtx` are two lists of dicts
    # zip() pairs up two lists element-by-element, that is, zip(nano, rtx) outputs WHEN NEEDED:
    # (nano[0], rtx[0]), (nano[1], rtx[1]), ...
    # enumerate() encapsulates a sequence, that is, enumerate(zip(nano, rtx) outputs the following WHEN NEEDED:
    # (0, (nano[0], rtx[0])), (1, (nano[1], rtx[1])), ...
    # Google Search: "zip() enumerate() lazy iterators" to understand the meaning of "WHEN NEEDED" above
    for p, (bn, br) in enumerate(zip(nano, rtx)):
        for k in keys:
            a, b = bn[k], br[k]
            if a.shape != b.shape:
                print(f"{k:<14}{p:>8}  SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            mx, mean, rel = stats(a, b)
            flag = "  <-- DIVERGES" if mx > args.tol else ""
            print(f"{k:<14}{p:>8}{mx:14.4e}{mean:14.4e}{rel:14.4e}{flag}")
            # worst.get(k, 0.0) returns the value corresponding to key `k`.
            # It returns 0.0 if the key doesn't exist.
            worst[k] = max(worst.get(k, 0.0), mx)

    print("\n=== worst max_abs per block across all patterns ===")
    for k, v in worst.items():
        flag = "  <-- DIVERGES" if v > args.tol else "  ok"
        print(f"{k:<14}{v:14.4e}{flag}")


if __name__ == "__main__":
    main()
