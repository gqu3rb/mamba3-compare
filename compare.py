"""Run after both results_nano.pt and results_rtx.pt exist. Only needs
torch (no mamba_ssm import), so it can run in either env.

python compare.py [--tol 1e-2]
"""

import argparse
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# nano accumulates RoPE angle via cumsum(Angles*DT) mod 2pi over the whole
# sequence; rtx6000-adapt's loop currently doesn't accumulate at all, so
# this row is expected to diverge by construction -- it's there to make
# that gap visible as a number, not to pass/fail against --tol.
DIAGNOSTIC_KEYS = {"rope_angle_final"}


def stats(a, b):
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    return diff.max().item(), diff.mean().item(), (diff / denom).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano", default=os.path.join(HERE, "results_nano.pt"))
    ap.add_argument("--rtx", default=os.path.join(HERE, "results_rtx.pt"))
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    nano = torch.load(args.nano)
    rtx = torch.load(args.rtx)
    assert len(nano) == len(rtx), f"pattern count mismatch: {len(nano)} vs {len(rtx)}"

    keys = list(nano[0].keys())
    header = f"{'block':<14}{'pattern':>8}{'max_abs':>14}{'mean_abs':>14}{'max_rel':>14}"
    print(header)
    print("-" * len(header))

    worst = {}
    for p, (bn, br) in enumerate(zip(nano, rtx)):
        for k in keys:
            a, b = bn[k], br[k]
            if a.shape != b.shape:
                print(f"{k:<14}{p:>8}  SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            mx, mean, rel = stats(a, b)
            if k in DIAGNOSTIC_KEYS:
                flag = "  (diagnostic, not accumulated on rtx side)"
            else:
                flag = "  <-- DIVERGES" if mx > args.tol else ""
            print(f"{k:<14}{p:>8}{mx:14.4e}{mean:14.4e}{rel:14.4e}{flag}")
            worst[k] = max(worst.get(k, 0.0), mx)

    print("\n=== worst max_abs per block across all patterns ===")
    for k, v in worst.items():
        if k in DIAGNOSTIC_KEYS:
            flag = "  (diagnostic)"
        else:
            flag = "  <-- DIVERGES" if v > args.tol else "  ok"
        print(f"{k:<14}{v:14.4e}{flag}")


if __name__ == "__main__":
    main()
