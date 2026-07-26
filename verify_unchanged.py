"""Check that a code change left every number unchanged.

Compares two results_*.pt files produced by run_side.py -- typically the SAME side before
and after a change to mamba_ssm -- and reports any tensor that differs. This is a
regression check, distinct from compare.py (which compares the two *implementations*
against each other and expects differences).

Originally written to prove that adding tap() calls to Mamba3.forward() perturbed nothing;
kept general so it can be reused for the next instrumentation change (e.g. surfacing the
Angles_Cumsum / Q_rot / K_scaled / SSM_States that mamba3_siso_combined currently discards).

IMPORTANT: both files must come from the SAME shared_data.pt. If shared_data.pt was
regenerated in between, the weights differ and every downstream tensor legitimately
changes -- this script cannot distinguish that from a real regression, so keep the
baseline and the new run on one generation of the data.

Keys present in only one file are listed and skipped, so adding or removing a tap is not
itself a failure.

May run where stdout is not visible (e.g. a SLURM compute node), so every message is
written to --log on the shared storage as well as printed.
"""

import argparse
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="results_*.pt from BEFORE the change")
    ap.add_argument("--new", required=True, help="results_*.pt from AFTER the change")
    ap.add_argument("--log", default=os.path.join(HERE, "verify_unchanged.log"))
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max allowed abs difference per tensor; 0.0 (default) demands "
                         "bit-identical results")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    try:
        log(f"baseline: {args.baseline}")
        log(f"new:      {args.new}")
        log(f"tol:      {args.tol:g}" + ("  (bit-identical required)" if args.tol == 0 else ""))
        log()

        old = torch.load(args.baseline)
        new = torch.load(args.new)

        if len(old) != len(new):
            log(f"FAIL: pattern count changed: {len(old)} vs {len(new)}")
            return 1

        ko, kn = set(old[0]), set(new[0])
        shared = sorted(ko & kn)
        log(f"only in baseline ({len(ko - kn)}): {sorted(ko - kn)}")
        log(f"only in new      ({len(kn - ko)}): {sorted(kn - ko)}")
        log(f"compared         ({len(shared)}): {shared}")
        log()

        if not shared:
            log("FAIL: no keys in common -- nothing was compared")
            return 1

        bad = []
        worst = {}
        for p, (o, n) in enumerate(zip(old, new)):
            for k in shared:
                if o[k].shape != n[k].shape:
                    bad.append(f"pattern {p} key {k}: shape "
                               f"{tuple(o[k].shape)} vs {tuple(n[k].shape)}")
                    continue
                d = (o[k].float() - n[k].float()).abs().max().item()
                worst[k] = max(worst.get(k, 0.0), d)
                if d > args.tol:
                    bad.append(f"pattern {p} key {k}: max_abs diff {d:.6e}")

        if bad:
            log(f"FAIL: {len(bad)} mismatch(es)")
            for b in bad[:20]:
                log("  " + b)
            if len(bad) > 20:
                log(f"  ... and {len(bad) - 20} more")
            log()
            log("worst max_abs per key across all patterns:")
            for k in sorted(worst, key=lambda k: -worst[k]):
                log(f"  {k:<16}{worst[k]:14.6e}")
            return 1

        log(f"OK: all {len(shared)} compared keys within tol across {len(old)} patterns")
        if args.tol > 0:
            log()
            log("worst max_abs per key across all patterns:")
            for k in sorted(worst, key=lambda k: -worst[k]):
                log(f"  {k:<16}{worst[k]:14.6e}")
        return 0
    except Exception as e:
        log(f"FAIL: {type(e).__name__}: {e}")
        return 1
    finally:
        with open(args.log, "w") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
