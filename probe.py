"""Shared block-extraction helper for comparing the `nano` (official) and
`rtx6000-adapt` Mamba3 implementations.

This file used to re-derive `Mamba3.forward()`'s math by hand so it could read out the
intermediate tensors. That duplication drifted from the shipped code: the rtx recurrence
replica applied `repeat_interleave(heads_per_group)` and indexed `B_t[:, 0, :, :]`, while
the real `rtx6000-adapt` loop does neither and indexes `B_t[:, 0, 0, :]`. The harness was
therefore reporting numbers for an algorithm that no branch actually runs.

Now each branch's real `forward()` records its own intermediates as it executes, via
`mamba_ssm/utils/tap.py`. `extract_blocks` just switches capture on and runs the model, so
the tensors compared are by construction the ones the shipped code computed. Keeping the
two branches comparable is now a matter of using the same 15 tap *names* on both sides
rather than keeping two hand-written reimplementations in sync.

Tap names recorded (identical on both branches):
    dd_dt, dd_A, trap_raw, angles_raw   -- straight out of in_proj's split
    z, x, B_raw, C_raw                  -- after the rearranges, before B_norm/C_norm
    A, DT                               -- DT tapped BEFORE nano transposes it for its kernel
    angles_exp                          -- after the per-head expand + float32 cast
    B_normed, C_normed                  -- first point where the two RMSNorm impls differ
    y_pre_outproj, out                  -- after the branches diverge; localizes the gap
"""

import torch

from mamba_ssm.utils import tap


@torch.no_grad()
def extract_blocks(model, u):
    """Run the real forward() with taps on and return everything it recorded.

    The returned tensors are already detached, float32 and on CPU (tap does that), so the
    caller can `torch.save` the dict directly.
    """
    with tap.capture() as blocks:
        model(u)
    return blocks
