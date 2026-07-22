"""Shared block-extraction helper for comparing the `nano` (official) and
`rtx6000-adapt` Mamba3 implementations.

This file is intentionally branch-agnostic: `Mamba3.forward()` is textually
identical between the two branches from `in_proj` through `B_norm`/`C_norm`
(only what happens *after* B_norm/C_norm diverges -- nano hands things off to
a fused Triton kernel, rtx6000-adapt runs a manual per-timestep loop). So the
exact same extraction code can be imported unmodified from either branch's
active `mamba_ssm` install; only which `mamba_ssm.Mamba3` is importable
changes between runs.

Deliberately stops BEFORE any branch-specific axis reordering:
  - nano transposes DT/ADT/trap to (b, n, l) for its kernel's layout.
  - rtx6000-adapt keeps everything as (b, l, n) for its Python loop.
Comparing the pre-transpose tensors avoids having to special-case that.

`extract_blocks` also captures two checkpoints from *after* B_norm/C_norm
(RoPE + recurrence), pulled directly out of each branch's own real code path
rather than a hand-derived approximation:
  - nano:          calls the real fused Triton kernel with
                    return_final_states=True and reads back the officially
                    accumulated RoPE angle state (cumsum(Angles*DT) mod 2pi
                    over the whole sequence) and the final SSM state.
  - rtx6000-adapt: mirrors Mamba3.forward()'s own per-timestep loop verbatim
                    (same order of ops, including its current `[:, 0, 0, :]`
                    head-0-only indexing after RoPE) to read out h_state
                    after the last timestep, plus the raw per-head angle
                    used at the last step (NOT accumulated -- the loop
                    doesn't carry an angle state across steps at all).

These two "rope_angle_final" values are NOT expected to match -- that's the
point. Reporting them side by side in compare.py is what turns "the RoPE
math looks different" into an actual number.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def extract_blocks(model, u):
    zxBCdtAtrap = model.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtAtrap,
        [
            model.d_inner, model.d_inner,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.nheads, model.nheads, model.nheads,
            model.num_rope_angles,
        ],
        dim=-1,
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=model.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)

    _A = -F.softplus(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)

    angles_exp = angles.unsqueeze(-2).expand(-1, -1, model.nheads, -1).to(torch.float32)

    B_normed = model.B_norm(B)
    C_normed = model.C_norm(C)

    blocks = {
        "z": z,
        "x": x,
        "B_raw": B,
        "C_raw": C,
        "dd_dt": dd_dt,
        "dd_A": dd_A,
        "trap_raw": trap,
        "angles_raw": angles,
        "angles_exp": angles_exp,
        "A": _A,
        "DT": DT,
        "B_normed": B_normed,
        "C_normed": C_normed,
    }
    blocks.update(_extract_recurrence(model, blocks))
    return blocks


@torch.no_grad()
def _extract_recurrence(model, blocks):
    if hasattr(model, "_apply_rope_pytorch"):
        return _extract_recurrence_rtx(model, blocks)
    return _extract_recurrence_nano(model, blocks)


def _extract_recurrence_nano(model, blocks):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    DT_t = rearrange(blocks["DT"], "b l n -> b n l")
    ADT_t = rearrange(blocks["A"] * blocks["DT"], "b l n -> b n l")
    trap_t = rearrange(blocks["trap_raw"], "b l h -> b h l")

    _, final_angle_state, final_ssm_state, _final_k_state, _final_v_state = mamba3_siso_combined(
        Q=blocks["C_normed"].squeeze(2),
        K=blocks["B_normed"].squeeze(2),
        V=blocks["x"],
        ADT=ADT_t,
        DT=DT_t,
        Trap=trap_t,
        Q_bias=model.C_bias.squeeze(1),
        K_bias=model.B_bias.squeeze(1),
        Angles=blocks["angles_exp"],
        D=model.D,
        Z=blocks["z"],
        chunk_size=model.chunk_size,
        Input_States=None,
        return_final_states=True,
        cu_seqlens=None,
    )
    return {
        "rope_angle_final": final_angle_state.detach().float().cpu(),
        "ssm_state_final": final_ssm_state.detach().float().cpu(),
    }


def _extract_recurrence_rtx(model, blocks):
    x, DT, A_, B_normed, angles_exp = (
        blocks["x"], blocks["DT"], blocks["A"], blocks["B_normed"], blocks["angles_exp"],
    )
    batch, seqlen = x.shape[:2]
    h_state = torch.zeros(
        batch, model.nheads, model.headdim, model.d_state,
        device=x.device, dtype=torch.float32,
    )
    last_angle = None

    for t in range(seqlen):
        x_t = x[:, t]
        dt_t = DT[:, t]
        A_t = A_[:, t]
        B_t = B_normed[:, t]
        angles_t = angles_exp[:, t]

        # expend the dimension of B_t and C_t
        # from
        # (batch, mimo_rank, num_bc_head, headdim)
        # to
        # (batch, mimo_rank, nheads, headdim)
        heads_per_group = model.nheads // model.num_bc_heads
        B_t = B_t.repeat_interleave(heads_per_group, dim=2)
            
        target_angles = angles_t.view(batch, 1, model.nheads, -1)
        B_t = model._apply_rope_pytorch(B_t, target_angles)

        dt_t_exp = dt_t.unsqueeze(-1).unsqueeze(-1)
        A_t_exp = A_t.unsqueeze(-1).unsqueeze(-1)
        A_bar = torch.exp(A_t_exp * dt_t_exp)

        B_t_aligned = B_t[:, 0, :, :].unsqueeze(1) if not model.is_mimo else B_t[:, :, :, :]
        update = (x_t.unsqueeze(-1) * dt_t_exp) * B_t_aligned.unsqueeze(2)
        h_state = h_state * A_bar + update
        last_angle = angles_t  # raw, per-head, NOT accumulated across t

    return {
        "rope_angle_final": last_angle.detach().float().cpu(),
        "ssm_state_final": h_state.detach().float().cpu(),
    }
