"""
VMonarch attention.

The N = m * b tokens are laid out on an (m, b) grid, token n -> (n // b, n % b).
The attention map is approximated by a Monarch matrix built from
  - R: m dense (b x b) blocks, mixing tokens inside each grid row;
  - L: b dense (m x m) blocks, mixing tokens inside each grid column;
which are fitted by a few steps of alternating minimization. The R step is a
batched attention whose row entropy is needed by the next L step, so it runs on
the fused flash-entropy kernel. The L step is cheap since m is small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import flash_entropy_attention


def _row_block_attention(x, k, w):
    """Attention inside every grid row.

    x, k: (B, H, m, b, D); w: (B, H, m, b, Dw).
    Returns softmax(x k^T) w and sum_j P_ij log P_ij for every row of P.
    """
    out, ent = flash_entropy_attention(*(t.flatten(1, 2) for t in (x, k, w)), sm_scale=1.0)
    return out.view(*w.shape[:-1], -1), -ent.view(x.shape[:-1]).detach()


def vmonarch_attention(q, k, v, num_blocks, block_size, num_iters=2, recompute_tokens=0, min_col_mass=0.01):
    """
    Args:
        q, k: (B, H, N, D); v: (B, H, N, Dv), with N == num_blocks * block_size.
        num_blocks: m, number of grid rows (keep it small).
        block_size: b, number of tokens per grid row.
        num_iters: alternating-minimization steps.
        recompute_tokens: the first `recompute_tokens` queries use exact attention.
        min_col_mass: lower bound of the column mass of L, for stability.
    Returns:
        (B, H, N, Dv)
    """
    B, H, N, D = q.shape
    assert N == num_blocks * block_size, f"N={N} != {num_blocks} * {block_size}"
    scale = D ** -0.5
    qg, kg, vg = (t.unflatten(2, (num_blocks, block_size)) for t in (q, k, v))

    # queries of the R step, i.e. L^T q normalized by the column mass of L
    u = qg * scale
    for it in range(num_iters):
        last = it == num_iters - 1
        out, neg_ent = _row_block_attention(u, kg, torch.cat([kg, vg], -1) if last else kg)
        kr = out[..., :D] * scale  # (B, H, m, b, D)

        # L step, independently for every grid column j:
        # L[j] = softmax_k(q[i, j] . kr[k, j] - neg_ent[k, j]) over grid rows i, k
        logits = torch.einsum("bhijd,bhkjd->bhjik", qg, kr).float()
        attn = (logits - neg_ent.transpose(-1, -2).unsqueeze(-2)).softmax(-1)
        if last:
            y = out[..., D:]
            z = torch.einsum("bhjik,bhkjd->bhijd", attn.to(y.dtype), y)
        else:
            col_mass = attn.sum(-2).transpose(-1, -2).clamp_min(min_col_mass)  # (B, H, m, b)
            u = torch.einsum("bhjik,bhijd->bhkjd", attn.to(q.dtype), qg)
            u = u * (scale / col_mass.unsqueeze(-1)).to(q.dtype)

    z = z.reshape(B, H, N, -1)
    if recompute_tokens > 0:
        exact = F.scaled_dot_product_attention(q[:, :, :recompute_tokens], k, v)
        z = torch.cat([exact, z[:, :, recompute_tokens:]], dim=2)
    return z


class VMonarchAttention(nn.Module):
    def __init__(self, num_blocks, block_size, num_iters=2, min_col_mass=0.01):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_iters = num_iters
        self.min_col_mass = min_col_mass

    def forward(self, q, k, v, recompute_tokens=0):
        return vmonarch_attention(q, k, v, self.num_blocks, self.block_size, self.num_iters,
                                  recompute_tokens, self.min_col_mass)

    def extra_repr(self):
        return f"num_blocks={self.num_blocks}, block_size={self.block_size}, num_iters={self.num_iters}"
