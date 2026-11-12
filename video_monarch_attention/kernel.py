# Flash attention that additionally returns the row-wise entropy of the attention map.
#
# Adapted from the Triton fused-attention tutorial (MIT License):
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
# Copyright 2018-2020 Philippe Tillet, Copyright 2020-2022 OpenAI.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

SEQ_ALIGN = 128  # sequences are zero-padded to a multiple of this
TORCH_TO_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def _capability():
    return torch.cuda.get_device_capability()[0]


def is_hopper():
    return _capability() == 9


def is_blackwell():
    return _capability() == 10


def supports_host_descriptor():
    return _capability() >= 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, h_i, q,  #
                    desc_k, desc_v, offset_y, dtype: tl.constexpr, qk_scale,  #
                    HEAD_DIM_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_n, N_BOUND, N_CTX,  #
                    warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    offsetkv_y = offset_y
    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetkv_y, 0]).T
        qk = tl.dot(q, k)
        if N_BOUND < N_CTX:
            qk = tl.where((start_n + offs_n[None, :]) < N_BOUND, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        # h_i = sum_j p_ij * (s_ij - m_i), rescaled online like l_i
        h_i = alpha * (h_i + (m_i - m_ij) * l_i) + tl.sum(p * qk, 1)
        l_ij = tl.sum(p, 1)
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM_V == 128:
            # split the accumulator to avoid register spilling
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        v = desc_v.load([offsetkv_y, 0])
        acc = tl.dot(p.to(dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return acc, l_i, m_i, h_i


def _host_descriptor_pre_hook(nargs):
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_M, BLOCK_N = nargs["BLOCK_M"], nargs["BLOCK_N"]
    nargs["desc_q"].block_shape = [BLOCK_M, nargs["HEAD_DIM_QK"]]
    nargs["desc_k"].block_shape = [BLOCK_N, nargs["HEAD_DIM_QK"]]
    nargs["desc_v"].block_shape = [BLOCK_N, nargs["HEAD_DIM_V"]]
    nargs["desc_o"].block_shape = [BLOCK_M, nargs["HEAD_DIM_V"]]


configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook)
    for BN in [32, 64]
    for s in [2, 3, 4]
    for w in [4, 8]
]


def prune_invalid_configs(configs, named_args, **kwargs):
    min_dim = min(kwargs["HEAD_DIM_QK"], kwargs["HEAD_DIM_V"])
    return [
        conf for conf in configs
        if conf.kwargs["BLOCK_M"] <= kwargs["N_CTX"] and conf.kwargs["BLOCK_N"] <= min_dim
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM_QK", "HEAD_DIM_V", "DTYPE", "warp_specialize"],
                 prune_configs_by={"early_config_prune": prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M, E,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_BOUND, N_CTX,  #
              HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,  #
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
              DTYPE: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996
    tl.static_assert(BLOCK_N <= HEAD_DIM_QK)
    tl.static_assert(BLOCK_N <= HEAD_DIM_V)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM_QK], strides=[HEAD_DIM_QK, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM_QK])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM_QK], strides=[HEAD_DIM_QK, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM_QK])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM_V], strides=[HEAD_DIM_V, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM_V])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM_V], strides=[HEAD_DIM_V, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM_V])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - torch.finfo(torch.float32).max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln(2)
    q = desc_q.load([qo_offset_y, 0])
    acc, l_i, m_i, h_i = _attn_fwd_inner(acc, l_i, m_i, h_i, q,  #
                                         desc_k, desc_v, offset_y, DTYPE, qk_scale,  #
                                         HEAD_DIM_V, BLOCK_M, BLOCK_N,  #
                                         offs_n, N_BOUND, N_CTX,  #
                                         warp_specialize, IS_HOPPER)
    # entropy H = -sum p log p = log l - h / l (converted from log2 to natural log)
    e_i = (tl.math.log2(l_i) - h_i / l_i) * LN2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX + offs_m, m_i)
    tl.store(E + off_hz * N_CTX + offs_m, e_i)
    desc_o.store([qo_offset_y, 0], acc.to(DTYPE))


@triton.jit
def _attn_bwd_preprocess(O, DO, Delta, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM_V: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM_V)
    o = tl.load(O + off_hz * HEAD_DIM_V * N_CTX + off_m[:, None] * HEAD_DIM_V + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM_V * N_CTX + off_m[:, None] * HEAD_DIM_V + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# Gradient w.r.t. a logit s_ij (natural units):
#   dS = P * (dP - delta) + dE * P * (-E - log P)
# where the second term comes from the entropy output E = -sum_j P_ij log P_ij.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, DO, DE, E, M, D,  #
                   stride_tok, stride_tov, stride_d,  #
                   N_BOUND, N_CTX,  #
                   BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,  #
                   start_n, start_m, num_steps):
    LN2: tl.constexpr = 0.6931471824645996
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_d_qk[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tov + offs_d_v[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    valid_n = (offs_n < N_BOUND)[:, None]
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        de = tl.load(DE + offs_m)
        e = tl.load(E + offs_m)
        qkT = tl.dot(k, qT)
        logpT = qkT - m[None, :]
        pT = tl.math.exp2(logpT)
        if N_BOUND < N_CTX:
            pT = tl.where(valid_n, pT, 0.0)
            logpT = tl.where(valid_n, logpT, 0.0)
        do = tl.load(do_ptrs)
        dv += tl.dot(pT, do.to(pT.dtype))
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :]) + de[None, :] * pT * (-e[None, :] - logpT * LN2)
        if N_BOUND < N_CTX:
            dsT = tl.where((offs_m < N_BOUND)[None, :], dsT, 0.0)
        dk += tl.dot(dsT, tl.trans(qT).to(dsT.dtype))  # fp32 for numerical stability
        curr_m += BLOCK_M1
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tov
    return dk, dv


@triton.jit
def _attn_bwd_dq(dq, q, K, V, do, de, e, m, D,  #
                 stride_tok, stride_tov, stride_d,  #
                 N_BOUND, N_CTX,  #
                 BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,  #
                 start_m, start_n, num_steps):
    LN2: tl.constexpr = 0.6931471824645996
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_d_qk[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tov + offs_d_v[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    valid_m = (offs_m < N_BOUND)[:, None]
    for _ in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        log_p = qk - m
        p = tl.math.exp2(log_p)
        if N_BOUND < N_CTX:
            valid_n = ((curr_n + tl.arange(0, BLOCK_N2)) < N_BOUND)[None, :]
            p = tl.where(valid_n, p, 0.0)
            log_p = tl.where(valid_n, log_p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None]) + de * p * (-e - log_p * LN2)
        if N_BOUND < N_CTX:
            ds = tl.where(valid_m, ds, 0.0)
        dq += tl.dot(ds, tl.trans(kT).to(ds.dtype))  # fp32 for numerical stability
        curr_n += BLOCK_N2
        kT_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tov
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale, DO, DE, E, DQ, DK, DV, M, D,  #
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX, N_BOUND,  #
              BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,  #
              HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996
    # strides are given for Q/K; V/DO rows are HEAD_DIM_V // HEAD_DIM_QK times wider
    tl.static_assert(HEAD_DIM_V % HEAD_DIM_QK == 0)
    v_scale: tl.constexpr = HEAD_DIM_V // HEAD_DIM_QK
    stride_tov = stride_tok * v_scale

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj * v_scale
    DO += adj * v_scale
    DQ += adj
    DK += adj
    DV += adj * v_scale
    M += off_chz
    D += off_chz
    DE += off_chz
    E += off_chz

    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)

    # dK and dV for one block of keys
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM_QK], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_d_qk[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tov + offs_d_v[None, :] * stride_d)
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, DO, DE, E, M, D,  #
                            stride_tok, stride_tov, stride_d, N_BOUND, N_CTX,  #
                            BLOCK_M1, BLOCK_N1, HEAD_DIM_QK, HEAD_DIM_V,  #
                            start_n, 0, N_CTX // BLOCK_M1)
    tl.store(DV + offs_n[:, None] * stride_tov + offs_d_v[None, :] * stride_d, dv)
    dk *= sm_scale
    tl.store(DK + offs_n[:, None] * stride_tok + offs_d_qk[None, :] * stride_d, dk)

    # dQ for one block of queries
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_d_qk[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tov + offs_d_v[None, :] * stride_d)
    de = tl.load(DE + offs_m)[:, None]
    e = tl.load(E + offs_m)[:, None]
    m = tl.load(M + offs_m)[:, None]
    dq = tl.zeros([BLOCK_M2, HEAD_DIM_QK], dtype=tl.float32)
    dq = _attn_bwd_dq(dq, q, K, V, do, de, e, m, D,  #
                      stride_tok, stride_tov, stride_d, N_BOUND, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM_QK, HEAD_DIM_V,  #
                      start_m, 0, N_CTX // BLOCK_N2)
    # K was pre-scaled by sm_scale / ln(2)
    tl.store(DQ + offs_m[:, None] * stride_tok + offs_d_qk[None, :] * stride_d, dq * LN2)


class _FlashEntropyAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, n_valid, warp_specialize):
        B, H, N, D = q.shape
        Dv = v.shape[-1]
        assert q.dtype in TORCH_TO_TL, f"unsupported dtype {q.dtype}"
        assert k.shape == q.shape and v.shape[:-1] == q.shape[:-1]
        assert D in {16, 32, 64, 128, 256} and Dv in {16, 32, 64, 128, 256}
        assert N % SEQ_ALIGN == 0
        o = torch.empty((B, H, N, Dv), device=q.device, dtype=q.dtype)
        M = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        E = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = B * H * N
            dummy = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, D], strides=[D, 1], block_shape=dummy)
            desc_k = TensorDescriptor(k, shape=[y_dim, D], strides=[D, 1], block_shape=dummy)
            desc_v = TensorDescriptor(v, shape=[y_dim, Dv], strides=[Dv, 1], block_shape=dummy)
            desc_o = TensorDescriptor(o, shape=[y_dim, Dv], strides=[Dv, 1], block_shape=dummy)
        else:
            desc_q, desc_k, desc_v, desc_o = q, k, v, o

        triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device=q.device))
        extra_kern_args = {}
        if is_blackwell() and warp_specialize:
            extra_kern_args["maxnreg"] = 168 if D == 128 and q.dtype == torch.float16 else 80

        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H, 1)
        _attn_fwd[grid](
            sm_scale, M, E, B, H, desc_q, desc_k, desc_v, desc_o,  #
            N_BOUND=n_valid, N_CTX=N,  #
            HEAD_DIM_QK=D, HEAD_DIM_V=Dv,  #
            DTYPE=TORCH_TO_TL[q.dtype], warp_specialize=warp_specialize, IS_HOPPER=is_hopper(),  #
            **extra_kern_args)
        o[..., n_valid:, :] = 0.0
        E[..., n_valid:] = 0.0

        ctx.save_for_backward(q, k, v, o, M, E)
        ctx.sm_scale = sm_scale
        ctx.n_valid = n_valid
        return o, E

    @staticmethod
    def backward(ctx, do, de):
        q, k, v, o, M, E = ctx.saved_tensors
        do, de = do.contiguous(), de.contiguous()
        B, H, N, D = q.shape
        Dv = v.shape[-1]
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        num_stages = 4 if Dv > D else 5  # fewer stages to fit shared memory

        RCP_LN2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[(N // SEQ_ALIGN, B * H)](o, do, delta, N, BLOCK_M=SEQ_ALIGN, HEAD_DIM_V=Dv)
        _attn_bwd[(N // BLOCK_N1, 1, B * H)](
            q, arg_k, v, ctx.sm_scale, do, de, E, dq, dk, dv, M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            H, N, ctx.n_valid,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            HEAD_DIM_QK=D, HEAD_DIM_V=Dv,  #
            num_warps=4, num_stages=num_stages)
        return dq, dk, dv, None, None, None


def flash_entropy_attention(q, k, v, sm_scale=None, warp_specialize=True):
    """
    Non-causal attention that also returns the entropy of every attention row.

    Args:
        q, k: (B, H, N, D)
        v: (B, H, N, Dv), Dv must be a multiple of D
        sm_scale: softmax scale, defaults to 1 / sqrt(D)
    Returns:
        o: (B, H, N, Dv) = softmax(sm_scale * q k^T) v
        e: (B, H, N) float32, e_i = -sum_j P_ij log P_ij
    """
    n = q.shape[-2]
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    pad = -n % SEQ_ALIGN
    q, k, v = (F.pad(t, (0, 0, 0, pad)).contiguous() for t in (q, k, v))
    o, e = _FlashEntropyAttention.apply(q, k, v, sm_scale, n, warp_specialize)
    return o[..., :n, :], e[..., :n]
