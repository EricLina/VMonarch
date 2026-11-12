import pytest
import torch
import torch.nn.functional as F

import video_monarch_attention.vmonarch as vm
from video_monarch_attention import flash_entropy_attention, vmonarch_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DTYPE = torch.bfloat16


def entropy_attention_ref(q, k, v, sm_scale):
    logp = (sm_scale * q.float() @ k.float().transpose(-1, -2)).log_softmax(-1)
    p = logp.exp()
    return p @ v.float(), -(p * logp).sum(-1)


def rand(*shape, requires_grad=True):
    return torch.randn(shape, device="cuda", dtype=DTYPE, requires_grad=requires_grad)


def assert_close(x, ref, tol):
    err = (x.float() - ref.float()).abs().max().item()
    assert err <= tol * ref.float().abs().max().item(), f"max abs err {err}"


@pytest.mark.parametrize("N", [128, 200, 1024])
@pytest.mark.parametrize("D, Dv", [(64, 64), (128, 128), (64, 128), (128, 256)])
def test_flash_entropy_attention(N, D, Dv):
    torch.manual_seed(0)
    q, k, v = rand(2, 3, N, D), rand(2, 3, N, D), rand(2, 3, N, Dv)
    do, de = torch.randn(2, 3, N, Dv, device="cuda"), torch.randn(2, 3, N, device="cuda")

    o, e = flash_entropy_attention(q, k, v)
    grads = torch.autograd.grad((o.float() * do).sum() + (e * de).sum(), (q, k, v))
    o_ref, e_ref = entropy_attention_ref(q, k, v, D ** -0.5)
    grads_ref = torch.autograd.grad((o_ref * do).sum() + (e_ref * de).sum(), (q, k, v))

    assert_close(o, o_ref, 2e-2)
    assert_close(e, e_ref, 2e-2)
    for g, g_ref in zip(grads, grads_ref):
        assert_close(g, g_ref, 3e-2)


@pytest.mark.parametrize("m, b", [(4, 256), (8, 100)])
@pytest.mark.parametrize("num_iters", [1, 2, 3])
def test_vmonarch_attention(monkeypatch, m, b, num_iters):
    torch.manual_seed(0)
    q, k, v = rand(1, 2, m * b, 64), rand(1, 2, m * b, 64), rand(1, 2, m * b, 64)
    do = torch.randn(1, 2, m * b, 64, device="cuda")

    z = vmonarch_attention(q, k, v, m, b, num_iters, recompute_tokens=b)
    grads = torch.autograd.grad((z.float() * do).sum(), (q, k, v))

    def row_block_attention_ref(x, k, w):
        out, ent = entropy_attention_ref(x, k, w, 1.0)
        return out, -ent.detach()

    monkeypatch.setattr(vm, "_row_block_attention", row_block_attention_ref)
    q32, k32, v32 = (t.detach().float().requires_grad_() for t in (q, k, v))
    z_ref = vmonarch_attention(q32, k32, v32, m, b, num_iters, recompute_tokens=b)
    grads_ref = torch.autograd.grad((z_ref * do).sum(), (q32, k32, v32))

    assert_close(z, z_ref, 3e-2)
    for g, g_ref in zip(grads, grads_ref):
        assert_close(g, g_ref, 5e-2)


def test_recompute_tokens_are_exact():
    torch.manual_seed(0)
    q, k, v = (rand(1, 2, 4 * 128, 64, requires_grad=False) for _ in range(3))
    z = vmonarch_attention(q, k, v, 4, 128, recompute_tokens=128)
    exact = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(z[:, :, :128], exact[:, :, :128])
