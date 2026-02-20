"""
Test that torch.compile with eager-numerics flags produces bit-exact results
vs eager execution for every op / subgraph / full-block in the Llama 3
training loop, across all model sizes and TP-sharded shapes.

Pass criterion: EXACT match (0 ULP) under emulate_precision_casts +
eager_numerics.division_rounding.  Speedup is printed but informational only.

Each test prints: ULP stats (max/mean), max absolute diff, timing, speedup.

Tiers (cumulative):
  @tier_30s   -- individual ops & tiny shapes       (~30 s)
  @tier_5min  -- subgraphs at debug/8B shapes       (~5 min)
  @tier_1hr   -- full blocks at 8B, small 70B       (~1 hr)
  @tier_full  -- full blocks at all sizes incl 405B (~1-2 hr est.)

Run a tier:
    pytest test_llama3_compile_shapes.py -m tier_30s -v -s
    pytest test_llama3_compile_shapes.py -m 'tier_30s or tier_5min' -v -s
    pytest test_llama3_compile_shapes.py -v -s       # everything
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch._inductor.config as inductor_cfg
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
tier_30s = pytest.mark.tier_30s
tier_5min = pytest.mark.tier_5min
tier_1hr = pytest.mark.tier_1hr
tier_full = pytest.mark.tier_full

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LlamaShape:
    name: str
    d: int
    nh: int
    nkv: int
    ff: int
    V: int
    S: int
    B: int
    tp: int

    @property
    def hd(self):  return self.d // self.nh
    @property
    def rhd(self): return self.hd // 2
    @property
    def nh_l(self): return self.nh // self.tp
    @property
    def nkv_l(self): return self.nkv // self.tp
    @property
    def d_q(self):  return self.nh_l * self.hd
    @property
    def d_kv(self): return self.nkv_l * self.hd
    @property
    def ff_l(self): return self.ff // self.tp
    @property
    def T(self):    return self.B * self.S
    @property
    def S_tp(self): return self.S // self.tp
    @property
    def T_tp(self): return self.B * self.S_tp

    def __str__(self): return self.name


TINY      = LlamaShape("tiny",  256,   16,  16, 768,   512,    128,  2, 1)
DEBUG     = LlamaShape("debug", 256,   16,  16, 768,   2048,   2048, 8, 1)
LLAMA_8B  = LlamaShape("8B",    4096,  32,  8,  14336, 128256, 8192, 1, 1)
LLAMA_70B = LlamaShape("70B",   8192,  64,  8,  28672, 128256, 8192, 8, 8)
LLAMA_405B= LlamaShape("405B",  16384, 128, 8,  53248, 128256, 8192, 2, 8)

DTYPES = [torch.bfloat16, torch.float32]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE = "cuda"
INIT_STD = 0.02


def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def randn(*shape, dtype=torch.bfloat16, std=1.0):
    return torch.randn(*shape, device=DEVICE, dtype=dtype) * std


def randn_c64(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(torch.complex64)


# ---------------------------------------------------------------------------
# Accuracy + timing harness
# ---------------------------------------------------------------------------
class AccuracyFailure(AssertionError):
    """Compiled output is not bit-exact with eager."""


def ulp_stats(eager, compiled):
    """Max and mean ULP distance.  Clamps reference magnitude to >= 1.0."""
    if isinstance(eager, tuple):
        stats = [ulp_stats(e, c) for e, c in zip(eager, compiled)]
        return max(s[0] for s in stats), max(s[1] for s in stats)
    dt = eager.dtype
    if dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        a, b = eager.float(), compiled.float()
        diff = (a - b).abs()
        fi = torch.finfo(dt)
        ref = torch.maximum(a.abs(), b.abs()).clamp(min=1.0)
        ulps = diff / (ref * fi.eps)
        return ulps.max().item(), ulps.mean().item()
    d = (eager.float() - compiled.float()).abs()
    return d.max().item(), d.mean().item()


def check(fn_and_args, *, label="", warmup=2, repeat=5):
    """Run eager vs compiled, print ULP + speedup, raise AccuracyFailure if not exact.

    fn_and_args: tuple of (callable, args_tuple) as returned by make_* factories.
    """
    fn, args = fn_and_args
    seed()
    eager_out = fn(*args)

    with inductor_cfg.patch({
        "emulate_precision_casts": True,
        "eager_numerics.division_rounding": True,
    }):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for _ in range(warmup):
            compiled(*args)
        torch.cuda.synchronize()
        seed()
        compiled_out = compiled(*args)

        exact = (
            all(torch.equal(e, c) for e, c in zip(eager_out, compiled_out))
            if isinstance(eager_out, tuple)
            else torch.equal(eager_out, compiled_out)
        )
        if exact:
            max_ulp = mean_ulp = diff = 0.0
            match_str = "EXACT (0 ULP)"
        else:
            max_ulp, mean_ulp = ulp_stats(eager_out, compiled_out)
            diff = max(
                (e.float() - c.float()).abs().max().item()
                for e, c in zip(eager_out, compiled_out)
            ) if isinstance(eager_out, tuple) else (
                (eager_out.float() - compiled_out.float()).abs().max().item()
            )
            match_str = f"max_diff={diff:.2e}  max_ulp={max_ulp:.1f}  mean_ulp={mean_ulp:.4f}"

        # Timing (same compile flags)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / repeat

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            compiled(*args)
        torch.cuda.synchronize()
        t_compiled = (time.perf_counter() - t0) / repeat

    speedup = t_eager / t_compiled if t_compiled > 0 else float("inf")
    print(
        f"  [{label}]  {match_str}  "
        f"eager={t_eager*1e3:.3f}ms  compiled={t_compiled*1e3:.3f}ms  "
        f"speedup={speedup:.2f}x"
    )
    if not exact:
        raise AccuracyFailure(
            f"not exact: max_diff={diff:.2e}, max_ulp={max_ulp:.1f}, mean_ulp={mean_ulp:.4f}"
        )


# ---------------------------------------------------------------------------
# Op / subgraph factories  (shape, dtype) -> (fn, args)
# ---------------------------------------------------------------------------

def make_rms_norm(s, dtype):
    x = randn(s.B, s.S_tp, s.d, dtype=dtype)
    w = randn(s.d, dtype=torch.float32 if dtype != torch.float32 else torch.float32)
    def fn(x, w):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-5) * w).to(x.dtype)
    return fn, (x, w)


def make_mm_wq(s, dtype):
    return torch.mm, (randn(s.T, s.d, dtype=dtype), randn(s.d, s.d_q, dtype=dtype))

def make_mm_wkv(s, dtype):
    return torch.mm, (randn(s.T, s.d, dtype=dtype), randn(s.d, s.d_kv, dtype=dtype))

def make_mm_w1(s, dtype):
    return torch.mm, (randn(s.T, s.d, dtype=dtype), randn(s.d, s.ff_l, dtype=dtype))

def make_mm_w2(s, dtype):
    return torch.mm, (randn(s.T, s.ff_l, dtype=dtype), randn(s.ff_l, s.d, dtype=dtype))

def make_mm_dwq(s, dtype):
    return torch.mm, (randn(s.d_q, s.T, dtype=dtype), randn(s.T, s.d, dtype=dtype))

def make_mm_dwkv(s, dtype):
    return torch.mm, (randn(s.d_kv, s.T, dtype=dtype), randn(s.T, s.d, dtype=dtype))

def make_mm_dw1(s, dtype):
    return torch.mm, (randn(s.ff_l, s.T, dtype=dtype), randn(s.T, s.d, dtype=dtype))

def make_mm_dw2(s, dtype):
    return torch.mm, (randn(s.d, s.T, dtype=dtype), randn(s.T, s.ff_l, dtype=dtype))

def make_mm_dxkv(s, dtype):
    return torch.mm, (randn(s.T, s.d_kv, dtype=dtype), randn(s.d_kv, s.d, dtype=dtype))


def make_rope(s, dtype):
    x = randn(s.B, s.S, s.nh_l, s.hd, dtype=dtype)
    freqs = randn_c64(1, s.S, 1, s.rhd)
    def fn(x, freqs):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs).flatten(3).to(x.dtype)
    return fn, (x, freqs)


def make_silu_gate(s, dtype):
    x = randn(s.B, s.S, s.ff_l, dtype=dtype)
    g = randn(s.B, s.S, s.ff_l, dtype=dtype)
    def fn(x, g): return F.silu(x) * g
    return fn, (x, g)


def make_flash_attn(s, dtype):
    q = randn(s.B, s.nh_l, s.S, s.hd, dtype=dtype)
    k = randn(s.B, s.nkv_l, s.S, s.hd, dtype=dtype)
    v = randn(s.B, s.nkv_l, s.S, s.hd, dtype=dtype)
    gqa = bool(s.nh_l != s.nkv_l)
    def fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    return fn, (q, k, v)


def make_cross_entropy(s, dtype):
    T = min(s.T_tp, 8192)
    pred = randn(T, s.V, dtype=dtype)
    labels = torch.randint(0, s.V, (T,), device=DEVICE)
    labels[::7] = -100
    def fn(pred, labels):
        return F.cross_entropy(pred.float(), labels, reduction="sum", ignore_index=-100)
    return fn, (pred, labels)


def make_attention_block(s, dtype):
    d, d_q, d_kv, hd = s.d, s.d_q, s.d_kv, s.hd
    nh_l, nkv_l, rhd, B, S = s.nh_l, s.nkv_l, s.rhd, s.B, s.S
    gqa = bool(nh_l != nkv_l)
    x = randn(B, S, d, dtype=dtype, std=INIT_STD)
    norm_w = torch.ones(d, device=DEVICE, dtype=torch.float32)
    wq = randn(d_q, d, dtype=dtype, std=INIT_STD)
    wk = randn(d_kv, d, dtype=dtype, std=INIT_STD)
    wv = randn(d_kv, d, dtype=dtype, std=INIT_STD)
    wo = randn(d, d_q, dtype=dtype, std=INIT_STD)
    freqs = randn_c64(S, rhd)
    def fn(x, norm_w, wq, wk, wv, wo, freqs):
        var = x.float().pow(2).mean(-1, keepdim=True)
        h = (x.float() * torch.rsqrt(var + 1e-5) * norm_w).to(x.dtype)
        T = B * S
        hf = h.view(T, d)
        xq = (hf @ wq.t()).view(B, S, nh_l, hd)
        xk = (hf @ wk.t()).view(B, S, nkv_l, hd)
        xv = (hf @ wv.t()).view(B, S, nkv_l, hd)
        fc = freqs[:S].view(1, S, 1, rhd)
        qc = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        kc = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq = torch.view_as_real(qc * fc).flatten(3).to(x.dtype)
        xk = torch.view_as_real(kc * fc).flatten(3).to(x.dtype)
        attn = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
            is_causal=True, enable_gqa=gqa,
        )
        out = attn.transpose(1, 2).contiguous().view(T, d_q)
        return (out @ wo.t()).view(B, S, d)
    return fn, (x, norm_w, wq, wk, wv, wo, freqs)


def make_ffn_block(s, dtype):
    d, ff_l, B, S = s.d, s.ff_l, s.B, s.S
    x = randn(B, S, d, dtype=dtype, std=INIT_STD)
    norm_w = torch.ones(d, device=DEVICE, dtype=torch.float32)
    w1 = randn(ff_l, d, dtype=dtype, std=INIT_STD)
    w3 = randn(ff_l, d, dtype=dtype, std=INIT_STD)
    w2 = randn(d, ff_l, dtype=dtype, std=INIT_STD)
    def fn(x, norm_w, w1, w3, w2):
        var = x.float().pow(2).mean(-1, keepdim=True)
        h = (x.float() * torch.rsqrt(var + 1e-5) * norm_w).to(x.dtype)
        T = B * S
        hf = h.view(T, d)
        return (F.silu(hf @ w1.t()) * (hf @ w3.t()) @ w2.t()).view(B, S, d)
    return fn, (x, norm_w, w1, w3, w2)


def make_transformer_block(s, dtype):
    d, d_q, d_kv, hd = s.d, s.d_q, s.d_kv, s.hd
    nh_l, nkv_l, rhd, ff_l = s.nh_l, s.nkv_l, s.rhd, s.ff_l
    B, S = s.B, s.S
    gqa = bool(nh_l != nkv_l)
    x = randn(B, S, d, dtype=dtype, std=INIT_STD)
    an_w = torch.ones(d, device=DEVICE, dtype=torch.float32)
    wq = randn(d_q, d, dtype=dtype, std=INIT_STD)
    wk = randn(d_kv, d, dtype=dtype, std=INIT_STD)
    wv = randn(d_kv, d, dtype=dtype, std=INIT_STD)
    wo = randn(d, d_q, dtype=dtype, std=INIT_STD)
    freqs = randn_c64(S, rhd)
    fn_w = torch.ones(d, device=DEVICE, dtype=torch.float32)
    w1 = randn(ff_l, d, dtype=dtype, std=INIT_STD)
    w3 = randn(ff_l, d, dtype=dtype, std=INIT_STD)
    w2 = randn(d, ff_l, dtype=dtype, std=INIT_STD)
    def fn(x, an_w, wq, wk, wv, wo, freqs, fn_w, w1, w3, w2):
        T = B * S
        var = x.float().pow(2).mean(-1, keepdim=True)
        h = (x.float() * torch.rsqrt(var + 1e-5) * an_w).to(x.dtype)
        hf = h.view(T, d)
        xq = (hf @ wq.t()).view(B, S, nh_l, hd)
        xk = (hf @ wk.t()).view(B, S, nkv_l, hd)
        xv = (hf @ wv.t()).view(B, S, nkv_l, hd)
        fc = freqs[:S].view(1, S, 1, rhd)
        qc = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        kc = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq = torch.view_as_real(qc * fc).flatten(3).to(x.dtype)
        xk = torch.view_as_real(kc * fc).flatten(3).to(x.dtype)
        attn = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
            is_causal=True, enable_gqa=gqa,
        )
        attn_out = (attn.transpose(1, 2).contiguous().view(T, d_q) @ wo.t()).view(B, S, d)
        r = x + attn_out
        var2 = r.float().pow(2).mean(-1, keepdim=True)
        h2 = (r.float() * torch.rsqrt(var2 + 1e-5) * fn_w).to(x.dtype)
        hf2 = h2.view(T, d)
        return r + (F.silu(hf2 @ w1.t()) * (hf2 @ w3.t()) @ w2.t()).view(B, S, d)
    return fn, (x, an_w, wq, wk, wv, wo, freqs, fn_w, w1, w3, w2)


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

def params(shapes, dtypes=DTYPES):
    """Build (shape, dtype) parametrize args for class-level decoration."""
    return pytest.mark.parametrize(
        "s,dtype",
        [(s, d) for s in shapes for d in dtypes],
        ids=[f"{s.name}-{str(d).split('.')[-1]}" for s in shapes for d in dtypes],
    )


# ===================================================================
# TIER 1: Individual ops (~30s)
# ===================================================================

@tier_30s
@params([TINY, DEBUG])
class TestOps:
    def test_rms_norm(self, s, dtype):
        check(make_rms_norm(s, dtype), label=f"rms_norm {s} {dtype}")

    def test_mm_wq(self, s, dtype):
        check(make_mm_wq(s, dtype), label=f"mm_wq {s} {dtype}")

    def test_mm_wkv(self, s, dtype):
        check(make_mm_wkv(s, dtype), label=f"mm_wkv {s} {dtype}")

    def test_mm_w1(self, s, dtype):
        check(make_mm_w1(s, dtype), label=f"mm_w1 {s} {dtype}")

    def test_mm_w2(self, s, dtype):
        check(make_mm_w2(s, dtype), label=f"mm_w2 {s} {dtype}")

    def test_rope(self, s, dtype):
        check(make_rope(s, dtype), label=f"rope {s} {dtype}")

    def test_silu_gate(self, s, dtype):
        check(make_silu_gate(s, dtype), label=f"silu_gate {s} {dtype}")

    def test_flash_attn(self, s, dtype):
        check(make_flash_attn(s, dtype), label=f"flash_attn {s} {dtype}")

    def test_cross_entropy(self, s, dtype):
        check(make_cross_entropy(s, dtype), label=f"cross_entropy {s} {dtype}")


# ===================================================================
# TIER 2: Subgraphs + 8B ops (~5 min)
# ===================================================================

@tier_5min
@params([DEBUG, LLAMA_8B])
class TestSubgraphs:
    def test_attention_block(self, s, dtype):
        check(make_attention_block(s, dtype), label=f"attn {s} {dtype}")

    def test_ffn_block(self, s, dtype):
        check(make_ffn_block(s, dtype), label=f"ffn {s} {dtype}")

    def test_cross_entropy(self, s, dtype):
        check(make_cross_entropy(s, dtype), label=f"xent {s} {dtype}")

    def test_transformer_block(self, s, dtype):
        check(make_transformer_block(s, dtype), label=f"block {s} {dtype}")

    def test_mm_wq(self, s, dtype):
        check(make_mm_wq(s, dtype), label=f"mm_wq {s} {dtype}")

    def test_mm_wkv(self, s, dtype):
        check(make_mm_wkv(s, dtype), label=f"mm_wkv {s} {dtype}")

    def test_mm_w1(self, s, dtype):
        check(make_mm_w1(s, dtype), label=f"mm_w1 {s} {dtype}")

    def test_mm_w2(self, s, dtype):
        check(make_mm_w2(s, dtype), label=f"mm_w2 {s} {dtype}")


# ===================================================================
# TIER 3: 70B shapes (~1 hr)
# ===================================================================

@tier_1hr
@params([DEBUG, LLAMA_8B, LLAMA_70B])
class TestLargeBlocks:
    def test_transformer_block(self, s, dtype):
        check(make_transformer_block(s, dtype), label=f"block {s} {dtype}")

    def test_attention_block(self, s, dtype):
        check(make_attention_block(s, dtype), label=f"attn {s} {dtype}")

    def test_ffn_block(self, s, dtype):
        check(make_ffn_block(s, dtype), label=f"ffn {s} {dtype}")

    def test_mm_wq(self, s, dtype):
        check(make_mm_wq(s, dtype), label=f"mm_wq {s} {dtype}")

    def test_mm_wkv(self, s, dtype):
        check(make_mm_wkv(s, dtype), label=f"mm_wkv {s} {dtype}")

    def test_mm_w1(self, s, dtype):
        check(make_mm_w1(s, dtype), label=f"mm_w1 {s} {dtype}")

    def test_mm_w2(self, s, dtype):
        check(make_mm_w2(s, dtype), label=f"mm_w2 {s} {dtype}")


# ===================================================================
# TIER 4: All sizes including 405B (~1-2 hr est.)
# ===================================================================

@tier_full
@params([DEBUG, LLAMA_8B, LLAMA_70B, LLAMA_405B])
class TestFull:
    def test_transformer_block(self, s, dtype):
        check(make_transformer_block(s, dtype), label=f"block {s} {dtype}")

    def test_attention_block(self, s, dtype):
        check(make_attention_block(s, dtype), label=f"attn {s} {dtype}")

    def test_ffn_block(self, s, dtype):
        check(make_ffn_block(s, dtype), label=f"ffn {s} {dtype}")

    def test_cross_entropy(self, s, dtype):
        check(make_cross_entropy(s, dtype), label=f"xent {s} {dtype}")

    def test_mm_all_fwd(self, s, dtype):
        """All forward mm shapes."""
        for factory, tag in [
            (make_mm_wq, "wq"), (make_mm_wkv, "wkv"),
            (make_mm_w1, "w1"), (make_mm_w2, "w2"),
        ]:
            check(*factory(s, dtype), label=f"mm_{tag} {s} {dtype}")

    def test_mm_all_bwd(self, s, dtype):
        """All backward mm shapes."""
        for factory, tag in [
            (make_mm_dwq, "dwq"), (make_mm_dwkv, "dwkv"),
            (make_mm_dw1, "dw1"), (make_mm_dw2, "dw2"),
            (make_mm_dxkv, "dxkv"),
        ]:
            check(*factory(s, dtype), label=f"mm_{tag} {s} {dtype}")

    def test_rope(self, s, dtype):
        check(make_rope(s, dtype), label=f"rope {s} {dtype}")

    def test_silu_gate(self, s, dtype):
        check(make_silu_gate(s, dtype), label=f"silu_gate {s} {dtype}")

    def test_flash_attn(self, s, dtype):
        check(make_flash_attn(s, dtype), label=f"flash_attn {s} {dtype}")
