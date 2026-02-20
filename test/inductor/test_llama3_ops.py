"""
Test torch.compile accuracy for individual ATen ops from the training loop.

Same harness as test_llama3_compile_shapes.py: exact match (0 ULP) required
with emulate_precision_casts + eager_numerics.division_rounding.

Run:
    pytest test_llama3_ops.py -m tier_30s -v -s
    pytest test_llama3_ops.py -v -s       # everything
"""

from __future__ import annotations

import pytest
import torch

from test_llama3_compile_shapes import (
    DEVICE,
    DEBUG,
    DTYPES,
    LLAMA_8B,
    TINY,
    check,
    params,
    randn,
    tier_30s,
    tier_5min,
)


# ---------------------------------------------------------------------------
# Op factories: (shape, dtype) -> (fn, args)
# ---------------------------------------------------------------------------

# --- Pointwise unary ---


def make_tanh(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.tanh, (x,)


def make_sigmoid(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.sigmoid, (x,)


def make_exp_out(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype) * 0.1
    def fn(x):
        out = torch.empty_like(x)
        torch.exp(x, out=out)
        return out
    return fn, (x,)


def make_log(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype).abs() + 1e-3
    return torch.log, (x,)


def make_clamp(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.clamp(min=-0.5, max=0.5)
    return fn, (x,)


def make_pow_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.pow(2)
    return fn, (x,)


def make_rsub_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return 1.0 - x
    return fn, (x,)


# --- Pointwise binary ---


def make_add(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.add, (x, y)


def make_sub(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.sub, (x, y)


def make_mul(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.mul, (x, y)


def make_div(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype).abs() + 0.1
    return torch.div, (x, y)


def make_maximum(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.maximum, (x, y)


# --- In-place ---


def make_add_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, y):
        x = x.clone()
        x.add_(y)
        return x
    return fn, (x, y)


def make_sub_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, y):
        x = x.clone()
        x.sub_(y)
        return x
    return fn, (x, y)


def make_mul_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, y):
        x = x.clone()
        x.mul_(y)
        return x
    return fn, (x, y)


def make_div_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype).abs() + 0.1
    def fn(x, y):
        x = x.clone()
        x.div_(y)
        return x
    return fn, (x, y)


def make_copy_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, y):
        x = x.clone()
        x.copy_(y)
        return x
    return fn, (x, y)


# --- Comparison ---


def make_lt_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.lt(0.0)
    return fn, (x,)


def make_ge_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.ge(0.0)
    return fn, (x,)


def make_eq_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.eq(0.0)
    return fn, (x,)


def make_gt_scalar(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.gt(0.0)
    return fn, (x,)


def make_lt_tensor(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.lt, (x, y)


def make_le_tensor(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.le, (x, y)


# --- Bitwise ---


def make_bitwise_or(s, dtype):
    x = torch.randint(0, 256, (s.B, s.S), device=DEVICE, dtype=torch.int64)
    y = torch.randint(0, 256, (s.B, s.S), device=DEVICE, dtype=torch.int64)
    return torch.bitwise_or, (x, y)


def make_bitwise_and(s, dtype):
    x = torch.randint(0, 256, (s.B, s.S), device=DEVICE, dtype=torch.int64)
    y = torch.randint(0, 256, (s.B, s.S), device=DEVICE, dtype=torch.int64)
    return torch.bitwise_and, (x, y)


# --- Reductions ---


def make_sum_dim(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.sum(dim=-1)
    return fn, (x,)


def make_sum_default(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.sum()
    return fn, (x,)


def make_max_dim(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.max(dim=-1)
    return fn, (x,)


def make_max_default(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.max()
    return fn, (x,)


def make_linalg_vector_norm(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return torch.linalg.vector_norm(x, dim=-1)
    return fn, (x,)


def make_sort_stable(s, dtype):
    x = randn(s.B, s.S, dtype=dtype)
    def fn(x):
        return torch.sort(x, dim=-1, stable=True)
    return fn, (x,)


# --- Shape manipulation ---


def make_view(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.view(s.B * s.S, s.d)
    return fn, (x,)


def make_unsafe_view(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return torch.ops.aten._unsafe_view(x, [s.B * s.S, s.d])
    return fn, (x,)


def make_unsqueeze(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.unsqueeze(1)
    return fn, (x,)


def make_expand(s, dtype):
    x = randn(1, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.expand(s.B, s.S, s.d)
    return fn, (x,)


def make_permute(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.permute(0, 2, 1)
    return fn, (x,)


def make_transpose(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.transpose(1, 2)
    return fn, (x,)


def make_select(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.select(1, 0)
    return fn, (x,)


def make_slice(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x[:, : s.S // 2, :]
    return fn, (x,)


def make_split(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    chunk = s.d // 4
    def fn(x):
        return x.split(chunk, dim=-1)
    return fn, (x,)


def make_split_with_sizes(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    sizes = [s.d // 2, s.d - s.d // 2]
    def fn(x):
        return x.split(sizes, dim=-1)
    return fn, (x,)


def make_cat(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, y):
        return torch.cat([x, y], dim=1)
    return fn, (x, y)


def make_clone(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    return torch.clone, (x,)


def make_detach(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.detach()
    return fn, (x,)


def make_constant_pad_nd(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return torch.nn.functional.pad(x, (0, 16))
    return fn, (x,)


# --- Indexing ---


def make_index(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    idx = torch.randint(0, s.B, (s.B,), device=DEVICE)
    def fn(x, idx):
        return x[idx]
    return fn, (x, idx)


def make_index_put(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    idx = torch.arange(s.B, device=DEVICE)
    values = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, idx, values):
        return torch.ops.aten.index_put(x, [idx], values)
    return fn, (x, idx, values)


def make_index_put_inplace(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    idx = torch.arange(s.B, device=DEVICE)
    values = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x, idx, values):
        x = x.clone()
        x.index_put_([idx], values)
        return x
    return fn, (x, idx, values)


# --- Creation ---


def make_arange_start(s, dtype):
    def fn():
        return torch.arange(0, s.S, device=DEVICE, dtype=torch.int64)
    return fn, ()


def make_arange_default(s, dtype):
    def fn():
        return torch.arange(s.S, device=DEVICE, dtype=torch.int64)
    return fn, ()


def make_zeros(s, dtype):
    def fn():
        return torch.zeros(s.B, s.S, device=DEVICE, dtype=dtype)
    return fn, ()


def make_new_zeros(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.new_zeros(s.B, s.S)
    return fn, (x,)


def make_new_ones(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return x.new_ones(s.B, s.S)
    return fn, (x,)


def make_scalar_tensor(s, dtype):
    def fn():
        return torch.scalar_tensor(1.0, device=DEVICE, dtype=dtype)
    return fn, ()


# --- Type conversion ---


def make_to_copy(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    target = torch.float32 if dtype != torch.float32 else torch.bfloat16
    def fn(x):
        return x.to(target)
    return fn, (x,)


# --- Special / backward ---


def make_where(s, dtype):
    cond = torch.randint(0, 2, (s.B, s.S, s.d), device=DEVICE, dtype=torch.bool)
    x = randn(s.B, s.S, s.d, dtype=dtype)
    y = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(cond, x, y):
        return torch.where(cond, x, y)
    return fn, (cond, x, y)


def make_embedding(s, dtype):
    V = min(s.V, 2048)
    weight = randn(V, s.d, dtype=dtype)
    indices = torch.randint(0, V, (s.B, s.S), device=DEVICE)
    def fn(weight, indices):
        return torch.ops.aten.embedding.default(weight, indices)
    return fn, (weight, indices)


def make_embedding_dense_backward(s, dtype):
    V = min(s.V, 2048)
    grad = randn(s.B, s.S, s.d, dtype=dtype)
    indices = torch.randint(0, V, (s.B, s.S), device=DEVICE)
    def fn(grad, indices):
        return torch.ops.aten.embedding_dense_backward(grad, indices, V, -1, False)
    return fn, (grad, indices)


def make_tanh_backward(s, dtype):
    grad = randn(s.B, s.S, s.d, dtype=dtype)
    output = randn(s.B, s.S, s.d, dtype=dtype).tanh()
    def fn(grad, output):
        return torch.ops.aten.tanh_backward(grad, output)
    return fn, (grad, output)


def make_sigmoid_backward(s, dtype):
    grad = randn(s.B, s.S, s.d, dtype=dtype)
    output = randn(s.B, s.S, s.d, dtype=dtype).sigmoid()
    def fn(grad, output):
        return torch.ops.aten.sigmoid_backward(grad, output)
    return fn, (grad, output)


def make_lift_fresh_copy(s, dtype):
    x = randn(s.B, s.S, s.d, dtype=dtype)
    def fn(x):
        return torch.ops.aten.lift_fresh_copy.default(x)
    return fn, (x,)


# _assert_scalar: guard op with no tensor output, not testable with this harness


# ===================================================================
# TIER 1: All ops at small shapes (~30s)
# ===================================================================

@tier_30s
@params([TINY, DEBUG])
class TestIndividualOps:
    # Pointwise unary
    def test_tanh(self, s, dtype):
        check(make_tanh(s, dtype), label=f"tanh {s} {dtype}")

    def test_sigmoid(self, s, dtype):
        check(make_sigmoid(s, dtype), label=f"sigmoid {s} {dtype}")

    def test_exp_out(self, s, dtype):
        check(make_exp_out(s, dtype), label=f"exp_out {s} {dtype}")

    def test_log(self, s, dtype):
        check(make_log(s, dtype), label=f"log {s} {dtype}")

    def test_clamp(self, s, dtype):
        check(make_clamp(s, dtype), label=f"clamp {s} {dtype}")

    def test_pow_scalar(self, s, dtype):
        check(make_pow_scalar(s, dtype), label=f"pow {s} {dtype}")

    def test_rsub_scalar(self, s, dtype):
        check(make_rsub_scalar(s, dtype), label=f"rsub {s} {dtype}")

    # Pointwise binary
    def test_add(self, s, dtype):
        check(make_add(s, dtype), label=f"add {s} {dtype}")

    def test_sub(self, s, dtype):
        check(make_sub(s, dtype), label=f"sub {s} {dtype}")

    def test_mul(self, s, dtype):
        check(make_mul(s, dtype), label=f"mul {s} {dtype}")

    def test_div(self, s, dtype):
        check(make_div(s, dtype), label=f"div {s} {dtype}")

    def test_maximum(self, s, dtype):
        check(make_maximum(s, dtype), label=f"maximum {s} {dtype}")

    # In-place
    def test_add_inplace(self, s, dtype):
        check(make_add_inplace(s, dtype), label=f"add_ {s} {dtype}")

    def test_sub_inplace(self, s, dtype):
        check(make_sub_inplace(s, dtype), label=f"sub_ {s} {dtype}")

    def test_mul_inplace(self, s, dtype):
        check(make_mul_inplace(s, dtype), label=f"mul_ {s} {dtype}")

    def test_div_inplace(self, s, dtype):
        check(make_div_inplace(s, dtype), label=f"div_ {s} {dtype}")

    def test_copy_inplace(self, s, dtype):
        check(make_copy_inplace(s, dtype), label=f"copy_ {s} {dtype}")

    # Comparison
    def test_lt_scalar(self, s, dtype):
        check(make_lt_scalar(s, dtype), label=f"lt_scalar {s} {dtype}")

    def test_ge_scalar(self, s, dtype):
        check(make_ge_scalar(s, dtype), label=f"ge_scalar {s} {dtype}")

    def test_eq_scalar(self, s, dtype):
        check(make_eq_scalar(s, dtype), label=f"eq_scalar {s} {dtype}")

    def test_gt_scalar(self, s, dtype):
        check(make_gt_scalar(s, dtype), label=f"gt_scalar {s} {dtype}")

    def test_lt_tensor(self, s, dtype):
        check(make_lt_tensor(s, dtype), label=f"lt_tensor {s} {dtype}")

    def test_le_tensor(self, s, dtype):
        check(make_le_tensor(s, dtype), label=f"le_tensor {s} {dtype}")

    # Bitwise
    def test_bitwise_or(self, s, dtype):
        check(make_bitwise_or(s, dtype), label=f"bitwise_or {s}")

    def test_bitwise_and(self, s, dtype):
        check(make_bitwise_and(s, dtype), label=f"bitwise_and {s}")

    # Reductions
    def test_sum_dim(self, s, dtype):
        check(make_sum_dim(s, dtype), label=f"sum_dim {s} {dtype}")

    def test_sum_default(self, s, dtype):
        check(make_sum_default(s, dtype), label=f"sum {s} {dtype}")

    def test_max_dim(self, s, dtype):
        check(make_max_dim(s, dtype), label=f"max_dim {s} {dtype}")

    def test_max_default(self, s, dtype):
        check(make_max_default(s, dtype), label=f"max {s} {dtype}")

    def test_linalg_vector_norm(self, s, dtype):
        check(make_linalg_vector_norm(s, dtype), label=f"vecnorm {s} {dtype}")

    def test_sort_stable(self, s, dtype):
        check(make_sort_stable(s, dtype), label=f"sort {s} {dtype}")

    # Shape manipulation
    def test_view(self, s, dtype):
        check(make_view(s, dtype), label=f"view {s} {dtype}")

    def test_unsafe_view(self, s, dtype):
        check(make_unsafe_view(s, dtype), label=f"_unsafe_view {s} {dtype}")

    def test_unsqueeze(self, s, dtype):
        check(make_unsqueeze(s, dtype), label=f"unsqueeze {s} {dtype}")

    def test_expand(self, s, dtype):
        check(make_expand(s, dtype), label=f"expand {s} {dtype}")

    def test_permute(self, s, dtype):
        check(make_permute(s, dtype), label=f"permute {s} {dtype}")

    def test_transpose(self, s, dtype):
        check(make_transpose(s, dtype), label=f"transpose {s} {dtype}")

    def test_select(self, s, dtype):
        check(make_select(s, dtype), label=f"select {s} {dtype}")

    def test_slice(self, s, dtype):
        check(make_slice(s, dtype), label=f"slice {s} {dtype}")

    def test_split(self, s, dtype):
        check(make_split(s, dtype), label=f"split {s} {dtype}")

    def test_split_with_sizes(self, s, dtype):
        check(make_split_with_sizes(s, dtype), label=f"split_sizes {s} {dtype}")

    def test_cat(self, s, dtype):
        check(make_cat(s, dtype), label=f"cat {s} {dtype}")

    def test_clone(self, s, dtype):
        check(make_clone(s, dtype), label=f"clone {s} {dtype}")

    def test_detach(self, s, dtype):
        check(make_detach(s, dtype), label=f"detach {s} {dtype}")

    def test_constant_pad_nd(self, s, dtype):
        check(make_constant_pad_nd(s, dtype), label=f"pad {s} {dtype}")

    # Indexing
    def test_index(self, s, dtype):
        check(make_index(s, dtype), label=f"index {s} {dtype}")

    def test_index_put(self, s, dtype):
        check(make_index_put(s, dtype), label=f"index_put {s} {dtype}")

    def test_index_put_inplace(self, s, dtype):
        check(make_index_put_inplace(s, dtype), label=f"index_put_ {s} {dtype}")

    # Creation
    def test_arange_start(self, s, dtype):
        check(make_arange_start(s, dtype), label=f"arange_start {s}")

    def test_arange_default(self, s, dtype):
        check(make_arange_default(s, dtype), label=f"arange {s}")

    def test_zeros(self, s, dtype):
        check(make_zeros(s, dtype), label=f"zeros {s} {dtype}")

    def test_new_zeros(self, s, dtype):
        check(make_new_zeros(s, dtype), label=f"new_zeros {s} {dtype}")

    def test_new_ones(self, s, dtype):
        check(make_new_ones(s, dtype), label=f"new_ones {s} {dtype}")

    def test_scalar_tensor(self, s, dtype):
        check(make_scalar_tensor(s, dtype), label=f"scalar_tensor {dtype}")

    # Type conversion
    def test_to_copy(self, s, dtype):
        check(make_to_copy(s, dtype), label=f"to_copy {s} {dtype}")

    # Special / backward
    def test_where(self, s, dtype):
        check(make_where(s, dtype), label=f"where {s} {dtype}")

    def test_embedding(self, s, dtype):
        check(make_embedding(s, dtype), label=f"embedding {s} {dtype}")

    def test_embedding_dense_backward(self, s, dtype):
        check(make_embedding_dense_backward(s, dtype), label=f"emb_bwd {s} {dtype}")

    def test_tanh_backward(self, s, dtype):
        check(make_tanh_backward(s, dtype), label=f"tanh_bwd {s} {dtype}")

    def test_sigmoid_backward(self, s, dtype):
        check(make_sigmoid_backward(s, dtype), label=f"sigmoid_bwd {s} {dtype}")

    def test_lift_fresh_copy(self, s, dtype):
        check(make_lift_fresh_copy(s, dtype), label=f"lift_fresh_copy {s} {dtype}")


# ===================================================================
# TIER 2: Numerically interesting ops at 8B shapes (~5 min)
# ===================================================================

@tier_5min
@params([DEBUG, LLAMA_8B])
class TestIndividualOps8B:
    def test_tanh(self, s, dtype):
        check(make_tanh(s, dtype), label=f"tanh {s} {dtype}")

    def test_sigmoid(self, s, dtype):
        check(make_sigmoid(s, dtype), label=f"sigmoid {s} {dtype}")

    def test_exp_out(self, s, dtype):
        check(make_exp_out(s, dtype), label=f"exp_out {s} {dtype}")

    def test_log(self, s, dtype):
        check(make_log(s, dtype), label=f"log {s} {dtype}")

    def test_pow_scalar(self, s, dtype):
        check(make_pow_scalar(s, dtype), label=f"pow {s} {dtype}")

    def test_add(self, s, dtype):
        check(make_add(s, dtype), label=f"add {s} {dtype}")

    def test_sub(self, s, dtype):
        check(make_sub(s, dtype), label=f"sub {s} {dtype}")

    def test_mul(self, s, dtype):
        check(make_mul(s, dtype), label=f"mul {s} {dtype}")

    def test_div(self, s, dtype):
        check(make_div(s, dtype), label=f"div {s} {dtype}")

    def test_sum_dim(self, s, dtype):
        check(make_sum_dim(s, dtype), label=f"sum_dim {s} {dtype}")

    def test_linalg_vector_norm(self, s, dtype):
        check(make_linalg_vector_norm(s, dtype), label=f"vecnorm {s} {dtype}")

    def test_where(self, s, dtype):
        check(make_where(s, dtype), label=f"where {s} {dtype}")

    def test_embedding(self, s, dtype):
        check(make_embedding(s, dtype), label=f"embedding {s} {dtype}")

    def test_embedding_dense_backward(self, s, dtype):
        check(make_embedding_dense_backward(s, dtype), label=f"emb_bwd {s} {dtype}")

    def test_tanh_backward(self, s, dtype):
        check(make_tanh_backward(s, dtype), label=f"tanh_bwd {s} {dtype}")

    def test_sigmoid_backward(self, s, dtype):
        check(make_sigmoid_backward(s, dtype), label=f"sigmoid_bwd {s} {dtype}")

    def test_sort_stable(self, s, dtype):
        check(make_sort_stable(s, dtype), label=f"sort {s} {dtype}")
