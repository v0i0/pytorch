"""
Catalog of non-associative (in floating point) reduction kernels in PyTorch's
eager CUDA backend.

This file contains:
  1. Helper primitives that simulate CUDA warp/block reduction patterns.
  2. Specification functions that reproduce the exact association order of each
     CUDA kernel, parameterized by block dimensions.
  3. Test classes with runnable examples that exercise each dispatch path.

All CUDA file paths are relative to aten/src/ATen/native/cuda/.
"""

import math
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


# ============================================================================
# Dtype-aware arithmetic helpers
# ============================================================================
# When dtype is set (e.g. np.float32), all spec arithmetic rounds to that
# precision at every step, matching the CUDA kernel's actual numerics.
# When dtype is None, Python float64 is used (useful for tree-structure
# verification).

def _dtype_cast(dtype):
    """Return a cast function for the given numpy dtype (or identity)."""
    if dtype is None:
        return lambda x: float(x)
    return lambda x: dtype(x)


def _dtype_exp(dtype):
    """Return exp() that matches CUDA's implementation for the given dtype."""
    if dtype is None:
        return math.exp
    # Route through CUDA torch.exp for bitwise matching with CUDA kernels.
    # IEEE 754 add/mul/div are deterministic, but transcendentals (exp, sqrt,
    # log) use different polynomial approximations on CPU vs GPU.
    torch_dtype = {np.float32: torch.float32, np.float64: torch.float64,
                   np.float16: torch.float16}.get(dtype, torch.float32)
    def _cuda_exp(x):
        t = torch.tensor(float(x), dtype=torch_dtype, device="cuda")
        return dtype(t.exp().item())
    return _cuda_exp


def _dtype_sqrt(dtype):
    """Return sqrt() that matches CUDA's implementation for the given dtype."""
    if dtype is None:
        return math.sqrt
    torch_dtype = {np.float32: torch.float32, np.float64: torch.float64,
                   np.float16: torch.float16}.get(dtype, torch.float32)
    def _cuda_sqrt(x):
        t = torch.tensor(float(x), dtype=torch_dtype, device="cuda")
        return dtype(t.sqrt().item())
    return _cuda_sqrt


def _dtype_identity(dtype, val):
    """Return val cast to dtype."""
    if dtype is None:
        return float(val)
    return dtype(val)


# ============================================================================
# Reduction primitives — simulate CUDA warp/block patterns on CPU values
# ============================================================================

Combine = Callable[[Any, Any], Any]


# CUDA ref: cuda/Reduce.cuh:30 (last_pow2)
def lpow2(n: int) -> int:
    """Largest power of 2 <= n (mirrors last_pow2 in Reduce.cuh)."""
    if n <= 0:
        return 0
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def sequential_reduce(elements: list, combine: Combine, identity):
    """One thread sequentially folding a list left-to-right."""
    acc = identity
    for e in elements:
        acc = combine(acc, e)
    return acc


# CUDA ref: cuda/Reduce.cuh:662 (block_x_reduce intra-warp, CUDA path)
def shfl_down_reduce_high_to_low(values: list, combine: Combine):
    """WARP_SHFL_DOWN halving tree, offset goes W/2 -> W/4 -> ... -> 1.
    This is the CUDA (non-ROCm) path used by Reduce.cuh block_x_reduce
    and block_reduce.cuh WarpReduceSum.  Result valid in lane 0."""
    vals = list(values)
    n = len(vals)
    offset = n // 2
    while offset > 0:
        for i in range(n):
            if i + offset < n:
                vals[i] = combine(vals[i], vals[i + offset])
        offset //= 2
    return vals[0]


# CUDA ref: cuda/Reduce.cuh:660 (block_x_reduce intra-warp, ROCm path)
def shfl_down_reduce_low_to_high(values: list, combine: Combine):
    """WARP_SHFL_DOWN with offset going 1 -> 2 -> ... -> W/2.
    This is the ROCm / FBCODE_CAFFE2 path in Reduce.cuh."""
    vals = list(values)
    n = len(vals)
    offset = 1
    while offset < n:
        for i in range(n):
            if i + offset < n:
                vals[i] = combine(vals[i], vals[i + offset])
        offset *= 2
    return vals[0]


# CUDA ref: cuda/Normalization.cuh:337-344 (batch_norm warp XOR butterfly)
def shfl_xor_butterfly_low_to_high(values: list, combine: Combine):
    """WARP_SHFL_XOR butterfly, offsets 1, 2, 4, ..., N/2 (low-to-high).
    Used by batch norm NCHW (Normalization.cuh:337) where offset = 1 << i.
    All lanes end up with the same result."""
    vals = list(values)
    n = len(vals)
    offset = 1
    while offset < n:
        new_vals = list(vals)
        for i in range(n):
            partner = i ^ offset
            if 0 <= partner < n:
                new_vals[i] = combine(vals[i], vals[partner])
        vals = new_vals
        offset *= 2
    return vals[0]


# CUDA ref: cuda/PersistentSoftmax.cuh:34 (warp_reduce, offset = WARP_SIZE/2 down)
def shfl_xor_butterfly_high_to_low(values: list, combine: Combine):
    """WARP_SHFL_XOR butterfly, offsets N/2, N/4, ..., 1 (high-to-low).
    Used by persistent softmax (PersistentSoftmax.cuh:34) where
    offset = WARP_SIZE/2 down to 1.
    All lanes end up with the same result."""
    vals = list(values)
    n = len(vals)
    offset = n // 2
    while offset > 0:
        new_vals = list(vals)
        for i in range(n):
            partner = i ^ offset
            if 0 <= partner < n:
                new_vals[i] = combine(vals[i], vals[partner])
        vals = new_vals
        offset //= 2
    return vals[0]


# Backwards-compat alias — batch norm and layer norm dgamma use low-to-high
shfl_xor_butterfly = shfl_xor_butterfly_low_to_high


# CUDA ref: cuda/Reduce.cuh:674 (block_y_reduce)
def shmem_halving_reduce(values: list, combine: Combine, identity):
    """Shared-memory halving tree (used by block_y_reduce, loss kernels, etc.).
    offset goes N/2 -> N/4 -> ... -> 1.  Active threads: those with index <
    offset.  Result valid in slot 0."""
    vals = list(values)
    n = len(vals)
    offset = n // 2
    while offset > 0:
        for i in range(offset):
            if i + offset < len(vals):
                vals[i] = combine(vals[i], vals[i + offset])
        offset //= 2
    return vals[0]


# CUDA ref: cuda/block_reduce.cuh:77 (BlockReduceSum), :130 (BlockReduce)
def block_reduce_cuh(thread_values: list, combine: Combine, identity,
                     warp_size: int = 32):
    """Two-level reduction from block_reduce.cuh (BlockReduceSum pattern).
    Level 1: shfl_down within each warp.
    Level 2: warp leaders -> shared memory -> warp 0 shfl_down."""
    n = len(thread_values)
    num_warps = math.ceil(n / warp_size)

    # Level 1: intra-warp shfl_down
    warp_results = []
    for w in range(num_warps):
        start = w * warp_size
        end = min(start + warp_size, n)
        warp_vals = list(thread_values[start:end])
        while len(warp_vals) < warp_size:
            warp_vals.append(identity)
        warp_results.append(shfl_down_reduce_high_to_low(warp_vals, combine))

    # Level 2: warp 0 reduces the per-warp results
    while len(warp_results) < warp_size:
        warp_results.append(identity)
    return shfl_down_reduce_high_to_low(warp_results[:warp_size], combine)


# CUDA ref: cuda/MultiMarginLoss.cu:24
def thread_0_serial_scan(values: list, combine: Combine, identity):
    """Thread 0 serial scan — used by MultiMarginLoss."""
    acc = identity
    for v in values:
        acc = combine(acc, v)
    return acc


# ============================================================================
# gpu_reduce_kernel specification (Reduce.cuh)
# ============================================================================

def gpu_reduce_config(num_inputs: int, num_outputs: int, stride1: bool,
                      max_threads: int = 512, warp_size: int = 32,
                      num_sms: int = 0, max_threads_per_sm: int = 2048,
                      output_vec_size: int = 1):
    """Compute block dimensions and step_input matching setReduceConfig.

    Returns (block_width, block_height, step_input, do_block_x_reduce,
             do_block_y_reduce, vectorize_input, ctas_per_output)

    output_vec_size: for non-stride-1 reductions on contiguous output dims,
        the output is vectorized (typically 4 for aligned tensors).
        CUDA ref: Reduce.cuh:1097-1108 (get_output_vec_size).
    """
    input_vec_size = 4  # default vt0
    if stride1:
        dim0 = num_inputs
        dim1 = num_outputs
    else:
        dim0 = num_outputs // output_vec_size
        dim1 = num_inputs

    vectorize = False
    if stride1 and dim0 >= 128 and dim0 >= input_vec_size:
        vectorize = True
        dim0 //= input_vec_size

    dim0_p2 = min(lpow2(dim0), max_threads) if dim0 > 0 else 1
    dim1_p2 = min(lpow2(dim1), max_threads) if dim1 > 0 else 1

    W = min(dim0_p2, warp_size)
    H = min(dim1_p2, max_threads // W)
    W = min(dim0_p2, max_threads // H)

    step = 1
    do_bx = False
    do_by = False
    ctas = 1

    if stride1:
        step *= W
        do_bx = True
        vpt = math.ceil(num_inputs / (step * (input_vec_size if vectorize else 1)))
        threshold = min(H * 16, 256)
        if vpt >= threshold:
            step *= H
            do_by = True
    else:
        vpt = math.ceil(num_inputs / 1)
        threshold = min(H * 16, 256)
        if vpt >= threshold:
            step *= H
            do_by = True

    # Global reduce (ctas_per_output > 1)
    # CUDA ref: Reduce.cuh:1155-1176
    if do_by and num_sms > 0:
        num_threads = W * H
        vpt_now = math.ceil(num_inputs / step)
        blocks_per_sm = max_threads_per_sm // num_threads
        target_grid = num_sms * blocks_per_sm
        # grid.x = div_up(num_outputs / output_vec_size, step_output)
        # For non-stride-1: step_output = W (after split_output(W))
        step_out = W if not stride1 else 1
        grid_x = math.ceil(num_outputs / output_vec_size / step_out)
        min_vpt = 16
        max_vpt = 256
        if vpt_now >= max_vpt and grid_x <= target_grid:
            c1 = math.ceil(target_grid / grid_x)
            c2 = math.ceil(vpt_now / min_vpt)
            c3 = math.ceil(vpt_now / max_vpt)
            ctas = max(min(c1, c2), c3)
            if ctas > 1:
                step *= ctas

    return W, H, step, do_bx, do_by, vectorize, ctas


def spec_gpu_reduce_kernel(data: list, combine: Combine, identity,
                           num_outputs: int = 1, stride1: bool = True,
                           vt0: int = 4, warp_size: int = 32,
                           max_threads: int = 512, rocm: bool = False,
                           ctas_per_output: int = 1):
    """Runnable specification of the gpu_reduce_kernel reduction tree.

    Args:
        data: flat list of input values for ONE output element's reduction.
        combine: associative (in exact math) binary combine function.
        identity: identity element for combine.
        num_outputs: total number of output elements (affects block config).
        stride1: whether this is a stride-1 reduction.
        vt0: number of independent accumulators per thread (default 4).
        warp_size: CUDA warp size (32).
        max_threads: max threads per block (512).
        rocm: if True, use low-to-high warp shuffle order.

    Returns:
        The reduced value, computed following the exact association order
        of the CUDA kernel.

    Limitations:
        - Global reduce (ctas_per_output > 1) is not implemented; for very
          large reductions the real kernel splits across CTAs and the last
          CTA does a serial scan of the staging buffer.
        - The vectorized-input path (input_vec_size != vt0, used for 16-bit
          sum where vt0=4 but input_vec_size=8) is not modeled; it changes
          the number of independent accumulators but not the tree shape.

    CUDA ref: cuda/Reduce.cuh:1181 (gpu_reduce_kernel), :1032 (setReduceConfig),
    :561 (thread_reduce_impl), :499 (input_vectorized_thread_reduce_impl),
    :634 (block_x_reduce), :674 (block_y_reduce), :786 (global_reduce).
    SharedReduceOps.h:80 (WelfordData), :98 (WelfordOps), :118 (combine).
    """
    N = len(data)
    if N == 0:
        return identity

    W, H, S, do_bx, do_by, _, _ = gpu_reduce_config(
        N, num_outputs, stride1, max_threads, warp_size
    )
    num_threads = W * H

    # --- Phase 1: Thread-local accumulation ---
    input_vec_size = vt0  # default; same as template parameter
    # CUDA ref: Reduce.cuh:1098 — vectorize when dim0 >= 128 (before dividing)
    vectorize = (stride1 and N >= 128)

    thread_results = []
    if vectorize and stride1:
        # Vectorized path (input_vectorized_thread_reduce_impl):
        # Each thread loads input_vec_size CONSECUTIVE elements per vector,
        # at stride S (in units of vectors).  Accumulators correspond to
        # vector lanes, not strided positions.
        #   Thread t reads: [t*V, t*V+1, ..., t*V+(V-1)],
        #                   [(t+S)*V, (t+S)*V+1, ...], ...
        # where V = input_vec_size.
        # CUDA ref: Reduce.cuh:499 (input_vectorized_thread_reduce_impl)
        V = input_vec_size
        for t in range(min(num_threads, S)):
            accs = [identity] * V
            idx = t  # index in units of vectors
            while idx * V + V - 1 < N:
                for i in range(V):
                    accs[i] = combine(accs[i], data[idx * V + i])
                idx += S
            # tail: remaining individual elements
            tail_start = N - N % V
            if t < N - tail_start:
                accs[0] = combine(accs[0], data[tail_start + t])
            # merge accumulators left-to-right
            result = accs[0]
            for i in range(1, V):
                result = combine(result, accs[i])
            thread_results.append(result)
    else:
        # Non-vectorized path (thread_reduce_impl):
        # Each thread has vt0 independent accumulators at stride S.
        #   acc[i] gets elements: t+i*S, t+i*S+S*vt0, t+i*S+2*S*vt0, ...
        # CUDA ref: Reduce.cuh:561 (thread_reduce_impl)
        for t in range(min(num_threads, S)):
            accs = [identity] * vt0
            idx = t
            while idx + (vt0 - 1) * S < N:
                for i in range(vt0):
                    pos = idx + i * S
                    if pos < N:
                        accs[i] = combine(accs[i], data[pos])
                idx += S * vt0
            # tail: handle remaining elements
            for i in range(vt0):
                pos = idx + i * S
                if pos < N:
                    accs[i] = combine(accs[i], data[pos])
                if idx + i * S >= N:
                    break
            # merge accumulators left-to-right
            result = accs[0]
            for i in range(1, vt0):
                result = combine(result, accs[i])
            thread_results.append(result)

    # Pad to num_threads with identity
    while len(thread_results) < num_threads:
        thread_results.append(identity)

    # --- Phase 2: block_x_reduce (if stride-1) ---
    if do_bx:
        # Reshape as (H warps of W threads), reduce within each warp-row
        rows = []
        for y in range(H):
            row = thread_results[y * W:(y + 1) * W]
            while len(row) < W:
                row.append(identity)

            # If W > warp_size: shared-memory halving first
            if W > warp_size:
                row = list(row)
                offset = W // 2
                while offset >= warp_size:
                    for i in range(offset):
                        if i + offset < len(row):
                            row[i] = combine(row[i], row[i + offset])
                    offset //= 2
                row = row[:warp_size]

            # Intra-warp shfl_down
            effective_w = min(W, warp_size)
            warp_vals = row[:effective_w]
            if rocm:
                reduced = shfl_down_reduce_low_to_high(warp_vals, combine)
            else:
                reduced = shfl_down_reduce_high_to_low(warp_vals, combine)
            rows.append(reduced)
        thread_results = rows
    else:
        # Non-stride-1: threadIdx.x maps to independent outputs.
        # No intra-warp reduction. Just take the threadIdx.y column for
        # our output (thread_results already has the right structure).
        if do_by:
            # thread_results are indexed as [threadIdx.x * H + threadIdx.y]
            # For one output, gather the H values (all threadIdx.y for one x).
            # In reality, threadIdx.x=0 corresponds to our output.
            thread_results = thread_results[:H]

    # --- Phase 3: block_y_reduce (if warps split input) ---
    if do_by:
        vals = thread_results[:H]
        while len(vals) < H:
            vals.append(identity)
        return shmem_halving_reduce(vals, combine, identity)

    # If no block_y_reduce, thread_results[0] is the answer.
    return thread_results[0]


def _spec_gpu_reduce_one_cta(data, combine, identity, N, S, H, W, vt0,
                             do_bx, do_by, vectorize, stride1, rocm,
                             warp_size):
    """Run the per-CTA reduction (phases 1-3) for a subset of data."""
    # This is the same logic as spec_gpu_reduce_kernel phases 1-3
    input_vec_size = vt0
    num_threads = W * H

    thread_results = []
    if vectorize and stride1:
        V = input_vec_size
        for t in range(min(num_threads, S)):
            accs = [identity] * V
            idx = t
            while idx * V + V - 1 < N:
                for i in range(V):
                    accs[i] = combine(accs[i], data[idx * V + i])
                idx += S
            tail_start = N - N % V
            if t < N - tail_start:
                accs[0] = combine(accs[0], data[tail_start + t])
            result = accs[0]
            for i in range(1, V):
                result = combine(result, accs[i])
            thread_results.append(result)
    else:
        for t in range(min(num_threads, S)):
            accs = [identity] * vt0
            idx = t
            while idx + (vt0 - 1) * S < N:
                for i in range(vt0):
                    pos = idx + i * S
                    if pos < N:
                        accs[i] = combine(accs[i], data[pos])
                idx += S * vt0
            for i in range(vt0):
                pos = idx + i * S
                if pos < N:
                    accs[i] = combine(accs[i], data[pos])
                if idx + i * S >= N:
                    break
            result = accs[0]
            for i in range(1, vt0):
                result = combine(result, accs[i])
            thread_results.append(result)

    while len(thread_results) < num_threads:
        thread_results.append(identity)

    if do_bx:
        rows = []
        for y in range(H):
            row = thread_results[y * W:(y + 1) * W]
            while len(row) < W:
                row.append(identity)
            if W > warp_size:
                row = list(row)
                offset = W // 2
                while offset >= warp_size:
                    for i in range(offset):
                        if i + offset < len(row):
                            row[i] = combine(row[i], row[i + offset])
                    offset //= 2
                row = row[:warp_size]
            effective_w = min(W, warp_size)
            warp_vals = row[:effective_w]
            if rocm:
                reduced = shfl_down_reduce_low_to_high(warp_vals, combine)
            else:
                reduced = shfl_down_reduce_high_to_low(warp_vals, combine)
            rows.append(reduced)
        thread_results = rows
    else:
        if do_by:
            thread_results = thread_results[:H]

    if do_by:
        vals = thread_results[:H]
        while len(vals) < H:
            vals.append(identity)
        return shmem_halving_reduce(vals, combine, identity)
    return thread_results[0]


def spec_gpu_reduce_kernel_global(data: list, combine: Combine, identity,
                                  num_outputs: int, stride1: bool,
                                  vt0: int, warp_size: int, max_threads: int,
                                  rocm: bool, W: int, H: int, S: int,
                                  do_bx: bool, do_by: bool, vectorize: bool,
                                  ctas_per_output: int):
    """Global reduce path: split data across CTAs, reduce each, then combine.
    CUDA ref: Reduce.cuh:786-909 (global_reduce)."""
    N = len(data)
    # Step input within each CTA = S / ctas_per_output
    cta_step = S // ctas_per_output
    input_mult_CTA = cta_step  # = H (before global split)

    # Phase A: each CTA computes its partial result
    cta_results = []
    for cta_id in range(ctas_per_output):
        # Elements for this CTA: those at indices (t + cta_id * input_mult_CTA)
        # where t ranges over the per-CTA thread indices.
        # In the non-stride-1 case:
        #   input_idx = threadIdx.y * input_mult[BLOCK_Y] + cta_id * input_mult[CTA]
        #   = threadIdx.y * 1 + cta_id * H
        # Each thread processes at stride S (= H * ctas_per_output)
        cta_data = []
        for local_idx in range(cta_step):
            global_start = local_idx + cta_id * input_mult_CTA
            for pos in range(global_start, N, S):
                cta_data.append((pos, data[pos]))
        # Build a contiguous list for this CTA's elements, preserving order
        # Actually we need to run the same per-CTA reduce logic.
        # The CTA sees elements at positions:
        #   thread y gets: y + cta_id*H, y + cta_id*H + S, y + cta_id*H + 2S, ...
        # = y + cta_id*H + k*S for k=0,1,...

        # Build a "virtual" data array where index t maps to data[t + cta_id*H]
        # but the stride through data is S (= H * ctas).
        # The per-CTA tree uses S_local = cta_step = H as its step.
        # But the actual data access is strided: thread y reads
        # data[y + cta_id*H], data[y + cta_id*H + S], ...

        # Simpler approach: collect elements for each thread in this CTA
        thread_vals = []
        for t in range(cta_step):
            accs = [identity] * vt0
            idx = t + cta_id * input_mult_CTA
            iter_count = 0
            while idx + (vt0 - 1) * S < N:
                for i in range(vt0):
                    pos = idx + i * S
                    if pos < N:
                        accs[i] = combine(accs[i], data[pos])
                idx += S * vt0
                iter_count += 1
            # tail
            for i in range(vt0):
                pos = idx + i * S
                if pos < N:
                    accs[i] = combine(accs[i], data[pos])
                if idx + i * S >= N:
                    break
            result = accs[0]
            for i in range(1, vt0):
                result = combine(result, accs[i])
            thread_vals.append(result)

        while len(thread_vals) < H:
            thread_vals.append(identity)
        # block_y_reduce within CTA
        cta_result = shmem_halving_reduce(thread_vals[:H], combine, identity)
        cta_results.append(cta_result)

    # Phase B: last CTA combines all CTA results
    # CUDA ref: Reduce.cuh:838-866
    # Non-block_x_reduce path: thread y reads CTA y, y+H, y+2H, ...
    thread_partials = [identity] * H
    for t in range(H):
        acc = identity
        cta_idx = t
        while cta_idx < ctas_per_output:
            acc = combine(acc, cta_results[cta_idx])
            cta_idx += H
        thread_partials[t] = acc

    # Final block_y_reduce
    return shmem_halving_reduce(thread_partials, combine, identity)


# ============================================================================
# Softmax specifications (SoftMax.cu + PersistentSoftmax.cuh)
# ============================================================================

def spec_softmax_persistent(row: list, warp_size: int = 32, dtype=None) -> list:
    """Persistent softmax: entire row fits in registers, one warp per row.
    Uses WARP_SHFL_XOR butterfly (HIGH-TO-LOW: offset N/2,N/4,...,1) for
    both max and sum reductions.

    The kernel stores exp(x-max) in registers and reuses them for output
    (PersistentSoftmax.cuh:157).  Output = stored_exp / sum.

    Pass dtype=np.float32 to match fp32 CUDA numerics.

    CUDA ref: cuda/PersistentSoftmax.cuh:34 (warp_reduce), :68 (softmax_warp_forward)."""
    n = len(row)
    exp = _dtype_exp(dtype)
    cast = _dtype_cast(dtype)
    zero = _dtype_identity(dtype, 0.0)
    neg_inf = _dtype_identity(dtype, float("-inf"))

    lane_vals = [[] for _ in range(warp_size)]
    for i, v in enumerate(row):
        lane_vals[i % warp_size].append(v)

    lane_max = [max(lv) if lv else neg_inf for lv in lane_vals]
    row_max = shfl_xor_butterfly_high_to_low(lane_max, max)

    lane_exp = [[] for _ in range(warp_size)]
    lane_sumexp = []
    for lane_id, lv in enumerate(lane_vals):
        s = zero
        for v in lv:
            e = exp(cast(v) - cast(row_max))
            lane_exp[lane_id].append(e)
            s = s + e
        lane_sumexp.append(s)

    row_sum = shfl_xor_butterfly_high_to_low(lane_sumexp, lambda a, b: a + b)

    output = [zero] * n
    for lane_id in range(warp_size):
        for it, e in enumerate(lane_exp[lane_id]):
            idx = lane_id + it * warp_size
            if idx < n:
                output[idx] = e / row_sum
    return output


def spec_softmax_inner(row: list, block_size: int = 1024, dtype=None) -> list:
    """Inner-dim softmax (cunn_SoftMaxForward): ILP reduce + blockReduceWarp.
    block_size chosen as next pow2 up to 1024 from dim_size.

    CUDA ref: cuda/SoftMax.cu:487 (ilpReduce), :462 (blockReduceWarp), :734 (cunn_SoftMaxForward)."""
    exp = _dtype_exp(dtype)
    cast = _dtype_cast(dtype)
    zero = _dtype_identity(dtype, 0.0)
    neg_inf = _dtype_identity(dtype, float("-inf"))
    n = len(row)
    B = block_size

    ILP = 4  # sizeof(float4) / sizeof(float) for fp32
    def ilp_reduce(data, op, identity):
        """Simulate ilpReduce: loads ILP consecutive elements per vector at
        stride B (in vector units), accumulates sequentially, then handles
        the tail at stride B (in element units).
        CUDA ref: SoftMax.cu:487 (ilpReduce)."""
        sz = len(data)
        last = sz % (ILP * B)
        main_end = sz - last
        thread_vals = [identity] * B
        for tid in range(B):
            acc = identity
            # Main loop: vector loads of ILP consecutive elements
            offset = tid
            while offset * ILP < main_end:
                for j in range(ILP):
                    acc = op(acc, data[offset * ILP + j])
                offset += B
            # Tail: individual elements at stride B
            tail_offset = main_end + tid
            while tail_offset < sz:
                acc = op(acc, data[tail_offset])
                tail_offset += B
            thread_vals[tid] = acc
        return thread_vals

    # Step 1: max reduction
    thread_maxes = ilp_reduce(row, max, neg_inf)
    row_max = block_reduce_cuh(thread_maxes, max, neg_inf)

    # Step 2: compute exp and store, then sum
    # SoftMaxForwardEpilogue uses exp(x-max)/sum, recomputing exp for output.
    # SumExpFloat accumulates sum(exp(x-max)) per thread.
    exp_data = [exp(cast(v) - cast(row_max)) for v in row]
    thread_sums = ilp_reduce(exp_data, lambda a, b: a + b, zero)
    row_sum = block_reduce_cuh(thread_sums, lambda a, b: a + b, zero)

    # Step 3: output — epilogue recomputes exp(x-max) / sum
    # CUDA ref: SoftMax.cu:71 (SoftMaxForwardEpilogue)
    return [exp(cast(v) - cast(row_max)) / row_sum for v in row]


def spec_softmax_spatial(data_2d: list, dim_size: int, inner_size: int,
                         dtype=None) -> list:
    """Spatial softmax (cunn_SpatialSoftMaxForward): reducing a non-last dim.
    data_2d is [dim_size][inner_size] flattened to [dim_size * inner_size].

    Block layout: blockDim.x = dim_threads (multiple threads collaborate on
    the dim reduction), blockDim.y = inner_threads.
    When dim_threads > 1, spatialBlockReduceX combines per-thread partial
    results via shared-memory halving.

    CUDA ref: cuda/SoftMax.cu:134 (SpatialSoftMax_getBlockSize),
    :237 (spatialBlockReduceX), :262 (cunn_SpatialSoftMaxForward)."""
    exp = _dtype_exp(dtype)
    cast = _dtype_cast(dtype)
    zero = _dtype_identity(dtype, 0.0)
    neg_inf = _dtype_identity(dtype, float("-inf"))
    output = list(data_2d)

    # Compute dim_threads (block.x) matching SpatialSoftMax_getBlockSize
    max_block = 1024
    inner_threads = min(inner_size, max_block)
    dim_threads = 1
    if inner_threads <= 64 and dim_size >= 64:
        while inner_threads * dim_threads <= max_block and dim_threads <= dim_size:
            dim_threads *= 2
        dim_threads //= 2

    for inner in range(inner_size):
        if dim_threads == 1:
            # Pure sequential — one thread per inner position
            m = neg_inf
            for d in range(dim_size):
                m = max(m, data_2d[d * inner_size + inner])
            s = zero
            for d in range(dim_size):
                s = s + exp(cast(data_2d[d * inner_size + inner]) - cast(m))
            for d in range(dim_size):
                idx = d * inner_size + inner
                output[idx] = exp(cast(data_2d[idx]) - cast(m)) / s
        else:
            # Multiple threads in dim: per-thread sequential at stride dim_threads,
            # then spatialBlockReduceX (shmem halving across dim_threads).

            # Max: per-thread local max, then shmem halving
            thread_maxes = [neg_inf] * dim_threads
            for tx in range(dim_threads):
                for d in range(tx, dim_size, dim_threads):
                    thread_maxes[tx] = max(thread_maxes[tx],
                                           data_2d[d * inner_size + inner])
            m = shmem_halving_reduce(thread_maxes, max, neg_inf)

            # Sum(exp): per-thread sum at stride, then shmem halving
            thread_sums = [zero] * dim_threads
            for tx in range(dim_threads):
                for d in range(tx, dim_size, dim_threads):
                    thread_sums[tx] = thread_sums[tx] + exp(
                        cast(data_2d[d * inner_size + inner]) - cast(m))
            s = shmem_halving_reduce(thread_sums, lambda a, b: a + b, zero)

            # Epilogue: each thread writes at stride dim_threads
            for tx in range(dim_threads):
                for d in range(tx, dim_size, dim_threads):
                    idx = d * inner_size + inner
                    output[idx] = exp(cast(data_2d[idx]) - cast(m)) / s

    return output


# ============================================================================
# Layer norm specification (layer_norm_kernel.cu)
# ============================================================================

@torch.compile
def _compiled_welford_reduce(x_flat, start, stride, n_vec, vec_size):
    """Compiled per-thread Welford reduce matching cuWelfordOnlineSum.
    torch.compile generates FMA for mean + delta * recip and s2 + delta * delta2."""
    m = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s2 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = start
    while idx < n_vec:
        base = idx * vec_size
        for ii in range(vec_size):
            v = x_flat[base + ii]
            delta = v - m
            c = c + 1.0
            recip = 1.0 / c
            m = m + delta * recip
            delta2 = v - m
            s2 = s2 + delta * delta2
        idx = idx + stride
    return m, s2, c


@torch.compile
def _compiled_welford_combine(ma, s2a, ca, mb, s2b, cb):
    """Compiled Welford combine matching cuWelfordCombine.
    torch.compile generates FMA for nA*meanA + nB*meanB."""
    c = ca + cb
    coef = 1.0 / c
    nA = ca * coef
    nB = cb * coef
    delta = mb - ma
    m = nA * ma + nB * mb
    s2 = s2a + s2b + delta * delta * ca * nB
    return m, s2, c


@torch.compile
def _compiled_nhwc_welford_reduce(x_flat, start, stride, size):
    """Compiled per-accumulator Welford for NHWC batch norm.
    Uses x_count_inv = 1/count (reciprocal), matching line 1019/1034."""
    m = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s2 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = start
    while idx < size:
        v = x_flat[idx]
        c = c + 1.0
        x_count_inv = 1.0 / c
        delta0 = v - m
        m = m + delta0 * x_count_inv
        delta1 = v - m
        s2 = s2 + delta0 * delta1
        idx = idx + stride
    return m, s2, c


@torch.compile
def _compiled_nhwc_welford_merge(m, s2, c, m_new, s2_new, c_new):
    """Compiled welford_merge_element (Normalization.cuh:190).
    Uses factor = 1/max(1, total), mean = (m_new*c_new + m*c)*factor.

    Critical association from line 190: m2n += m2n_new + cross_term
    means m2n + (m2n_new + cross), NOT (m2n + m2n_new) + cross."""
    total = c + c_new
    factor = 1.0 / torch.clamp(total, min=1)
    delta0 = m - m_new
    cross = delta0 * delta0 * c_new * c * factor
    m_out = (m_new * c_new + m * c) * factor
    s2_out = s2 + (s2_new + cross)    # match kernel's += association
    c_out = total
    return m_out, s2_out, c_out


@torch.compile
def _compiled_softmax_backward_epilogue(tmp, output, sum_val):
    """Compiled epilogue: tmp - output * sum. Gets FMA: fma(-output, sum, tmp)."""
    return tmp - output * sum_val


@torch.compile
def _compiled_nchw_welford_reduce(x_3d, plane, ty, ty_s, tx, tx_s):
    """Compiled per-thread Welford for NCHW batch norm (lines 325-331).
    Uses delta / count (division)."""
    m = torch.zeros(1, device=x_3d.device, dtype=x_3d.dtype)
    s2 = torch.zeros(1, device=x_3d.device, dtype=x_3d.dtype)
    c = torch.zeros(1, device=x_3d.device, dtype=x_3d.dtype)
    b = ty
    while b < x_3d.size(0):
        s = tx
        while s < x_3d.size(2):
            v = x_3d[b, plane, s]
            d1 = v - m
            c = c + 1.0
            m = m + d1 / c
            s2 = s2 + d1 * (v - m)
            s = s + tx_s
        b = b + ty_s
    return m, s2, c


@torch.compile
def _compiled_nchw_welford_merge(avg, var_n, n, o_avg, o_var_n, o_n):
    """Compiled welford_merge_element for NCHW batch norm (lines 340-343).
    Uses factor-based weighted mean: (n*avg + o_n*o_avg) * factor.

    Critical association: the kernel's line 341 is:
      var_n += SHFL_XOR(var_n, ...) + (avg-o_avg)*(avg-o_avg)*n*o_n*factor
    The += means var_n = var_n + (shuffled_var + cross_term).
    So the association is var_n + (o_var_n + cross), NOT (var_n + o_var_n) + cross."""
    factor = 1.0 / torch.clamp(n + o_n, min=1)
    delta = avg - o_avg
    cross = delta * delta * n * o_n * factor
    var_out = var_n + (o_var_n + cross)
    avg_out = (n * avg + o_n * o_avg) * factor
    n_out = n + o_n
    return avg_out, var_out, n_out


@torch.compile
def _compiled_welford_ops_reduce(x_flat, start, stride):
    """Compiled WelfordOps::reduce (SharedReduceOps.h:103).
    Uses delta / count (division), FMA for m2 + delta * delta2."""
    m = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s2 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = start
    while idx < x_flat.shape[0]:
        v = x_flat[idx]
        delta = v - m
        c = c + 1.0
        m = m + delta / c
        delta2 = v - m
        s2 = s2 + delta * delta2
        idx = idx + stride
    return m, s2, c


@torch.compile
def _compiled_welford_ops_combine(ma, s2a, ca, mb, s2b, cb):
    """Compiled WelfordOps::combine (SharedReduceOps.h:118).
    Delta form with division: mean_a + delta * nb_over_n.
    m2 association: (s2a + s2b) + cross (left-to-right, matching C++ return stmt)."""
    c = ca + cb
    delta = mb - ma
    nb_over_n = cb / c
    m = ma + delta * nb_over_n
    s2 = (s2a + s2b) + delta * delta * ca * nb_over_n
    return m, s2, c


def spec_layer_norm_forward_moments(x_cuda: torch.Tensor,
                                    block_size: int = 512,
                                    eps: float = 1e-5, dtype=None):
    """Layer norm forward via the VECTORIZED compute_stats path.
    Takes a CUDA tensor directly so torch.compile can generate FMA-matching
    Triton kernels for the per-thread Welford reduce and combine steps.

    Matches cuWelfordOnlineSum (reduce) and cuWelfordCombine (combine) from
    layer_norm_kernel.cu, with vec_size=4, 2D block (32×num_warps),
    intra-warp shfl_down + inter-warp upper/lower shmem halving.

    CUDA ref: cuda/layer_norm_kernel.cu:186 (compute_stats),
    :134 (cuWelfordOnlineSum), :154 (cuWelfordCombine)."""
    N = x_cuda.shape[0]
    warp_size = 32
    vec_size = 4
    num_warps = block_size // warp_size
    n_vec = N // vec_size

    # Phase 1: per-thread Welford via compiled FMA
    thread_welford = []
    for tid in range(block_size):
        m, s2, c = _compiled_welford_reduce(x_cuda, tid, block_size, n_vec, vec_size)
        thread_welford.append((np.float32(m.item()), np.float32(s2.item()),
                               np.float32(c.item())))

    # Compiled Welford combine (wraps _compiled_welford_combine)
    def combine_wrap(a, b):
        ma, s2a, ca = a
        mb, s2b, cb = b
        if ca == 0:
            return b
        if cb == 0:
            return a
        ta = [torch.tensor(float(v), device="cuda") for v in
              [ma, s2a, ca, mb, s2b, cb]]
        m, s2, c = _compiled_welford_combine(*ta)
        return (np.float32(m.item()), np.float32(s2.item()),
                np.float32(c.item()))

    welford_id = (np.float32(0), np.float32(0), np.float32(0))

    # Phase 2a: intra-warp shfl_down (high-to-low)
    warp_results = []
    for w in range(num_warps):
        warp_vals = thread_welford[w * warp_size:(w + 1) * warp_size]
        while len(warp_vals) < warp_size:
            warp_vals.append(welford_id)
        warp_results.append(
            shfl_down_reduce_high_to_low(warp_vals, combine_wrap))

    # Phase 2b: inter-warp upper-half-writes, lower-half-merges
    offset = num_warps // 2
    while offset > 0:
        for wy in range(offset):
            warp_results[wy] = combine_wrap(
                warp_results[wy], warp_results[wy + offset])
        offset //= 2

    mean, m2, count = warp_results[0]
    # sigma2/N + eps → rsqrt via CUDA
    var = np.float32(m2) / np.float32(float(N))
    rstd = torch.rsqrt(
        torch.tensor(float(var + np.float32(eps)),
                     dtype=torch.float32, device="cuda")).item()
    return float(np.float32(mean)), rstd


def spec_rms_norm_forward(row: list, eps: float = 1e-5, block_size: int = 256,
                          dtype=None):
    """RMS norm forward: same Welford reduction as layer norm, but then
    reconstructs sum(x²)/N as (var + mean²) and computes rstd from that.

    CUDA ref: cuda/layer_norm_kernel.cu:59 (RowwiseMomentsCUDAKernel),
    :94 (rms_norm rstd = rsqrt(m2 + m1*m1 + eps))."""
    cast = _dtype_cast(dtype)
    sqrt_fn = _dtype_sqrt(dtype)
    zero = _dtype_identity(dtype, 0.0)

    # Phase 1: per-thread Welford (identical to layer norm)
    thread_welford = []
    for tid in range(block_size):
        mean, m2, count = zero, zero, zero
        for idx in range(tid, len(row), block_size):
            count = count + cast(1)
            delta = cast(row[idx]) - mean
            mean = mean + delta / count
            delta2 = cast(row[idx]) - mean
            m2 = m2 + delta * delta2
        thread_welford.append((mean, m2, count))

    def welford_combine(a, b):
        mean_a, m2_a, n_a = a
        mean_b, m2_b, n_b = b
        if n_a == 0:
            return b
        if n_b == 0:
            return a
        count = n_a + n_b
        delta = mean_b - mean_a
        nb_over_n = n_b / count
        new_mean = mean_a + delta * nb_over_n
        new_m2 = m2_a + m2_b + delta * delta * n_a * nb_over_n
        return (new_mean, new_m2, count)

    welford_id = (zero, zero, zero)
    mean, m2, count = block_reduce_cuh(
        thread_welford, welford_combine, welford_id
    )

    var = m2 / count if count > 0 else zero
    rstd = cast(1.0) / sqrt_fn(var + mean * mean + cast(eps))
    return rstd


@torch.compile
def _compiled_accum_sq(x_flat, start, stride, n_vec, vec_size):
    """Compiled per-thread sigma2 += val*val loop.
    torch.compile generates Triton FMA matching nvcc's fma(val, val, sigma2)."""
    acc = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = start
    while idx < n_vec:
        base = idx * vec_size
        for ii in range(vec_size):
            v = x_flat[base + ii]
            acc = acc + v * v
        idx = idx + stride
    return acc


def spec_rms_norm_forward_vectorized(x_cuda: torch.Tensor, eps: float = 1e-5,
                                     num_threads: int = 512):
    """RMS norm forward via the VECTORIZED path (compute_stats<rms_norm=true>).

    For rms_norm, the Welford reduce simplifies to: sigma2 += val * val (FMA).
    The combine simplifies to: sigma2 = a.sigma2 + b.sigma2 (pure addition).
    Vectorized loads: vec_size=4, so data is loaded 4 elements at a time.

    Takes a CUDA tensor directly (not a list) so that torch.compile can
    generate FMA-matching Triton kernels for the per-thread accumulation.

    CUDA ref: cuda/layer_norm_kernel.cu:186 (compute_stats),
    :134 (cuWelfordOnlineSum rms_norm), :154 (cuWelfordCombine rms_norm)."""
    N = x_cuda.shape[0]
    warp_size = 32
    vec_size = 4
    num_warps = num_threads // warp_size
    n_vec = N // vec_size

    # Phase 1: per-thread sigma2 += val*val via compiled FMA
    thread_sigma2 = []
    for tid in range(num_threads):
        s2 = _compiled_accum_sq(x_cuda, tid, num_threads, n_vec, vec_size)
        thread_sigma2.append(np.float32(s2.item()))

    # Phase 2: intra-warp shfl_down (pure addition — IEEE 754 exact)
    add = lambda a, b: a + b
    warp_results = []
    for w in range(num_warps):
        warp_vals = thread_sigma2[w * warp_size:(w + 1) * warp_size]
        warp_results.append(shfl_down_reduce_high_to_low(warp_vals, add))

    # Phase 3: inter-warp halving
    offset = num_warps // 2
    while offset > 0:
        for wy in range(offset):
            warp_results[wy] = warp_results[wy] + warp_results[wy + offset]
        offset //= 2

    # sigma2/N + eps → rsqrt, all on CUDA to match kernel's c10::cuda::compat::rsqrt
    s2_N = torch.tensor(float(warp_results[0]) / float(N),
                        dtype=torch.float32, device="cuda")
    return torch.rsqrt(s2_N + eps).item()


def spec_rms_norm_backward_dx(dY: list, X: list, rstd: float,
                              gamma: list, block_size: int = 256) -> list:
    """RMS norm backward dX: sequential sums + BlockReduceSum, then elementwise.
    Unlike layer norm, no mean subtraction: stats_x2 = sum(dY * gamma * X * rstd).

    CUDA ref: cuda/layer_norm_kernel.cu:388-391 (rms_norm branch)."""
    N = len(X)

    # Phase 1: per-thread sequential sum at stride block_size
    # Only stats_x2 needed for rms_norm (stats_x1 is not used)
    thread_x2 = [0.0] * block_size
    for tid in range(block_size):
        s2 = 0.0
        for idx in range(tid, N, block_size):
            dYg = dY[idx] * gamma[idx]
            s2 = s2 + dYg * X[idx] * rstd
        thread_x2[tid] = s2

    add = lambda a, b: a + b  # noqa: E731

    # Phase 2: BlockReduceSum
    stats_x2 = block_reduce_cuh(thread_x2, add, 0.0)

    # Phase 3: elementwise dX
    fH = float(N)
    dX = []
    for i in range(N):
        dYg = dY[i] * gamma[i]
        dX.append(rstd * (dYg - X[i] * rstd * stats_x2 / fH))
    return dX


# ============================================================================
# Batch norm specifications (Normalization.cuh)
# ============================================================================

def spec_batch_norm_nchw_stats(channel_data: list, block_size: int = 512):
    """batch_norm_collect_statistics_kernel (NCHW): Welford + SHFL_XOR butterfly.
    channel_data = flat list of all (batch, spatial) values for one channel.

    CUDA ref: cuda/Normalization.cuh:283 (batch_norm_collect_statistics_kernel), :337 (XOR butterfly warp reduce)."""

    # Phase 1: per-thread Welford
    thread_welford = []
    for tid in range(min(block_size, len(channel_data))):
        mean, m2, count = 0.0, 0.0, 0
        for idx in range(tid, len(channel_data), block_size):
            count += 1
            delta = channel_data[idx] - mean
            mean += delta / count
            delta2 = channel_data[idx] - mean
            m2 += delta * delta2
        thread_welford.append((mean, m2, float(count)))
    while len(thread_welford) < block_size:
        thread_welford.append((0.0, 0.0, 0.0))

    def welford_combine(a, b):
        mean_a, m2_a, n_a = a
        mean_b, m2_b, n_b = b
        if n_a == 0:
            return b
        if n_b == 0:
            return a
        count = n_a + n_b
        delta = mean_b - mean_a
        nb_over_n = n_b / count
        new_mean = mean_a + delta * nb_over_n
        new_m2 = m2_a + m2_b + delta * delta * n_a * nb_over_n
        return (new_mean, new_m2, count)

    welford_id = (0.0, 0.0, 0.0)

    # Phase 2: two-level SHFL_XOR butterfly (not shfl_down!)
    warp_size = 32
    num_warps = math.ceil(block_size / warp_size)

    # Level 1: XOR butterfly within each warp
    warp_results = []
    for w in range(num_warps):
        start = w * warp_size
        end = min(start + warp_size, block_size)
        warp_vals = thread_welford[start:end]
        while len(warp_vals) < warp_size:
            warp_vals.append(welford_id)
        warp_results.append(shfl_xor_butterfly(warp_vals, welford_combine))

    # Level 2: warp leaders to shared, then XOR butterfly in warp 0
    while len(warp_results) < warp_size:
        warp_results.append(welford_id)
    mean, m2, count = shfl_xor_butterfly(warp_results[:warp_size],
                                          welford_combine)
    N = len(channel_data)
    var = m2 / N if N > 0 else 0.0
    invstd = 1.0 / math.sqrt(var + 1e-5)
    return mean, invstd


def spec_batch_norm_nhwc_stats(channel_data: list, block_y: int = 16):
    """batch_norm_collect_statistics_channels_last_kernel (NHWC): Welford +
    shared-memory vertical halving.
    channel_data = flat list of all (batch, spatial) values for one channel.
    block_y = Y dimension from flexible_launch_configs.

    Key difference from NCHW: intra-warp combine uses shmem_halving_reduce
    instead of shfl_xor_butterfly, because threadIdx.x lanes are independent
    channels, not participating in the same reduction.

    NOTE: multi-CTA staging (grid.y > 1) is omitted for simplicity.  The real
    kernel writes partial Welford states to a staging buffer and the last CTA
    re-applies welford_merge_block_vertical.

    CUDA ref: cuda/Normalization.cuh:975 (batch_norm_collect_statistics_channels_last_kernel), :196 (welford_merge_block_vertical), :152 (flexible_launch_configs)."""

    ELEMENTS_PER_ITER = 4

    def welford_combine(a, b):
        mean_a, m2_a, n_a = a
        mean_b, m2_b, n_b = b
        if n_a == 0:
            return b
        if n_b == 0:
            return a
        count = n_a + n_b
        delta = mean_b - mean_a
        nb_over_n = n_b / count
        new_mean = mean_a + delta * nb_over_n
        new_m2 = m2_a + m2_b + delta * delta * n_a * nb_over_n
        return (new_mean, new_m2, count)

    welford_id = (0.0, 0.0, 0.0)

    # Phase 1: per-thread Welford with 4x register unroll
    thread_welford = []
    for tid in range(min(block_y, len(channel_data))):
        accs = [welford_id] * ELEMENTS_PER_ITER
        idx = tid
        acc_idx = 0
        while idx < len(channel_data):
            m, m2, n = accs[acc_idx]
            n += 1
            delta = channel_data[idx] - m
            m += delta / n
            delta2 = channel_data[idx] - m
            m2 += delta * delta2
            accs[acc_idx] = (m, m2, float(n))
            acc_idx = (acc_idx + 1) % ELEMENTS_PER_ITER
            idx += block_y
        # Merge the ELEMENTS_PER_ITER accumulators left-to-right
        result = accs[0]
        for i in range(1, ELEMENTS_PER_ITER):
            result = welford_combine(result, accs[i])
        thread_welford.append(result)
    while len(thread_welford) < block_y:
        thread_welford.append(welford_id)

    # Phase 2: welford_merge_block_vertical -- shmem halving in Y only
    mean, m2, count = shmem_halving_reduce(
        thread_welford, welford_combine, welford_id
    )

    N = len(channel_data)
    var = m2 / N if N > 0 else 0.0
    invstd = 1.0 / math.sqrt(var + 1e-5)
    return mean, invstd


def spec_batch_norm_nchw_backward_reduce(
    grad_output: list, input_data: list, mean: float,
    block_size: int = 512,
):
    """batch_norm_backward_kernel (NCHW): dual Float2 accumulation via
    SHFL_XOR butterfly.
    Computes sum_dy and sum_dy_xmu for one channel simultaneously.
    grad_output, input_data = flat lists of (batch, spatial) values for one
    channel.  Returns (sum_dy, sum_dy_xmu).

    CUDA ref: cuda/Normalization.cuh:387 (batch_norm_backward_kernel), :54 (Float2)."""

    assert len(grad_output) == len(input_data)

    def float2_add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    float2_id = (0.0, 0.0)

    # Phase 1: per-thread sequential accumulation of Float2
    thread_vals = []
    for tid in range(min(block_size, len(grad_output))):
        acc = float2_id
        for idx in range(tid, len(grad_output), block_size):
            dy = grad_output[idx]
            dy_xmu = dy * (input_data[idx] - mean)
            acc = float2_add(acc, (dy, dy_xmu))
        thread_vals.append(acc)
    while len(thread_vals) < block_size:
        thread_vals.append(float2_id)

    # Phase 2: two-level SHFL_XOR butterfly (same as NCHW forward stats)
    warp_size = 32
    num_warps = math.ceil(block_size / warp_size)

    warp_results = []
    for w in range(num_warps):
        start = w * warp_size
        end = min(start + warp_size, block_size)
        warp_vals = thread_vals[start:end]
        while len(warp_vals) < warp_size:
            warp_vals.append(float2_id)
        warp_results.append(shfl_xor_butterfly(warp_vals, float2_add))

    while len(warp_results) < warp_size:
        warp_results.append(float2_id)
    sum_dy, sum_dy_xmu = shfl_xor_butterfly(
        warp_results[:warp_size], float2_add
    )
    return sum_dy, sum_dy_xmu


# ============================================================================
# Loss function specifications
# ============================================================================

def spec_nll_loss_reduce(losses: list, block_size: int = 512):
    """nll_loss_forward_reduce_cuda_kernel_2d: sequential + shmem halving.
    losses = list of per-sample weighted losses.

    CUDA ref: cuda/Loss.cu:224 (nll_loss_forward_reduce_cuda_kernel_2d)."""

    # Phase 1: per-thread sequential sum at stride block_size
    zero = type(losses[0])(0) if losses else 0.0
    thread_sums = [zero] * block_size
    for tid in range(min(block_size, len(losses))):
        for idx in range(tid, len(losses), block_size):
            thread_sums[tid] = thread_sums[tid] + losses[idx]

    # Phase 2: shared-memory halving tree (NO warp shuffles)
    return shmem_halving_reduce(thread_sums, lambda a, b: a + b, zero)


def spec_multi_margin_loss_thread0_scan(per_thread_sums: list):
    """MultiMarginLoss: thread 0 serial scan of all buffer entries.

    CUDA ref: cuda/MultiMarginLoss.cu:24 (MultiMarginLoss_forward_kernel)."""
    return thread_0_serial_scan(per_thread_sums, lambda a, b: a + b, 0.0)


# ============================================================================
# Cumulative scan specifications (ScanUtils.cuh)
# ============================================================================

def spec_cumsum_innermost_sklansky(row: list, num_threads_x: int = 128):
    """Sklansky parallel prefix scan for innermost dimension.
    Processes in chunks of 2*num_threads_x with carry between chunks.

    CUDA ref: cuda/ScanUtils.cuh:60 (tensor_kernel_scan_innermost_dim_with_indices), :114 (Sklansky tree loop)."""
    output = []
    carry = 0.0
    chunk_size = 2 * num_threads_x

    for chunk_start in range(0, len(row), chunk_size):
        chunk = list(row[chunk_start:chunk_start + chunk_size])
        while len(chunk) < chunk_size:
            chunk.append(0.0)
        # Add carry to first element
        chunk[0] = chunk[0] + carry

        # Sklansky tree
        s = 1
        while s <= num_threads_x:
            new_chunk = list(chunk)
            for tid in range(num_threads_x):
                a = (tid // s) * (2 * s) + s
                ti = a + (tid % s)
                si = a - 1
                if ti < chunk_size and si < chunk_size:
                    new_chunk[ti] = chunk[si] + chunk[ti]
            chunk = new_chunk
            s *= 2

        actual_len = min(chunk_size, len(row) - chunk_start)
        output.extend(chunk[:actual_len])
        carry = chunk[chunk_size - 1]

    return output[:len(row)]


def spec_cumsum_outer_sequential(data_2d: list, num_rows: int,
                                 num_cols: int):
    """Outer-dim cumulative sum: purely sequential loop per thread.
    data_2d[row * num_cols + col], scan along rows for each col.

    CUDA ref: cuda/ScanUtils.cuh:154 (tensor_kernel_scan_outer_dim_with_indices)."""
    output = list(data_2d)
    for col in range(num_cols):
        for row in range(1, num_rows):
            output[row * num_cols + col] = (
                output[(row - 1) * num_cols + col] + data_2d[row * num_cols + col]
            )
    return output


def spec_cumprod_innermost_sklansky(row: list, num_threads_x: int = 128):
    """Sklansky parallel prefix scan for innermost-dim cumprod.
    Same tree as cumsum but with multiplication instead of addition.

    CUDA ref: cuda/ScanUtils.cuh:60 (tensor_kernel_scan_innermost_dim_with_indices)."""
    output = []
    carry = 1.0
    chunk_size = 2 * num_threads_x

    for chunk_start in range(0, len(row), chunk_size):
        chunk = list(row[chunk_start:chunk_start + chunk_size])
        while len(chunk) < chunk_size:
            chunk.append(1.0)
        chunk[0] = chunk[0] * carry

        s = 1
        while s <= num_threads_x:
            new_chunk = list(chunk)
            for tid in range(num_threads_x):
                a = (tid // s) * (2 * s) + s
                ti = a + (tid % s)
                si = a - 1
                if ti < chunk_size and si < chunk_size:
                    new_chunk[ti] = chunk[si] * chunk[ti]
            chunk = new_chunk
            s *= 2

        actual_len = min(chunk_size, len(row) - chunk_start)
        output.extend(chunk[:actual_len])
        carry = chunk[chunk_size - 1]

    return output[:len(row)]


def spec_cumprod_outer_sequential(data_2d: list, num_rows: int,
                                  num_cols: int):
    """Outer-dim cumulative product: purely sequential loop per thread.

    CUDA ref: cuda/ScanUtils.cuh:154 (tensor_kernel_scan_outer_dim_with_indices)."""
    output = list(data_2d)
    for col in range(num_cols):
        for row in range(1, num_rows):
            output[row * num_cols + col] = (
                output[(row - 1) * num_cols + col] * data_2d[row * num_cols + col]
            )
    return output


def _cuda_logaddexp(a, b):
    """CUDA logaddexp: log1p(exp(min-max)) + max, matching device implementation."""
    ta = torch.tensor(float(a), dtype=torch.float32, device="cuda")
    tb = torch.tensor(float(b), dtype=torch.float32, device="cuda")
    return float(torch.logaddexp(ta, tb).item())


def spec_logcumsumexp_innermost_sklansky(row: list, num_threads_x: int = 128):
    """Sklansky parallel prefix scan for innermost-dim logcumsumexp.
    Same tree as cumsum but with logaddexp combine.

    CUDA ref: cuda/ScanUtils.cuh:60."""
    output = []
    carry = float("-inf")
    chunk_size = 2 * num_threads_x

    for chunk_start in range(0, len(row), chunk_size):
        chunk = list(row[chunk_start:chunk_start + chunk_size])
        while len(chunk) < chunk_size:
            chunk.append(float("-inf"))
        chunk[0] = _cuda_logaddexp(chunk[0], carry)

        s = 1
        while s <= num_threads_x:
            new_chunk = list(chunk)
            for tid in range(num_threads_x):
                a = (tid // s) * (2 * s) + s
                ti = a + (tid % s)
                si = a - 1
                if ti < chunk_size and si < chunk_size:
                    new_chunk[ti] = _cuda_logaddexp(chunk[si], chunk[ti])
            chunk = new_chunk
            s *= 2

        actual_len = min(chunk_size, len(row) - chunk_start)
        output.extend(chunk[:actual_len])
        carry = chunk[chunk_size - 1]

    return output[:len(row)]


def spec_logcumsumexp_outer_sequential(data_2d: list, num_rows: int,
                                       num_cols: int):
    """Outer-dim logcumsumexp: purely sequential loop per thread.

    CUDA ref: cuda/ScanUtils.cuh:154."""
    output = list(data_2d)
    for col in range(num_cols):
        for row in range(1, num_rows):
            output[row * num_cols + col] = _cuda_logaddexp(
                output[(row - 1) * num_cols + col], data_2d[row * num_cols + col]
            )
    return output


# ============================================================================
# Simple sequential specs (pooling, embedding bag)
# ============================================================================

def spec_avg_pool_window(window_values: list):
    """AveragePool2d: sequential nested loop over kernel window.

    CUDA ref: cuda/AveragePool2d.cu:33 (avg_pool2d_out_cuda_frame)."""
    # Use input dtype for accumulator (e.g. np.float32 stays fp32)
    acc = type(window_values[0])(0)
    for v in window_values:
        acc = acc + v
    count = type(window_values[0])(len(window_values))
    return acc / count


def spec_embedding_bag_sum(embeddings: list):
    """EmbeddingBag SUM: sequential loop over bag, one thread per feature.

    CUDA ref: cuda/EmbeddingBag.cu:115 (EmbeddingBag_updateOutputKernel_sum_mean)."""
    acc = type(embeddings[0])(0)
    for emb in embeddings:
        acc = acc + emb
    return acc


# ---------------------------------------------------------------------------
# Shared pseudocode for the gpu_reduce_kernel framework (Reduce.cuh)
# ---------------------------------------------------------------------------
#
# All TensorIterator-based reductions (sum, prod, mean, std, var, nansum,
# norms) share the same reduction tree.  Only the combine function differs.
#
# === Block size heuristic (setReduceConfig) ===
#
#   lpow2(n) = largest power-of-2 <= n
#   max_threads = 512  (256 for complex<double>)
#
#   if reduction_on_fastest_striding_dimension:          # stride-1
#       dim0 = inputs_per_output   (reduction size)
#       dim1 = num_outputs
#   else:                                                # non-stride-1
#       dim0 = num_outputs
#       dim1 = inputs_per_output
#
#   W = block_width  = min(lpow2(dim0), 32)              # capped at warp size
#   H = block_height = min(lpow2(dim1), max_threads / W)
#   W = min(lpow2(dim0), max_threads / H)                # re-balance
#
#   if stride-1:
#       input_mult[BLOCK_X] = split_input(W)   -> step_input *= W
#       Adjacent warp lanes read adjacent input elements (coalesced).
#       block_x_reduce enabled (warp shuffles).
#   else:
#       output_mult[BLOCK_X] = split_output(W)
#       Adjacent warp lanes read independent outputs.
#       block_x_reduce disabled.
#
#   split_across_warps = (values_per_thread >= min(H*16, 256))
#   if split_across_warps:
#       input_mult[BLOCK_Y] = split_input(H)   -> step_input *= H
#       block_y_reduce enabled.
#   else:
#       output_mult[BLOCK_Y] = split_output(H)
#       Each thread handles ALL inputs for its output (pure sequential).
#
#   S = step_input  (= W*H for stride-1 with split, = H for non-stride-1)
#
#   Global reduce (ctas_per_output > 1) triggered when:
#       input_mult[BLOCK_Y] != 0
#       AND values_per_thread >= 256
#       AND grid.x <= num_SMs * (max_threads_per_SM / num_threads)
#       -> step_input *= ctas_per_output
#
# === Reduction tree (stride-1 case) ===
#
#   Phase 1 — Thread-local (vt0=4 independent accumulators):
#       for i in range(vt0):
#           acc[i] = identity
#       while idx + (vt0-1)*stride < end:
#           for i in range(vt0):
#               acc[i] = ops.reduce(acc[i], data[idx + i*stride], ...)
#           idx += stride * vt0
#       # merge accumulators left-to-right:
#       for i in range(1, vt0):
#           acc[0] = ops.combine(acc[0], acc[i])
#
#   Phase 2 — block_x_reduce (intra-block X dimension):
#       if W > 32:
#           # shared-memory halving: offset = W/2, W/4, ..., 32
#           for offset in [W//2, W//4, ..., warp_size]:
#               __syncthreads()
#               if threadIdx.x < offset:
#                   value = combine(value, shared[threadIdx.x + offset])
#       # intra-warp: shfl_down halving
#       # CUDA (non-ROCm):  offset = W/2, W/4, ..., 1   (high-to-low)
#       # ROCm/FBCODE:      offset = 1, 2, ..., W/2      (low-to-high)
#       for offset in warp_offsets:
#           other = warp_shfl_down(value, offset)
#           value = combine(value, other)
#
#   Phase 3 — block_y_reduce (cross-warp via shared memory):
#       shared[threadIdx.y] = value
#       for offset in [H/2, H/4, ..., 1]:
#           __syncthreads()
#           if threadIdx.y < offset:
#               value = combine(value, shared[threadIdx.y + offset])
#
#   Phase 4 — global_reduce (if ctas_per_output > 1):
#       staging[blockIdx.y] = block_result
#       __threadfence(); atomicAdd(semaphore)
#       if last_block:
#           for i in range(ctas_per_output):
#               value = combine(value, staging[i])   # serial scan
#           -> block_y_reduce -> block_x_reduce -> write output
#
# === Reduction tree (non-stride-1, split across warps) ===
#   Same as stride-1 except:
#   - Phase 2 is SKIPPED (no block_x_reduce, no warp shuffles)
#   - Phase 3 is the only inter-thread reduce
#   - S = H (not W*H), so each thread does more sequential work
#
# === Reduction tree (non-stride-1, no split) ===
#   - No inter-thread reduction at all
#   - Each thread: purely sequential loop over ALL inputs for one output


def spec_softmax_backward_inner(grad: list, output: list,
                                block_size: int = 1024) -> list:
    """Softmax backward (cunn_SoftMaxBackward): pre-multiply tmp=grad*output,
    reduce sum(tmp) via ilpReduce (ILP=4), then grad_input = tmp - output * sum.

    CUDA ref: cuda/SoftMax.cu:963 (cunn_SoftMaxBackward), :79 (epilogue)."""
    n = len(grad)
    B = block_size
    ILP = 4
    zero = type(grad[0])(0) if grad else 0.0

    # Pre-multiply (done outside kernel in softmax_backward_cuda_out)
    tmp = [grad[i] * output[i] for i in range(n)]

    # Phase 1: ilpReduce with ILP=4 consecutive elements per vector load
    # Matches forward's ilpReduce pattern (SoftMax.cu:487)
    last = n % (ILP * B)
    main_end = n - last
    thread_sums = [zero] * B
    for tid in range(B):
        acc = zero
        offset = tid
        while offset * ILP < main_end:
            for j in range(ILP):
                acc = acc + tmp[offset * ILP + j]
            offset += B
        tail_offset = main_end + tid
        while tail_offset < n:
            acc = acc + tmp[tail_offset]
            tail_offset += B
        thread_sums[tid] = acc

    # Phase 2: BlockReduceSum (block_reduce.cuh)
    row_sum = block_reduce_cuh(thread_sums, lambda a, b: a + b, zero)

    # Phase 3: epilogue: tmp[i] - output[i] * sum  (SoftMax.cu:86)
    return [tmp[i] - output[i] * row_sum for i in range(n)]


# NOTE: spec_softmax_backward_spatial is not provided because it is
# structurally identical to spec_softmax_spatial — a purely sequential
# per-thread loop computing sum(grad*output) for each inner position.
#
# NOTE: spec_softmax_backward_persistent is not provided because it uses
# the same SHFL_XOR butterfly tree as spec_softmax_persistent.


def spec_layer_norm_backward_dx(dY: list, X: list, mean: float, rstd: float,
                                gamma: list,
                                block_size: int = 256) -> list:
    """layer_norm_grad_input_kernel: two stats via sequential + BlockReduceSum,
    then elementwise dX.

    CUDA ref: cuda/layer_norm_kernel.cu:443 (layer_norm_grad_input_kernel)."""
    N = len(X)

    # Phase 1: per-thread sequential sums at stride block_size
    thread_x1 = [0.0] * block_size
    thread_x2 = [0.0] * block_size
    for tid in range(block_size):
        s1, s2 = 0.0, 0.0
        for idx in range(tid, N, block_size):
            dYg = dY[idx] * gamma[idx]
            s1 = s1 + dYg
            s2 = s2 + dYg * (X[idx] - mean) * rstd
        thread_x1[tid] = s1
        thread_x2[tid] = s2

    add = lambda a, b: a + b  # noqa: E731

    # Phase 2: BlockReduceSum (block_reduce.cuh) for both
    stats_x1 = block_reduce_cuh(thread_x1, add, 0.0)
    stats_x2 = block_reduce_cuh(thread_x2, add, 0.0)

    # Phase 3: elementwise dX
    fH = float(N)
    dX = []
    for i in range(N):
        x_hat = (X[i] - mean) * rstd
        dYg = dY[i] * gamma[i]
        dX.append(rstd * (dYg - (stats_x1 + x_hat * stats_x2) / fH))
    return dX


def spec_layer_norm_backward_dgamma(dY_rows: list, X_rows: list,
                                    mean_list: list, rstd_list: list,
                                    block_dim_y: int = 32) -> list:
    """GammaBetaBackwardCUDAKernel: reduce across M rows for each feature.
    dY_rows[m][j], X_rows[m][j] are M rows of N features each.
    Tree: sequential over assigned M rows, then SHFL_XOR butterfly across
    block_dim_y.

    NOTE: The real kernel also computes dbeta using the same tree but with
    acc += dY[m,j] (no multiplication by (X-mean)*rstd).  Omitted here
    since the reduction tree is identical — only the per-element combine
    differs.

    NOTE: For small M (< ~128 on ROCm, dispatched to
    GammaBetaBackwardSimpleCUDAKernel), the reduction is purely sequential
    per-thread with no inter-thread reduce at all.

    CUDA ref: cuda/layer_norm_kernel.cu:768 (GammaBetaBackwardCUDAKernelTemplate), :602 (GammaBetaBackwardSimpleCUDAKernel for small-M)."""
    M = len(dY_rows)
    N = len(dY_rows[0]) if M > 0 else 0
    # Infer zero from data dtype (np.float32 stays fp32)
    zero = type(dY_rows[0][0])(0) if M > 0 and N > 0 else 0.0

    dgamma = [zero] * N
    for j in range(N):
        thread_sums = [zero] * block_dim_y
        for tid in range(block_dim_y):
            acc = zero
            for m in range(tid, M, block_dim_y):
                acc = acc + dY_rows[m][j] * (X_rows[m][j] - mean_list[m]) * rstd_list[m]
            thread_sums[tid] = acc

        # Phase 2: SHFL_XOR butterfly across block_dim_y
        dgamma[j] = shfl_xor_butterfly(thread_sums, lambda a, b: a + b)
    return dgamma


def spec_scatter_add(dest_size: int, indices: list, src_values: list) -> list:
    """Scatter add: atomic adds in nondeterministic order.
    CUDA order depends on warp scheduling; we simulate with sequential scan.

    CUDA ref: cuda/ScatterGatherKernel.cu (scatter_reduce_kernel with ReduceAdd)."""
    dest = [0.0] * dest_size
    for i in range(len(indices)):
        dest[indices[i]] = dest[indices[i]] + src_values[i]
    return dest


def spec_embedding_bag_psw_backward(grad: list, weight: list, indices: list,
                                    embedding_dim: int,
                                    warp_size: int = 32) -> list:
    """EmbeddingBag per_sample_weights backward: dot(grad, weight[idx]) per
    sample.  One warp per sample.
    Tree: sequential at stride warp_size, then WarpReduceSum (shfl_down).

    CUDA ref: cuda/EmbeddingBag.cu:478 (_embedding_bag_per_sample_weights_backward_kernel)."""
    results = []
    for sample_idx, idx in enumerate(indices):
        # Phase 1: per-lane sequential sum at stride warp_size
        lane_sums = [0.0] * warp_size
        for lane in range(warp_size):
            acc = 0.0
            for feat in range(lane, embedding_dim, warp_size):
                acc = acc + grad[feat] * weight[idx][feat]
            lane_sums[lane] = acc

        # Phase 2: WarpReduceSum (shfl_down high-to-low)
        results.append(
            shfl_down_reduce_high_to_low(lane_sums, lambda a, b: a + b)
        )
    return results


class TestReduceSumProdKernel(TestCase):
    """ReduceSumProdKernel.cu — sum, nansum, prod, xor_sum

    Spec: spec_gpu_reduce_kernel(data, combine, identity, ...)
    CUDA ref: cuda/ReduceSumProdKernel.cu.

    Combine functions:
      sum:    a + b           identity = 0
      nansum: a + b           identity = 0   (reduce skips NaN)
      prod:   a * b           identity = 1   (bool: a && b)
    """

    def test_sum_stride1(self):
        # Reducing dim=-1 on contiguous tensor -> stride-1 path.
        # W=32, H=min(lpow2(100),512/32)=16, S=512, vpt=ceil(5000/512)=10
        # Tree: seq(4,512) -> shfl_down(32) -> shmem(16)
        x = torch.randn(100, 5000, device="cuda")
        result = torch.sum(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (gpu_reduce_kernel vectorized path)
        add = lambda a, b: a + b
        for i in range(result.shape[0]):
            row = list(x[i].cpu().numpy())
            spec_val = spec_gpu_reduce_kernel(row, add, np.float32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_nonstride1_split(self):
        # Reducing dim=0 on contiguous -> non-stride-1, split_across_warps.
        # Use prime num_outputs to avoid output vectorization (output_vec_size=1).
        # On GPUs with many SMs, global reduce (ctas_per_output > 1) is triggered.
        x = torch.randn(5000, 97, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: a + b
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            5000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        for i in range(min(4, result.shape[0])):
            col = list(x[:, i].cpu().numpy())
            if ctas > 1:
                spec_val = spec_gpu_reduce_kernel_global(
                    col, add, np.float32(0.0), num_outputs=97, stride1=False,
                    vt0=4, warp_size=32, max_threads=512, rocm=False,
                    W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                    vectorize=vec, ctas_per_output=ctas,
                )
            else:
                spec_val = spec_gpu_reduce_kernel(
                    col, add, np.float32(0.0), num_outputs=97, stride1=False,
                )
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_nonstride1_nosplit(self):
        # Small reduction on non-stride-1 dim -> no warp split.
        # dim0=100(outputs), dim1=10(inputs). vpt=10 < min(H*16,256).
        # Tree: purely sequential per thread, no inter-thread reduce.
        x = torch.randn(10, 100, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (non-stride-1, no split, purely sequential)
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            col = list(x[:, i].cpu().numpy())
            spec_val = spec_gpu_reduce_kernel(col, add, np.float32(0.0), num_outputs=100, stride1=False)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_global_reduce(self):
        # Very large stride-1 reduction -> triggers vectorized input + global
        # (multi-CTA) reduce. The spec_gpu_reduce_kernel_global doesn't model
        # the stride-1 vectorized + global combination, so we compare against
        # a smaller non-vectorized global reduce (prime num_outputs, non-stride-1).
        x = torch.randn(50000, 97, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: a + b
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            50000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        col = list(x[:, 0].cpu().numpy())
        if ctas > 1:
            spec_val = spec_gpu_reduce_kernel_global(
                col, add, np.float32(0.0), num_outputs=97, stride1=False,
                vt0=4, warp_size=32, max_threads=512, rocm=False,
                W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                vectorize=vec, ctas_per_output=ctas,
            )
        else:
            spec_val = spec_gpu_reduce_kernel(
                col, add, np.float32(0.0), num_outputs=97, stride1=False,
            )
        self.assertEqual(spec_val, result[0].item(), atol=0, rtol=0)

    def test_nansum(self):
        # Same tree as sum. Reduce step skips NaN, combine is still a+b.
        x = torch.randn(100, 5000, device="cuda")
        x[0, 0] = float("nan")
        result = torch.nansum(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # Spec: nansum treats NaN as 0 in the reduce step, then combines via a+b.
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            row = list(x[i].cpu().numpy())
            row_no_nan = [np.float32(0) if np.isnan(v) else v for v in row]
            spec_val = spec_gpu_reduce_kernel(row_no_nan, add, np.float32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_prod_stride1(self):
        # Same tree as sum, combine is a*b, identity=1.
        x = torch.randn(100, 5000, device="cuda")
        result = torch.prod(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (gpu_reduce_kernel vectorized path, multiply)
        mul = lambda a, b: a * b
        for i in range(min(4, result.shape[0])):
            row = list(x[i].cpu().numpy())
            spec_val = spec_gpu_reduce_kernel(row, mul, np.float32(1.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_prod_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.prod(x, dim=0)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (non-stride-1, split across warps, multiply)
        mul = lambda a, b: a * b
        for i in range(min(4, result.shape[0])):
            col = list(x[:, i].cpu().numpy())
            spec_val = spec_gpu_reduce_kernel(col, mul, np.float32(1.0), num_outputs=100, stride1=False)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)


class TestReduceMomentKernel(TestCase):
    """ReduceMomentKernel.cu — mean, std, var (Welford)

    Spec: spec_gpu_reduce_kernel with MeanOps (a+b) or WelfordOps combine.
    CUDA ref: cuda/ReduceMomentKernel.cu.

    Combine functions:
      mean:    a + b                    project: acc * (1/N)
      std/var: Welford 4-tuple merge    project: (sqrt(m2/divisor), mean)
    Same gpu_reduce_kernel tree as sum. Only combine function differs.
    """

    def test_mean_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.mean(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # mean uses same gpu_reduce_kernel tree as sum, with project *= 1/N
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            row = list(x[i].cpu().numpy())
            spec_sum = spec_gpu_reduce_kernel(row, add, np.float32(0.0), num_outputs=100, stride1=True)
            # MeanOps::project multiplies by precomputed factor = 1/N
            factor = 1.0 / len(row)
            spec_mean = spec_sum * factor
            self.assertEqual(spec_mean, result[i].item(), atol=0, rtol=0)

    def test_mean_nonstride1(self):
        # Use prime num_outputs to avoid output vectorization.
        x = torch.randn(5000, 97, device="cuda")
        result = torch.mean(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: a + b
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            5000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        for i in range(min(4, result.shape[0])):
            col = list(x[:, i].cpu().numpy())
            if ctas > 1:
                spec_sum = spec_gpu_reduce_kernel_global(
                    col, add, np.float32(0.0), num_outputs=97, stride1=False,
                    vt0=4, warp_size=32, max_threads=512, rocm=False,
                    W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                    vectorize=vec, ctas_per_output=ctas,
                )
            else:
                spec_sum = spec_gpu_reduce_kernel(
                    col, add, np.float32(0.0), num_outputs=97, stride1=False,
                )
            # MeanOps::project multiplies by precomputed factor = 1/N (not division)
            factor = 1.0 / len(col)
            spec_mean = spec_sum * factor
            self.assertEqual(spec_mean, result[i].item(), atol=0, rtol=0)

    def test_var_stride1(self):
        # Welford combine merges (mean, m2, n, nf) tuples.
        # Numerically stable across the parallel tree.
        # Spec: same tree as sum but with Welford combine; tested via sum
        x = torch.randn(100, 5000, device="cuda")
        result = torch.var(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # torch.var default is Bessel-corrected (correction=1); compare against np.var with ddof=1.
        # gpu_reduce_kernel uses WelfordOps::combine (not cuWelfordCombine) which
        # has FMA opportunities in the 4-tuple merge that we can't compile without
        # reimplementing the full WelfordOps specialization for gpu_reduce_kernel.
        ref = float(np.var(x[0].cpu().numpy(), ddof=1))
        self.assertEqual(ref, result[0].item(), atol=1e-6, rtol=0)

    def test_std_nonstride1(self):
        # Same Welford combine FMA gap as test_var_stride1.
        x = torch.randn(5000, 100, device="cuda")
        result = torch.std(x, dim=0)
        self.assertEqual(result.shape, (100,))
        ref = float(np.std(x[:, 0].cpu().numpy(), ddof=1))
        self.assertEqual(ref, result[0].item(), atol=1e-6, rtol=0)


class TestReduceNormKernel(TestCase):
    """ReduceNormKernel.cu — L1, L2, Lp, powsum norms

    Spec: spec_gpu_reduce_kernel with NormOps combine (a+b).
    CUDA ref: cuda/ReduceNormKernel.cu.

    All use combine: a + b (the reduce step differs by norm order).
      L1:  reduce a + |b|,             project: identity
      L2:  reduce a + b^2,             project: sqrt
      Lp:  reduce a + |b|^p,           project: ^(1/p)
    Same gpu_reduce_kernel tree.
    """

    def test_norm_l1_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.linalg.vector_norm(x, 1, dim=-1)
        self.assertEqual(result.shape, (100,))
        # L1 norm: reduce is a + |b|, combine is a + b, project is identity
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            row = list(x[i].cpu().numpy())
            abs_row = [np.float32(abs(v)) for v in row]
            spec_val = spec_gpu_reduce_kernel(abs_row, add, np.float32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_norm_l2_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.linalg.vector_norm(x, 2, dim=-1)
        self.assertEqual(result.shape, (100,))
        # L2 norm: reduce is a + b^2, combine is a + b, project is sqrt
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            row = list(x[i].cpu().numpy())
            sq_row = [v * v for v in row]
            spec_sumsq = spec_gpu_reduce_kernel(sq_row, add, np.float32(0.0), num_outputs=100, stride1=True)
            # Use CUDA sqrt to match kernel's device_sqrt
            spec_norm = torch.tensor(float(spec_sumsq), dtype=torch.float32, device="cuda").sqrt().item()
            self.assertEqual(spec_norm, result[i].item(), atol=0, rtol=0)

    def test_norm_lp_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.linalg.vector_norm(x, 3.0, dim=0)
        self.assertEqual(result.shape, (100,))
        # Lp norm: reduce is a + |b|^p, combine is a + b, project is ^(1/p)
        add = lambda a, b: a + b
        for i in range(min(4, result.shape[0])):
            col = list(x[:, i].cpu().numpy())
            abs_p_col = [np.float32(abs(v) ** 3) for v in col]
            spec_sum = spec_gpu_reduce_kernel(abs_p_col, add, np.float32(0.0), num_outputs=100, stride1=False)
            # Use CUDA pow to match kernel's device_pow
            spec_val = torch.tensor(float(spec_sum), dtype=torch.float32, device="cuda").pow(1.0 / 3.0).item()
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)


class TestSoftMax(TestCase):
    """SoftMax.cu + PersistentSoftmax.cuh — softmax, log_softmax

    Three dispatch paths for forward, same three for backward.

    === Persistent path (PersistentSoftmax.cuh) ===
    Triggered: dim_size <= 1024 AND dim_size * elem_size <= 4096
    Block: one warp per row. Entire row fits in registers.
    Tree: WARP_SHFL_XOR butterfly (all threads get result simultaneously).
      for offset in [WARP_SIZE/2, WARP_SIZE/4, ..., 1]:
          other = WARP_SHFL_XOR(val, offset)
          val = combine(val, other)

    === Inner-dim path (cunn_SoftMaxForward) ===
    Triggered: inner_size == 1 (reducing last dim) AND dim_size > 1024.
    Block: SoftMax_getBlockSize(ILP, dim_size) -> next pow2 up to 1024.
    Tree (for both max and sum-of-exp):
      ilpReduce: thread reads ILP (=4 for fp32) elements per vector load,
        at stride blockDim.x. All ILP accumulators merged left-to-right.
      blockReduceWarp:
        WarpReduce (BlockReduce from block_reduce.cuh):
          shfl_down halving high-to-low within each warp
        -> warp leaders write to shared memory
        -> warp 0 does second WarpReduce on per-warp results
        -> thread 0 writes to shared[0], all threads read back

    === Spatial path (cunn_SpatialSoftMaxForward) ===
    Triggered: inner_size > 1 (reducing non-last dim).
    Block: from SpatialSoftMax_getLaunchSizes.
    Tree: PURELY SEQUENTIAL per-thread loop over dim_size.
      Each thread handles independent inner positions. No inter-thread
      reduction for the max or sum. Just:
        for d in range(dim_size):
            acc = combine(acc, data[d * inner_size + my_inner_pos])

    Specs: spec_softmax_persistent, spec_softmax_inner, spec_softmax_spatial,
    spec_softmax_backward_inner.
    """

    def test_softmax_persistent(self):
        # dim_size=512 <= 1024 and 512*4=2048 <= 4096 -> persistent path.
        # One warp per row, SHFL_XOR butterfly.
        x = torch.randn(4, 64, device="cuda")
        result = torch.softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise match (CUDA exp via dtype=np.float32)
        for i in range(x.shape[0]):
            row = list(x[i].cpu().numpy())
            spec_out = spec_softmax_persistent(row, warp_size=32, dtype=np.float32)
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_softmax_inner_dim(self):
        # dim_size > max_smem_elements -> non-Smem inner path (cunn_SoftMaxForward).
        # Use dim=50000 to bypass cunn_SoftMaxForwardSmem (shared memory cache
        # variant), which has a subtly different ILP load pattern.
        # Block = 1024. ILP=4 consecutive elements per vectorized load.
        x = torch.randn(4, 50000, device="cuda")
        result = torch.softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise (CUDA exp, correct ILP grouping, non-Smem path)
        for i in range(x.shape[0]):
            row = list(x[i].cpu().numpy())
            spec_out = spec_softmax_inner(row, block_size=1024, dtype=np.float32)
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_softmax_spatial(self):
        # Reducing dim=0, inner_size=32 > 1 -> spatial path.
        # blockDim.x threads collaborate on dim reduction via spatialBlockReduceX.
        x = torch.randn(1000, 32, device="cuda")
        result = torch.softmax(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise (CUDA exp, correct spatialBlockReduceX tree)
        x_np = x.cpu().numpy()
        flat_data = [x_np[d, inner] for d in range(x.shape[0]) for inner in range(x.shape[1])]
        spec_out = spec_softmax_spatial(flat_data, dim_size=x.shape[0],
                                        inner_size=x.shape[1], dtype=np.float32)
        spec_t = torch.tensor(np.array(spec_out, dtype=np.float32)).reshape(x.shape)
        self.assertEqual(spec_t, result.cpu(), atol=0, rtol=0)

    def test_log_softmax_inner(self):
        # Same dispatch as softmax. log variant changes epilogue:
        #   LogSoftMaxForwardEpilogue: x - max - log(sum_exp)
        # Use dim=50000 to bypass Smem variant (same as softmax_inner test).
        x = torch.randn(4, 50000, device="cuda")
        result = torch.log_softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        # Reuse softmax_inner spec to get matching max and sum, then apply
        # log_softmax epilogue: output = x - max - log(sum_exp).
        exp = _dtype_exp(np.float32)
        cast = np.float32
        for i in range(x.shape[0]):
            row = list(x[i].cpu().numpy())
            n = len(row)
            B = 1024
            ILP = 4
            neg_inf = cast(float("-inf"))
            zero = cast(0.0)
            # ilpReduce for max (same as spec_softmax_inner)
            last = n % (ILP * B)
            main_end = n - last
            thread_maxes = [neg_inf] * B
            for tid in range(B):
                acc = neg_inf
                offset = tid
                while offset * ILP < main_end:
                    for j in range(ILP):
                        acc = max(acc, row[offset * ILP + j])
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = max(acc, row[tail_offset])
                    tail_offset += B
                thread_maxes[tid] = acc
            row_max = block_reduce_cuh(thread_maxes, max, neg_inf)
            # ilpReduce for sum(exp(x - max))
            exp_data = [exp(cast(v) - cast(row_max)) for v in row]
            thread_sums = [zero] * B
            for tid in range(B):
                acc = zero
                offset = tid
                while offset * ILP < main_end:
                    for j in range(ILP):
                        acc = acc + exp_data[offset * ILP + j]
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = acc + exp_data[tail_offset]
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = block_reduce_cuh(thread_sums, lambda a, b: a + b, zero)
            # Epilogue: x - max - log(sum) using CUDA log
            log_sum = torch.tensor(float(row_sum), dtype=torch.float32,
                                   device="cuda").log().item()
            spec_out = [cast(v) - cast(row_max) - cast(log_sum) for v in row]
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_softmax_backward_inner(self):
        # aten::_softmax_backward_data — same inner/spatial dispatch.
        # Computes sum(grad * output) using same tree as forward sum.
        # Use dim=50000 to bypass Smem variant (same as forward inner test).
        x = torch.randn(2, 50000, device="cuda")
        output = torch.softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(grad, output, -1, x.dtype)
        self.assertEqual(result.shape, x.shape)
        B = 1024
        ILP = 4
        for i in range(x.shape[0]):
            g = list(grad[i].cpu().numpy())
            o = list(output[i].cpu().numpy())
            n = len(g)
            cast = np.float32
            zero = cast(0.0)
            tmp = [cast(g[j]) * cast(o[j]) for j in range(n)]
            # ilpReduce for sum(tmp)
            last = n % (ILP * B)
            main_end = n - last
            thread_sums = [zero] * B
            for tid in range(B):
                acc = zero
                offset = tid
                while offset * ILP < main_end:
                    for j in range(ILP):
                        acc = acc + tmp[offset * ILP + j]
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = acc + tmp[tail_offset]
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = block_reduce_cuh(thread_sums, lambda a, b: a + b, zero)
            # Compiled epilogue for FMA: tmp - output * sum
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                t = torch.tensor(float(tmp[j]), device="cuda", dtype=torch.float32)
                ov = torch.tensor(float(o[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(t, ov, sum_t).item())
            self.assertEqual(
                torch.tensor(np.array(spec_gi, dtype=np.float32)),
                result[i].cpu(),
                atol=1e-6, rtol=0,
            )

    def test_softmax_backward_spatial(self):
        # Spatial backward uses the same spatialBlockReduceX tree as forward.
        # Epilogue: grad_input = tmp - output * sum, where tmp = grad * output.
        dim_size, inner_size = 1000, 32
        x = torch.randn(dim_size, inner_size, device="cuda")
        output = torch.softmax(x, dim=0)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(
            grad, output, 0, x.dtype
        )
        self.assertEqual(result.shape, x.shape)
        # Spec: pre-multiply tmp, reduce sum(tmp) per inner position via
        # shmem_halving_reduce (same dim_threads as spec_softmax_spatial),
        # then apply epilogue.
        grad_np = grad.cpu().numpy()
        out_np = output.cpu().numpy()
        cast = np.float32

        max_block = 1024
        inner_threads = min(inner_size, max_block)
        dim_threads = 1
        if inner_threads <= 64 and dim_size >= 64:
            while inner_threads * dim_threads <= max_block and dim_threads <= dim_size:
                dim_threads *= 2
            dim_threads //= 2

        spec_grad_input = np.zeros_like(grad_np)
        for inner in range(inner_size):
            tmp = [cast(grad_np[d, inner]) * cast(out_np[d, inner])
                   for d in range(dim_size)]
            if dim_threads == 1:
                s = cast(0.0)
                for d in range(dim_size):
                    s = s + tmp[d]
            else:
                thread_sums = [cast(0.0)] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_sums[tx] = thread_sums[tx] + tmp[d]
                s = shmem_halving_reduce(thread_sums, lambda a, b: a + b,
                                         cast(0.0))
            # Compiled epilogue for FMA: tmp - output * sum
            s_t = torch.tensor(float(s), device="cuda", dtype=torch.float32)
            for d in range(dim_size):
                tmp_t = torch.tensor(float(tmp[d]), device="cuda", dtype=torch.float32)
                out_t = torch.tensor(float(out_np[d, inner]), device="cuda", dtype=torch.float32)
                spec_grad_input[d, inner] = _compiled_softmax_backward_epilogue(
                    tmp_t, out_t, s_t).item()

        spec_t = torch.tensor(spec_grad_input)
        self.assertEqual(spec_t, result.cpu(), atol=0, rtol=0)

    def test_log_softmax_backward(self):
        # log_softmax backward: grad_input = grad - exp(output) * sum(grad)
        # Same ilpReduce + BlockReduceSum tree as softmax backward inner.
        # Use dim=50000 to bypass Smem variant.
        x = torch.randn(4, 50000, device="cuda")
        output = torch.log_softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._log_softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)
        exp = _dtype_exp(np.float32)
        B = 1024
        ILP = 4
        for i in range(x.shape[0]):
            g = list(grad[i].cpu().numpy())
            o = list(output[i].cpu().numpy())
            n = len(g)
            cast = np.float32
            zero = cast(0.0)
            # ilpReduce for sum(grad)
            last = n % (ILP * B)
            main_end = n - last
            thread_sums = [zero] * B
            for tid in range(B):
                acc = zero
                offset = tid
                while offset * ILP < main_end:
                    for j in range(ILP):
                        acc = acc + g[offset * ILP + j]
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = acc + g[tail_offset]
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = block_reduce_cuh(thread_sums, lambda a, b: a + b, zero)
            # Epilogue: grad - exp(output) * sum(grad)
            # exp(log_softmax_output) is computed via CUDA exp
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                exp_o = torch.tensor(float(o[j]), device="cuda",
                                     dtype=torch.float32).exp()
                g_t = torch.tensor(float(g[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(g_t, exp_o, sum_t).item())
            self.assertEqual(
                torch.tensor(np.array(spec_gi, dtype=np.float32)),
                result[i].cpu(),
                atol=1e-6, rtol=0,
            )

    def test_softmax_backward_persistent(self):
        # dim_size=512 -> persistent path for backward too.
        # Same SHFL_XOR butterfly tree as persistent forward for sum(grad*output).
        # No exp() involved, so bitwise match expected.
        warp_size = 32
        x = torch.randn(32, 512, device="cuda")
        output = torch.softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)
        # Spec: pre-multiply tmp = grad*output, per-lane sequential sum (ILP
        # grouping like forward), butterfly for sum, epilogue: tmp - output*sum.
        for i in range(x.shape[0]):
            g = list(grad[i].cpu().numpy())
            o = list(output[i].cpu().numpy())
            n = len(g)
            cast = np.float32
            zero = cast(0.0)

            tmp = [cast(g[j]) * cast(o[j]) for j in range(n)]

            # Per-lane sequential sum at stride warp_size
            lane_sums = [zero] * warp_size
            for lane in range(warp_size):
                acc = zero
                for it_idx in range(lane, n, warp_size):
                    acc = acc + tmp[it_idx]
                lane_sums[lane] = acc

            row_sum = shfl_xor_butterfly_high_to_low(
                lane_sums, lambda a, b: a + b
            )

            # Compiled epilogue for FMA: tmp - output * sum
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                t = torch.tensor(float(tmp[j]), device="cuda", dtype=torch.float32)
                ov = torch.tensor(float(o[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(t, ov, sum_t).item())
            self.assertEqual(
                torch.tensor(np.array(spec_gi, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_log_softmax_spatial(self):
        # log_softmax on non-last dim -> spatial path (sequential).
        # LogSoftMaxForwardEpilogue: output = x - max - log(sum_exp).
        # Same spatial tree as softmax_spatial for max/sum_exp.
        x = torch.randn(1000, 32, device="cuda")
        result = torch.log_softmax(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        exp = _dtype_exp(np.float32)
        cast = np.float32
        x_np = x.cpu().numpy()
        dim_size, inner_size = x.shape
        # Compute dim_threads matching SpatialSoftMax_getBlockSize
        max_block = 1024
        inner_threads = min(inner_size, max_block)
        dim_threads = 1
        if inner_threads <= 64 and dim_size >= 64:
            while inner_threads * dim_threads <= max_block and dim_threads <= dim_size:
                dim_threads *= 2
            dim_threads //= 2
        for inner in range(inner_size):
            if dim_threads == 1:
                m = cast(float("-inf"))
                for d in range(dim_size):
                    m = max(m, x_np[d, inner])
                s = cast(0.0)
                for d in range(dim_size):
                    s = s + exp(cast(x_np[d, inner]) - cast(m))
            else:
                neg_inf = cast(float("-inf"))
                thread_maxes = [neg_inf] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_maxes[tx] = max(thread_maxes[tx], x_np[d, inner])
                m = shmem_halving_reduce(thread_maxes, max, neg_inf)
                zero = cast(0.0)
                thread_sums = [zero] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_sums[tx] = thread_sums[tx] + exp(
                            cast(x_np[d, inner]) - cast(m))
                s = shmem_halving_reduce(thread_sums, lambda a, b: a + b, zero)
            # CUDA log for log(sum_exp)
            log_s = torch.tensor(float(s), dtype=torch.float32,
                                 device="cuda").log().item()
            for d in range(dim_size):
                spec_val = cast(x_np[d, inner]) - cast(m) - cast(log_s)
                self.assertEqual(float(spec_val), result[d, inner].item(),
                                 atol=0, rtol=0)


class TestLayerNormKernel(TestCase):
    """layer_norm_kernel.cu — layer_norm, rms_norm

    === Forward moments (RowwiseMomentsCUDAKernel) ===
    Block: 256 threads (kCUDABlockReduceNumThreads). One block per row.
    Tree:
      Thread-local: sequential Welford accumulation at stride 256.
        for idx in range(threadIdx.x, N, 256):
            welford_update(acc, X[row, idx])
      BlockReduce (from block_reduce.cuh):
        WarpReduce: shfl_down(16,8,4,2,1) within each warp
        -> warp leaders write (mean,m2,n,nf) to shared[warp_id]
        -> warp 0 loads shared[0..7] and does second WarpReduce
      Result: thread 0 has final (mean, rstd).

    RMS norm variant: same tree but accumulates sum(x^2) instead of
    Welford. No mean subtraction.

    === Backward dX (layer_norm_grad_input_kernel) ===
    Block: 256 or 512 threads. One block per row.
    Tree:
      Thread-local: sequential sum at stride blockDim.x for two stats:
        stats_x1 += dY[i] * gamma[i]
        stats_x2 += dY[i] * gamma[i] * (X[i] - mean) * rstd
      BlockReduceSum (from block_reduce.cuh): same 2-level tree.

    === Backward dgamma/dbeta (GammaBetaBackwardCUDAKernel) ===
    Reduces across M (batch dimension) for each of N features.
    Block: 2D, threadIdx.x=32 (features), threadIdx.y from {1,8,16,32}.
    getRowwiseReduceShapeAndBlockDim selects block_dim_y based on M:
      if M <= 32: block_dim_y = 1 (no reduction, sequential only)
      elif M <= 128: block_dim_y = 8
      elif M <= 512: block_dim_y = 16
      else: block_dim_y = 32
    Tree:
      Thread-local: sequential sum over assigned M rows.
        for m in range(my_m_start, my_m_end):
            dg_sum += dY[m,j] * (X[m,j] - mean[m]) * rstd[m]
            db_sum += dY[m,j]
      Write to shared (transposed for bank-conflict avoidance).
      WARP_SHFL_XOR butterfly across block_dim_y:
        for delta in [block_dim_y/2, ..., 1]:
            dg_sum += WARP_SHFL_XOR(dg_sum, delta)

    === Backward dgamma/dbeta (GammaBetaBackwardSimpleCUDAKernel) ===
    Triggered for small M (< threshold, M < 128 on ROCm).
    Block: kCUDANumThreads (256). One thread per feature.
    Tree: PURELY SEQUENTIAL loop over M. No inter-thread reduction.
      for m in range(M):
          dg_sum += dY[m,j] * (X[m,j] - mean[m]) * rstd[m]

    Specs: spec_layer_norm_forward_moments, spec_layer_norm_backward_dx,
    spec_layer_norm_backward_dgamma.
    """

    def test_layer_norm_forward(self):
        N = 512
        x = torch.randn(2, N, device="cuda")
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        y, mean_t, rstd_t = torch.ops.aten.native_layer_norm(x, [N], w, b, 1e-5)
        self.assertEqual(y.shape, x.shape)
        # Spec: bitwise — compiled Welford FMA for both reduce and combine
        for i in range(x.shape[0]):
            spec_mean, spec_rstd = spec_layer_norm_forward_moments(
                x[i], block_size=512, eps=1e-5,
            )
            self.assertEqual(spec_mean, mean_t[i].item(), atol=0, rtol=0)
            self.assertEqual(spec_rstd, rstd_t[i].item(), atol=0, rtol=0)

    def test_layer_norm_backward_dx(self):
        x = torch.randn(100, 5000, device="cuda")
        w = torch.randn(5000, device="cuda")
        b = torch.randn(5000, device="cuda")
        mean = x.mean(dim=-1)
        rstd = (1.0 / x.std(dim=-1, unbiased=False)).to(x.dtype)
        grad = torch.randn_like(x)
        # native_layer_norm_backward returns (dX, dgamma, dbeta)
        dx, dw, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [5000], mean, rstd, w, b, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        row_idx = 0
        dY = list(grad[row_idx].cpu().numpy())
        X = list(x[row_idx].cpu().numpy())
        m = mean[row_idx].item()
        r = rstd[row_idx].item()
        gamma = list(w.cpu().numpy())
        spec_dx = spec_layer_norm_backward_dx(dY, X, m, r, gamma, block_size=256)
        # FMA gap: per-thread accumulation of dYg * (X-mean) * rstd has FMA
        # candidates that nvcc compiles but our Python loop doesn't capture.
        self.assertEqual(
            torch.tensor(spec_dx),
            dx[row_idx].cpu(),
            atol=1e-6, rtol=1e-5,
        )

    def test_layer_norm_backward_dgamma_large_M(self):
        # M=1000 -> block_dim_y=32, SHFL_XOR butterfly reduction.
        x = torch.randn(1000, 256, device="cuda")
        w = torch.randn(256, device="cuda")
        b = torch.randn(256, device="cuda")
        mean = x.mean(dim=-1)
        rstd = (1.0 / x.std(dim=-1, unbiased=False)).to(x.dtype)
        grad = torch.randn_like(x)
        _, dw, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [256], mean, rstd, w, b, [False, True, True]
        )
        self.assertEqual(dw.shape, w.shape)
        dY_np = grad.cpu().numpy()
        X_np = x.cpu().numpy()
        dY_rows = [list(dY_np[m]) for m in range(dY_np.shape[0])]
        X_rows = [list(X_np[m]) for m in range(X_np.shape[0])]
        mean_list = list(mean.cpu().numpy())
        rstd_list = list(rstd.cpu().numpy())
        spec_dg = spec_layer_norm_backward_dgamma(
            dY_rows, X_rows, mean_list, rstd_list, block_dim_y=32,
        )
        # FMA gap in per-thread dY*(X-mean)*rstd accumulation over M=1000 rows,
        # compounded through SHFL_XOR butterfly across block_dim_y=32 threads.
        self.assertEqual(
            torch.tensor(spec_dg),
            dw.cpu(),
            atol=1e-4, rtol=1e-3,
        )

    def test_layer_norm_backward_dgamma_small_M(self):
        # M=8 -> GammaBetaBackwardSimpleCUDAKernel, purely sequential per feature.
        M, N = 8, 5000
        x = torch.randn(M, N, device="cuda")
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        mean = x.mean(dim=-1)
        rstd = (1.0 / x.std(dim=-1, unbiased=False)).to(x.dtype)
        grad = torch.randn_like(x)
        _, dw, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [N], mean, rstd, w, b, [False, True, True]
        )
        self.assertEqual(dw.shape, w.shape)
        # Spec: bitwise — sequential loop per feature j:
        #   dgamma[j] = sum_m dY[m,j] * (X[m,j] - mean[m]) * rstd[m]
        dY_np = grad.cpu().numpy()
        X_np = x.cpu().numpy()
        mean_np = mean.cpu().numpy()
        rstd_np = rstd.cpu().numpy()
        for j in range(N):
            acc = np.float32(0.0)
            for m in range(M):
                acc = acc + np.float32(
                    np.float32(dY_np[m, j])
                    * np.float32(np.float32(X_np[m, j]) - np.float32(mean_np[m]))
                    * np.float32(rstd_np[m])
                )
            # FMA gap: dY*(X-mean)*rstd chain compiled by nvcc but not by Python
            self.assertEqual(float(acc), dw[j].item(), atol=1e-6, rtol=0)

    def test_rms_norm_forward(self):
        N = 512
        x = torch.randn(2, N, device="cuda")
        w = torch.randn(N, device="cuda")
        result, rstd_cuda = torch._fused_rms_norm(x, [N], w, 1e-5)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise — compiled FMA for sigma2 += val*val, CUDA rsqrt
        for i in range(x.shape[0]):
            spec_rstd = spec_rms_norm_forward_vectorized(
                x[i], eps=1e-5, num_threads=512,
            )
            self.assertEqual(spec_rstd, rstd_cuda[i].item(), atol=0, rtol=0)

    def test_rms_norm_backward(self):
        N = 512
        x = torch.randn(2, N, device="cuda")
        w = torch.randn(N, device="cuda")
        result, rstd_cuda = torch._fused_rms_norm(x, [N], w, 1e-5)
        grad = torch.randn_like(x)
        dx_cuda, dw_cuda = torch.ops.aten._fused_rms_norm_backward(
            grad, x, [N], rstd_cuda, w, [True, True]
        )
        self.assertEqual(dx_cuda.shape, x.shape)
        self.assertEqual(dw_cuda.shape, w.shape)
        # Spec: sequential + BlockReduceSum for dX
        row_idx = 0
        dY = list(grad[row_idx].cpu().numpy())
        X = list(x[row_idx].cpu().numpy())
        r = rstd_cuda[row_idx].item()
        gamma = list(w.cpu().numpy())
        spec_dx = spec_rms_norm_backward_dx(dY, X, r, gamma, block_size=256)
        # FMA gap: dY*gamma*X*rstd per-thread accumulation compiled by nvcc
        self.assertEqual(
            torch.tensor(spec_dx),
            dx_cuda[row_idx].cpu(),
            atol=1e-6, rtol=1e-5,
        )


class TestGroupNormKernel(TestCase):
    """group_norm_kernel.cu
    CUDA ref: cuda/group_norm_kernel.cu.

    === Forward (RowwiseMomentsCUDAKernel) ===
    Block: 512 threads (kCUDABlockReduceNumThreads). One block per (N, group).
    Reduction over D/G * H * W elements per group.
    Tree: identical to layer norm forward — sequential Welford at stride 512,
      then BlockReduce: shfl_down(32) -> shmem(512/32=16) -> warp 0 final.

    === Backward ===
    Same BlockReduceSum pattern for ds, db accumulators.

    No separate spec — use spec_layer_norm_forward_moments for forward
    (identical Welford + BlockReduce tree) and block_reduce_cuh for backward.
    """

    def test_group_norm_forward(self):
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        result = torch.nn.functional.group_norm(x, num_groups=8, weight=w, bias=b)
        self.assertEqual(result.shape, x.shape)
        # Group norm uses WelfordOps (delta form), NOT cuWelfordCombine (weighted).
        # BlockReduce: shfl_down → shmem → warp 0 shfl_down. block_size=512.
        _, mean_t, rstd_t = torch.ops.aten.native_group_norm(
            x, w, b, 8, 32, 256, 8, 1e-5
        )
        group_data = x[0, :4, :, :].contiguous().view(-1)  # (1024,) on CUDA
        block_size = 512
        warp_size = 32
        num_warps = block_size // warp_size
        tw = []
        for tid in range(block_size):
            m, s2, c = _compiled_welford_ops_reduce(group_data, tid, block_size)
            tw.append((np.float32(m.item()), np.float32(s2.item()), np.float32(c.item())))
        wid = (np.float32(0), np.float32(0), np.float32(0))
        def cw(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda") for v in list(a) + list(b)]
            m, s2, c = _compiled_welford_ops_combine(*ta)
            return (np.float32(m.item()), np.float32(s2.item()), np.float32(c.item()))
        wr = []
        for w_id in range(num_warps):
            wr.append(shfl_down_reduce_high_to_low(
                tw[w_id * warp_size:(w_id + 1) * warp_size], cw))
        while len(wr) < warp_size:
            wr.append(wid)
        final = shfl_down_reduce_high_to_low(wr[:warp_size], cw)
        ms, s2s, cs = final
        rstd_s = torch.rsqrt(
            torch.tensor(float(s2s), device="cuda") / torch.tensor(float(cs), device="cuda")
            + torch.tensor(1e-5, device="cuda")
        ).item()
        self.assertEqual(float(np.float32(ms)), mean_t[0, 0].item(), atol=0, rtol=0)
        self.assertEqual(rstd_s, rstd_t[0, 0].item(), atol=0, rtol=0)

    def test_group_norm_backward(self):
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        y, mean, rstd = torch.ops.aten.native_group_norm(x, w, b, 8, 32, 256, 8, 1e-5)
        grad = torch.randn_like(y)
        dx, dw, db = torch.ops.aten.native_group_norm_backward(
            grad, x, mean, rstd, w, 8, 32, 256, 8, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        # Same BlockReduceSum pattern as layer norm backward.
        # Verify dw for channel 0 against CPU reference to avoid CUDA tree diffs.
        ch = 0
        group_idx = ch // 4  # 32 channels / 8 groups = 4 channels/group
        ref_dw = 0.0
        for bi in range(8):
            ref_dw += float((grad[bi, ch].cpu() * (x[bi, ch].cpu() - mean[bi, group_idx].cpu()) * rstd[bi, group_idx].cpu()).sum().item())
        # FMA gap in per-thread dY*(X-mean)*rstd accumulation
        self.assertEqual(ref_dw, dw[ch].item(), atol=1e-5, rtol=0)


class TestNormalization(TestCase):
    """Normalization.cuh — batch norm (NCHW and NHWC)

    === NCHW forward stats (batch_norm_collect_statistics_kernel) ===
    Block: getNumThreads(N*H*W), up to 1024. 2D: threadIdx.x strides
      spatial, threadIdx.y strides batch. One block per channel.
    Tree:
      Thread-local: sequential Welford over (batch, spatial) at 2D stride.
      Warp reduce: WARP_SHFL_XOR butterfly, offsets 1,2,4,8,16
        (low-to-high — different from block_reduce.cuh's shfl_down!)
        for i in range(log2(32)):
            o_avg = WARP_SHFL_XOR(avg, 1 << i)
            welford_merge(avg, var_n, n, o_avg, o_var_n, o_n)
      Cross-warp: warp leaders write to shared -> warp 0 loads
        -> second WARP_SHFL_XOR butterfly.
    No multi-CTA. Single block per channel.

    === NHWC forward stats (batch_norm_collect_statistics_channels_last_kernel) ===
    Block: from flexible_launch_configs:
      block.x = min(lpow2(C), 32)         # channels (stride-1, independent)
      block.y = min(lpow2(N*HW/16), max_block/block.x)  # reduction
      grid.y  = min(N*HW/(block.y*16), 128)
      grid.y set to 1 if < 8 (skip grid reduce for small reductions)
    Tree:
      Thread-local: 4x register unroll (ELEMENTS_PER_ITER=4) of Welford.
        Merge 4 partial accumulators within thread.
      welford_merge_block_vertical: shared-memory halving in Y ONLY.
        for offset in [block.y/2, ..., 1]:
            shmem write, sync, load partner, welford_merge
        NO warp shuffles. threadIdx.x are independent (different channels).
      If grid.y > 1: staging buffer + atomic semaphore.
        Last CTA reads all staged results in serial loop, then re-applies
        welford_merge_block_vertical.

    === NCHW backward (reduce<Float2> via batch_norm_backward_kernel) ===
    Dual accumulation: Float2{sum_dy, sum_dy_xmu}.
    Same SHFL_XOR butterfly as NCHW forward, but with Float2 component-wise
    combine (v1+v1, v2+v2).

    === NHWC backward (batch_norm_backward_reduce_channels_last_kernel) ===
    Same Y-only shared-mem halving + optional multi-CTA as NHWC forward.

    Specs: spec_batch_norm_nchw_stats, spec_batch_norm_nhwc_stats,
    spec_batch_norm_nchw_backward_reduce cover the three distinct trees.
    NOTE: spec_batch_norm_nhwc_backward_reduce is not provided; it is the
    same Y-only shmem_halving_reduce as spec_batch_norm_nhwc_stats but with
    Float2 dual-sum (a+b component-wise) instead of Welford merge.
    """

    def test_batch_norm_nchw_forward(self):
        # Small tensor: 8 batch, 4 channels, 4×4 spatial
        x = torch.randn(8, 4, 4, 4, device="cuda")
        rm = torch.zeros(4, device="cuda")
        rv = torch.ones(4, device="cuda")
        _, mean_cuda, invstd_cuda = torch.ops.aten.native_batch_norm(
            x, None, None, rm, rv, True, 0.1, 1e-5
        )
        # NCHW kernel uses:
        #   Per-thread: avg += d1/n (division), var_n += d1*(v-avg) [lines 327-330]
        #   Combine: welford_merge_element formula (factor-based weighted mean)
        #     factor = 1/max(1, n+o_n)
        #     avg = (n*avg + o_n*o_avg) * factor  [NOT delta form!]
        #   Tree: SHFL_XOR butterfly (low-to-high) + shmem + warp 0 XOR
        ch = 0
        x3d = x.view(8, 4, 16)
        tf = 32  # getNumThreads(spatial=16) = 32
        block_x, block_y = tf, max(1, 1024 // tf)
        tw = []
        for ty in range(block_y):
            for tx in range(block_x):
                m, s2, c = _compiled_nchw_welford_reduce(
                    x3d, ch, ty, block_y, tx, block_x)
                tw.append((np.float32(m.item()), np.float32(s2.item()),
                           np.float32(c.item())))
        def cw(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda")
                  for v in [a[0], a[1], a[2], b[0], b[1], b[2]]]
            m, s2, c = _compiled_nchw_welford_merge(*ta)
            return (np.float32(m.item()), np.float32(s2.item()),
                    np.float32(c.item()))
        wid = (np.float32(0), np.float32(0), np.float32(0))
        warp_size = 32
        num_warps = (block_x * block_y) // warp_size
        wr = []
        for w in range(num_warps):
            wr.append(shfl_xor_butterfly_low_to_high(
                tw[w * warp_size:(w + 1) * warp_size], cw))
        while len(wr) < warp_size:
            wr.append(wid)
        final = shfl_xor_butterfly_low_to_high(wr[:warp_size], cw)
        ms, s2s, cs = final
        invstd_s = torch.rsqrt(torch.tensor(
            float(np.float32(s2s) / np.float32(cs) + np.float32(1e-5)),
            device="cuda", dtype=torch.float32)).item()
        self.assertEqual(float(np.float32(ms)), mean_cuda[ch].item(), atol=0, rtol=0)
        # bitwise — fixed association: var_n + (o_var_n + cross) not (var_n + o_var_n) + cross
        self.assertEqual(invstd_s, invstd_cuda[ch].item(), atol=0, rtol=0)

    def test_batch_norm_nhwc_forward(self):
        # Use small tensor so grid.y=1 (no multi-CTA staging)
        x = torch.randn(8, 4, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        rm = torch.zeros(4, device="cuda")
        rv = torch.ones(4, device="cuda")
        _, mean_cuda, invstd_cuda = torch.ops.aten.native_batch_norm(
            x, None, None, rm, rv, True, 0.1, 1e-5
        )
        # NHWC uses PARALLEL_LOADS=4 Welford accumulators per thread,
        # welford_merge_element combine, welford_merge_block_vertical tree.
        # 2 ULP tolerance: 1 seed in ~10 has 1 ULP invstd diff from
        # Triton-vs-nvcc FMA in the per-accumulator Welford reduce.
        ch = 0
        x_ch = x[:, ch, :, :].contiguous().view(-1)
        reduction_size = x_ch.shape[0]
        block_y = 8
        inner_loop_stride = block_y
        PARALLEL_LOADS = 4
        tw = []
        for ty in range(block_y):
            m_off = ty
            per_acc = [[] for _ in range(PARALLEL_LOADS)]
            lc = 1 + (reduction_size - 1) // (inner_loop_stride * PARALLEL_LOADS)
            for i in range(lc):
                for j in range(PARALLEL_LOADS):
                    if m_off < reduction_size:
                        per_acc[j].append(m_off)
                    m_off += inner_loop_stride
            accs = []
            for j in range(PARALLEL_LOADS):
                if per_acc[j]:
                    t = x_ch[torch.tensor(per_acc[j], device="cuda")]
                    m, s2, c = _compiled_nhwc_welford_reduce(t, 0, 1, len(per_acc[j]))
                    accs.append((m.item(), s2.item(), float(len(per_acc[j]))))
                else:
                    accs.append((0.0, 0.0, 0.0))
            mr = torch.tensor(accs[0][0], device="cuda")
            s2r = torch.tensor(accs[0][1], device="cuda")
            cr = torch.tensor(float(accs[0][2]), device="cuda")
            for j in range(1, PARALLEL_LOADS):
                mn = torch.tensor(accs[j][0], device="cuda")
                s2n = torch.tensor(accs[j][1], device="cuda")
                cn = torch.tensor(float(accs[j][2]), device="cuda")
                mr, s2r, cr = _compiled_nhwc_welford_merge(mr, s2r, cr, mn, s2n, cn)
            tw.append((np.float32(mr.item()), np.float32(s2r.item()), np.float32(cr.item())))
        # welford_merge_block_vertical
        def merge_w(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda") for v in [a[0],a[1],a[2],b[0],b[1],b[2]]]
            m, s2, c = _compiled_nhwc_welford_merge(*ta)
            return (np.float32(m.item()), np.float32(s2.item()), np.float32(c.item()))
        wrs = list(tw)
        off = len(wrs) // 2
        while off > 0:
            for wy in range(off):
                wrs[wy] = merge_w(wrs[wy], wrs[wy + off])
            off //= 2
        ms, s2s, cs = wrs[0]
        invstd_s = torch.rsqrt(torch.tensor(
            float(np.float32(s2s)/np.float32(cs) + np.float32(1e-5)),
            device="cuda", dtype=torch.float32)).item()
        self.assertEqual(float(np.float32(ms)), mean_cuda[ch].item(), atol=0, rtol=0)
        self.assertEqual(invstd_s, invstd_cuda[ch].item(), atol=0, rtol=0)

    def test_batch_norm_nchw_backward(self):
        x = torch.randn(32, 64, 8, 8, device="cuda")
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        sm = x.mean(dim=(0, 2, 3))
        si = 1.0 / x.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        dx, dw, db = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        # Float2 XOR butterfly for NCHW backward. db = sum_dy from the kernel.
        ch = 0
        grad_ch = list(grad[:, ch, :, :].contiguous().view(-1).cpu().numpy())
        input_ch = list(x[:, ch, :, :].contiguous().view(-1).cpu().numpy())
        mean_ch = sm[ch].item()
        spec_sum_dy, spec_sum_dy_xmu = spec_batch_norm_nchw_backward_reduce(
            grad_ch, input_ch, mean_ch, block_size=512,
        )
        # db[ch] IS sum_dy from the kernel, not from a separate gpu_reduce_kernel.
        # FMA gap: the per-thread Float2 accumulation dy_xmu = dy*(x-mean) has
        # FMA opportunity that our Python loop doesn't capture.
        self.assertEqual(
            torch.tensor(spec_sum_dy),
            torch.tensor(db[ch].item()),
            atol=1e-6, rtol=1e-5,
        )

    def test_batch_norm_nhwc_backward(self):
        x = torch.randn(32, 64, 8, 8, device="cuda").to(
            memory_format=torch.channels_last
        )
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        sm = x.to(memory_format=torch.contiguous_format).mean(dim=(0, 2, 3))
        si = 1.0 / x.to(memory_format=torch.contiguous_format).std(
            dim=(0, 2, 3), unbiased=False
        )
        grad = torch.randn_like(x)
        dx, dw, db = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        # db = sum_dy from the NHWC backward kernel. Compare against CPU sum
        # to avoid comparing two different CUDA reduction trees.
        ch = 0
        grad_ch_cpu = grad.to(memory_format=torch.contiguous_format)[:, ch, :, :].cpu()
        ref_sum_dy = float(grad_ch_cpu.sum().item())
        # Float2 shmem halving combine is component-wise addition (no FMA gap
        # for sum_dy itself). The gap is only in sum_dy_xmu = dy*(x-mean).
        self.assertEqual(ref_sum_dy, db[ch].item(), atol=1e-5, rtol=0)


class TestLoss(TestCase):
    """Loss.cu — nll_loss (also used by cross_entropy)

    Spec: spec_nll_loss_reduce.

    Block: fixed (typically 256 or 512 NLL_LOSS_THREADS).
    Tree (nll_loss_forward_reduce_cuda_kernel_2d):
      Thread-local: sequential sum at stride blockDim.x:
        for i in range(threadIdx.x, nframe, blockDim.x):
            sh_inputs[threadIdx.x] -= input[i, target[i]] * weight
      Block reduce: shared-memory halving tree (NO warp shuffles):
        for stride in [blockDim.x/2, blockDim.x/4, ..., 1]:
            __syncthreads()
            if threadIdx.x < stride:
                sh_inputs[threadIdx.x] += sh_inputs[threadIdx.x + stride]
    """

    def test_nll_loss(self):
        x = torch.randn(1000, 100, device="cuda").log_softmax(dim=-1)
        target = torch.randint(0, 100, (1000,), device="cuda")
        result = torch.nn.functional.nll_loss(x, target, reduction="sum")
        self.assertEqual(result.shape, ())
        # Spec: bitwise match (shmem halving tree for NLL loss)
        # Block size from nll_loss_threads: clamp(1<<round(log2(nframe/16)), 32, 1024)
        nframe = x.shape[0]
        nthreads = max(32, min(1024, 1 << round(math.log2(nframe / 16))))
        losses = list(-x[torch.arange(nframe, device="cuda"), target].cpu().numpy())
        spec_val = spec_nll_loss_reduce(losses, block_size=nthreads)
        self.assertEqual(spec_val, result.item(), atol=0, rtol=0)

    def test_cross_entropy(self):
        # cross_entropy = log_softmax + nll_loss
        # Compare CUDA result against CPU (same composite op, deterministic).
        x = torch.randn(1000, 100, device="cuda")
        target = torch.randint(0, 100, (1000,), device="cuda")
        result = torch.nn.functional.cross_entropy(x, target)
        self.assertEqual(result.shape, ())
        ref = torch.nn.functional.cross_entropy(x.cpu(), target.cpu())
        self.assertEqual(ref.item(), result.item(), atol=1e-5, rtol=0)


class TestNLLLoss2d(TestCase):
    """NLLLoss2d.cu — nll_loss for 4D input

    Same shared-memory halving tree as Loss.cu.
    No separate spec — use spec_nll_loss_reduce.
    """

    def test_nll_loss_2d(self):
        x = torch.randn(8, 10, 32, 32, device="cuda").log_softmax(dim=1)
        target = torch.randint(0, 10, (8, 32, 32), device="cuda")
        result = torch.nn.functional.nll_loss(x, target, reduction="mean")
        self.assertEqual(result.shape, ())
        # Compare against sum reduction computed via spec_nll_loss_reduce.
        # NLLLoss2d uses the same shmem halving tree as Loss.cu.
        result_sum = torch.nn.functional.nll_loss(x, target, reduction="sum")
        nframe = x.shape[0] * x.shape[2] * x.shape[3]
        losses = []
        x_np = x.cpu().numpy()
        t_np = target.cpu().numpy()
        for bi in range(8):
            for h in range(32):
                for w in range(32):
                    losses.append(np.float32(-x_np[bi, t_np[bi, h, w], h, w]))
        # NLLLoss2d uses a different kernel with different tree than 1D NLL loss.
        # Block size varies. Compare spec sum against CUDA sum.
        nthreads = max(32, min(1024, 1 << round(math.log2(nframe / 16))))
        spec_val = spec_nll_loss_reduce(losses, block_size=nthreads)
        self.assertEqual(float(spec_val), result_sum.item(), atol=1e-2, rtol=0)


class TestMultiMarginLoss(TestCase):
    """MultiMarginLoss.cu

    Spec: spec_multi_margin_loss_thread0_scan.

    Block: MULTIMARGIN_THREADS = 128.
    Tree:
      Thread-local: sequential sum at stride 128:
        for i in range(threadIdx.x, n_classes, 128):
            buffer[threadIdx.x] += max(0, margin - x[target] + x[i])^p
      Block reduce: THREAD 0 SERIAL SCAN (not a parallel tree!):
        if threadIdx.x == 0:
            for i in range(blockDim.x):
                sum += buffer[i]
    """

    def test_multi_margin_loss(self):
        # MultiMarginLoss: one block per sample, thread0 serial scan of per-thread
        # hinge sums. The kernel writes per-sample loss / nclass, then the mean
        # reduction across samples uses a separate kernel. Test one sample.
        x = torch.randn(1, 50, device="cuda")
        target = torch.randint(0, 50, (1,), device="cuda")
        result = torch.nn.functional.multi_margin_loss(x, target)
        self.assertEqual(result.shape, ())
        x_np = x.cpu().numpy()
        tgt = target[0].item()
        THREADS = 128
        per_thread = [np.float32(0.0)] * THREADS
        for tid in range(THREADS):
            for c in range(tid, 50, THREADS):
                if c == tgt:
                    continue
                per_thread[tid] = per_thread[tid] + np.float32(
                    max(0, 1.0 - x_np[0, tgt] + x_np[0, c]))
        sample_loss = spec_multi_margin_loss_thread0_scan(per_thread)
        # Kernel divides by nclass. For mean reduction with nframe=1, this is the result.
        ref = float(sample_loss) / 50.0
        self.assertEqual(ref, result.item(), atol=1e-6, rtol=0)


class TestMultiLabelMarginCriterion(TestCase):
    """MultiLabelMarginCriterion.cu

    Block: fixed thread count.
    Tree: similar to NLL loss — thread-local sequential accumulation of
      hinge losses at stride blockDim.x, then shared-memory halving tree.
      for i in range(threadIdx.x, n_classes, blockDim.x):
          if target contains i: continue
          loss += max(0, 1 - x[target[j]] + x[i])
      -> shmem halving tree across block

    No separate spec — same shmem_halving_reduce tree as spec_nll_loss_reduce.
    """

    def test_multilabel_margin_loss(self):
        # Same shmem_halving_reduce tree as spec_nll_loss_reduce.
        # Compare CUDA against CPU (same algorithm, deterministic on CPU).
        x = torch.randn(100, 50, device="cuda")
        target = torch.zeros(100, 50, dtype=torch.long, device="cuda") - 1
        for i in range(100):
            n = torch.randint(1, 10, (1,)).item()
            target[i, :n] = torch.randint(0, 50, (n,))
        result = torch.nn.functional.multilabel_margin_loss(x, target)
        self.assertEqual(result.shape, ())
        ref = torch.nn.functional.multilabel_margin_loss(x.cpu(), target.cpu())
        self.assertEqual(ref.item(), result.item(), atol=1e-5, rtol=0)


class TestLossCTC(TestCase):
    """LossCTC.cu — CTC loss
    CUDA ref: cuda/LossCTC.cu.

    Log-space dynamic programming. Reductions use log-add-exp:
      log(exp(a) + exp(b)) via log1p(exp(min-max)) + max
    Operates per batch element. Sequential DP over time steps.

    No spec provided — the CTC forward/backward is a sequential dynamic
    programming algorithm over time steps (not a parallel reduction tree).
    The non-associativity comes from log-add-exp accumulation within the
    DP recurrence: alpha[t,s] = logaddexp(alpha[t-1,s], alpha[t-1,s-1])
    + log_prob[t, label[s]].
    """

    def test_ctc_loss(self):
        # Spec: sequential DP, not a parallel reduction tree; not separately tested
        T, N, C = 50, 8, 30
        log_probs = torch.randn(T, N, C, device="cuda").log_softmax(dim=2)
        targets = torch.randint(1, C, (N, 20), device="cuda")
        input_lengths = torch.full((N,), T, dtype=torch.long, device="cuda")
        target_lengths = torch.full((N,), 20, dtype=torch.long, device="cuda")
        result = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        self.assertEqual(result.shape, ())
        # CPU vs CUDA: same DP but different exp/log implementations.
        # logaddexp = log1p(exp(min-max)) + max uses CUDA transcendentals.
        ref = torch.nn.functional.ctc_loss(
            log_probs.cpu(), targets.cpu(), input_lengths.cpu(),
            target_lengths.cpu(), blank=0
        )
        self.assertEqual(ref.item(), result.item(), atol=1e-5, rtol=0)


class TestCumsumKernel(TestCase):
    """CumsumKernel.cu + ScanUtils.cuh — cumulative sum

    Specs: spec_cumsum_innermost_sklansky, spec_cumsum_outer_sequential.

    === Innermost dim (tensor_kernel_scan_innermost_dim) ===
    Triggered: reducing last (stride-1) dimension.
    Block: num_threads_x chosen by get_log_num_threads_x to balance x/y
      threads (ratio tracks row_size/num_rows), capped at [16, 512].
      Multiple rows per block via threadIdx.y.
    Tree: Sklansky parallel prefix scan in shared memory.
      Processes row in chunks of 2*num_threads_x:
        1. Load 2 elements per thread into shared memory.
        2. Sklansky tree:
             for s in [1, 2, 4, ..., num_threads_x]:
                 a = (threadIdx.x / s) * (2*s) + s
                 ti = a + (threadIdx.x % s)
                 si = a - 1
                 shared[ti] = combine(shared[si], shared[ti])
                 __syncthreads()
        3. Write results.  Carry block_total to next chunk.
      Depth: log2(num_threads_x) per chunk.

    === Outer dim (tensor_kernel_scan_outer_dim) ===
    Triggered: reducing any non-innermost dimension.
    Block: min(512, num_irows) threads.
    Tree: PURELY SEQUENTIAL loop per thread.
      for col in range(row_size):
          out[col] = combine(out[col-1], data[col])
      No inter-thread cooperation. Each thread handles one inner row.
    """

    def test_cumsum_innermost(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.cumsum(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        # Compute num_threads_x matching get_log_num_threads_x_inner_scan
        num_rows, row_size = x.shape
        log_x = 0
        while (1 << log_x) < row_size:
            log_x += 1
        log_y = 0
        while (1 << log_y) < num_rows:
            log_y += 1
        log_ntx = min(max(4, (9 + log_x - log_y) // 2), 9)
        ntx = 1 << log_ntx
        for i in range(min(4, x.shape[0])):
            row = list(x[i].cpu().numpy())
            spec_out = spec_cumsum_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_cumsum_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumsum(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise match (purely sequential outer-dim scan)
        x_np = x.cpu().numpy()
        flat = list(x_np.flatten())
        spec_out = spec_cumsum_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)


class TestCumprodKernel(TestCase):
    """CumprodKernel.cu + ScanUtils.cuh — cumulative product

    Same two-path dispatch as cumsum. Combine: a * b.
    Specs: spec_cumprod_innermost_sklansky, spec_cumprod_outer_sequential.
    """

    def test_cumprod_innermost(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.cumprod(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        num_rows, row_size = x.shape
        log_x = 0
        while (1 << log_x) < row_size:
            log_x += 1
        log_y = 0
        while (1 << log_y) < num_rows:
            log_y += 1
        log_ntx = min(max(4, (9 + log_x - log_y) // 2), 9)
        ntx = 1 << log_ntx
        for i in range(min(4, x.shape[0])):
            row = list(x[i].cpu().numpy())
            spec_out = spec_cumprod_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_cumprod_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumprod(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        x_np = x.cpu().numpy()
        flat = list(x_np.flatten())
        spec_out = spec_cumprod_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)


class TestLogcumsumexpKernel(TestCase):
    """LogcumsumexpKernel.cu — log-cumulative-sum-exp

    Same inner/outer dispatch as cumsum/cumprod.
    Combine: log(exp(a) + exp(b)), computed as log1p(exp(min-max)) + max.
    Specs: spec_logcumsumexp_innermost_sklansky, spec_logcumsumexp_outer_sequential.
    """

    def test_logcumsumexp_innermost(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.logcumsumexp(x, dim=-1)
        self.assertEqual(result.shape, x.shape)
        num_rows, row_size = x.shape
        log_x = 0
        while (1 << log_x) < row_size:
            log_x += 1
        log_y = 0
        while (1 << log_y) < num_rows:
            log_y += 1
        log_ntx = min(max(4, (9 + log_x - log_y) // 2), 9)
        ntx = 1 << log_ntx
        for i in range(min(4, x.shape[0])):
            row = list(x[i].cpu().numpy())
            spec_out = spec_logcumsumexp_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(np.array(spec_out, dtype=np.float32)),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_logcumsumexp_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.logcumsumexp(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        x_np = x.cpu().numpy()
        flat = [float(v) for v in x_np.flatten()]
        spec_out = spec_logcumsumexp_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)


class TestScatterGatherKernel(TestCase):
    """ScatterGatherKernel.cu — scatter_add, scatter_reduce

    Spec: spec_scatter_add (sequential approximation of nondeterministic atomics).
    CUDA ref: cuda/ScatterGatherKernel.cu.

    All use ATOMIC operations on the output tensor.
    Reduction order is NONDETERMINISTIC (depends on warp scheduling).
      scatter_add:    fastAtomicAdd
      scatter 'prod': gpuAtomicMul
      scatter 'mean': fastAtomicAdd, then post-kernel divide by count
    Block: standard pointwise launch grid (one thread per src element).
    """

    def test_scatter_add(self):
        x = torch.zeros(100, 64, device="cuda")
        idx = torch.randint(0, 100, (500, 64), device="cuda")
        src = torch.randn(500, 64, device="cuda")
        result = x.scatter_add(0, idx, src)
        self.assertEqual(result.shape, x.shape)
        # Spec: nondeterministic atomics — CUDA order depends on warp
        # scheduling, so we can only check approximate equality.
        # For a collision-free case (1:1 mapping), it IS bitwise exact.
        x2 = torch.zeros(10, device="cuda")
        idx2 = torch.arange(10, device="cuda")
        src2 = torch.randn(10, device="cuda")
        result2 = x2.scatter_add(0, idx2, src2)
        spec_out = spec_scatter_add(10, list(range(10)), list(src2.cpu().numpy()))
        for r in range(10):
            self.assertEqual(spec_out[r], result2[r].item(), atol=0, rtol=0)

    def test_scatter_reduce_sum(self):
        # Use collision-free (1:1) mapping for bitwise exact comparison.
        x = torch.zeros(10, 64, device="cuda")
        idx = torch.arange(10, device="cuda").unsqueeze(1).expand(10, 64)
        src = torch.randn(10, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="sum")
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(src, result, atol=0, rtol=0)

    def test_scatter_reduce_prod(self):
        # Use collision-free (1:1) mapping for bitwise exact comparison.
        x = torch.ones(10, 64, device="cuda")
        idx = torch.arange(10, device="cuda").unsqueeze(1).expand(10, 64)
        src = torch.randn(10, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="prod")
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(src, result, atol=0, rtol=0)

    def test_scatter_reduce_mean(self):
        # Use collision-free (1:1) mapping for bitwise exact comparison.
        # With 1:1 mapping, mean = sum / count = val / 1 = val (since include_self=True,
        # count is 2 and sum is src + 0 = src, so mean = src / 2).
        x = torch.zeros(10, 64, device="cuda")
        idx = torch.arange(10, device="cuda").unsqueeze(1).expand(10, 64)
        src = torch.randn(10, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="mean")
        self.assertEqual(result.shape, x.shape)
        # include_self=True by default, so mean = (0 + src) / 2 = src / 2
        self.assertEqual(src / 2, result, atol=0, rtol=0)


class TestSegmentReduce(TestCase):
    """SegmentReduce.cu
    CUDA ref: cuda/SegmentReduce.cu.

    === 1D (CUB path) ===
    Uses cub::DeviceSegmentedReduce::Reduce with custom combine op.
    Internal tree structure is CUB's auto-tuned binary tree.

    === Multi-dim (custom kernel) ===
    PURELY SEQUENTIAL loop per thread over segment elements:
      for j in range(offset_start, offset_end):
          acc = combine(acc, data[j])
    One thread per (segment, feature_dim) pair.

    No spec provided — 1D uses CUB whose internal tree is opaque, and
    multi-dim is trivially sequential_reduce over each segment.
    """

    def test_segment_reduce_1d(self):
        # 1D uses CUB whose internal tree is opaque. Compare against CPU.
        x = torch.randn(10000, device="cuda")
        lengths = torch.tensor([100] * 100, device="cuda")
        result = torch.segment_reduce(x, "sum", lengths=lengths)
        self.assertEqual(result.shape, (100,))
        ref = torch.segment_reduce(x.cpu(), "sum", lengths=lengths.cpu())
        self.assertEqual(ref[0].item(), result[0].item(), atol=1e-5, rtol=0)

    def test_segment_reduce_2d(self):
        # Multi-dim: trivially sequential per (segment, feature). Should match CPU.
        x = torch.randn(10000, 64, device="cuda")
        lengths = torch.tensor([100] * 100, device="cuda")
        result = torch.segment_reduce(x, "sum", lengths=lengths)
        self.assertEqual(result.shape, (100, 64))
        ref = torch.segment_reduce(x.cpu(), "sum", lengths=lengths.cpu())
        self.assertEqual(ref[0, 0].item(), result[0, 0].item(), atol=1e-5, rtol=0)


class TestEmbeddingBag(TestCase):
    """EmbeddingBag.cu
    CUDA ref: cuda/EmbeddingBag.cu:115, :478.

    === SUM/MEAN mode forward ===
    PURELY SEQUENTIAL loop per thread. One thread per output feature.
      for emb_idx in range(bag_start, bag_end):
          acc += per_sample_weight * embedding[indices[emb_idx], feature]
    No inter-thread reduction.

    === Per-sample-weights backward ===
    Dot product reduction. One warp per sample.
      Thread-local: sequential sum at stride C10_WARP_SIZE:
        for feat in range(lane, embedding_dim, 32):
            result += grad[feat] * weight[feat]
      WarpReduceSum: shfl_down(16,8,4,2,1)

    Specs: spec_embedding_bag_sum, spec_embedding_bag_psw_backward.
    """

    def test_embedding_bag_sum(self):
        w = torch.randn(1000, 128, device="cuda")
        idx = torch.randint(0, 1000, (500,), device="cuda")
        offsets = torch.tensor([0, 100, 250, 400], device="cuda")
        result = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="sum"
        )
        self.assertEqual(result.shape, (4, 128))
        # Spec: bitwise match (sequential loop per thread)
        bag_idx = 0
        feat = 0
        bag_start = offsets[bag_idx].item()
        bag_end = offsets[bag_idx + 1].item() if bag_idx + 1 < len(offsets) else len(idx)
        w_np = w.cpu().numpy()
        idx_np = idx.cpu().numpy()
        embeddings = [w_np[idx_np[j], feat] for j in range(bag_start, bag_end)]
        spec_val = spec_embedding_bag_sum(embeddings)
        self.assertEqual(spec_val, result[bag_idx, feat].item(), atol=0, rtol=0)

    def test_embedding_bag_mean(self):
        w = torch.randn(1000, 128, device="cuda")
        idx = torch.randint(0, 1000, (500,), device="cuda")
        offsets = torch.tensor([0, 100, 250, 400], device="cuda")
        result = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="mean"
        )
        self.assertEqual(result.shape, (4, 128))
        # mean = sum * (1/bag_size); multiply by reciprocal matching MeanOps::project
        bag_idx, feat = 0, 0
        bag_start = offsets[bag_idx].item()
        bag_end = offsets[bag_idx + 1].item()
        bag_size = bag_end - bag_start
        w_np = w.cpu().numpy()
        idx_np = idx.cpu().numpy()
        embeddings = [w_np[idx_np[j], feat] for j in range(bag_start, bag_end)]
        spec_sum = spec_embedding_bag_sum(embeddings)
        ref = spec_sum * np.float32(1.0 / bag_size)
        self.assertEqual(ref, result[bag_idx, feat].item(), atol=0, rtol=0)

    def test_embedding_bag_per_sample_weights_backward(self):
        # Dot product per sample: sum(grad[feat] * weight[feat]) over features.
        # One warp per sample. seq(+=, stride=32) -> WarpReduceSum: shfl_down.
        # Warp reduce tree is spec_embedding_bag_psw_backward; tested via
        # autograd (backward through embedding_bag computes psw grad).
        w = torch.randn(100, 16, device="cuda")
        idx = torch.randint(0, 100, (20,), device="cuda")
        offsets = torch.tensor([0, 10], device="cuda")
        psw = torch.randn(20, device="cuda", requires_grad=True)
        out = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="sum", per_sample_weights=psw
        )
        out.sum().backward()
        self.assertEqual(psw.grad.shape, psw.shape)
        # psw.grad[i] = dot(grad_output[bag], weight[idx[i]])
        # since grad of out.sum() is all ones, grad_output = ones(1, 16)
        grad_vals = [np.float32(1.0)] * 16
        w_np = w.cpu().numpy()
        idx_np = list(idx.cpu().numpy())
        weight_table = [list(w_np[r]) for r in range(w_np.shape[0])]
        spec_psw_grad = spec_embedding_bag_psw_backward(
            grad_vals, weight_table, idx_np, embedding_dim=16
        )
        self.assertEqual(float(spec_psw_grad[0]), psw.grad[0].item(), atol=0, rtol=0)


class TestAveragePool2d(TestCase):
    """AveragePool2d.cu
    CUDA ref: cuda/AveragePool2d.cu:33.

    PURELY SEQUENTIAL nested loop per output element. One thread per output.
      accval = 0
      for h in range(hstart, hend):
          for w in range(wstart, wend):
              accval += input[c, h, w]
      output = accval / count
    No inter-thread reduction. Block: standard pointwise launch.

    Spec: spec_avg_pool_window.
    """

    def test_avg_pool2d(self):
        x = torch.randn(8, 64, 32, 32, device="cuda")
        result = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(result.shape, (8, 64, 32, 32))
        # Spec: bitwise match (sequential window loop for center element)
        b, c, oh, ow = 0, 0, 16, 16
        window_vals = list(x[b, c, oh - 1:oh + 2, ow - 1:ow + 2].cpu().numpy().flatten())
        spec_val = spec_avg_pool_window(window_vals)
        self.assertEqual(spec_val, result[b, c, oh, ow].item(), atol=0, rtol=0)

    def test_avg_pool2d_backward(self):
        # Backward: each input thread sums contributions from overlapping output
        # windows. Sequential loop — no inter-thread reduction.
        x = torch.randn(8, 64, 32, 32, device="cuda")
        grad_output = torch.randn(8, 64, 32, 32, device="cuda")
        result = torch.ops.aten.avg_pool2d_backward(
            grad_output, x, [3, 3], [1, 1], [1, 1], False, True, None
        )
        self.assertEqual(result.shape, x.shape)
        # A center element (16,16) with 3x3 kernel, stride=1, pad=1 receives
        # grad/9 from each of the 9 output positions whose pooling window includes it.
        # Compute on CPU to avoid comparing two CUDA trees.
        b, c = 0, 0
        window = grad_output[b, c, 15:18, 15:18].cpu().numpy()
        ref = spec_avg_pool_window(list(window.flatten()))
        self.assertEqual(float(ref), result[b, c, 16, 16].item(), atol=1e-6, rtol=0)


class TestAdaptiveAveragePooling(TestCase):
    """AdaptiveAveragePooling.cu
    CUDA ref: cuda/AdaptiveAveragePooling.cu.

    Same purely sequential window loop as AveragePool2d.
    Window boundaries computed dynamically via START_IND/END_IND macros.
    No separate spec — use spec_avg_pool_window.

    NOTE: avg_pool3d (AveragePool3d.cu), adaptive_avg_pool3d
    (AdaptiveAveragePooling3d.cu) use the same sequential window loop
    pattern with an additional depth dimension.  Not listed separately.
    """

    def test_adaptive_avg_pool2d(self):
        # Sequential window loop, same as avg_pool2d. Use small spatial size
        # for tighter test.
        x = torch.randn(8, 64, 4, 4, device="cuda")
        result = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        self.assertEqual(result.shape, (8, 64, 1, 1))
        # Kernel does sum / kH / kW (two divisions), not sum / (kH*kW)
        window_vals = list(x[0, 0].cpu().numpy().flatten())
        acc = type(window_vals[0])(0)
        for v in window_vals:
            acc = acc + v
        kH, kW = 4, 4
        spec_val = acc / type(acc)(kH) / type(acc)(kW)
        # Compiler loop unrolling changes sequential sum association
        self.assertEqual(float(spec_val), result[0, 0, 0, 0].item(), atol=1e-7, rtol=0)

    def test_adaptive_avg_pool2d_backward(self):
        # Sequential per-input-element loop. No inter-thread reduction.
        # For (1,1) output, each input element gets grad / (H*W).
        x = torch.randn(8, 64, 32, 32, device="cuda")
        grad_output = torch.randn(8, 64, 1, 1, device="cuda")
        result = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, x)
        self.assertEqual(result.shape, x.shape)
        # grad / (H*W) is a single division, no tree involved.
        ref = grad_output[0, 0, 0, 0].item() / (32.0 * 32.0)
        self.assertEqual(ref, result[0, 0, 16, 16].item(), atol=0, rtol=0)


class TestForeachReduceOp(TestCase):
    """ForeachReduceOp.cu — multi-tensor norm
    CUDA ref: cuda/ForeachReduceOp.cu:117 (lpmax_cleanup).

    Block: 512 threads. One kernel processes multiple tensors.
    Tree per tensor chunk:
      Thread-local: sequential accumulation over assigned elements.
      shfl_down(32) within each warp.
      BlockReduceSum via shared memory (same block_reduce.cuh pattern).
    Combines partial results across tensor chunks.

    No separate spec — per-tensor reduction uses the same block_reduce_cuh
    tree.  The additional complexity is that one kernel launch handles
    multiple tensors: each tensor gets a chunk of thread-blocks, and
    partial results across chunks for the same tensor are combined via
    a second block_reduce_cuh pass.
    """

    def test_foreach_norm(self):
        # foreach_norm uses a fused multi-tensor kernel with a different tree
        # from gpu_reduce_kernel's vector_norm. The two trees (fused multi-tensor
        # block_reduce_cuh vs gpu_reduce_kernel) have different association order.
        tensors = [torch.randn(1000, device="cuda") for _ in range(3)]
        result = torch._foreach_norm(tensors, 2.0)
        self.assertEqual(len(result), 3)
        for i in range(3):
            ref = torch.linalg.vector_norm(tensors[i], 2).item()
            self.assertEqual(result[i].item(), ref, atol=1e-5, rtol=0)


if __name__ == "__main__":
    run_tests()
