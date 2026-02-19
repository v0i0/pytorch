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
import sys
from typing import Any, Callable, List, Tuple

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

torch._dynamo.config.recompile_limit = sys.maxsize


# ============================================================================
# Dtype-aware arithmetic helpers
# ============================================================================
# When dtype is set (e.g. torch.float32), all spec arithmetic rounds to that
# precision at every step, matching the CUDA kernel's actual numerics.
# When dtype is None, Python float64 is used (useful for tree-structure
# verification).

def _f32(x):
    """Cast to float32 precision via CUDA round-trip."""
    return torch.tensor(float(x), dtype=torch.float32, device="cuda").item()


def _fma_f32(a, b, c):
    """Fused multiply-add in float32: round(a*b + c) with single rounding.
    Uses float64 intermediate since float64 exactly represents float32*float32."""
    return torch.tensor(
        float(a) * float(b) + float(c), dtype=torch.float32, device="cuda"
    ).item()


def _dtype_cast(dtype):
    """Return a cast function for the given torch dtype (or identity)."""
    if dtype is None:
        return lambda x: float(x)
    return lambda x: torch.tensor(float(x), dtype=dtype, device="cuda").item()


def _dtype_exp(dtype):
    """Return exp() that matches CUDA's implementation for the given dtype."""
    if dtype is None:
        return math.exp
    # Route through CUDA torch.exp for bitwise matching with CUDA kernels.
    # IEEE 754 add/mul/div are deterministic, but transcendentals (exp, sqrt,
    # log) use different polynomial approximations on CPU vs GPU.
    def _cuda_exp(x):
        t = torch.tensor(float(x), dtype=dtype, device="cuda")
        return torch.tensor(t.exp().item(), dtype=dtype, device="cuda").item()
    return _cuda_exp


def _dtype_sqrt(dtype):
    """Return sqrt() that matches CUDA's implementation for the given dtype."""
    if dtype is None:
        return math.sqrt
    def _cuda_sqrt(x):
        t = torch.tensor(float(x), dtype=dtype, device="cuda")
        return torch.tensor(t.sqrt().item(), dtype=dtype, device="cuda").item()
    return _cuda_sqrt


def _dtype_identity(dtype, val):
    """Return val cast to dtype."""
    if dtype is None:
        return float(val)
    return torch.tensor(float(val), dtype=dtype, device="cuda").item()


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


# CUDA ref: cuda/SoftMax.cu:412 (blockReduce)
def softmax_block_reduce(thread_values: list, combine: Combine, identity,
                         warp_size: int = 32):
    """Shared-memory sequential reduce from SoftMax.cu (NOT shfl_down).
    Level 1: first-warp lane l sequentially sums smem[l*32..l*32+31].
    Level 2: thread 0 sequentially sums per-warp results."""
    n = len(thread_values)
    num_warps = n // warp_size
    warp_results = [identity] * num_warps
    for lane in range(min(warp_size, num_warps)):
        val = identity
        for i in range(warp_size):
            val = combine(val, thread_values[lane * warp_size + i])
        warp_results[lane] = val
    result = identity
    for i in range(num_warps):
        result = combine(result, warp_results[i])
    return result


# CUDA ref: cuda/SoftMax.cu:163 (SoftMax_getBlockSize)
def softmax_getblocksize(ILP: int, dim_size: int, max_threads: int = 1024,
                         warp_size: int = 32) -> int:
    """Block size for cunn_SoftMaxBackward."""
    max_block_size = min(dim_size // ILP, max_threads)
    if ILP > 1:
        max_block_size //= 2
    block_size = 1
    while block_size < max_block_size:
        block_size *= 2
    return max(block_size, warp_size)


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

    Pass dtype=torch.float32 to match fp32 CUDA numerics.

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
            s = cast(s + e)
        lane_sumexp.append(s)

    row_sum = shfl_xor_butterfly_high_to_low(lane_sumexp, lambda a, b: cast(a + b))

    output = [zero] * n
    for lane_id in range(warp_size):
        for it, e in enumerate(lane_exp[lane_id]):
            idx = lane_id + it * warp_size
            if idx < n:
                output[idx] = cast(e / row_sum)
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
    thread_sums = ilp_reduce(exp_data, lambda a, b: _f32(a + b), zero)
    row_sum = block_reduce_cuh(thread_sums, lambda a, b: _f32(a + b), zero)

    # Step 3: output — epilogue recomputes exp(x-max) / sum
    # CUDA ref: SoftMax.cu:71 (SoftMaxForwardEpilogue)
    return [cast(exp(cast(v) - cast(row_max)) / row_sum) for v in row]


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
                s = cast(s + exp(cast(data_2d[d * inner_size + inner]) - cast(m)))
            for d in range(dim_size):
                idx = d * inner_size + inner
                output[idx] = cast(exp(cast(data_2d[idx]) - cast(m)) / s)
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
                    thread_sums[tx] = cast(thread_sums[tx] + exp(
                        cast(data_2d[d * inner_size + inner]) - cast(m)))
            s = shmem_halving_reduce(thread_sums, lambda a, b: cast(a + b), zero)

            # Epilogue: each thread writes at stride dim_threads
            for tx in range(dim_threads):
                for d in range(tx, dim_size, dim_threads):
                    idx = d * inner_size + inner
                    output[idx] = cast(exp(cast(data_2d[idx]) - cast(m)) / s)

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


@torch.compile
def _compiled_ln_bwd_stats(dY, X, gamma, mean_val, rstd_val, start, stride):
    """Compiled per-thread accumulation for layer_norm_grad_input_kernel.
    Manually unrolled (no for-k) so Triton generates FMA.

    CUDA ref: layer_norm_kernel.cu:496-517."""
    s1 = torch.zeros(1, device=dY.device, dtype=dY.dtype)
    s2 = torch.zeros(1, device=dY.device, dtype=dY.dtype)
    N = dY.shape[0]
    l = start * 4
    while l + 3 < N:
        s1 = s1 + dY[l] * gamma[l]
        s2 = s2 + dY[l] * gamma[l] * (X[l] - mean_val) * rstd_val
        s1 = s1 + dY[l + 1] * gamma[l + 1]
        s2 = s2 + dY[l + 1] * gamma[l + 1] * (X[l + 1] - mean_val) * rstd_val
        s1 = s1 + dY[l + 2] * gamma[l + 2]
        s2 = s2 + dY[l + 2] * gamma[l + 2] * (X[l + 2] - mean_val) * rstd_val
        s1 = s1 + dY[l + 3] * gamma[l + 3]
        s2 = s2 + dY[l + 3] * gamma[l + 3] * (X[l + 3] - mean_val) * rstd_val
        l = l + stride * 4
    while l < N:
        s1 = s1 + dY[l] * gamma[l]
        s2 = s2 + dY[l] * gamma[l] * (X[l] - mean_val) * rstd_val
        l = l + 1
    return s1, s2


@torch.compile
def _compiled_rms_bwd_stats(dY, X, gamma, rstd_val, start, stride):
    """Compiled per-thread accumulation for rms_norm backward.
    Manually unrolled (no for-k) so Triton generates FMA.

    CUDA ref: layer_norm_kernel.cu:514-516."""
    s2 = torch.zeros(1, device=dY.device, dtype=dY.dtype)
    N = dY.shape[0]
    l = start * 4
    while l + 3 < N:
        s2 = s2 + dY[l] * gamma[l] * X[l] * rstd_val
        s2 = s2 + dY[l + 1] * gamma[l + 1] * X[l + 1] * rstd_val
        s2 = s2 + dY[l + 2] * gamma[l + 2] * X[l + 2] * rstd_val
        s2 = s2 + dY[l + 3] * gamma[l + 3] * X[l + 3] * rstd_val
        l = l + stride * 4
    while l < N:
        s2 = s2 + dY[l] * gamma[l] * X[l] * rstd_val
        l = l + 1
    return s2


@torch.compile
def _compiled_ln_dx_elementwise(dY, X, gamma, mean_val, rstd_val,
                                 stats_x1, stats_x2, fH):
    """Compiled elementwise dX for layer_norm backward.
    FMA: f_grad -= (X-mean)*rstd * stats_x2 -> fma(-(X-mean)*rstd, stats_x2, f_grad)

    CUDA ref: layer_norm_kernel.cu:553-576 (vectorized elementwise)."""
    N = dY.shape[0]
    term1 = (1.0 / fH) * rstd_val
    dx = torch.empty_like(dY)
    i = 0
    while i < N:
        f_grad = fH * gamma[i] * dY[i]
        f_grad = f_grad - (X[i] - mean_val) * rstd_val * stats_x2
        f_grad = f_grad - stats_x1
        f_grad = f_grad * term1
        dx[i] = f_grad
        i = i + 1
    return dx


@torch.compile
def _compiled_rms_dx_elementwise(dY, X, gamma, rstd_val, stats_x2, fH):
    """Compiled elementwise dX for rms_norm backward.
    FMA: f_grad -= X*rstd * stats_x2 -> fma(-X*rstd, stats_x2, f_grad)

    CUDA ref: layer_norm_kernel.cu:553-576 (vectorized, rms_norm=true)."""
    N = dY.shape[0]
    term1 = (1.0 / fH) * rstd_val
    dx = torch.empty_like(dY)
    i = 0
    while i < N:
        f_grad = fH * gamma[i] * dY[i]
        f_grad = f_grad - X[i] * rstd_val * stats_x2
        f_grad = f_grad * term1
        dx[i] = f_grad
        i = i + 1
    return dx


@torch.compile
def _compiled_dgamma_accum(dY, X, mean_val, rstd_val, start, end):
    """Compiled per-thread dgamma accumulation for GammaBetaBackwardCUDAKernel.
    acc += dY * (X - mean) * rstd — generates FMA: fma(dY*(X-mean), rstd, acc).

    CUDA ref: layer_norm_kernel.cu:650-707."""
    acc = torch.zeros(1, device=dY.device, dtype=dY.dtype)
    m = start
    while m < end:
        acc = acc + dY[m] * (X[m] - mean_val[m]) * rstd_val[m]
        m = m + 1
    return acc


@torch.compile
def _compiled_rms_dgamma_accum(dY, X, rstd_val, start, end):
    """Compiled per-thread dgamma accumulation for rms_norm backward.
    acc += dY * X * rstd — generates FMA: fma(dY*X, rstd, acc).

    CUDA ref: layer_norm_kernel.cu:650-707 (rms_norm path)."""
    acc = torch.zeros(1, device=dY.device, dtype=dY.dtype)
    m = start
    while m < end:
        acc = acc + dY[m] * X[m] * rstd_val[m]
        m = m + 1
    return acc


@torch.compile
def _compiled_welford_vec2_reduce(x_flat, tid, stride, N):
    """Compiled vectorized (V=2) Welford thread reduce for gpu_reduce_kernel.
    Two independent accumulators, each processing every other element of
    consecutive pairs at stride `stride` (in units of vectors).
    Returns merged Welford state (mean, m2, count).

    CUDA ref: Reduce.cuh:499 (input_vectorized_thread_reduce_impl) with
    WelfordOps::reduce (SharedReduceOps.h:103)."""
    m0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    m1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = tid
    while idx * 2 + 1 < N:
        v0 = x_flat[idx * 2]
        delta0 = v0 - m0
        c0 = c0 + 1.0
        m0 = m0 + delta0 / c0
        d0b = v0 - m0
        s0 = s0 + delta0 * d0b
        v1 = x_flat[idx * 2 + 1]
        delta1 = v1 - m1
        c1 = c1 + 1.0
        m1 = m1 + delta1 / c1
        d1b = v1 - m1
        s1 = s1 + delta1 * d1b
        idx = idx + stride
    # Merge acc[0] and acc[1] via WelfordOps::combine
    c = c0 + c1
    delta = m1 - m0
    nb_over_n = c1 / c
    m = m0 + delta * nb_over_n
    s2 = (s0 + s1) + delta * delta * c0 * nb_over_n
    return m, s2, c


@torch.compile
def _compiled_welford_nonvec_vt2_reduce(x_flat, tid, S, N):
    """Compiled non-vectorized (vt0=2) Welford thread reduce for gpu_reduce_kernel.
    Two independent accumulators at stride S: acc[i] gets elements at positions
    tid + i*S, tid + i*S + 2S, tid + i*S + 4S, ...
    Returns merged Welford state (mean, m2, count).

    CUDA ref: Reduce.cuh:561 (thread_reduce_impl) with WelfordOps::reduce."""
    m0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c0 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    m1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    s1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    c1 = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
    idx = tid
    # Main loop: both accumulators get data
    while idx + S < N:
        v0 = x_flat[idx]
        delta0 = v0 - m0
        c0 = c0 + 1.0
        m0 = m0 + delta0 / c0
        d0b = v0 - m0
        s0 = s0 + delta0 * d0b
        v1 = x_flat[idx + S]
        delta1 = v1 - m1
        c1 = c1 + 1.0
        m1 = m1 + delta1 / c1
        d1b = v1 - m1
        s1 = s1 + delta1 * d1b
        idx = idx + S * 2
    # Tail: acc[0] may still have data
    if idx < N:
        v0 = x_flat[idx]
        delta0 = v0 - m0
        c0 = c0 + 1.0
        m0 = m0 + delta0 / c0
        d0b = v0 - m0
        s0 = s0 + delta0 * d0b
    # Merge acc[0] and acc[1] via WelfordOps::combine
    c = c0 + c1
    delta = m1 - m0
    nb_over_n = c1 / c
    m = m0 + delta * nb_over_n
    s2 = (s0 + s1) + delta * delta * c0 * nb_over_n
    return m, s2, c


def _welford_combine_wrap(a, b):
    """Welford combine via compiled function, with identity short-circuit."""
    if a[2] == 0:
        return b
    if b[2] == 0:
        return a
    ta = [torch.tensor(float(v), device="cuda") for v in list(a) + list(b)]
    m, s2, c = _compiled_welford_ops_combine(*ta)
    return (_f32(m.item()), _f32(s2.item()), _f32(c.item()))


def _welford_cta_reduce(x_cuda, N, S, H, W, do_bx, do_by, vectorize,
                         stride1, warp_size, cta_offset, welford_id):
    """Run phases 1-3 (thread reduce + block reduce) for one CTA."""
    vt0 = 2
    cw = _welford_combine_wrap
    num_threads = W * H

    # Phase 1: per-thread Welford via compiled reduce
    thread_welford = []
    active_threads = min(num_threads, H if (not stride1 and do_by) else S)
    for t in range(active_threads):
        start_idx = t + cta_offset
        if vectorize and stride1:
            m, s2, c = _compiled_welford_vec2_reduce(x_cuda, start_idx, S, N)
        else:
            m, s2, c = _compiled_welford_nonvec_vt2_reduce(
                x_cuda, start_idx, S, N)
        thread_welford.append(
            (_f32(m.item()), _f32(s2.item()), _f32(c.item())))

    while len(thread_welford) < num_threads:
        thread_welford.append(welford_id)

    # Phase 2: block_x_reduce (stride-1 only)
    if do_bx:
        rows = []
        for y in range(H):
            row = thread_welford[y * W:(y + 1) * W]
            while len(row) < W:
                row.append(welford_id)
            if W > warp_size:
                row = list(row)
                offset = W // 2
                while offset >= warp_size:
                    for i in range(offset):
                        if i + offset < len(row):
                            row[i] = cw(row[i], row[i + offset])
                    offset //= 2
                row = row[:warp_size]
            effective_w = min(W, warp_size)
            rows.append(
                shfl_down_reduce_high_to_low(row[:effective_w], cw))
        thread_welford = rows
    else:
        if do_by:
            thread_welford = thread_welford[:H]

    # Phase 3: block_y_reduce
    if do_by:
        vals = thread_welford[:H]
        while len(vals) < H:
            vals.append(welford_id)
        return shmem_halving_reduce(vals, cw, welford_id)
    return thread_welford[0]


def spec_welford_gpu_reduce(x_cuda: torch.Tensor, num_outputs: int,
                            stride1: bool, correction: int = 1,
                            take_sqrt: bool = False,
                            output_vec_size: int = 1):
    """Spec for torch.var/torch.std via gpu_reduce_kernel with WelfordOps.

    Models the exact CUDA tree with vt0=2, including global reduce (multi-CTA)
    when the GPU has enough SMs and the reduction is large enough.

    CUDA ref: Reduce.cuh + SharedReduceOps.h (WelfordOps with vt0=2)."""
    N = x_cuda.shape[0]
    warp_size = 32
    props = torch.cuda.get_device_properties(0)

    # Compute block config with vt0=2 for Welford
    vectorize = stride1 and N >= 128
    if stride1:
        dim0 = N // 2 if vectorize else N
        dim1 = num_outputs
    else:
        dim0 = num_outputs // output_vec_size
        dim1 = N
    dim0_p2 = min(lpow2(dim0), 512) if dim0 > 0 else 1
    dim1_p2 = min(lpow2(dim1), 512) if dim1 > 0 else 1
    W = min(dim0_p2, warp_size)
    H = min(dim1_p2, 512 // W)
    W = min(dim0_p2, 512 // H)
    num_threads = W * H
    do_bx = False
    do_by = False
    ctas = 1
    S = 1
    if stride1:
        S = W
        do_bx = True
        vpt = math.ceil(N / (S * (2 if vectorize else 1)))
        if vpt >= min(H * 16, 256):
            S *= H
            do_by = True
    else:
        vpt = math.ceil(N / 1)
        if vpt >= min(H * 16, 256):
            S = H
            do_by = True

    # Check for global reduce (multi-CTA)
    if do_by:
        vpt_now = math.ceil(N / S)
        blocks_per_sm = props.max_threads_per_multi_processor // num_threads
        target_grid = props.multi_processor_count * blocks_per_sm
        step_out = W if not stride1 else 1
        grid_x = math.ceil(num_outputs / output_vec_size / step_out)
        if vpt_now >= 256 and grid_x <= target_grid:
            c1 = math.ceil(target_grid / grid_x)
            c2 = math.ceil(vpt_now / 16)
            c3 = math.ceil(vpt_now / 256)
            ctas = max(min(c1, c2), c3)
            if ctas > 1:
                S *= ctas

    welford_id = (_f32(0), _f32(0), _f32(0))
    cw = _welford_combine_wrap

    if ctas <= 1:
        # Single CTA
        final = _welford_cta_reduce(
            x_cuda, N, S, H, W, do_bx, do_by, vectorize, stride1,
            warp_size, 0, welford_id)
    else:
        # Global reduce: each CTA gets a slice of the input
        input_mult_CTA = S // ctas  # = H
        cta_results = []
        for cta_id in range(ctas):
            cta_offset = cta_id * input_mult_CTA
            cta_result = _welford_cta_reduce(
                x_cuda, N, S, H, W, do_bx, do_by, vectorize, stride1,
                warp_size, cta_offset, welford_id)
            cta_results.append(cta_result)

        # Last CTA combines: thread y reads cta_results[y, y+H, y+2H, ...]
        thread_partials = [welford_id] * H
        for t in range(H):
            acc = welford_id
            cta_idx = t
            while cta_idx < ctas:
                acc = cw(acc, cta_results[cta_idx])
                cta_idx += H
            thread_partials[t] = acc
        final = shmem_halving_reduce(thread_partials, cw, welford_id)

    # Project: var or std
    mean_val, m2_val, count_val = final
    nf = _f32(float(count_val))
    divisor = _f32(nf - _f32(float(correction))) if nf > correction else _f32(0.0)
    var_val = _f32(float(m2_val) / float(divisor))
    if take_sqrt:
        return torch.tensor(float(var_val), dtype=torch.float32,
                            device="cuda").sqrt().item()
    return var_val


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
        thread_welford.append((_f32(m.item()), _f32(s2.item()),
                               _f32(c.item())))

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
        return (_f32(m.item()), _f32(s2.item()),
                _f32(c.item()))

    welford_id = (_f32(0), _f32(0), _f32(0))

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
    var = _f32(m2) / _f32(float(N))
    rstd = torch.rsqrt(
        torch.tensor(float(var + _f32(eps)),
                     dtype=torch.float32, device="cuda")).item()
    return float(_f32(mean)), rstd


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
        thread_sigma2.append(_f32(s2.item()))

    # Phase 2: intra-warp shfl_down (pure addition — IEEE 754 exact)
    add = lambda a, b: _f32(a + b)
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

    add = lambda a, b: _f32(a + b)  # noqa: E731

    # Phase 2: BlockReduceSum
    stats_x2 = block_reduce_cuh(thread_x2, add, 0.0)

    # Phase 3: elementwise dX
    fH = float(N)
    dX = []
    for i in range(N):
        dYg = dY[i] * gamma[i]
        dX.append(rstd * (dYg - X[i] * rstd * stats_x2 / fH))
    return dX


def spec_rms_norm_backward_dgamma(dY_rows: list, X_rows: list,
                                   rstd_list: list,
                                   block_dim_y: int = 32) -> list:
    """RMS norm backward dgamma: same GammaBetaBackwardCUDAKernel tree as
    layer norm dgamma but without mean subtraction.
    dgamma[j] = sum_m dY[m,j] * X[m,j] * rstd[m].

    CUDA ref: cuda/layer_norm_kernel.cu:768 (GammaBetaBackwardCUDAKernelTemplate)."""
    M = len(dY_rows)
    N = len(dY_rows[0]) if M > 0 else 0
    zero = type(dY_rows[0][0])(0) if M > 0 and N > 0 else 0.0

    dgamma = [zero] * N
    for j in range(N):
        thread_sums = [zero] * block_dim_y
        for tid in range(block_dim_y):
            acc = zero
            for m in range(tid, M, block_dim_y):
                acc = acc + dY_rows[m][j] * X_rows[m][j] * rstd_list[m]
            thread_sums[tid] = acc

        dgamma[j] = shfl_xor_butterfly(thread_sums, lambda a, b: _f32(a + b))
    return dgamma


def _ln_bwd_per_thread_stats(dY_cuda, X_cuda, gamma_cuda, mean_val, rstd_val,
                              tid, block_size, N):
    """Per-thread stats matching nvcc FMA contractions + CSE.
    nvcc applies CSE on c_loss*gamma_val (shared between stats_x1 and stats_x2),
    so stats_x1 uses plain add (not FMA), and stats_x2 reuses the rounded product.

    CUDA ref: layer_norm_kernel.cu:496-531."""
    mi = float(mean_val.item())
    ri = float(rstd_val.item())
    s1 = _f32(0.0)
    s2 = _f32(0.0)
    l = tid * 4
    while l + 3 < N:
        for k in range(4):
            cl = float(dY_cuda[l + k].item())
            gv = float(gamma_cuda[l + k].item())
            ch = float(X_cuda[l + k].item())
            t = _f32(cl * gv)
            s1 = _f32(s1 + t)
            s2 = _fma_f32(_f32(t * _f32(ch - mi)), ri, s2)
        l += block_size * 4
    while l < N:
        cl = float(dY_cuda[l].item())
        gv = float(gamma_cuda[l].item())
        ch = float(X_cuda[l].item())
        t = _f32(cl * gv)
        s1 = _f32(s1 + t)
        s2 = _fma_f32(_f32(t * _f32(ch - mi)), ri, s2)
        l += 1
    return s1, s2


def _rms_bwd_per_thread_stats(dY_cuda, X_cuda, gamma_cuda, rstd_val,
                               tid, block_size, N):
    """Per-thread stats for rms_norm backward using _fma_f32.

    CUDA ref: layer_norm_kernel.cu:514-516."""
    ri = float(rstd_val.item())
    s2 = _f32(0.0)
    l = tid * 4
    while l + 3 < N:
        for k in range(4):
            cl = float(dY_cuda[l + k].item())
            gv = float(gamma_cuda[l + k].item())
            ch = float(X_cuda[l + k].item())
            s2 = _fma_f32(_f32(_f32(cl * gv) * ch), ri, s2)
        l += block_size * 4
    while l < N:
        cl = float(dY_cuda[l].item())
        gv = float(gamma_cuda[l].item())
        ch = float(X_cuda[l].item())
        s2 = _fma_f32(_f32(_f32(cl * gv) * ch), ri, s2)
        l += 1
    return s2


def spec_layer_norm_backward_dx_compiled(
        dY_cuda, X_cuda, mean_val, rstd_val, gamma_cuda,
        block_size: int = 128):
    """Layer norm backward dX with FMA-precise stats + elementwise.

    block_size: num_threads() = C10_WARP_SIZE * 4 = 128 on CUDA.
    CUDA ref: layer_norm_kernel.cu:464 (vectorized), :357 (compute_gI)."""
    N = dY_cuda.shape[0]
    add = lambda a, b: _f32(a + b)  # noqa: E731

    thread_x1, thread_x2 = [], []
    for tid in range(block_size):
        s1, s2 = _ln_bwd_per_thread_stats(
            dY_cuda, X_cuda, gamma_cuda, mean_val, rstd_val,
            tid, block_size, N)
        thread_x1.append(s1)
        thread_x2.append(s2)

    stats_x1 = block_reduce_cuh(thread_x1, add, 0.0)
    stats_x2 = block_reduce_cuh(thread_x2, add, 0.0)

    # Elementwise dX: nvcc cross-statement FMA contraction merges
    # the `*dy` from line 1 with the `-=` from line 2 into a single FMA.
    #   t1 = fH * gamma                                       [mul]
    #   t2 = ((x - mean) * rstd) * stats_x2                   [sub, two muls]
    #   f_grad = fma(t1, dY, -t2)                              [FMA]
    #   f_grad = f_grad - stats_x1                             [sub]
    #   f_grad = f_grad * term1                                [mul]
    mi = float(mean_val.item())
    ri = float(rstd_val.item())
    sx1 = float(stats_x1)
    sx2 = float(stats_x2)
    fH = _f32(float(N))
    term1 = _f32(_f32(1.0 / fH) * ri)
    dx = []
    for i in range(N):
        dy = float(dY_cuda[i].item())
        gv = float(gamma_cuda[i].item())
        x = float(X_cuda[i].item())
        t1 = _f32(fH * gv)
        t2 = _f32(_f32(_f32(x - mi) * ri) * sx2)
        f_grad = _fma_f32(t1, dy, -t2)
        f_grad = _f32(f_grad - sx1)
        f_grad = _f32(f_grad * term1)
        dx.append(f_grad)
    return dx


def spec_rms_norm_backward_dx_compiled(
        dY_cuda, X_cuda, rstd_val, gamma_cuda, block_size: int = 128):
    """RMS norm backward dX with FMA-precise stats + elementwise.

    CUDA ref: layer_norm_kernel.cu:464 (vectorized, rms_norm=true)."""
    N = dY_cuda.shape[0]
    add = lambda a, b: _f32(a + b)  # noqa: E731

    thread_x2 = []
    for tid in range(block_size):
        s2 = _rms_bwd_per_thread_stats(
            dY_cuda, X_cuda, gamma_cuda, rstd_val, tid, block_size, N)
        thread_x2.append(s2)

    stats_x2 = block_reduce_cuh(thread_x2, add, 0.0)

    # Elementwise dX: nvcc cross-statement FMA contraction merges
    # the `*dy` from line 1 with the `-=` from line 2 into a single FMA.
    #   t1 = fH * gamma                                  [mul]
    #   t2 = (x * rstd) * stats_x2                       [two muls]
    #   f_grad = fma(t1, dY, -t2)                         [FMA]
    #   f_grad = f_grad * term1                           [mul]
    ri = float(rstd_val.item())
    sx2 = float(stats_x2)
    fH = _f32(float(N))
    term1 = _f32(_f32(1.0 / fH) * ri)
    dx = []
    for i in range(N):
        dy = float(dY_cuda[i].item())
        gv = float(gamma_cuda[i].item())
        x = float(X_cuda[i].item())
        t1 = _f32(fH * gv)
        t2 = _f32(_f32(x * ri) * sx2)
        f_grad = _fma_f32(t1, dy, -t2)
        f_grad = _f32(f_grad * term1)
        dx.append(f_grad)
    return dx


def spec_dgamma_compiled(dY_cuda, X_cuda, mean_cuda, rstd_cuda,
                         block_dim_y: int = 32, rows_per_block_y: int = 256,
                         rms_norm: bool = False):
    """GammaBetaBackwardCUDAKernel dgamma with _fma_f32 accumulation.

    Per-thread accumulates over contiguous row blocks, then XOR butterfly.
    Takes CUDA tensors (1D slices for a single feature j).

    CUDA ref: layer_norm_kernel.cu:650-707, 768-866."""
    M = dY_cuda.shape[0]
    rows_per_thread_y = rows_per_block_y // block_dim_y

    thread_sums = [_f32(0.0)] * block_dim_y
    for M_start in range(0, M, rows_per_block_y):
        for tid_y in range(block_dim_y):
            start_row = M_start + tid_y * rows_per_thread_y
            end_row = min(start_row + rows_per_thread_y, M)
            if start_row >= M:
                continue
            for m in range(start_row, end_row):
                dy_val = float(dY_cuda[m].item())
                x_val = float(X_cuda[m].item())
                rstd_val = float(rstd_cuda[m].item())
                if rms_norm:
                    t = _f32(dy_val * x_val)
                    thread_sums[tid_y] = _fma_f32(t, rstd_val,
                                                   thread_sums[tid_y])
                else:
                    mean_val = float(mean_cuda[m].item())
                    t = _f32(_f32(dy_val) * _f32(x_val - mean_val))
                    thread_sums[tid_y] = _fma_f32(t, rstd_val,
                                                   thread_sums[tid_y])

    # CUDA ref: layer_norm_kernel.cu:851 — delta = block_dim_y >> 1 down to 1
    return shfl_xor_butterfly_high_to_low(thread_sums, lambda a, b: _f32(a + b))


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
    block_x: int, block_y: int, B: int, S: int,
):
    """reduce<Float2> from Normalization.cuh: Block2D + BlockReduce (shfl_down).
    Models the 2D per-thread accumulation loop over (batch, spatial).
    Per-thread dy_xmu uses fma(dy, x-mean, acc).
    Returns (sum_dy, sum_dy_xmu).

    CUDA ref: cuda/Normalization.cuh:114 (reduce), :71 (GradOp),
              cuda/block_reduce.cuh:130 (BlockReduce)."""

    assert len(grad_output) == len(input_data)

    def float2_add(a, b):
        return (_f32(a[0] + b[0]), _f32(a[1] + b[1]))

    float2_id = (_f32(0.0), _f32(0.0))
    block_size = block_x * block_y

    # Phase 1: 2D per-thread sequential accumulation (Block2D layout)
    thread_vals = [float2_id] * block_size
    for ty in range(block_y):
        for tx in range(block_x):
            acc_v1 = _f32(0.0)
            acc_v2 = _f32(0.0)
            for batch in range(ty, B, block_y):
                for x in range(tx, S, block_x):
                    flat_idx = batch * S + x
                    dy = _f32(grad_output[flat_idx])
                    c = _f32(_f32(input_data[flat_idx]) - _f32(mean))
                    acc_v1 = _f32(acc_v1 + dy)
                    acc_v2 = _fma_f32(dy, c, acc_v2)
            thread_vals[tx + ty * block_x] = (acc_v1, acc_v2)

    # Phase 2: BlockReduce from block_reduce.cuh (two-level shfl_down)
    return block_reduce_cuh(thread_vals, float2_add, float2_id)


@torch.compile
def _compiled_bn_nchw_dx(go, inp, mean, proj_scale, grad_mean, grad_scale):
    """Elementwise dx from batch_norm_backward_kernel (NCHW).
    dx = (go - (inp - mean)*proj_scale - grad_mean) * grad_scale"""
    proj = (inp - mean) * proj_scale
    return (go - proj - grad_mean) * grad_scale


def _bn_nchw_block_config_fused(spatial):
    """Block config for the fused batch_norm_backward_kernel.
    CUDA ref: cuda/Normalization.cuh:679-681."""
    tf = 512
    for s in [32, 64, 128, 256, 512]:
        if spatial <= s:
            tf = s
            break
    return tf, max(1, 512 // tf)


def _bn_nchw_block_config_reduce(batch_size, feature_size):
    """Block config for the separate batch_norm_backward_reduce_kernel.
    CUDA ref: cuda/Normalization.cuh:854-858."""
    def _last_pow2(n):
        n |= (n >> 1); n |= (n >> 2); n |= (n >> 4)
        n |= (n >> 8); n |= (n >> 16)
        return max(1, n - (n >> 1))
    block_y = min(_last_pow2(batch_size), 512 // 32)
    tf = 512
    for s in [32, 64, 128, 256, 512]:
        if feature_size <= s:
            tf = s
            break
    block_x = min(max(tf, 32), 512 // block_y)
    return block_x, block_y


def spec_batch_norm_nchw_backward(grad, x, mean_t, invstd_t, weight,
                                  fused=True):
    """Full NCHW batch norm backward.
    fused=True: batch_norm_backward_kernel (reduce+elemt in one, when dx requested).
    fused=False: batch_norm_backward_reduce_kernel (separate reduce).
    CUDA ref: cuda/Normalization.cuh:387 (fused), :504 (reduce-only)."""
    B, C = x.shape[0], x.shape[1]
    spatial = 1
    for d in x.shape[2:]:
        spatial *= d
    N = B * spatial

    g = grad.detach().float().reshape(B, C, spatial)
    xv = x.detach().float().reshape(B, C, spatial)
    w = weight.detach().float()

    if fused:
        block_x, block_y = _bn_nchw_block_config_fused(spatial)
    else:
        block_x, block_y = _bn_nchw_block_config_reduce(B, spatial)

    dx = torch.zeros_like(g)
    dw = torch.zeros(C, device=x.device, dtype=torch.float32)
    db = torch.zeros(C, device=x.device, dtype=torch.float32)

    for c in range(C):
        m = mean_t[c].float().item()
        ist = invstd_t[c].float().item()
        wc = w[c].item()

        grad_ch = g[:, c, :].contiguous().view(-1).cpu().tolist()
        input_ch = xv[:, c, :].contiguous().view(-1).cpu().tolist()
        sum_dy, sum_dy_xmu = spec_batch_norm_nchw_backward_reduce(
            grad_ch, input_ch, m,
            block_x=block_x, block_y=block_y, B=B, S=spatial,
        )

        db[c] = sum_dy
        dw[c] = _f32(sum_dy_xmu * ist)

        # Elementwise phase
        norm = _f32(1.0 / N)
        grad_mean = _f32(sum_dy * norm)
        proj_scale = _f32(_f32(_f32(sum_dy_xmu * norm) * ist) * ist)
        grad_scale = _f32(ist * wc)

        dx[:, c, :] = _compiled_bn_nchw_dx(
            g[:, c, :], xv[:, c, :],
            torch.tensor(m, device=x.device, dtype=torch.float32),
            torch.tensor(proj_scale, device=x.device, dtype=torch.float32),
            torch.tensor(grad_mean, device=x.device, dtype=torch.float32),
            torch.tensor(grad_scale, device=x.device, dtype=torch.float32),
        )

    return (dx.reshape(x.shape), dw, db)


def spec_batch_norm_nhwc_backward_reduce(
    grad_output: list, input_data: list, mean: float,
    block_y: int = 16,
):
    """batch_norm_backward_reduce_channels_last_kernel (NHWC): Float2
    (sum_dy, sum_dy_xmu) accumulation via shmem_halving_reduce.
    Per-thread dy_xmu uses fma(dy, x-mean, acc).
    Returns (sum_dy, sum_dy_xmu).

    CUDA ref: cuda/Normalization.cuh:1198 (batch_norm_backward_reduce_channels_last_kernel)."""
    assert len(grad_output) == len(input_data)

    def float2_add(a, b):
        return (_f32(a[0] + b[0]), _f32(a[1] + b[1]))

    float2_id = (_f32(0.0), _f32(0.0))
    ELEMENTS_PER_ITER = 4

    thread_vals = []
    for tid in range(min(block_y, len(grad_output))):
        accs = [float2_id] * ELEMENTS_PER_ITER
        idx = tid
        acc_idx = 0
        while idx < len(grad_output):
            dy = _f32(grad_output[idx])
            c = _f32(_f32(input_data[idx]) - _f32(mean))
            a0, a1 = accs[acc_idx]
            accs[acc_idx] = (_f32(a0 + dy), _fma_f32(dy, c, a1))
            acc_idx = (acc_idx + 1) % ELEMENTS_PER_ITER
            idx += block_y
        result = accs[0]
        for i in range(1, ELEMENTS_PER_ITER):
            result = float2_add(result, accs[i])
        thread_vals.append(result)
    while len(thread_vals) < block_y:
        thread_vals.append(float2_id)

    return shmem_halving_reduce(thread_vals, float2_add, float2_id)


@torch.compile
def _compiled_bn_nhwc_dx(go, inp, m_c, factor_1_c, m_dy_c, factor_2_c):
    """Elementwise dx from batch_norm_backward_elemt_channels_last_kernel.
    dx = (go - m_dy_c - (inp - m_c)*factor_1_c) * factor_2_c"""
    return (go - m_dy_c - (inp - m_c) * factor_1_c) * factor_2_c


def spec_batch_norm_nhwc_backward(grad, x, mean_t, invstd_t, weight):
    """Full NHWC batch norm backward (separate reduce + elemt, grid.y=1).
    CUDA ref: cuda/Normalization.cuh:1198 (reduce), :1355 (elemt)."""
    B, C = x.shape[0], x.shape[1]
    spatial = 1
    for d in x.shape[2:]:
        spatial *= d
    N = B * spatial

    x_contig = x.to(memory_format=torch.contiguous_format)
    g_contig = grad.to(memory_format=torch.contiguous_format)
    g = g_contig.detach().float().reshape(B, C, spatial)
    xv = x_contig.detach().float().reshape(B, C, spatial)
    w = weight.detach().float()

    # NHWC reduce block config: flexible_launch_configs
    stride = C
    red = N  # reduction_size = B * H * W
    def _last_pow2(n):
        n |= (n >> 1); n |= (n >> 2); n |= (n >> 4)
        n |= (n >> 8); n |= (n >> 16)
        return max(1, n - (n >> 1))
    bx = min(_last_pow2(stride), 32)
    by = min(_last_pow2((red + 15) // 16), 512 // bx)
    if bx * by != 512:
        bx = min(_last_pow2(stride), 512 // by)

    dx = torch.zeros_like(g)
    dw = torch.zeros(C, device=x.device, dtype=torch.float32)
    db = torch.zeros(C, device=x.device, dtype=torch.float32)

    for c in range(C):
        m = mean_t[c].float().item()
        ist = invstd_t[c].float().item()
        wc = w[c].item()

        grad_ch = g[:, c, :].contiguous().view(-1).cpu().tolist()
        input_ch = xv[:, c, :].contiguous().view(-1).cpu().tolist()
        sum_dy, sum_dy_xmu = spec_batch_norm_nhwc_backward_reduce(
            grad_ch, input_ch, m, block_y=by,
        )

        db[c] = sum_dy
        dw[c] = _f32(sum_dy_xmu * ist)

        # Elementwise: same formula as NCHW elemt kernel
        norm_fct = _f32(1.0 / N)
        m_dy_c = _f32(sum_dy * norm_fct)
        factor_1_c = _f32(_f32(ist * ist) * _f32(sum_dy_xmu * norm_fct))
        factor_2_c = _f32(wc * ist)

        dx[:, c, :] = _compiled_bn_nhwc_dx(
            g[:, c, :], xv[:, c, :],
            torch.tensor(m, device=x.device, dtype=torch.float32),
            torch.tensor(factor_1_c, device=x.device, dtype=torch.float32),
            torch.tensor(m_dy_c, device=x.device, dtype=torch.float32),
            torch.tensor(factor_2_c, device=x.device, dtype=torch.float32),
        )

    return (dx.reshape(x_contig.shape), dw, db)


# ============================================================================
# Loss function specifications
# ============================================================================

def spec_nll_loss_reduce(losses: list, block_size: int = 512):
    """nll_loss_forward_reduce_cuda_kernel_2d: sequential + shmem halving.
    losses = list of per-sample weighted losses.

    CUDA ref: cuda/Loss.cu:224 (nll_loss_forward_reduce_cuda_kernel_2d)."""

    # Phase 1: per-thread sequential sum at stride block_size
    zero = _f32(0.0) if losses else 0.0
    thread_sums = [zero] * block_size
    for tid in range(min(block_size, len(losses))):
        for idx in range(tid, len(losses), block_size):
            thread_sums[tid] = _f32(thread_sums[tid] + losses[idx])

    # Phase 2: shared-memory halving tree (NO warp shuffles)
    return shmem_halving_reduce(thread_sums, lambda a, b: _f32(a + b), zero)


def spec_multi_margin_loss_thread0_scan(per_thread_sums: list):
    """MultiMarginLoss: thread 0 serial scan of all buffer entries.

    CUDA ref: cuda/MultiMarginLoss.cu:24 (MultiMarginLoss_forward_kernel)."""
    return thread_0_serial_scan(per_thread_sums, lambda a, b: _f32(a + b), 0.0)


# ============================================================================
# Cumulative scan specifications (ScanUtils.cuh)
# ============================================================================

def spec_cumsum_innermost_sklansky(row: list, num_threads_x: int = 128):
    """Sklansky parallel prefix scan for innermost dimension.
    Processes in chunks of 2*num_threads_x with carry between chunks.

    CUDA ref: cuda/ScanUtils.cuh:60 (tensor_kernel_scan_innermost_dim_with_indices), :114 (Sklansky tree loop)."""
    output = []
    carry = _f32(0.0)
    chunk_size = 2 * num_threads_x

    for chunk_start in range(0, len(row), chunk_size):
        chunk = list(row[chunk_start:chunk_start + chunk_size])
        while len(chunk) < chunk_size:
            chunk.append(_f32(0.0))
        # Add carry to first element
        chunk[0] = _f32(chunk[0] + carry)

        # Sklansky tree
        s = 1
        while s <= num_threads_x:
            new_chunk = list(chunk)
            for tid in range(num_threads_x):
                a = (tid // s) * (2 * s) + s
                ti = a + (tid % s)
                si = a - 1
                if ti < chunk_size and si < chunk_size:
                    new_chunk[ti] = _f32(chunk[si] + chunk[ti])
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
            output[row * num_cols + col] = _f32(
                output[(row - 1) * num_cols + col] + data_2d[row * num_cols + col]
            )
    return output


def spec_cumprod_innermost_sklansky(row: list, num_threads_x: int = 128):
    """Sklansky parallel prefix scan for innermost-dim cumprod.
    Same tree as cumsum but with multiplication instead of addition.

    CUDA ref: cuda/ScanUtils.cuh:60 (tensor_kernel_scan_innermost_dim_with_indices)."""
    output = []
    carry = _f32(1.0)
    chunk_size = 2 * num_threads_x

    for chunk_start in range(0, len(row), chunk_size):
        chunk = list(row[chunk_start:chunk_start + chunk_size])
        while len(chunk) < chunk_size:
            chunk.append(_f32(1.0))
        chunk[0] = _f32(chunk[0] * carry)

        s = 1
        while s <= num_threads_x:
            new_chunk = list(chunk)
            for tid in range(num_threads_x):
                a = (tid // s) * (2 * s) + s
                ti = a + (tid % s)
                si = a - 1
                if ti < chunk_size and si < chunk_size:
                    new_chunk[ti] = _f32(chunk[si] * chunk[ti])
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
            output[row * num_cols + col] = _f32(
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
    acc = _f32(0.0)
    for v in window_values:
        acc = _f32(acc + v)
    count = _f32(len(window_values))
    return _f32(acc / count)


def spec_embedding_bag_sum(embeddings: list):
    """EmbeddingBag SUM: sequential loop over bag, one thread per feature.

    CUDA ref: cuda/EmbeddingBag.cu:115 (EmbeddingBag_updateOutputKernel_sum_mean)."""
    acc = _f32(0.0)
    for emb in embeddings:
        acc = _f32(acc + emb)
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
    zero = _f32(0.0) if grad else 0.0

    # Pre-multiply (done outside kernel in softmax_backward_cuda_out)
    tmp = [_f32(grad[i] * output[i]) for i in range(n)]

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
                acc = _f32(acc + tmp[offset * ILP + j])
            offset += B
        tail_offset = main_end + tid
        while tail_offset < n:
            acc = _f32(acc + tmp[tail_offset])
            tail_offset += B
        thread_sums[tid] = acc

    # Phase 2: BlockReduceSum (block_reduce.cuh)
    row_sum = block_reduce_cuh(thread_sums, lambda a, b: _f32(a + b), zero)

    # Phase 3: epilogue: tmp[i] - output[i] * sum  (SoftMax.cu:86)
    return [_f32(tmp[i] - _f32(output[i] * row_sum)) for i in range(n)]


# NOTE: spec_softmax_backward_spatial is not provided because it is
# structurally identical to spec_softmax_spatial — a purely sequential
# per-thread loop computing sum(grad*output) for each inner position.
#
# NOTE: spec_softmax_backward_persistent is not provided because it uses
# the same SHFL_XOR butterfly tree as spec_softmax_persistent.


def spec_log_softmax_backward_spatial(grad_2d: list, output_2d: list,
                                       dim_size: int, inner_size: int,
                                       dtype=None) -> list:
    """Spatial log_softmax backward: sum(grad) per inner position via
    spatialBlockReduceX, then grad_input = grad - exp(output) * sum(grad).
    Same spatial tree as forward for the sum reduction.

    CUDA ref: cuda/SoftMax.cu:262 (cunn_SpatialSoftMaxBackward)."""
    exp = _dtype_exp(dtype)
    cast = _dtype_cast(dtype)
    zero = _dtype_identity(dtype, 0.0)
    result = [zero] * len(grad_2d)

    max_block = 1024
    inner_threads = min(inner_size, max_block)
    dim_threads = 1
    if inner_threads <= 64 and dim_size >= 64:
        while inner_threads * dim_threads <= max_block and dim_threads <= dim_size:
            dim_threads *= 2
        dim_threads //= 2

    for inner in range(inner_size):
        if dim_threads == 1:
            s = zero
            for d in range(dim_size):
                s = cast(s + grad_2d[d * inner_size + inner])
        else:
            thread_sums = [zero] * dim_threads
            for tx in range(dim_threads):
                for d in range(tx, dim_size, dim_threads):
                    thread_sums[tx] = cast(thread_sums[tx] + grad_2d[d * inner_size + inner])
            s = shmem_halving_reduce(thread_sums, lambda a, b: cast(a + b), zero)

        s_t = torch.tensor(float(s), device="cuda", dtype=torch.float32)
        for d in range(dim_size):
            idx = d * inner_size + inner
            g_t = torch.tensor(float(grad_2d[idx]), device="cuda",
                               dtype=torch.float32)
            exp_o = torch.tensor(float(output_2d[idx]), device="cuda",
                                 dtype=torch.float32).exp()
            result[idx] = _compiled_softmax_backward_epilogue(
                g_t, exp_o, s_t).item()
    return result


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

    add = lambda a, b: _f32(a + b)  # noqa: E731

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
    # Infer zero from data dtype (float stays fp32)
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
        dgamma[j] = shfl_xor_butterfly(thread_sums, lambda a, b: _f32(a + b))
    return dgamma


def spec_layer_norm_backward_dbeta(dY: torch.Tensor,
                                   block_dim_y: int = 32,
                                   rows_per_block_y: int = 256) -> torch.Tensor:
    """GammaBetaBackwardCUDAKernelTemplate: dbeta reduction (tensor-based).
    Per-thread sequential sum over contiguous row blocks, then SHFL_XOR
    butterfly high-to-low across block_dim_y.

    CUDA ref: cuda/layer_norm_kernel.cu:768 (GammaBetaBackwardCUDAKernelTemplate)."""
    M, N = dY.shape
    rows_per_thread_y = rows_per_block_y // block_dim_y

    # Phase 1: per-thread sequential accumulation (contiguous row blocks)
    thread_sums = torch.zeros(block_dim_y, N, device=dY.device, dtype=dY.dtype)
    for M_start in range(0, M, rows_per_block_y):
        for tid_y in range(block_dim_y):
            for i in range(rows_per_thread_y):
                m = M_start + tid_y * rows_per_thread_y + i
                if m < M:
                    thread_sums[tid_y] = thread_sums[tid_y] + dY[m]

    # Phase 2: SHFL_XOR butterfly high-to-low
    if block_dim_y <= 1:
        return thread_sums[0]
    vals = thread_sums.clone()
    delta = block_dim_y // 2
    while delta >= 1:
        new_vals = vals.clone()
        for i in range(block_dim_y):
            partner = i ^ delta
            if 0 <= partner < block_dim_y:
                new_vals[i] = vals[i] + vals[partner]
        vals = new_vals
        delta //= 2
    return vals[0]


def spec_gamma_beta_backward_dgamma_simple(
    dY_rows: list, X_rows: list, mean_list: list, rstd_list: list,
) -> list:
    """GammaBetaBackwardCUDAKernelTemplate with block_dim_y=1: purely sequential
    dgamma per feature.  Uses _fma_f32 to match nvcc's FMA contraction:
    fma(dY*(X-mean), rstd, acc).

    CUDA ref: cuda/layer_norm_kernel.cu:768 (with block_dim_y=1, partial_reduction=true)."""
    M = len(dY_rows)
    N = len(dY_rows[0]) if M > 0 else 0
    dgamma = [_f32(0.0)] * N
    for j in range(N):
        acc = _f32(0.0)
        for m in range(M):
            t = _f32(_f32(dY_rows[m][j]) * _f32(_f32(X_rows[m][j]) - _f32(mean_list[m])))
            acc = _fma_f32(t, _f32(rstd_list[m]), acc)
        dgamma[j] = acc
    return dgamma


def spec_group_norm_backward_internal(grad_3d, x_3d, N_batch, C, HxW, num_threads=32):
    """ComputeInternalGradientsCUDAKernel: per-(n,c) pair compute
    ds[n,c] = sum_hw(dY*X) and db[n,c] = sum_hw(dY).
    Tree: sequential at stride num_threads, then WarpReduceSum (shfl_down).
    FMA in ds: fma(dY, X, acc).
    Returns (ds, db) as lists of shape [N_batch][C].

    CUDA ref: cuda/group_norm_kernel.cu:276 (ComputeInternalGradientsCUDAKernel)."""
    g = grad_3d.reshape(N_batch, C, HxW).cpu().tolist()
    xv = x_3d.reshape(N_batch, C, HxW).cpu().tolist()
    ds = [[0.0] * C for _ in range(N_batch)]
    db = [[0.0] * C for _ in range(N_batch)]
    add = lambda a, b: _f32(a + b)  # noqa: E731
    for n in range(N_batch):
        for c in range(C):
            ts_ds = [_f32(0.0)] * num_threads
            ts_db = [_f32(0.0)] * num_threads
            for tid in range(num_threads):
                acc_ds, acc_db = _f32(0.0), _f32(0.0)
                for hw in range(tid, HxW, num_threads):
                    acc_ds = _fma_f32(_f32(g[n][c][hw]), _f32(xv[n][c][hw]), acc_ds)
                    acc_db = _f32(acc_db + _f32(g[n][c][hw]))
                ts_ds[tid] = acc_ds
                ts_db[tid] = acc_db
            ds[n][c] = shfl_down_reduce_high_to_low(ts_ds, add)
            db[n][c] = shfl_down_reduce_high_to_low(ts_db, add)
    return ds, db


def spec_group_norm_backward_dw_db(ds, db, mean_t, rstd_t, N_batch, C, G):
    """GammaBetaBackwardCUDAKernel1: sequential loop over N for dgamma and dbeta.
    dgamma[c] = sum_n((ds[n,c] - db[n,c]*mean[n,g]) * rstd[n,g])
    dbeta[c] = sum_n(db[n,c])
    FMA contractions: fma(-db, mean, ds) and fma(result, rstd, sum1).
    Returns (dgamma, dbeta) lists of length C.

    CUDA ref: cuda/group_norm_kernel.cu:355 (GammaBetaBackwardCUDAKernel1)."""
    D = C // G
    ml = mean_t.cpu().tolist()
    rl = rstd_t.cpu().tolist()
    dgamma = [_f32(0.0)] * C
    dbeta = [_f32(0.0)] * C
    for c in range(C):
        g = c // D
        dg_acc = _f32(0.0)
        db_acc = _f32(0.0)
        for n in range(N_batch):
            t = _fma_f32(-_f32(db[n][c]), _f32(ml[n][g]), _f32(ds[n][c]))
            dg_acc = _fma_f32(t, _f32(rl[n][g]), dg_acc)
            db_acc = _f32(db_acc + _f32(db[n][c]))
        dgamma[c] = dg_acc
        dbeta[c] = db_acc
    return dgamma, dbeta


@torch.compile
def _compiled_gn_c2c3(sum1, sum2, mean_ng, rstd_ng, s):
    """ComputeBackwardFusedParamsCUDAKernel: c2, c3 formula (thread 0).
    FMA matching: (sum2*mean - sum1) * rstd^3 * s and -c2*mean - sum2*rstd*s.

    CUDA ref: cuda/group_norm_kernel.cu:344-351."""
    t = sum2 * mean_ng - sum1
    c2 = t * rstd_ng * rstd_ng * rstd_ng * s
    c3 = -c2 * mean_ng - sum2 * rstd_ng * s
    return c2, c3


@torch.compile
def _compiled_gn_dx_elem(c1, dy, c2, x, c3):
    """Group norm elementwise dx = c1*dy + c2*x + c3 with FMA matching.

    CUDA ref: cuda/group_norm_kernel.cu:903."""
    return c1 * dy + c2 * x + c3


def spec_group_norm_backward_dx(grad, x, mean_t, rstd_t, weight, ds, db,
                                N_batch, C, HxW, G, num_threads=32):
    """Group norm backward dx: WarpReduceSum tree for sum1/sum2 over D channels,
    compiled c2/c3 formula, compiled elementwise dx = c1*dy + c2*x + c3.

    CUDA ref: cuda/group_norm_kernel.cu:307, :891-923."""
    D = C // G
    wl = weight.cpu().tolist()
    add = lambda a, b: _f32(a + b)  # noqa: E731
    nt2 = 32 if D < 512 else 512

    c2_t = torch.zeros(N_batch, G, device=grad.device, dtype=torch.float32)
    c3_t = torch.zeros(N_batch, G, device=grad.device, dtype=torch.float32)

    for n in range(N_batch):
        for gi in range(G):
            vals1 = [_f32(0.0)] * nt2
            vals2 = [_f32(0.0)] * nt2
            for tid in range(nt2):
                a1, a2 = _f32(0.0), _f32(0.0)
                for i in range(tid, D, nt2):
                    c_idx = gi * D + i
                    gamma_v = _f32(wl[c_idx])
                    a1 = _f32(a1 + _f32(_f32(ds[n][c_idx]) * gamma_v))
                    a2 = _f32(a2 + _f32(_f32(db[n][c_idx]) * gamma_v))
                vals1[tid] = a1
                vals2[tid] = a2
            if nt2 <= 32:
                s1 = shfl_down_reduce_high_to_low(vals1, add)
                s2 = shfl_down_reduce_high_to_low(vals2, add)
            else:
                s1 = block_reduce_cuh(vals1, add, _f32(0.0))
                s2 = block_reduce_cuh(vals2, add, _f32(0.0))

            s_val = 1.0 / (D * HxW)
            c2v, c3v = _compiled_gn_c2c3(
                torch.tensor(s1, device='cuda'), torch.tensor(s2, device='cuda'),
                mean_t[n, gi], rstd_t[n, gi],
                torch.tensor(s_val, device='cuda'),
            )
            c2_t[n, gi] = c2v
            c3_t[n, gi] = c3v

    c1_t = rstd_t.view(N_batch, G, 1) * weight.view(1, G, D)
    return _compiled_gn_dx_elem(
        c1_t.unsqueeze(-1),
        grad.reshape(N_batch, G, D, HxW),
        c2_t.view(N_batch, G, 1, 1),
        x.reshape(N_batch, G, D, HxW),
        c3_t.view(N_batch, G, 1, 1),
    ).reshape(grad.shape)


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
            shfl_down_reduce_high_to_low(lane_sums, lambda a, b: _f32(a + b))
        )
    return results


# ============================================================================
# Reference backward specifications (mathematical gradient formulas)
# ============================================================================
# Sequential torch-CUDA implementations of gradient formulas.  These don't
# model the CUDA kernel's reduction tree but compute mathematically correct
# gradients on the same device, in float64, as independent references.

def ref_prod_backward(grad, x):
    """Gradient of prod(x, dim=-1).  Matches C++ autograd:
    grad * (result / input), which is used when no zeros present."""
    xf = x.detach().float()
    gf = grad.detach().float()
    result = torch.prod(xf, dim=-1)
    return gf.unsqueeze(-1) * (result.unsqueeze(-1) / xf)


def ref_var_backward(grad, x, correction=1):
    """Gradient of var(x, dim=-1).  Matches autograd decomposition:
    (grad * scalar(2/(N-correction))).unsqueeze(-1) * (x - mean)."""
    xf = x.detach().float()
    gf = grad.detach().float()
    N = xf.shape[-1]
    mean = xf.mean(dim=-1, keepdim=True)
    return (gf * (2.0 / (N - correction))).unsqueeze(-1) * (xf - mean)


def ref_std_backward(grad, x, correction=1):
    """Gradient of std(x, dim=-1).  Matches autograd decomposition:
    coeff = grad / (2*std), masked to 0 where std==0, then
    (coeff * scalar(2/(N-correction))).unsqueeze(-1) * (x - mean)."""
    xf = x.detach().float()
    gf = grad.detach().float()
    N = xf.shape[-1]
    std_fwd = torch.std(xf, dim=-1)
    coeff = gf / (std_fwd * 2)
    coeff = coeff.masked_fill(std_fwd == 0, 0)
    mean = xf.mean(dim=-1, keepdim=True)
    return (coeff * (2.0 / (N - correction))).unsqueeze(-1) * (xf - mean)


def ref_norm_l2_backward(grad, x):
    """Gradient of ||x||_2 along dim=-1.  Matches autograd decomposition:
    grad * (x / norm), where division happens first."""
    xf = x.detach().float()
    gf = grad.detach().float()
    norm = torch.linalg.vector_norm(xf, 2, dim=-1)
    div_result = xf / norm.unsqueeze(-1)
    div_result = div_result.masked_fill(norm.unsqueeze(-1) == 0, 0)
    return gf.unsqueeze(-1) * div_result


def ref_norm_lp_backward(grad, x, p):
    """Gradient of ||x||_p along dim=0.  Matches autograd decomposition:
    (x * |x|^(p-2)) * (grad / norm^(p-1))."""
    xf = x.detach().float()
    gf = grad.detach().float()
    norm = torch.linalg.vector_norm(xf, p, dim=0)
    signed = xf * xf.abs().pow(p - 2)
    coeff = gf / norm.pow(p - 1)
    coeff = coeff.masked_fill(norm == 0, 0)
    return signed * coeff.unsqueeze(0)


def ref_nll_loss_backward(num_samples, num_classes, target, reduction="mean"):
    """Gradient of nll_loss (input = log_probs).  Scatter -1 at target."""
    dx = torch.zeros(num_samples, num_classes, device=target.device)
    dx.scatter_(1, target.unsqueeze(1), -1.0)
    if reduction == "mean":
        dx /= num_samples
    return dx


def ref_nll_loss_2d_backward(x_shape, target, reduction="mean"):
    """Gradient of nll_loss for 4D input.  Scatter -1 at target positions."""
    B, C, H, W = x_shape
    dx = torch.zeros(x_shape, device=target.device)
    N = B * H * W
    for b in range(B):
        for h in range(H):
            for w in range(W):
                dx[b, target[b, h, w], h, w] = -1.0
    if reduction == "mean":
        dx /= N
    return dx


def ref_cross_entropy_forward(x, target):
    """Cross entropy = log_softmax + nll_loss.  Uses CUDA kernels directly."""
    xf = x.detach().float()
    log_soft = torch.nn.functional.log_softmax(xf, dim=-1)
    return torch.nn.functional.nll_loss(log_soft, target).item()


def ref_cross_entropy_backward(x, target):
    """Gradient of cross_entropy (mean).  Matches autograd decomposition:
    log_softmax forward, nll_loss_backward, _log_softmax_backward_data."""
    xf = x.detach().float()
    N = xf.shape[0]
    log_soft = torch.ops.aten._log_softmax.default(xf, 1, False)
    nll_grad = torch.zeros_like(xf)
    nll_grad.scatter_(1, target.unsqueeze(1), -1.0 / N)
    return torch.ops.aten._log_softmax_backward_data.default(
        nll_grad, log_soft, 1, torch.float32)


def ref_multi_margin_loss_backward(x, target, p=1, margin=1.0):
    """Gradient of multi_margin_loss (p=1, mean reduction)."""
    xf = x.detach().float()
    N, C = xf.shape
    dx = torch.zeros_like(xf)
    for i in range(N):
        tgt = target[i].item()
        for c in range(C):
            if c == tgt:
                continue
            if margin - xf[i, tgt].item() + xf[i, c].item() > 0:
                dx[i, c] += 1.0 / C
                dx[i, tgt] -= 1.0 / C
    dx /= N
    return dx


def ref_multilabel_margin_loss_forward(x, target):
    """Multilabel margin loss matching CUDA kernel (MultiLabelMarginCriterion.cu).
    Per-sample: 128-thread sequential hinge + BlockReduceSum, /dim /nframe.
    Final: sum per-sample values via gpu_reduce_kernel."""
    xf = x.detach().float()
    B, C = xf.shape
    THREADS = 128
    per_sample = []
    for bi in range(B):
        target_set = set()
        for j in range(C):
            if target[bi, j].item() < 0:
                break
            target_set.add(target[bi, j].item())
        thread_sums = [_f32(0.0)] * THREADS
        for dt in range(C):
            if target[bi, dt].item() < 0:
                break
            tgt_label = target[bi, dt].item()
            input_target_k = xf[bi, tgt_label].item()
            for tid in range(THREADS):
                for d in range(tid, C, THREADS):
                    if d not in target_set:
                        z = _f32(_f32(1 - input_target_k) + xf[bi, d].item())
                        if z > 0:
                            thread_sums[tid] = _f32(thread_sums[tid] + float(z))
        total = block_reduce_cuh(
            thread_sums, lambda a, b: _f32(a + b), _f32(0.0))
        per_sample.append(_f32(_f32(float(total) / C) / B))
    add = lambda a, b: _f32(a + b)
    return float(spec_gpu_reduce_kernel(
        per_sample, add, _f32(0.0), num_outputs=1, stride1=True))


def ref_multilabel_margin_loss_backward(x, target):
    """Gradient of multilabel_margin_loss matching CUDA backward kernel.
    Per-sample: g = 1/(nframe*dim), thread-local grad_input += g or -= g,
    then BlockReduceSum for target gradient, then multiply by grad_output."""
    xf = x.detach().float()
    B, C = xf.shape
    THREADS = 128
    dx = torch.zeros_like(xf)
    g = _f32(1.0 / _f32(B * C))
    for bi in range(B):
        target_set = set()
        for j in range(C):
            if target[bi, j].item() < 0:
                break
            target_set.add(target[bi, j].item())
        for dt in range(C):
            if target[bi, dt].item() < 0:
                break
            tgt_label = target[bi, dt].item()
            input_target_k = xf[bi, tgt_label].item()
            thread_sums = [_f32(0.0)] * THREADS
            for tid in range(THREADS):
                for d in range(tid, C, THREADS):
                    if d not in target_set:
                        z = _f32(_f32(1 - input_target_k) + xf[bi, d].item())
                        if z > 0:
                            thread_sums[tid] = _f32(thread_sums[tid] - g)
                            dx[bi, d] = _f32(dx[bi, d].item() + g)
            total_sum = block_reduce_cuh(
                thread_sums, lambda a, b: _f32(a + b), _f32(0.0))
            dx[bi, tgt_label] = _f32(dx[bi, tgt_label].item() + float(total_sum))
    return dx


def ref_ctc_loss_forward_backward(log_probs, targets, input_lengths,
                                   target_lengths, blank=0):
    """CTC forward + backward via alpha-beta DP.  Returns (loss, grad).
    All computations in float32 CUDA matching LossCTC.cu kernel exactly:
    -INFINITY, 3-way logaddexp, kernel beta convention, naive collect."""
    T, N, C = log_probs.shape
    dev = log_probs.device
    lp = log_probs.detach().float()
    NEG_INF = float('-inf')
    targets_cpu = targets.cpu()
    il_cpu = input_lengths.cpu()
    tl_cpu = target_lengths.cpu()

    grad = torch.zeros(T, N, C, dtype=torch.float32, device=dev)
    losses = []

    for b in range(N):
        inp_len = il_cpu[b].item()
        tgt_len = tl_cpu[b].item()
        tgt = targets_cpu[b, :tgt_len].long()
        S = 2 * tgt_len + 1
        labels = [blank] * S
        for k in range(tgt_len):
            labels[2 * k + 1] = tgt[k].item()
        labels_t = torch.tensor(labels, dtype=torch.long, device=dev)

        # Precompute have_three masks
        have_three_fwd = torch.zeros(S, dtype=torch.bool, device=dev)
        for s in range(2, S):
            if labels[s] != blank and labels[s] != labels[s - 2]:
                have_three_fwd[s] = True
        have_three_bwd = torch.zeros(S, dtype=torch.bool, device=dev)
        for s in range(S):
            if s + 2 < S and labels[s + 2] != blank and labels[s + 2] != labels[s]:
                have_three_bwd[s] = True

        # lp[t, label'[s]] for all t and s
        lp_labels = lp[:inp_len, b, :].gather(
            1, labels_t.unsqueeze(0).expand(inp_len, S))

        # Forward alpha DP: la[t,s] = logaddexp3(la[t-1,s],la[t-1,s-1],la[t-1,s-2]) + lp[t,label'[s]]
        la = torch.full((inp_len, S), NEG_INF, dtype=torch.float32, device=dev)
        la[0, 0] = lp_labels[0, 0]
        if S > 1:
            la[0, 1] = lp_labels[0, 1]
        for t in range(1, inp_len):
            la1 = la[t - 1]
            la2 = torch.full((S,), NEG_INF, dtype=torch.float32, device=dev)
            if S > 1:
                la2[1:] = la[t - 1, :S - 1]
            la3 = torch.full((S,), NEG_INF, dtype=torch.float32, device=dev)
            if S > 2:
                la3[2:][have_three_fwd[2:]] = la[t - 1, :S - 2][have_three_fwd[2:]]
            lamax = torch.max(la1, torch.max(la2, la3))
            lamax = torch.where(lamax == NEG_INF, torch.zeros_like(lamax), lamax)
            la[t] = (torch.log(torch.exp(la1 - lamax)
                               + torch.exp(la2 - lamax)
                               + torch.exp(la3 - lamax))
                     + lamax + lp_labels[t])

        # Loss (eq 8)
        l1 = la[inp_len - 1, S - 1]
        l2 = (la[inp_len - 1, S - 2] if S > 1
              else torch.tensor(NEG_INF, dtype=torch.float32, device=dev))
        m = torch.max(l1, l2)
        m = torch.where(m == NEG_INF, torch.zeros_like(m), m)
        log_p = torch.log(torch.exp(l1 - m) + torch.exp(l2 - m)) + m
        nll = -log_p

        # Backward beta DP (kernel convention: includes lp at each timestep)
        lb = torch.full((inp_len, S), NEG_INF, dtype=torch.float32, device=dev)
        lb[inp_len - 1, S - 1] = lp_labels[inp_len - 1, S - 1]
        if S > 1:
            lb[inp_len - 1, S - 2] = lp_labels[inp_len - 1, S - 2]
        for t in range(inp_len - 2, -1, -1):
            lb1 = lb[t + 1]
            lb2 = torch.full((S,), NEG_INF, dtype=torch.float32, device=dev)
            if S > 1:
                lb2[:S - 1] = lb[t + 1, 1:]
            lb3 = torch.full((S,), NEG_INF, dtype=torch.float32, device=dev)
            if S > 2:
                lb3[:S - 2][have_three_bwd[:S - 2]] = lb[t + 1, 2:][have_three_bwd[:S - 2]]
            lbmax = torch.max(lb1, torch.max(lb2, lb3))
            lbmax = torch.where(lbmax == NEG_INF, torch.zeros_like(lbmax), lbmax)
            lb[t] = (torch.log(torch.exp(lb1 - lbmax)
                               + torch.exp(lb2 - lbmax)
                               + torch.exp(lb3 - lbmax))
                     + lbmax + lp_labels[t])

        # Gradient collection (naive collect kernel: sequential logaddexp over s)
        collected = torch.full((inp_len, C), NEG_INF, dtype=torch.float32, device=dev)
        for s in range(S):
            c = labels[s]
            log_ab = la[:, s] + lb[:, s]  # (inp_len,)
            cur = collected[:, c]
            is_neg = (cur == NEG_INF)
            mx = torch.max(cur, log_ab)
            safe = torch.log(torch.exp(cur - mx) + torch.exp(log_ab - mx)) + mx
            collected[:, c] = torch.where(is_neg, log_ab, safe)

        # gradient = exp(lp) - exp(collected + nll - lp)
        lp_slice = lp[:inp_len, b, :]
        grad_b = lp_slice.exp() - (collected + nll - lp_slice).exp()
        grad[:inp_len, b, :] = grad_b
        losses.append(nll)

    # Mean reduction: divide by target_lengths then mean over batch
    # Use CUDA tensor operations to match (res / target_lengths_t).mean()
    loss_t = torch.stack(losses)  # (N,) float32 CUDA
    tl_float = target_lengths[:N].float().clamp(min=1)
    loss_t = loss_t / tl_float
    mean_loss = loss_t.mean().item()
    for b in range(N):
        tl_b = float(tl_cpu[b].item())
        grad[:, b, :] = grad[:, b, :] / tl_b
    grad = grad / N
    return mean_loss, grad


def ref_cumprod_backward(grad, x, dim):
    """Gradient of cumprod.  Uses the native aten decomposition which
    decomposes into per-position prod/cumprod/sum calls (O(n^2) total)."""
    return torch.ops.aten.cumprod_backward.default(
        grad, x.detach(), dim, torch.cumprod(x.detach(), dim=dim))


def ref_logcumsumexp_backward(grad, x, output, dim):
    """Gradient of logcumsumexp.  Matches autograd decomposition:
    split grad into positive/negative parts in log-space, reverse-logcumsumexp
    each, add x, exp, then subtract."""
    xf = x.detach().float()
    gf = grad.detach().float()
    of = output.detach().float()
    NEG_INF = torch.tensor(-3.4028234663852886e+38, dtype=torch.float32,
                           device=x.device)
    abs_g = gf.abs()
    log_abs_g = abs_g.log()
    log_pos = torch.where(gf > 0, log_abs_g, NEG_INF)
    log_neg = torch.where(gf < 0, log_abs_g, NEG_INF)
    rev_pos = (log_pos - of).flip(dim)
    exp_pos = (torch.logcumsumexp(rev_pos, dim=dim).flip(dim) + xf).exp()
    rev_neg = (log_neg - of).flip(dim)
    exp_neg = (torch.logcumsumexp(rev_neg, dim=dim).flip(dim) + xf).exp()
    return exp_pos - exp_neg


def ref_scatter_add_backward(grad, idx):
    """Gradient of scatter_add w.r.t. src: gather from grad at idx."""
    return grad.gather(0, idx)


def ref_embedding_bag_backward_weight(grad_output, idx, offsets,
                                       num_embeddings, embedding_dim):
    """Gradient of embedding_bag(mode=sum) w.r.t. weight: scatter_add."""
    dw = torch.zeros(num_embeddings, embedding_dim, device=grad_output.device,
                     dtype=torch.float32)
    num_bags = offsets.shape[0]
    for bag in range(num_bags):
        start = offsets[bag].item()
        end = offsets[bag + 1].item() if bag + 1 < num_bags else idx.shape[0]
        for j in range(start, end):
            dw[idx[j].item()] += grad_output[bag].float()
    return dw


def ref_group_norm_backward(grad, x, mean_t, rstd_t, weight,
                             num_groups):
    """Full group norm backward: dx, dw, db."""
    orig_shape = x.shape
    B, C = orig_shape[0], orig_shape[1]
    HxW = 1
    for d in orig_shape[2:]:
        HxW *= d
    g = grad.detach().double().reshape(B, C, HxW)
    xv = x.detach().double().reshape(B, C, HxW)
    w = weight.detach().double()
    cpg = C // num_groups

    dx = torch.zeros_like(g)
    dw = torch.zeros(C, device=x.device, dtype=torch.float64)
    db = torch.zeros(C, device=x.device, dtype=torch.float64)

    for bi in range(B):
        for gi in range(num_groups):
            cs, ce = gi * cpg, (gi + 1) * cpg
            m = mean_t[bi, gi].double()
            r = rstd_t[bi, gi].double()
            gs = cpg * HxW
            ds = torch.tensor(0.0, device=x.device, dtype=torch.float64)
            db_l = torch.tensor(0.0, device=x.device, dtype=torch.float64)
            for c in range(cs, ce):
                xh = (xv[bi, c] - m) * r
                ds += (g[bi, c] * w[c] * xh).sum()
                db_l += (g[bi, c] * w[c]).sum()
            for c in range(cs, ce):
                xh = (xv[bi, c] - m) * r
                dx[bi, c] = r * (w[c] * g[bi, c] - (db_l + xh * ds) / gs)

    for c in range(C):
        gi = c // cpg
        for bi in range(B):
            m = mean_t[bi, gi].double()
            r = rstd_t[bi, gi].double()
            xh = (xv[bi, c] - m) * r
            dw[c] += (g[bi, c] * xh).sum()
            db[c] += g[bi, c].sum()

    return (dx.reshape(orig_shape).float(), dw.float(), db.float())


def ref_batch_norm_backward(grad, x, mean_t, invstd_t, weight):
    """Full batch norm backward: dx, dw, db."""
    orig_shape = x.shape
    B, C = orig_shape[0], orig_shape[1]
    spatial = 1
    for d in orig_shape[2:]:
        spatial *= d
    N = B * spatial
    g = grad.detach().double().reshape(B, C, spatial)
    xv = x.detach().double().reshape(B, C, spatial)
    w = weight.detach().double()

    dx = torch.zeros_like(g)
    dw = torch.zeros(C, device=x.device, dtype=torch.float64)
    db = torch.zeros(C, device=x.device, dtype=torch.float64)

    for c in range(C):
        m = mean_t[c].double()
        ist = invstd_t[c].double()
        xh = (xv[:, c, :] - m) * ist
        sum_dy = g[:, c, :].sum()
        sum_dy_xhat = (g[:, c, :] * xh).sum()
        dw[c] = sum_dy_xhat
        db[c] = sum_dy
        dx[:, c, :] = w[c] * ist * (g[:, c, :] - (sum_dy + xh * sum_dy_xhat) / N)

    return (dx.reshape(orig_shape).float(), dw.float(), db.float())


def ref_avg_pool2d_backward(grad_output, input_shape, kH, kW,
                             sH, sW, pH, pW, count_include_pad=False):
    """Gradient of avg_pool2d matching CUDA kernel gather loop (float32).

    The kernel computes each input gradient independently: for each input
    position, iterate over contributing output positions and accumulate
    top_diff[ph][pw] / divide_factor in float32.
    """
    B, C, iH, iW = input_shape
    oH = (iH + 2 * pH - kH) // sH + 1
    oW = (iW + 2 * pW - kW) // sW + 1
    g_list = grad_output.detach().cpu().tolist()
    dx = [[[[0.0] * iW for _ in range(iH)] for _ in range(C)]
          for _ in range(B)]
    for b in range(B):
        for c in range(C):
            for ih in range(iH):
                for iw in range(iW):
                    h = ih + pH
                    w = iw + pW
                    phstart = 0 if h < kH else (h - kH) // sH + 1
                    phend = min(h // sH + 1, oH)
                    pwstart = 0 if w < kW else (w - kW) // sW + 1
                    pwend = min(w // sW + 1, oW)
                    gradient = _f32(0.0)
                    for ph in range(phstart, phend):
                        for pw in range(pwstart, pwend):
                            hstart = ph * sH - pH
                            wstart = pw * sW - pW
                            hend = min(hstart + kH, iH + pH)
                            wend = min(wstart + kW, iW + pW)
                            if count_include_pad:
                                divide_factor = (hend - hstart) * (wend - wstart)
                            else:
                                hs = max(hstart, 0)
                                ws = max(wstart, 0)
                                he = min(hend, iH)
                                we = min(wend, iW)
                                divide_factor = (he - hs) * (we - ws)
                            gradient = _f32(gradient + _f32(
                                g_list[b][c][ph][pw] / divide_factor))
                    dx[b][c][ih][iw] = gradient
    return torch.tensor(dx, device=grad_output.device, dtype=torch.float32)


def ref_adaptive_avg_pool2d_backward(grad_output, input_shape):
    """Gradient of adaptive_avg_pool2d(x, (1,1)): grad / (H*W)."""
    B, C, H, W = input_shape
    g = grad_output.detach().float()
    dx = torch.zeros(input_shape, device=grad_output.device, dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            dx[b, c, :, :] = g[b, c, 0, 0] / (H * W)
    return dx


def ref_segment_reduce_sum_1d(data, lengths):
    """CUB DeviceSegmentedReduce::Reduce — opaque tree reduction."""
    d64 = data.detach().double()
    n_seg = lengths.shape[0]
    result = torch.zeros(n_seg, device=data.device, dtype=torch.float64)
    off = 0
    for i in range(n_seg):
        length = lengths[i].item()
        for j in range(length):
            result[i] += d64[off + j]
        off += length
    return result.float()


def ref_segment_reduce_sum_2d(data, lengths):
    """segment_reduce_forward_kernel: one thread per (segment, feature),
    sequential float32 accumulation."""
    d = data.detach().cpu().tolist()
    n_seg = lengths.shape[0]
    n_feat = data.shape[1]
    result = [[_f32(0.0)] * n_feat for _ in range(n_seg)]
    off = 0
    for i in range(n_seg):
        length = lengths[i].item()
        for f in range(n_feat):
            acc = _f32(0.0)
            for j in range(length):
                acc = _f32(acc + d[off + j][f])
            result[i][f] = acc
        off += length
    return torch.tensor(result, device=data.device, dtype=torch.float32)


def ref_foreach_norm_l2(t):
    """ForeachReduceOp.cu L2 norm: kBlockSize=512, kILP=4, BlockReduceSum.

    Aligned path: thread tid loads elements [tid*4 .. tid*4+3], accumulates
    4 independent squared values, then sums them left-to-right.
    BlockReduceSum across 512 threads, then sqrt.
    """
    import math
    vals = t.detach().cpu().tolist()
    n = len(vals)
    BLOCK = 512
    ILP = 4
    thread_vals = [_f32(0.0)] * BLOCK
    for tid in range(BLOCK):
        accs = [_f32(0.0)] * ILP
        base = tid * ILP
        if base < n:
            for ii in range(ILP):
                idx = base + ii
                if idx < n:
                    v = _f32(vals[idx])
                    accs[ii] = _f32(accs[ii] + _f32(v * v))
        val = _f32(0.0)
        for ii in range(ILP):
            val = _f32(val + accs[ii])
        thread_vals[tid] = val
    total = block_reduce_cuh(
        thread_vals, lambda a, b: _f32(a + b), _f32(0.0))
    return _f32(math.sqrt(float(total)))


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
        add = lambda a, b: _f32(a + b)
        for i in range(result.shape[0]):
            row = x[i].cpu().tolist()
            spec_val = spec_gpu_reduce_kernel(row, add, _f32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_nonstride1_split(self):
        # Reducing dim=0 on contiguous -> non-stride-1, split_across_warps.
        # Use prime num_outputs to avoid output vectorization (output_vec_size=1).
        # On GPUs with many SMs, global reduce (ctas_per_output > 1) is triggered.
        x = torch.randn(5000, 97, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: _f32(a + b)
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            5000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        for i in range(min(4, result.shape[0])):
            col = x[:, i].cpu().tolist()
            if ctas > 1:
                spec_val = spec_gpu_reduce_kernel_global(
                    col, add, _f32(0.0), num_outputs=97, stride1=False,
                    vt0=4, warp_size=32, max_threads=512, rocm=False,
                    W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                    vectorize=vec, ctas_per_output=ctas,
                )
            else:
                spec_val = spec_gpu_reduce_kernel(
                    col, add, _f32(0.0), num_outputs=97, stride1=False,
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
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            col = x[:, i].cpu().tolist()
            spec_val = spec_gpu_reduce_kernel(col, add, _f32(0.0), num_outputs=100, stride1=False)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_global_reduce(self):
        # Very large stride-1 reduction -> triggers vectorized input + global
        # (multi-CTA) reduce. The spec_gpu_reduce_kernel_global doesn't model
        # the stride-1 vectorized + global combination, so we compare against
        # a smaller non-vectorized global reduce (prime num_outputs, non-stride-1).
        x = torch.randn(50000, 97, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: _f32(a + b)
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            50000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        col = x[:, 0].cpu().tolist()
        if ctas > 1:
            spec_val = spec_gpu_reduce_kernel_global(
                col, add, _f32(0.0), num_outputs=97, stride1=False,
                vt0=4, warp_size=32, max_threads=512, rocm=False,
                W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                vectorize=vec, ctas_per_output=ctas,
            )
        else:
            spec_val = spec_gpu_reduce_kernel(
                col, add, _f32(0.0), num_outputs=97, stride1=False,
            )
        self.assertEqual(spec_val, result[0].item(), atol=0, rtol=0)

    def test_nansum(self):
        # Same tree as sum. Reduce step skips NaN, combine is still a+b.
        x = torch.randn(100, 5000, device="cuda")
        x[0, 0] = float("nan")
        result = torch.nansum(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # Spec: nansum treats NaN as 0 in the reduce step, then combines via a+b.
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            row = x[i].cpu().tolist()
            row_no_nan = [_f32(0) if math.isnan(float(v)) else v for v in row]
            spec_val = spec_gpu_reduce_kernel(row_no_nan, add, _f32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_prod_stride1(self):
        # Same tree as sum, combine is a*b, identity=1.
        x = torch.randn(100, 5000, device="cuda")
        result = torch.prod(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (gpu_reduce_kernel vectorized path, multiply)
        mul = lambda a, b: _f32(a * b)
        for i in range(min(4, result.shape[0])):
            row = x[i].cpu().tolist()
            spec_val = spec_gpu_reduce_kernel(row, mul, _f32(1.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_prod_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.prod(x, dim=0)
        self.assertEqual(result.shape, (100,))
        # Spec: bitwise match (non-stride-1, split across warps, multiply)
        mul = lambda a, b: _f32(a * b)
        for i in range(min(4, result.shape[0])):
            col = x[:, i].cpu().tolist()
            spec_val = spec_gpu_reduce_kernel(col, mul, _f32(1.0), num_outputs=100, stride1=False)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_sum_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.sum(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = grad.unsqueeze(-1).expand_as(x)
        self.assertEqual(x.grad, ref, atol=0, rtol=0)

    def test_prod_backward(self):
        x = torch.randn(10, 100, device="cuda", requires_grad=True)
        result = torch.prod(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_prod_backward(grad, x)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_nansum_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        x.data[0, 0] = float("nan")
        result = torch.nansum(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = grad.unsqueeze(-1).expand_as(x).clone()
        ref[x.isnan()] = 0
        self.assertEqual(x.grad, ref, atol=0, rtol=0)


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
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            row = x[i].cpu().tolist()
            spec_sum = spec_gpu_reduce_kernel(row, add, _f32(0.0), num_outputs=100, stride1=True)
            # MeanOps::project multiplies by precomputed factor = 1/N (in float32)
            factor = _f32(1.0 / _f32(len(row)))
            spec_mean = _f32(spec_sum * factor)
            self.assertEqual(spec_mean, result[i].item(), atol=0, rtol=0)

    def test_mean_nonstride1(self):
        # Use prime num_outputs to avoid output vectorization.
        x = torch.randn(5000, 97, device="cuda")
        result = torch.mean(x, dim=0)
        self.assertEqual(result.shape, (97,))
        add = lambda a, b: _f32(a + b)
        props = torch.cuda.get_device_properties(0)
        W, H, S, do_bx, do_by, vec, ctas = gpu_reduce_config(
            5000, 97, stride1=False, num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            output_vec_size=1,
        )
        for i in range(min(4, result.shape[0])):
            col = x[:, i].cpu().tolist()
            if ctas > 1:
                spec_sum = spec_gpu_reduce_kernel_global(
                    col, add, _f32(0.0), num_outputs=97, stride1=False,
                    vt0=4, warp_size=32, max_threads=512, rocm=False,
                    W=W, H=H, S=S, do_bx=do_bx, do_by=do_by,
                    vectorize=vec, ctas_per_output=ctas,
                )
            else:
                spec_sum = spec_gpu_reduce_kernel(
                    col, add, _f32(0.0), num_outputs=97, stride1=False,
                )
            # MeanOps::project multiplies by precomputed factor = 1/N (in float32)
            factor = _f32(1.0 / _f32(len(col)))
            spec_mean = _f32(spec_sum * factor)
            self.assertEqual(spec_mean, result[i].item(), atol=0, rtol=0)

    def test_var_stride1(self):
        # Welford combine merges (mean, m2, nf) tuples via WelfordOps (vt0=2).
        # Spec: compiled Welford reduce/combine matching exact CUDA FMA pattern.
        x = torch.randn(100, 5000, device="cuda")
        result = torch.var(x, dim=-1)
        self.assertEqual(result.shape, (100,))
        for i in range(min(4, result.shape[0])):
            spec_val = spec_welford_gpu_reduce(
                x[i], num_outputs=100, stride1=True, correction=1,
                take_sqrt=False,
            )
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_std_nonstride1(self):
        # Welford non-stride-1 reduce with vt0=2 + block_y_reduce.
        # output_vec_size=1 because input stride in output dim is 1 element.
        x = torch.randn(5000, 100, device="cuda")
        result = torch.std(x, dim=0)
        self.assertEqual(result.shape, (100,))
        for i in range(min(4, result.shape[0])):
            spec_val = spec_welford_gpu_reduce(
                x[:, i].contiguous(), num_outputs=100, stride1=False,
                correction=1, take_sqrt=True,
            )
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_mean_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.mean(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = grad.unsqueeze(-1).expand_as(x) / x.shape[-1]
        self.assertEqual(x.grad, ref, atol=0, rtol=0)

    def test_var_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.var(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_var_backward(grad, x)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_std_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.std(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_std_backward(grad, x)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            row = x[i].cpu().tolist()
            abs_row = [_f32(abs(v)) for v in row]
            spec_val = spec_gpu_reduce_kernel(abs_row, add, _f32(0.0), num_outputs=100, stride1=True)
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_norm_l2_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.linalg.vector_norm(x, 2, dim=-1)
        self.assertEqual(result.shape, (100,))
        # L2 norm: reduce is a + b^2, combine is a + b, project is sqrt
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            row = x[i].cpu().tolist()
            sq_row = [v * v for v in row]
            spec_sumsq = spec_gpu_reduce_kernel(sq_row, add, _f32(0.0), num_outputs=100, stride1=True)
            # Use CUDA sqrt to match kernel's device_sqrt
            spec_norm = torch.tensor(float(spec_sumsq), dtype=torch.float32, device="cuda").sqrt().item()
            self.assertEqual(spec_norm, result[i].item(), atol=0, rtol=0)

    def test_norm_lp_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.linalg.vector_norm(x, 3.0, dim=0)
        self.assertEqual(result.shape, (100,))
        # Lp norm: reduce is a + |b|^p, combine is a + b, project is ^(1/p)
        add = lambda a, b: _f32(a + b)
        for i in range(min(4, result.shape[0])):
            col = x[:, i].cpu().tolist()
            abs_p_col = [_f32(abs(v) ** 3) for v in col]
            spec_sum = spec_gpu_reduce_kernel(abs_p_col, add, _f32(0.0), num_outputs=100, stride1=False)
            # Use CUDA pow to match kernel's device_pow
            spec_val = torch.tensor(float(spec_sum), dtype=torch.float32, device="cuda").pow(1.0 / 3.0).item()
            self.assertEqual(spec_val, result[i].item(), atol=0, rtol=0)

    def test_norm_l2_backward(self):
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.linalg.vector_norm(x, 2, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_norm_l2_backward(grad, x)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_norm_lp_backward(self):
        x = torch.randn(5000, 100, device="cuda", requires_grad=True)
        result = torch.linalg.vector_norm(x, 3.0, dim=0)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_norm_lp_backward(grad, x, 3.0)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
        # Spec: bitwise match (CUDA exp via dtype=torch.float32)
        for i in range(x.shape[0]):
            row = x[i].cpu().tolist()
            spec_out = spec_softmax_persistent(row, warp_size=32, dtype=torch.float32)
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
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
            row = x[i].cpu().tolist()
            spec_out = spec_softmax_inner(row, block_size=1024, dtype=torch.float32)
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
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
        x_list = x.cpu().tolist()
        flat_data = [x_list[d][inner] for d in range(x.shape[0]) for inner in range(x.shape[1])]
        spec_out = spec_softmax_spatial(flat_data, dim_size=x.shape[0],
                                        inner_size=x.shape[1], dtype=torch.float32)
        spec_t = torch.tensor(spec_out, dtype=torch.float32).reshape(x.shape)
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
        exp = _dtype_exp(torch.float32)
        cast = _f32
        for i in range(x.shape[0]):
            row = x[i].cpu().tolist()
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
                        acc = cast(acc + exp_data[offset * ILP + j])
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = cast(acc + exp_data[tail_offset])
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = block_reduce_cuh(thread_sums, lambda a, b: cast(a + b), zero)
            # Epilogue: x - max - log(sum) using CUDA log
            log_sum = torch.tensor(float(row_sum), dtype=torch.float32,
                                   device="cuda").log().item()
            spec_out = [cast(cast(cast(v) - cast(row_max)) - cast(log_sum)) for v in row]
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_softmax_backward_inner(self):
        # aten::_softmax_backward_data — same inner/spatial dispatch.
        # Computes sum(grad * output) using same tree as forward sum.
        # Use dim=50000 to bypass Smem variant (same as forward inner test).
        # Block reduce uses SoftMax.cu's sequential blockReduce (NOT shfl_down).
        x = torch.randn(2, 50000, device="cuda")
        output = torch.softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(grad, output, -1, x.dtype)
        self.assertEqual(result.shape, x.shape)
        ILP = 4
        B = softmax_getblocksize(ILP, x.shape[-1])
        for i in range(x.shape[0]):
            g = grad[i].cpu().tolist()
            o = output[i].cpu().tolist()
            n = len(g)
            cast = _f32
            zero = cast(0.0)
            tmp = [cast(cast(g[j]) * cast(o[j])) for j in range(n)]
            # ilpReduce for sum(tmp)
            last = n % (ILP * B)
            main_end = n - last
            thread_sums = [zero] * B
            for tid in range(B):
                acc = zero
                offset = tid
                while offset * ILP < main_end:
                    for j in range(ILP):
                        acc = cast(acc + tmp[offset * ILP + j])
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = cast(acc + tmp[tail_offset])
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = softmax_block_reduce(
                thread_sums, lambda a, b: cast(a + b), zero)
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                t = torch.tensor(float(tmp[j]), device="cuda", dtype=torch.float32)
                ov = torch.tensor(float(o[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(t, ov, sum_t).item())
            self.assertEqual(
                torch.tensor(spec_gi, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
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
        grad_list = grad.cpu().tolist()
        out_list = output.cpu().tolist()
        cast = _f32

        max_block = 1024
        inner_threads = min(inner_size, max_block)
        dim_threads = 1
        if inner_threads <= 64 and dim_size >= 64:
            while inner_threads * dim_threads <= max_block and dim_threads <= dim_size:
                dim_threads *= 2
            dim_threads //= 2

        spec_grad_input = [[0.0] * inner_size for _ in range(dim_size)]
        for inner in range(inner_size):
            tmp = [cast(cast(grad_list[d][inner]) * cast(out_list[d][inner]))
                   for d in range(dim_size)]
            if dim_threads == 1:
                s = cast(0.0)
                for d in range(dim_size):
                    s = cast(s + tmp[d])
            else:
                thread_sums = [cast(0.0)] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_sums[tx] = cast(thread_sums[tx] + tmp[d])
                s = shmem_halving_reduce(thread_sums, lambda a, b: cast(a + b),
                                         cast(0.0))
            # Compiled epilogue for FMA: tmp - output * sum
            s_t = torch.tensor(float(s), device="cuda", dtype=torch.float32)
            for d in range(dim_size):
                tmp_t = torch.tensor(float(tmp[d]), device="cuda", dtype=torch.float32)
                out_t = torch.tensor(float(out_list[d][inner]), device="cuda", dtype=torch.float32)
                spec_grad_input[d][inner] = _compiled_softmax_backward_epilogue(
                    tmp_t, out_t, s_t).item()

        spec_t = torch.tensor(spec_grad_input, dtype=torch.float32)
        self.assertEqual(spec_t, result.cpu(), atol=0, rtol=0)

    def test_log_softmax_backward(self):
        # log_softmax backward: grad_input = grad - exp(output) * sum(grad)
        # Same ilpReduce + SoftMax.cu sequential blockReduce as softmax backward.
        # Use dim=50000 to bypass Smem variant.
        x = torch.randn(4, 50000, device="cuda")
        output = torch.log_softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._log_softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)
        ILP = 4
        B = softmax_getblocksize(ILP, x.shape[-1])
        for i in range(x.shape[0]):
            g = grad[i].cpu().tolist()
            n = len(g)
            cast = _f32
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
                        acc = cast(acc + g[offset * ILP + j])
                    offset += B
                tail_offset = main_end + tid
                while tail_offset < n:
                    acc = cast(acc + g[tail_offset])
                    tail_offset += B
                thread_sums[tid] = acc
            row_sum = softmax_block_reduce(
                thread_sums, lambda a, b: cast(a + b), zero)
            # Epilogue: grad - exp(output) * sum(grad)
            # exp(output) computed on CUDA to match kernel's inline std::exp
            exp_output = output[i].exp()
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                exp_o = torch.tensor(float(exp_output[j].item()), device="cuda",
                                     dtype=torch.float32)
                g_t = torch.tensor(float(g[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(g_t, exp_o, sum_t).item())
            self.assertEqual(
                torch.tensor(spec_gi, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
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
            g = grad[i].cpu().tolist()
            o = output[i].cpu().tolist()
            n = len(g)
            cast = _f32
            zero = cast(0.0)

            tmp = [cast(cast(g[j]) * cast(o[j])) for j in range(n)]

            # Per-lane sequential sum at stride warp_size
            lane_sums = [zero] * warp_size
            for lane in range(warp_size):
                acc = zero
                for it_idx in range(lane, n, warp_size):
                    acc = cast(acc + tmp[it_idx])
                lane_sums[lane] = acc

            row_sum = shfl_xor_butterfly_high_to_low(
                lane_sums, lambda a, b: cast(a + b)
            )

            # Compiled epilogue for FMA: tmp - output * sum
            sum_t = torch.tensor(float(row_sum), device="cuda", dtype=torch.float32)
            spec_gi = []
            for j in range(n):
                t = torch.tensor(float(tmp[j]), device="cuda", dtype=torch.float32)
                ov = torch.tensor(float(o[j]), device="cuda", dtype=torch.float32)
                spec_gi.append(_compiled_softmax_backward_epilogue(t, ov, sum_t).item())
            self.assertEqual(
                torch.tensor(spec_gi, dtype=torch.float32),
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
        exp = _dtype_exp(torch.float32)
        cast = _f32
        x_list = x.cpu().tolist()
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
                    m = max(m, x_list[d][inner])
                s = cast(0.0)
                for d in range(dim_size):
                    s = cast(s + exp(cast(x_list[d][inner]) - cast(m)))
            else:
                neg_inf = cast(float("-inf"))
                thread_maxes = [neg_inf] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_maxes[tx] = max(thread_maxes[tx], x_list[d][inner])
                m = shmem_halving_reduce(thread_maxes, max, neg_inf)
                zero = cast(0.0)
                thread_sums = [zero] * dim_threads
                for tx in range(dim_threads):
                    for d in range(tx, dim_size, dim_threads):
                        thread_sums[tx] = cast(thread_sums[tx] + exp(
                            cast(x_list[d][inner]) - cast(m)))
                s = shmem_halving_reduce(thread_sums, lambda a, b: cast(a + b), zero)
            # CUDA log for log(sum_exp)
            log_s = torch.tensor(float(s), dtype=torch.float32,
                                 device="cuda").log().item()
            for d in range(dim_size):
                spec_val = cast(cast(cast(x_list[d][inner]) - cast(m)) - cast(log_s))
                self.assertEqual(float(spec_val), result[d, inner].item(),
                                 atol=0, rtol=0)

    def test_log_softmax_backward_spatial(self):
        # log_softmax backward on non-last dim -> spatial path.
        # sum(grad) per inner position, then grad - exp(output) * sum(grad).
        dim_size, inner_size = 1000, 32
        x = torch.randn(dim_size, inner_size, device="cuda")
        output = torch.log_softmax(x, dim=0)
        grad = torch.randn_like(output)
        result = torch.ops.aten._log_softmax_backward_data(
            grad, output, 0, x.dtype
        )
        self.assertEqual(result.shape, x.shape)
        grad_list = grad.cpu().tolist()
        out_list = output.cpu().tolist()
        flat_grad = [grad_list[d][inner]
                     for d in range(dim_size) for inner in range(inner_size)]
        flat_out = [out_list[d][inner]
                    for d in range(dim_size) for inner in range(inner_size)]
        spec_out = spec_log_softmax_backward_spatial(
            flat_grad, flat_out, dim_size, inner_size, dtype=torch.float32,
        )
        spec_t = torch.tensor(
            spec_out, dtype=torch.float32
        ).reshape(dim_size, inner_size)
        self.assertEqual(spec_t, result.cpu(), atol=0, rtol=0)


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
        dx, dw, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [5000], mean, rstd, w, b, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        row_idx = 0
        # Compiled spec: block_size=128 (num_threads()=C10_WARP_SIZE*4)
        spec_dx = spec_layer_norm_backward_dx_compiled(
            grad[row_idx], x[row_idx], mean[row_idx], rstd[row_idx], w,
            block_size=128,
        )
        self.assertEqual(
            torch.tensor(spec_dx),
            dx[row_idx].cpu(),
            atol=0, rtol=0,
        )

    def test_layer_norm_backward_dgamma_large_M(self):
        # M=1000 >= 256 -> block_dim_y=32, rows_per_block_y=256.
        # Compiled FMA accumulation + XOR butterfly reduction.
        M, N = 1000, 256
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
        spec_dg = []
        for j in range(N):
            val = spec_dgamma_compiled(
                grad[:, j].contiguous(), x[:, j].contiguous(),
                mean, rstd,
                block_dim_y=32, rows_per_block_y=256, rms_norm=False,
            )
            spec_dg.append(val)
        self.assertEqual(
            torch.tensor(spec_dg),
            dw.cpu(),
            atol=0, rtol=0,
        )

    def test_layer_norm_backward_dgamma_small_M(self):
        # M=8 < 64 → block_dim_y=1, rows_per_block_y=8, partial_reduction=true.
        # Purely sequential: one thread per feature, loops over all M rows.
        # FMA: acc = fma(dY*(X-mean), rstd, acc).
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
        spec_dw = spec_gamma_beta_backward_dgamma_simple(
            grad.cpu().tolist(), x.cpu().tolist(),
            mean.cpu().tolist(), rstd.cpu().tolist(),
        )
        self.assertEqual(torch.tensor(spec_dw), dw.cpu(), atol=0, rtol=0)

    def test_layer_norm_backward_dbeta(self):
        # dbeta uses the GammaBetaBackwardCUDAKernelTemplate tree:
        # per-thread sequential sum over contiguous row blocks, then
        # SHFL_XOR butterfly high-to-low across block_dim_y threads.
        # M=1000 >= 256 → block_dim_y=32, rows_per_block_y=256.
        M, N = 1000, 256
        x = torch.randn(M, N, device="cuda")
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        mean = x.mean(dim=-1)
        rstd = (1.0 / x.std(dim=-1, unbiased=False)).to(x.dtype)
        grad = torch.randn_like(x)
        _, _, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [N], mean, rstd, w, b, [False, False, True]
        )
        self.assertEqual(db.shape, b.shape)
        spec_db = spec_layer_norm_backward_dbeta(
            grad, block_dim_y=32, rows_per_block_y=256,
        )
        self.assertEqual(spec_db, db, atol=0, rtol=0)

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
        # Compiled spec: block_size=128 (num_threads()=C10_WARP_SIZE*4)
        row_idx = 0
        spec_dx = spec_rms_norm_backward_dx_compiled(
            grad[row_idx], x[row_idx], rstd_cuda[row_idx], w,
            block_size=128,
        )
        self.assertEqual(
            torch.tensor(spec_dx),
            dx_cuda[row_idx].cpu(),
            atol=0, rtol=0,
        )

    def test_rms_norm_backward_dgamma(self):
        M, N = 1000, 256
        x = torch.randn(M, N, device="cuda")
        w = torch.randn(N, device="cuda")
        result, rstd_cuda = torch._fused_rms_norm(x, [N], w, 1e-5)
        grad = torch.randn_like(x)
        _, dw_cuda = torch.ops.aten._fused_rms_norm_backward(
            grad, x, [N], rstd_cuda, w, [False, True]
        )
        self.assertEqual(dw_cuda.shape, w.shape)
        spec_dg = []
        for j in range(N):
            val = spec_dgamma_compiled(
                grad[:, j].contiguous(), x[:, j].contiguous(),
                torch.zeros(M, device="cuda"),  # not used for rms_norm
                rstd_cuda.flatten(),
                block_dim_y=32, rows_per_block_y=256, rms_norm=True,
            )
            spec_dg.append(val)
        self.assertEqual(
            torch.tensor(spec_dg),
            dw_cuda.cpu(),
            atol=0, rtol=0,
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
            tw.append((_f32(m.item()), _f32(s2.item()), _f32(c.item())))
        wid = (_f32(0), _f32(0), _f32(0))
        def cw(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda") for v in list(a) + list(b)]
            m, s2, c = _compiled_welford_ops_combine(*ta)
            return (_f32(m.item()), _f32(s2.item()), _f32(c.item()))
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
        self.assertEqual(float(_f32(ms)), mean_t[0, 0].item(), atol=0, rtol=0)
        self.assertEqual(rstd_s, rstd_t[0, 0].item(), atol=0, rtol=0)

    def test_group_norm_backward(self):
        # Group norm backward dw uses ComputeInternalGradientsCUDAKernel
        # (WarpReduceSum tree) + GammaBetaBackwardCUDAKernel1 (sequential + FMA).
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        N_batch, C, HxW, G = 8, 32, 256, 8
        y, mean, rstd = torch.ops.aten.native_group_norm(
            x, w, b, N_batch, C, HxW, G, 1e-5)
        grad = torch.randn_like(y)
        dx, dw, db = torch.ops.aten.native_group_norm_backward(
            grad, x, mean, rstd, w, N_batch, C, HxW, G, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        ds, db_int = spec_group_norm_backward_internal(
            grad, x, N_batch, C, HxW, num_threads=32)
        spec_dw, _ = spec_group_norm_backward_dw_db(
            ds, db_int, mean, rstd, N_batch, C, G)
        self.assertEqual(torch.tensor(spec_dw), dw.cpu(), atol=0, rtol=0)

    def test_group_norm_backward_dx(self):
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        N_batch, C, HxW, G = 8, 32, 256, 8
        y, mean, rstd = torch.ops.aten.native_group_norm(
            x, w, b, N_batch, C, HxW, G, 1e-5)
        grad = torch.randn_like(y)
        dx, _, _ = torch.ops.aten.native_group_norm_backward(
            grad, x, mean, rstd, w, N_batch, C, HxW, G, [True, False, False]
        )
        ds, db_int = spec_group_norm_backward_internal(
            grad, x, N_batch, C, HxW, num_threads=32)
        spec_dx = spec_group_norm_backward_dx(
            grad, x, mean, rstd, w, ds, db_int, N_batch, C, HxW, G)
        self.assertEqual(spec_dx, dx, atol=0, rtol=0)

    def test_group_norm_backward_db(self):
        # dbeta[c] = sum_n(db_internal[n,c]) where db_internal is from
        # ComputeInternalGradientsCUDAKernel (WarpReduceSum tree).
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        N_batch, C, HxW, G = 8, 32, 256, 8
        y, mean, rstd = torch.ops.aten.native_group_norm(
            x, w, b, N_batch, C, HxW, G, 1e-5)
        grad = torch.randn_like(y)
        _, _, db = torch.ops.aten.native_group_norm_backward(
            grad, x, mean, rstd, w, N_batch, C, HxW, G, [False, False, True]
        )
        self.assertEqual(db.shape, b.shape)
        ds, db_int = spec_group_norm_backward_internal(
            grad, x, N_batch, C, HxW, num_threads=32)
        _, spec_db = spec_group_norm_backward_dw_db(
            ds, db_int, mean, rstd, N_batch, C, G)
        self.assertEqual(torch.tensor(spec_db), db.cpu(), atol=0, rtol=0)


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
                tw.append((_f32(m.item()), _f32(s2.item()),
                           _f32(c.item())))
        def cw(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda")
                  for v in [a[0], a[1], a[2], b[0], b[1], b[2]]]
            m, s2, c = _compiled_nchw_welford_merge(*ta)
            return (_f32(m.item()), _f32(s2.item()),
                    _f32(c.item()))
        wid = (_f32(0), _f32(0), _f32(0))
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
            float(_f32(s2s) / _f32(cs) + _f32(1e-5)),
            device="cuda", dtype=torch.float32)).item()
        self.assertEqual(float(_f32(ms)), mean_cuda[ch].item(), atol=0, rtol=0)
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
            tw.append((_f32(mr.item()), _f32(s2r.item()), _f32(cr.item())))
        # welford_merge_block_vertical
        def merge_w(a, b):
            if a[2] == 0: return b
            if b[2] == 0: return a
            ta = [torch.tensor(float(v), device="cuda") for v in [a[0],a[1],a[2],b[0],b[1],b[2]]]
            m, s2, c = _compiled_nhwc_welford_merge(*ta)
            return (_f32(m.item()), _f32(s2.item()), _f32(c.item()))
        wrs = list(tw)
        off = len(wrs) // 2
        while off > 0:
            for wy in range(off):
                wrs[wy] = merge_w(wrs[wy], wrs[wy + off])
            off //= 2
        ms, s2s, cs = wrs[0]
        invstd_s = torch.rsqrt(torch.tensor(
            float(_f32(s2s)/_f32(cs) + _f32(1e-5)),
            device="cuda", dtype=torch.float32)).item()
        self.assertEqual(float(_f32(ms)), mean_cuda[ch].item(), atol=0, rtol=0)
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
        # Fused kernel (grad_input_mask[0]=True): block_x=getNumThreads(64)=64,
        # block_y=max(1,512/64)=8. B=32, S=64.
        ch = 0
        grad_ch = grad[:, ch, :, :].contiguous().view(-1).cpu().tolist()
        input_ch = x[:, ch, :, :].contiguous().view(-1).cpu().tolist()
        mean_ch = sm[ch].item()
        spec_sum_dy, spec_sum_dy_xmu = spec_batch_norm_nchw_backward_reduce(
            grad_ch, input_ch, mean_ch,
            block_x=64, block_y=8, B=32, S=64,
        )
        self.assertEqual(
            torch.tensor(spec_sum_dy),
            torch.tensor(db[ch].item()),
            atol=0, rtol=0,
        )

    def test_batch_norm_nhwc_backward(self):
        # Small tensor to ensure grid.y=1 in NHWC backward reduce.
        x = torch.randn(8, 64, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        x_contig = x.to(memory_format=torch.contiguous_format)
        sm = x_contig.mean(dim=(0, 2, 3))
        si = 1.0 / x_contig.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        dx, dw, db = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [True, True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        _, _, spec_db = spec_batch_norm_nhwc_backward(
            grad, x, sm, si, w)
        self.assertEqual(spec_db, db, atol=0, rtol=0)

    def test_batch_norm_nchw_backward_dx(self):
        x = torch.randn(32, 64, 8, 8, device="cuda")
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        sm = x.mean(dim=(0, 2, 3))
        si = 1.0 / x.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        dx, _, _ = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [True, False, False]
        )
        self.assertEqual(dx.shape, x.shape)
        spec_dx, _, _ = spec_batch_norm_nchw_backward(
            grad, x, sm, si, w)
        self.assertEqual(spec_dx, dx, atol=0, rtol=0)

    def test_batch_norm_nhwc_backward_dw(self):
        x = torch.randn(8, 64, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        x_contig = x.to(memory_format=torch.contiguous_format)
        sm = x_contig.mean(dim=(0, 2, 3))
        si = 1.0 / x_contig.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        _, dw, _ = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [False, True, False]
        )
        self.assertEqual(dw.shape, w.shape)
        _, spec_dw, _ = spec_batch_norm_nhwc_backward(
            grad, x, sm, si, w)
        self.assertEqual(spec_dw, dw, atol=0, rtol=0)

    def test_batch_norm_nhwc_backward_reduce_spec(self):
        # Small tensor so grid.y=1 (no multi-CTA staging).
        x = torch.randn(8, 4, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        w = torch.ones(4, device="cuda")
        rm = torch.zeros(4, device="cuda")
        rv = torch.ones(4, device="cuda")
        sm = x.to(memory_format=torch.contiguous_format).mean(dim=(0, 2, 3))
        si = 1.0 / x.to(memory_format=torch.contiguous_format).std(
            dim=(0, 2, 3), unbiased=False
        )
        grad = torch.randn_like(x)
        _, _, db = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [False, False, True]
        )
        ch = 0
        grad_ch = grad.to(
            memory_format=torch.contiguous_format
        )[:, ch, :, :].contiguous().view(-1)
        input_ch = x.to(
            memory_format=torch.contiguous_format
        )[:, ch, :, :].contiguous().view(-1)
        mean_ch = sm[ch].item()
        grad_list = grad_ch.cpu().tolist()
        input_list = input_ch.cpu().tolist()
        spec_sum_dy, _ = spec_batch_norm_nhwc_backward_reduce(
            grad_list, input_list, mean_ch, block_y=8,
        )
        self.assertEqual(spec_sum_dy, db[ch].item(), atol=0, rtol=0)

    def test_batch_norm_nchw_backward_dw(self):
        x = torch.randn(32, 64, 8, 8, device="cuda")
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        sm = x.mean(dim=(0, 2, 3))
        si = 1.0 / x.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        _, dw, _ = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [False, True, False]
        )
        self.assertEqual(dw.shape, w.shape)
        _, spec_dw, _ = spec_batch_norm_nchw_backward(
            grad, x, sm, si, w, fused=False)
        self.assertEqual(spec_dw, dw, atol=0, rtol=0)

    def test_batch_norm_nhwc_backward_dx(self):
        x = torch.randn(8, 64, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        w = torch.ones(64, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        x_contig = x.to(memory_format=torch.contiguous_format)
        sm = x_contig.mean(dim=(0, 2, 3))
        si = 1.0 / x_contig.std(dim=(0, 2, 3), unbiased=False)
        grad = torch.randn_like(x)
        dx, _, _ = torch.ops.aten.native_batch_norm_backward(
            grad, x, w, rm, rv, sm, si, True, 1e-5, [True, False, False]
        )
        self.assertEqual(dx.shape, x.shape)
        spec_dx, _, _ = spec_batch_norm_nhwc_backward(
            grad, x, sm, si, w)
        self.assertEqual(spec_dx, dx, atol=0, rtol=0)


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
        losses = (-x[torch.arange(nframe, device="cuda"), target]).cpu().tolist()
        spec_val = spec_nll_loss_reduce(losses, block_size=nthreads)
        self.assertEqual(spec_val, result.item(), atol=0, rtol=0)

    def test_cross_entropy(self):
        # cross_entropy = log_softmax + nll_loss
        x = torch.randn(1000, 100, device="cuda")
        target = torch.randint(0, 100, (1000,), device="cuda")
        result = torch.nn.functional.cross_entropy(x, target)
        self.assertEqual(result.shape, ())
        ref = ref_cross_entropy_forward(x, target)
        self.assertEqual(ref, result.item(), atol=0, rtol=0)

    def test_nll_loss_backward(self):
        x = torch.randn(1000, 100, device="cuda", requires_grad=True)
        target = torch.randint(0, 100, (1000,), device="cuda")
        loss = torch.nn.functional.nll_loss(x, target, reduction="mean")
        loss.backward()
        ref = ref_nll_loss_backward(1000, 100, target)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_cross_entropy_backward(self):
        x = torch.randn(1000, 100, device="cuda", requires_grad=True)
        target = torch.randint(0, 100, (1000,), device="cuda")
        loss = torch.nn.functional.cross_entropy(x, target)
        loss.backward()
        ref = ref_cross_entropy_backward(x, target)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


class TestNLLLoss2d(TestCase):
    """NLLLoss2d.cu — nll_loss for 4D input

    Same shared-memory halving tree as Loss.cu.
    No separate spec — use spec_nll_loss_reduce.
    """

    # test_nll_loss_2d_forward omitted: NLLLoss2d uses gpuAtomicAdd across
    # blocks for the final sum, making the reduction order non-deterministic
    # (depends on GPU scheduling). No fixed association order to model.

    def test_nll_loss_2d_backward(self):
        x = torch.randn(8, 10, 32, 32, device="cuda", requires_grad=True)
        target = torch.randint(0, 10, (8, 32, 32), device="cuda")
        loss = torch.nn.functional.nll_loss(x, target, reduction="mean")
        loss.backward()
        ref = ref_nll_loss_2d_backward(x.shape, target)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
        x_list = x.cpu().tolist()
        tgt = target[0].item()
        THREADS = 128
        per_thread = [_f32(0.0)] * THREADS
        for tid in range(THREADS):
            for c in range(tid, 50, THREADS):
                if c == tgt:
                    continue
                per_thread[tid] = _f32(per_thread[tid] + _f32(
                    max(0, _f32(_f32(1.0 - x_list[0][tgt]) + x_list[0][c]))))
        sample_loss = spec_multi_margin_loss_thread0_scan(per_thread)
        # Kernel divides by nclass in float32: static_cast<scalar_t>(sum / denom)
        ref = _f32(float(sample_loss) / 50.0)
        self.assertEqual(ref, result.item(), atol=0, rtol=0)

    def test_multi_margin_loss_backward(self):
        x = torch.randn(1, 50, device="cuda", requires_grad=True)
        target = torch.randint(0, 50, (1,), device="cuda")
        loss = torch.nn.functional.multi_margin_loss(x, target)
        loss.backward()
        ref = ref_multi_margin_loss_backward(
            x.detach(), target)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
        x = torch.randn(100, 50, device="cuda")
        target = torch.zeros(100, 50, dtype=torch.long, device="cuda") - 1
        for i in range(100):
            n = torch.randint(1, 10, (1,)).item()
            target[i, :n] = torch.randint(0, 50, (n,))
        result = torch.nn.functional.multilabel_margin_loss(x, target)
        self.assertEqual(result.shape, ())
        ref = ref_multilabel_margin_loss_forward(
            x, target)
        self.assertEqual(ref, result.item(), atol=0, rtol=0)

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
        T, N, C = 50, 8, 30
        log_probs = torch.randn(T, N, C, device="cuda").log_softmax(dim=2)
        targets = torch.randint(1, C, (N, 20), device="cuda")
        input_lengths = torch.full((N,), T, dtype=torch.long, device="cuda")
        target_lengths = torch.full((N,), 20, dtype=torch.long, device="cuda")
        result = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        self.assertEqual(result.shape, ())
        ref_loss, _ = ref_ctc_loss_forward_backward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        self.assertEqual(ref_loss, result.item(), atol=0, rtol=0)

    def test_ctc_loss_backward(self):
        T, N, C = 50, 8, 30
        log_probs = torch.randn(T, N, C, device="cuda").log_softmax(
            dim=2
        ).requires_grad_(True)
        targets = torch.randint(1, C, (N, 20), device="cuda")
        input_lengths = torch.full((N,), T, dtype=torch.long, device="cuda")
        target_lengths = torch.full((N,), 20, dtype=torch.long, device="cuda")
        loss = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        loss.backward()
        _, ref_grad = ref_ctc_loss_forward_backward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        self.assertEqual(ref_grad, log_probs.grad, atol=0, rtol=0)


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
            row = x[i].cpu().tolist()
            spec_out = spec_cumsum_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_cumsum_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumsum(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        # Spec: bitwise match (purely sequential outer-dim scan)
        flat = x.cpu().flatten().tolist()
        spec_out = spec_cumsum_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)

    def test_cumsum_backward_innermost(self):
        # cumsum backward = flip(cumsum(flip(grad))). Same Sklansky tree.
        x = torch.randn(100, 5000, device="cuda", requires_grad=True)
        result = torch.cumsum(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
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
            grad_row = grad[i].flip(0).cpu().tolist()
            spec_out = spec_cumsum_innermost_sklansky(grad_row,
                                                      num_threads_x=ntx)
            spec_flipped = list(reversed(spec_out))
            self.assertEqual(
                torch.tensor(spec_flipped, dtype=torch.float32),
                x.grad[i].cpu(),
                atol=0, rtol=0,
            )

    def test_cumsum_backward_outer(self):
        # cumsum backward along dim=0 = flip(cumsum(flip(grad, 0), 0), 0)
        x = torch.randn(5000, 100, device="cuda", requires_grad=True)
        result = torch.cumsum(x, dim=0)
        grad = torch.randn_like(result)
        result.backward(grad)
        grad_flipped = grad.flip(0).cpu()
        flat = grad_flipped.flatten().tolist()
        spec_out = spec_cumsum_outer_sequential(flat, x.shape[0], x.shape[1])
        spec_t = torch.tensor(spec_out, dtype=torch.float32).reshape(
            x.shape[0], x.shape[1]
        )
        spec_flipped = spec_t.flip(0)
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(
                    spec_flipped[r, c].item(),
                    x.grad[r, c].item(),
                    atol=0, rtol=0,
                )


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
            row = x[i].cpu().tolist()
            spec_out = spec_cumprod_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_cumprod_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumprod(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        flat = x.cpu().flatten().tolist()
        spec_out = spec_cumprod_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)

    def test_cumprod_backward_innermost(self):
        x = torch.randn(10, 100, device="cuda", requires_grad=True)
        result = torch.cumprod(x, dim=-1)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_cumprod_backward(grad, x, dim=-1)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_cumprod_backward_outer(self):
        x = torch.randn(100, 10, device="cuda", requires_grad=True)
        result = torch.cumprod(x, dim=0)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_cumprod_backward(grad, x, dim=0)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
            row = x[i].cpu().tolist()
            spec_out = spec_logcumsumexp_innermost_sklansky(row, num_threads_x=ntx)
            self.assertEqual(
                torch.tensor(spec_out, dtype=torch.float32),
                result[i].cpu(),
                atol=0, rtol=0,
            )

    def test_logcumsumexp_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.logcumsumexp(x, dim=0)
        self.assertEqual(result.shape, x.shape)
        flat = x.cpu().flatten().tolist()
        spec_out = spec_logcumsumexp_outer_sequential(flat, x.shape[0], x.shape[1])
        for c in range(min(4, x.shape[1])):
            for r in range(x.shape[0]):
                self.assertEqual(spec_out[r * x.shape[1] + c], result[r, c].item(), atol=0, rtol=0)

    def test_logcumsumexp_backward_innermost(self):
        x = torch.randn(10, 100, device="cuda", requires_grad=True)
        output = torch.logcumsumexp(x, dim=-1)
        grad = torch.randn_like(output)
        output.backward(grad)
        ref = ref_logcumsumexp_backward(
            grad, x.detach(), output.detach(), dim=-1)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)

    def test_logcumsumexp_backward_outer(self):
        x = torch.randn(100, 10, device="cuda", requires_grad=True)
        output = torch.logcumsumexp(x, dim=0)
        grad = torch.randn_like(output)
        output.backward(grad)
        ref = ref_logcumsumexp_backward(
            grad, x.detach(), output.detach(), dim=0)
        self.assertEqual(ref, x.grad, atol=0, rtol=0)


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
        spec_out = spec_scatter_add(10, list(range(10)), src2.cpu().tolist())
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

    def test_scatter_add_backward(self):
        # scatter_add backward w.r.t. src is gather: grad_src[i] = grad_out[idx[i]]
        x = torch.zeros(100, 64, device="cuda")
        idx = torch.randint(0, 100, (500, 64), device="cuda")
        src = torch.randn(500, 64, device="cuda", requires_grad=True)
        result = x.scatter_add(0, idx, src)
        grad = torch.randn_like(result)
        result.backward(grad)
        ref = ref_scatter_add_backward(grad, idx)
        self.assertEqual(ref, src.grad, atol=0, rtol=0)


class TestSegmentReduce(TestCase):
    """SegmentReduce.cu
    CUDA ref: cuda/SegmentReduce.cu.

    === 1D (CUB path) ===
    Uses cub::DeviceSegmentedReduce::Reduce.
    sm_100 policy: 512 threads, 16 items/thread, striped load,
    BLOCK_REDUCE_WARP_REDUCTIONS with shfl_down low-to-high (offsets 1,2,4,8,16).
    Thread 0 then sequentially combines valid warp aggregates.
    CUDA ref: cub/device/dispatch/tuning/tuning_reduce.cuh:181 (sm100 float+plus),
    cub/block/specializations/block_reduce_warp_reductions.cuh,
    cub/warp/specializations/warp_reduce_shfl.cuh:217 (float plus ReduceStep).

    === Multi-dim (custom kernel) ===
    PURELY SEQUENTIAL loop per thread over segment elements:
      for j in range(offset_start, offset_end):
          acc = combine(acc, data[j])
    One thread per (segment, feature_dim) pair.
    """

    def test_segment_reduce_1d(self):
        x = torch.randn(10000, device="cuda")
        lengths = torch.tensor([100] * 100, device="cuda")
        result = torch.segment_reduce(x, "sum", lengths=lengths)
        self.assertEqual(result.shape, (100,))
        # CUB sm_100 policy: 512 threads, striped load, shfl_down low-to-high
        CUB_THREADS = 512
        WARP = 32
        add = lambda a, b: _f32(a + b)
        zero = _f32(0.0)
        x_list = x.cpu().tolist()
        spec = []
        off = 0
        for seg in range(100):
            N = lengths[seg].item()
            data = x_list[off:off + N]
            thread_vals = [zero] * CUB_THREADS
            for t in range(N):
                thread_vals[t] = _f32(data[t])
            warp_aggs = []
            for w in range(CUB_THREADS // WARP):
                ws = w * WARP
                wv = thread_vals[ws:ws + WARP]
                nv = min(WARP, max(0, N - ws))
                if nv <= 0:
                    warp_aggs.append(zero)
                    continue
                last = nv - 1
                offset = 1
                while offset < WARP:
                    for i in range(WARP):
                        src = i + offset
                        if src < WARP and src <= last:
                            wv[i] = _f32(wv[i] + wv[src])
                    offset *= 2
                warp_aggs.append(wv[0])
            total = warp_aggs[0]
            for w in range(1, CUB_THREADS // WARP):
                if w * WARP < N:
                    total = _f32(total + warp_aggs[w])
            spec.append(total)
            off += N
        self.assertEqual(
            torch.tensor(spec, dtype=torch.float32),
            result.cpu(),
            atol=0, rtol=0,
        )

    def test_segment_reduce_2d(self):
        x = torch.randn(10000, 64, device="cuda")
        lengths = torch.tensor([100] * 100, device="cuda")
        result = torch.segment_reduce(x, "sum", lengths=lengths)
        self.assertEqual(result.shape, (100, 64))
        ref = ref_segment_reduce_sum_2d(x, lengths)
        self.assertEqual(ref, result, atol=0, rtol=0)


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
        w_list = w.cpu().tolist()
        idx_list = idx.cpu().tolist()
        embeddings = [w_list[idx_list[j]][feat] for j in range(bag_start, bag_end)]
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
        w_list = w.cpu().tolist()
        idx_list = idx.cpu().tolist()
        embeddings = [w_list[idx_list[j]][feat] for j in range(bag_start, bag_end)]
        spec_sum = spec_embedding_bag_sum(embeddings)
        ref = _f32(spec_sum * _f32(1.0 / bag_size))
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
        grad_vals = [_f32(1.0)] * 16
        weight_table = w.cpu().tolist()
        idx_list = idx.cpu().tolist()
        spec_psw_grad = spec_embedding_bag_psw_backward(
            grad_vals, weight_table, idx_list, embedding_dim=16
        )
        self.assertEqual(float(spec_psw_grad[0]), psw.grad[0].item(), atol=0, rtol=0)

    def test_embedding_bag_backward_weight(self):
        # Weight gradient uses scatter_add of per-sample contributions.
        w = torch.randn(100, 16, device="cuda", requires_grad=True)
        idx = torch.randint(0, 100, (20,), device="cuda")
        offsets = torch.tensor([0, 10], device="cuda")
        out = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="sum"
        )
        out.sum().backward()
        grad_output = torch.ones(2, 16, device="cuda")  # grad of sum() is all ones
        ref = ref_embedding_bag_backward_weight(
            grad_output, idx, offsets,
            num_embeddings=100, embedding_dim=16)
        self.assertEqual(ref, w.grad, atol=0, rtol=0)


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
        window_vals = x[b, c, oh - 1:oh + 2, ow - 1:ow + 2].cpu().flatten().tolist()
        spec_val = spec_avg_pool_window(window_vals)
        self.assertEqual(spec_val, result[b, c, oh, ow].item(), atol=0, rtol=0)

    def test_avg_pool2d_backward(self):
        x = torch.randn(2, 4, 8, 8, device="cuda")
        grad_output = torch.randn(2, 4, 8, 8, device="cuda")
        result = torch.ops.aten.avg_pool2d_backward(
            grad_output, x, [3, 3], [1, 1], [1, 1], False, True, None
        )
        self.assertEqual(result.shape, x.shape)
        ref = ref_avg_pool2d_backward(
            grad_output, x.shape, kH=3, kW=3,
            sH=1, sW=1, pH=1, pW=1, count_include_pad=True)
        self.assertEqual(ref, result, atol=0, rtol=0)


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
        # Kernel source shows sequential sum += val, but nvcc compiles the
        # double loop (kH=4, kW=4) into a halving-tree reduction.
        x = torch.randn(8, 64, 4, 4, device="cuda")
        result = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        self.assertEqual(result.shape, (8, 64, 1, 1))
        window_vals = x[0, 0].cpu().flatten().tolist()
        # shfl_down-style halving tree: offset 8, 4, 2, 1 on 16 values
        vals = [_f32(v) for v in window_vals]
        offset = 8
        while offset > 0:
            for i in range(16):
                if i + offset < 16:
                    vals[i] = _f32(vals[i] + vals[i + offset])
            offset //= 2
        kH, kW = 4, 4
        spec_val = _f32(_f32(vals[0] / _f32(kH)) / _f32(kW))
        self.assertEqual(float(spec_val), result[0, 0, 0, 0].item(), atol=0, rtol=0)

    def test_adaptive_avg_pool2d_backward(self):
        x = torch.randn(8, 64, 32, 32, device="cuda")
        grad_output = torch.randn(8, 64, 1, 1, device="cuda")
        result = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, x)
        self.assertEqual(result.shape, x.shape)
        ref = ref_adaptive_avg_pool2d_backward(
            grad_output, x.shape)
        self.assertEqual(ref, result, atol=0, rtol=0)


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
        tensors = [torch.randn(1000, device="cuda") for _ in range(3)]
        result = torch._foreach_norm(tensors, 2.0)
        self.assertEqual(len(result), 3)
        for i in range(3):
            ref = ref_foreach_norm_l2(tensors[i])
            self.assertEqual(result[i].item(), ref, atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
