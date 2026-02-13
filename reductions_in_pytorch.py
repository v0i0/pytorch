"""
Catalog of non-associative (in floating point) reduction kernels in PyTorch's
eager CUDA backend. Each test function corresponds to one CUDA source file and
documents the reduction tree structure, block-size heuristics, and provides
runnable examples that exercise each dispatch path.

All file paths are relative to aten/src/ATen/native/cuda/.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


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


class TestReduceSumProdKernel(TestCase):
    """ReduceSumProdKernel.cu — sum, nansum, prod, xor_sum

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

    def test_sum_nonstride1_split(self):
        # Reducing dim=0 on contiguous -> non-stride-1, split_across_warps.
        # dim0=100(outputs), dim1=5000(inputs). W=32,H=16,S=16, vpt=313.
        # split_across_warps: 313 >= min(16*16,256)=256 -> yes.
        # Tree: seq(4,16) -> shmem(16) only, NO warp shuffles.
        # May also trigger global reduce (vpt>=256, depends on GPU SM count).
        x = torch.randn(5000, 100, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (100,))

    def test_sum_nonstride1_nosplit(self):
        # Small reduction on non-stride-1 dim -> no warp split.
        # dim0=100(outputs), dim1=10(inputs). vpt=10 < min(H*16,256).
        # Tree: purely sequential per thread, no inter-thread reduce.
        x = torch.randn(10, 100, device="cuda")
        result = torch.sum(x, dim=0)
        self.assertEqual(result.shape, (100,))

    def test_sum_global_reduce(self):
        # Very large stride-1 reduction -> triggers global (multi-CTA) reduce.
        # S=512, vpt=ceil(500000/512)=977 >= 256, grid likely undersubscribed.
        # Tree: ...same phases... + global serial scan of CTA staging buffer.
        x = torch.randn(2, 500000, device="cuda")
        result = torch.sum(x, dim=-1)
        self.assertEqual(result.shape, (2,))

    def test_nansum(self):
        # Same tree as sum. Reduce step skips NaN, combine is still a+b.
        x = torch.randn(100, 5000, device="cuda")
        x[0, 0] = float("nan")
        result = torch.nansum(x, dim=-1)
        self.assertEqual(result.shape, (100,))

    def test_prod_stride1(self):
        # Same tree as sum, combine is a*b, identity=1.
        x = torch.randn(100, 5000, device="cuda")
        result = torch.prod(x, dim=-1)
        self.assertEqual(result.shape, (100,))

    def test_prod_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.prod(x, dim=0)
        self.assertEqual(result.shape, (100,))


class TestReduceMomentKernel(TestCase):
    """ReduceMomentKernel.cu — mean, std, var (Welford)

    Combine functions:
      mean:    a + b                    project: acc * (1/N)
      std/var: Welford 4-tuple merge    project: (sqrt(m2/divisor), mean)
    Same gpu_reduce_kernel tree as sum. Only combine function differs.
    """

    def test_mean_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.mean(x, dim=-1)
        self.assertEqual(result.shape, (100,))

    def test_mean_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.mean(x, dim=0)
        self.assertEqual(result.shape, (100,))

    def test_var_stride1(self):
        # Welford combine merges (mean, m2, n, nf) tuples.
        # Numerically stable across the parallel tree.
        x = torch.randn(100, 5000, device="cuda")
        result = torch.var(x, dim=-1)
        self.assertEqual(result.shape, (100,))

    def test_std_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.std(x, dim=0)
        self.assertEqual(result.shape, (100,))


class TestReduceNormKernel(TestCase):
    """ReduceNormKernel.cu — L1, L2, Lp, powsum norms

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

    def test_norm_l2_stride1(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.linalg.vector_norm(x, 2, dim=-1)
        self.assertEqual(result.shape, (100,))

    def test_norm_lp_nonstride1(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.linalg.vector_norm(x, 3.0, dim=0)
        self.assertEqual(result.shape, (100,))


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
    """

    def test_softmax_persistent(self):
        # dim_size=512 <= 1024 and 512*4=2048 <= 4096 -> persistent path.
        # One warp per row, SHFL_XOR butterfly.
        x = torch.randn(32, 512, device="cuda")
        result = torch.softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)

    def test_softmax_inner_dim(self):
        # dim_size=10000 > 1024 -> inner-dim path (cunn_SoftMaxForward).
        # Block = 1024 threads. ilpReduce + blockReduceWarp.
        x = torch.randn(32, 10000, device="cuda")
        result = torch.softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)

    def test_softmax_spatial(self):
        # Reducing dim=0, inner_size=32 > 1 -> spatial path.
        # Sequential per-thread loop over 1000 elements.
        x = torch.randn(1000, 32, device="cuda")
        result = torch.softmax(x, dim=0)
        self.assertEqual(result.shape, x.shape)

    def test_log_softmax_inner(self):
        # Same dispatch as softmax. log variant just changes epilogue.
        x = torch.randn(32, 10000, device="cuda")
        result = torch.log_softmax(x, dim=-1)
        self.assertEqual(result.shape, x.shape)

    def test_softmax_backward_inner(self):
        # aten::_softmax_backward_data — same inner/spatial dispatch.
        # Computes sum(grad * output) using same tree as forward sum.
        x = torch.randn(32, 10000, device="cuda")
        output = torch.softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)

    def test_softmax_backward_spatial(self):
        x = torch.randn(1000, 32, device="cuda")
        output = torch.softmax(x, dim=0)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(
            grad, output, 0, x.dtype
        )
        self.assertEqual(result.shape, x.shape)

    def test_log_softmax_backward(self):
        x = torch.randn(32, 10000, device="cuda")
        output = torch.log_softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._log_softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)

    def test_softmax_backward_persistent(self):
        # dim_size=512 -> persistent path for backward too.
        # Same SHFL_XOR butterfly tree as persistent forward.
        x = torch.randn(32, 512, device="cuda")
        output = torch.softmax(x, dim=-1)
        grad = torch.randn_like(output)
        result = torch.ops.aten._softmax_backward_data(
            grad, output, -1, x.dtype
        )
        self.assertEqual(result.shape, x.shape)

    def test_log_softmax_spatial(self):
        # log_softmax on non-last dim -> spatial path (sequential).
        x = torch.randn(1000, 32, device="cuda")
        result = torch.log_softmax(x, dim=0)
        self.assertEqual(result.shape, x.shape)


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
    """

    def test_layer_norm_forward(self):
        x = torch.randn(100, 5000, device="cuda")
        w = torch.randn(5000, device="cuda")
        b = torch.randn(5000, device="cuda")
        result = torch.nn.functional.layer_norm(x, [5000], w, b)
        self.assertEqual(result.shape, x.shape)

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

    def test_layer_norm_backward_dgamma_small_M(self):
        # M=8 -> GammaBetaBackwardSimpleCUDAKernel, purely sequential.
        x = torch.randn(8, 5000, device="cuda")
        w = torch.randn(5000, device="cuda")
        b = torch.randn(5000, device="cuda")
        mean = x.mean(dim=-1)
        rstd = (1.0 / x.std(dim=-1, unbiased=False)).to(x.dtype)
        grad = torch.randn_like(x)
        _, dw, db = torch.ops.aten.native_layer_norm_backward(
            grad, x, [5000], mean, rstd, w, b, [False, True, True]
        )
        self.assertEqual(dw.shape, w.shape)

    def test_rms_norm_forward(self):
        x = torch.randn(100, 5000, device="cuda")
        w = torch.randn(5000, device="cuda")
        # Same tree as layer norm forward but sum(x^2) not Welford.
        result, rstd = torch._fused_rms_norm(x, [5000], w)
        self.assertEqual(result.shape, x.shape)

    def test_rms_norm_backward(self):
        # Same tree as layer norm backward (both dX and dgamma parts).
        x = torch.randn(100, 5000, device="cuda")
        w = torch.randn(5000, device="cuda")
        _, rstd = torch._fused_rms_norm(x, [5000], w)
        grad = torch.randn_like(x)
        dx, dw = torch.ops.aten._fused_rms_norm_backward(
            grad, x, [5000], rstd, w, [True, True]
        )
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(dw.shape, w.shape)


class TestGroupNormKernel(TestCase):
    """group_norm_kernel.cu

    === Forward (RowwiseMomentsCUDAKernel) ===
    Block: 512 threads (kCUDABlockReduceNumThreads). One block per (N, group).
    Reduction over D/G * H * W elements per group.
    Tree: identical to layer norm forward — sequential Welford at stride 512,
      then BlockReduce: shfl_down(32) -> shmem(512/32=16) -> warp 0 final.

    === Backward ===
    Same BlockReduceSum pattern for ds, db accumulators.
    """

    def test_group_norm_forward(self):
        x = torch.randn(8, 32, 16, 16, device="cuda")
        w = torch.randn(32, device="cuda")
        b = torch.randn(32, device="cuda")
        result = torch.nn.functional.group_norm(x, num_groups=8, weight=w, bias=b)
        self.assertEqual(result.shape, x.shape)

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
    """

    def test_batch_norm_nchw_forward(self):
        x = torch.randn(32, 64, 8, 8, device="cuda")
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        result = torch.nn.functional.batch_norm(x, rm, rv, training=True)
        self.assertEqual(result.shape, x.shape)

    def test_batch_norm_nhwc_forward(self):
        x = torch.randn(32, 64, 8, 8, device="cuda").to(
            memory_format=torch.channels_last
        )
        rm = torch.zeros(64, device="cuda")
        rv = torch.ones(64, device="cuda")
        result = torch.nn.functional.batch_norm(x, rm, rv, training=True)
        self.assertEqual(result.shape, x.shape)

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


class TestLoss(TestCase):
    """Loss.cu — nll_loss (also used by cross_entropy)

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
        result = torch.nn.functional.nll_loss(x, target, reduction="mean")
        self.assertEqual(result.shape, ())

    def test_cross_entropy(self):
        # cross_entropy = log_softmax + nll_loss
        x = torch.randn(1000, 100, device="cuda")
        target = torch.randint(0, 100, (1000,), device="cuda")
        result = torch.nn.functional.cross_entropy(x, target)
        self.assertEqual(result.shape, ())


class TestNLLLoss2d(TestCase):
    """NLLLoss2d.cu — nll_loss for 4D input

    Same shared-memory halving tree as Loss.cu.
    """

    def test_nll_loss_2d(self):
        x = torch.randn(8, 10, 32, 32, device="cuda").log_softmax(dim=1)
        target = torch.randint(0, 10, (8, 32, 32), device="cuda")
        result = torch.nn.functional.nll_loss(x, target, reduction="mean")
        self.assertEqual(result.shape, ())


class TestMultiMarginLoss(TestCase):
    """MultiMarginLoss.cu

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
        x = torch.randn(100, 50, device="cuda")
        target = torch.randint(0, 50, (100,), device="cuda")
        result = torch.nn.functional.multi_margin_loss(x, target)
        self.assertEqual(result.shape, ())


class TestMultiLabelMarginCriterion(TestCase):
    """MultiLabelMarginCriterion.cu

    Block: fixed thread count.
    Tree: similar to NLL loss — thread-local sequential accumulation of
      hinge losses at stride blockDim.x, then shared-memory halving tree.
      for i in range(threadIdx.x, n_classes, blockDim.x):
          if target contains i: continue
          loss += max(0, 1 - x[target[j]] + x[i])
      -> shmem halving tree across block
    """

    def test_multilabel_margin_loss(self):
        x = torch.randn(100, 50, device="cuda")
        # targets: positive labels followed by -1 padding
        target = torch.zeros(100, 50, dtype=torch.long, device="cuda") - 1
        for i in range(100):
            n = torch.randint(1, 10, (1,)).item()
            target[i, :n] = torch.randint(0, 50, (n,))
        result = torch.nn.functional.multilabel_margin_loss(x, target)
        self.assertEqual(result.shape, ())


class TestLossCTC(TestCase):
    """LossCTC.cu — CTC loss

    Log-space dynamic programming. Reductions use log-add-exp:
      log(exp(a) + exp(b)) via log1p(exp(min-max)) + max
    Operates per batch element. Sequential DP over time steps.
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


class TestCumsumKernel(TestCase):
    """CumsumKernel.cu + ScanUtils.cuh — cumulative sum

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

    def test_cumsum_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumsum(x, dim=0)
        self.assertEqual(result.shape, x.shape)


class TestCumprodKernel(TestCase):
    """CumprodKernel.cu + ScanUtils.cuh — cumulative product

    Same two-path dispatch as cumsum. Combine: a * b.
    """

    def test_cumprod_innermost(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.cumprod(x, dim=-1)
        self.assertEqual(result.shape, x.shape)

    def test_cumprod_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.cumprod(x, dim=0)
        self.assertEqual(result.shape, x.shape)


class TestLogcumsumexpKernel(TestCase):
    """LogcumsumexpKernel.cu — log-cumulative-sum-exp

    Same inner/outer dispatch as cumsum/cumprod.
    Combine: log(exp(a) + exp(b)), computed as log1p(exp(min-max)) + max.
    """

    def test_logcumsumexp_innermost(self):
        x = torch.randn(100, 5000, device="cuda")
        result = torch.logcumsumexp(x, dim=-1)
        self.assertEqual(result.shape, x.shape)

    def test_logcumsumexp_outer(self):
        x = torch.randn(5000, 100, device="cuda")
        result = torch.logcumsumexp(x, dim=0)
        self.assertEqual(result.shape, x.shape)


class TestScatterGatherKernel(TestCase):
    """ScatterGatherKernel.cu — scatter_add, scatter_reduce

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

    def test_scatter_reduce_sum(self):
        x = torch.zeros(100, 64, device="cuda")
        idx = torch.randint(0, 100, (500, 64), device="cuda")
        src = torch.randn(500, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="sum")
        self.assertEqual(result.shape, x.shape)

    def test_scatter_reduce_prod(self):
        x = torch.ones(100, 64, device="cuda")
        idx = torch.randint(0, 100, (500, 64), device="cuda")
        src = torch.randn(500, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="prod")
        self.assertEqual(result.shape, x.shape)

    def test_scatter_reduce_mean(self):
        # Atomic add + post-kernel divide by count.
        x = torch.zeros(100, 64, device="cuda")
        idx = torch.randint(0, 100, (500, 64), device="cuda")
        src = torch.randn(500, 64, device="cuda")
        result = x.scatter_reduce(0, idx, src, reduce="mean")
        self.assertEqual(result.shape, x.shape)


class TestSegmentReduce(TestCase):
    """SegmentReduce.cu

    === 1D (CUB path) ===
    Uses cub::DeviceSegmentedReduce::Reduce with custom combine op.
    Internal tree structure is CUB's auto-tuned binary tree.

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

    def test_segment_reduce_2d(self):
        x = torch.randn(10000, 64, device="cuda")
        lengths = torch.tensor([100] * 100, device="cuda")
        result = torch.segment_reduce(x, "sum", lengths=lengths)
        self.assertEqual(result.shape, (100, 64))


class TestEmbeddingBag(TestCase):
    """EmbeddingBag.cu

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
    """

    def test_embedding_bag_sum(self):
        w = torch.randn(1000, 128, device="cuda")
        idx = torch.randint(0, 1000, (500,), device="cuda")
        offsets = torch.tensor([0, 100, 250, 400], device="cuda")
        result = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="sum"
        )
        self.assertEqual(result.shape, (4, 128))

    def test_embedding_bag_mean(self):
        w = torch.randn(1000, 128, device="cuda")
        idx = torch.randint(0, 1000, (500,), device="cuda")
        offsets = torch.tensor([0, 100, 250, 400], device="cuda")
        result = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="mean"
        )
        self.assertEqual(result.shape, (4, 128))

    def test_embedding_bag_per_sample_weights_backward(self):
        # Dot product per sample: sum(grad[feat] * weight[feat]) over features.
        # One warp per sample. seq(+=, stride=32) -> WarpReduceSum: shfl_down.
        w = torch.randn(1000, 128, device="cuda")
        idx = torch.randint(0, 1000, (500,), device="cuda")
        offsets = torch.tensor([0, 100, 250, 400], device="cuda")
        per_sample_weights = torch.randn(500, device="cuda")
        result = torch.nn.functional.embedding_bag(
            idx, w, offsets, mode="sum", per_sample_weights=per_sample_weights
        )
        # The backward for per_sample_weights uses the warp-reduce dot product.
        grad = torch.randn_like(result)
        psw_grad = torch.ops.aten._embedding_bag_per_sample_weights_backward(
            grad, w, idx, offsets, per_sample_weights, 500, 0
        )
        self.assertEqual(psw_grad.shape, per_sample_weights.shape)


class TestAveragePool2d(TestCase):
    """AveragePool2d.cu

    PURELY SEQUENTIAL nested loop per output element. One thread per output.
      accval = 0
      for h in range(hstart, hend):
          for w in range(wstart, wend):
              accval += input[c, h, w]
      output = accval / count
    No inter-thread reduction. Block: standard pointwise launch.
    """

    def test_avg_pool2d(self):
        x = torch.randn(8, 64, 32, 32, device="cuda")
        result = torch.nn.functional.avg_pool2d(x, kernel_size=3, padding=1)
        self.assertEqual(result.shape, (8, 64, 32, 32))

    def test_avg_pool2d_backward(self):
        # Backward: each input thread sums over contributing output positions.
        # Same sequential loop pattern, one thread per input element.
        x = torch.randn(8, 64, 32, 32, device="cuda")
        grad_output = torch.randn(8, 64, 32, 32, device="cuda")
        result = torch.ops.aten.avg_pool2d_backward(
            grad_output, x, [3, 3], [1, 1], [1, 1], False, True, None
        )
        self.assertEqual(result.shape, x.shape)


class TestAdaptiveAveragePooling(TestCase):
    """AdaptiveAveragePooling.cu

    Same purely sequential window loop as AveragePool2d.
    Window boundaries computed dynamically via START_IND/END_IND macros.
    """

    def test_adaptive_avg_pool2d(self):
        x = torch.randn(8, 64, 32, 32, device="cuda")
        result = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        self.assertEqual(result.shape, (8, 64, 1, 1))

    def test_adaptive_avg_pool2d_backward(self):
        x = torch.randn(8, 64, 32, 32, device="cuda")
        grad_output = torch.randn(8, 64, 1, 1, device="cuda")
        result = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, x)
        self.assertEqual(result.shape, x.shape)


class TestForeachReduceOp(TestCase):
    """ForeachReduceOp.cu — multi-tensor norm

    Block: 512 threads. One kernel processes multiple tensors.
    Tree per tensor chunk:
      Thread-local: sequential accumulation over assigned elements.
      shfl_down(32) within each warp.
      BlockReduceSum via shared memory (same block_reduce.cuh pattern).
    Combines partial results across tensor chunks.
    """

    def test_foreach_norm(self):
        tensors = [torch.randn(1000, device="cuda") for _ in range(3)]
        result = torch._foreach_norm(tensors, 2.0)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    run_tests()
