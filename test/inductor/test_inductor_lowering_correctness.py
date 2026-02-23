# Owner(s): ["module: inductor"]

"""
Test that inductor produces numerically correct results vs eager on CUDA.

Two test methods:
  test_lowering: ops with direct lowerings (ATen → inductor IR → codegen)
  test_full_pipeline: ops that are decomposed first then lowered (ATen → decomp
      → primitive ATen ops → lowering → codegen). This catches accumulated error
      from decomposition + codegen together — a gap that neither the decomp test
      (which runs decomps eagerly) nor the lowering test (which tests primitives
      in isolation) would catch.

Both compile with all eager-numerics-matching flags and point triton's libdevice
at the CUDA toolkit's copy to minimize sources of divergence.
"""

import os
import unittest

import torch
import torch._inductor.config
import torch._inductor.decomposition
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.lowering import fallbacks, lowerings
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensorMode,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TestCase,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    unMarkDynamoStrictTest,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


f16 = torch.float16
f32 = torch.float32
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool

ALL_SAMPLES = os.getenv("PYTORCH_ALL_SAMPLES", "0") == "1"

# Point triton's libdevice at the CUDA toolkit's copy so that special-function
# implementations (exp, log, sin, ...) match what eager's CUDA kernels link
# against, rather than triton's bundled (potentially different) version.
_cuda_version = torch.version.cuda
if _cuda_version and "TRITON_LIBDEVICE_PATH" not in os.environ:
    _cuda_home = f"/usr/local/cuda-{_cuda_version}"
    _libdevice = os.path.join(_cuda_home, "nvvm", "libdevice", "libdevice.10.bc")
    if os.path.exists(_libdevice):
        os.environ["TRITON_LIBDEVICE_PATH"] = _libdevice

# Inductor config that makes codegen match eager numerics as closely as
# possible.  Each flag addresses a specific source of divergence:
_EAGER_NUMERICS_CONFIG = {
    # Error on missing lowerings (don't silently fall back to eager)
    "implicit_fallbacks": False,
    # Avoid nondeterminism from autotuning
    "triton.autotune_pointwise": False,
    # Preserve downcast-upcast pairs between fused low-precision ops so that
    # intermediate precision truncation matches eager's cast-per-op semantics.
    "emulate_precision_casts": True,
    # Use round-to-nearest division (div_rn) instead of triton's default
    # approximate div.full, matching eager's fp32 division.
    "eager_numerics.division_rounding": True,
    # Disable flush-to-zero so subnormals are preserved, matching eager CUDA.
    "eager_numerics.disable_ftz": True,
    # Use fp64 for unbacked float scalars (from .item()) to match eager
    # precision.  (Default True in OSS, but set explicitly for clarity.)
    "_use_fp64_for_unbacked_floats": True,
    # Upcast fp16/bf16 to fp32 in triton codegen to match eager's per-op
    # upcast behavior.
    "triton.codegen_upcast_to_fp32": True,
}

# Same config but allow fallbacks — decomposed ops may decompose into ops
# that don't have lowerings (e.g. _fused_rms_norm).
_EAGER_NUMERICS_CONFIG_WITH_FALLBACKS = {
    **_EAGER_NUMERICS_CONFIG,
    "implicit_fallbacks": True,
}


def overload_to_aten_name(op):
    return op._schema.name.split("::")[1]


# --- Op lists ---

_lowering_names = {
    overload_to_aten_name(k)
    for k in lowerings
    if isinstance(k, torch._ops.OpOverload) and k not in fallbacks
}
_lowering_test_ops = [op for op in op_db if op.aten_name in _lowering_names]

_decomp_table = torch._inductor.decomposition.select_decomp_table()
_decomp_names = {
    overload_to_aten_name(k)
    for k in _decomp_table
    if isinstance(k, torch._ops.OpOverload)
}
# Decomp-only: ops whose aten_name is in the decomp table but NOT in lowerings.
# These are fully decomposed before reaching the lowering stage.
_decomp_only_names = _decomp_names - _lowering_names
_decomp_only_test_ops = [
    op
    for op in op_db
    if op.aten_name in _decomp_only_names
    and op.aten_name not in _lowering_names
]

# Tolerances per dtype. float32 is more generous than decomp tests because
# codegen introduces rounding from fused ops and different reduction ordering.
_dtype_precisions = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-5, 1.5e-5),
    torch.float64: (1e-7, 1e-7),
}

# Ops where the test is not meaningful (uninitialized memory,
# nondeterministic, untraceable, etc.)
EXCLUDE_SET = {
    # Uninitialized memory
    (None, None, "empty"),
    (None, None, "empty_like"),
    (None, None, "empty_permuted"),
    (None, None, "empty_strided"),
    (None, None, "new_empty"),
    (None, None, "new_empty_strided"),
    # Data-dependent / dynamic output shape
    (None, None, "nonzero"),
    (None, None, "nonzero_static"),
    (None, None, "item"),
    (None, None, "argwhere"),
    # Sparse ops
    (None, None, "to_sparse"),
    (None, None, "sparse.sampled_addmm"),
    # In-place mutations that don't go through compile well
    (None, None, "resize_"),
    (None, None, "resize_as_"),
    (None, None, "zero_"),
    # RNG ops whose OpInfo sample_inputs use untraceable wrappers
    (None, None, "bernoulli"),
    (None, None, "randn"),
    (None, None, "normal"),
    (None, None, "multinomial"),
    (None, None, "cauchy"),
    (None, None, "exponential"),
    (None, None, "geometric"),
    (None, None, "log_normal"),
    (None, None, "rand_like"),
    (None, None, "randint"),
    (None, None, "randint_like"),
    (None, None, "randn_like"),
    # CUDA f16 precision: these ops need reference_in_float or wider tolerances
    # in test_torchinductor_opinfo.py; not worth special-casing here.
    ("cuda", torch.float16, "round.decimals_3"),
    ("cuda", torch.float16, "cumsum"),
    ("cuda", torch.float16, "_unsafe_masked_index_put_accumulate"),
    # Composed f16 precision: decomp + codegen errors compound beyond tolerance
    ("cuda", torch.float16, "addr"),
    ("cuda", torch.float16, "nn.functional.group_norm"),
    # STFT: FFT decomposition accumulates rounding across trig + reduction chain
    (None, None, "stft"),
}


class HasRngOp(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.has_rng_op = False

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs if kwargs else {}
        if torch.Tag.nondeterministic_seeded in func.tags:
            self.has_rng_op = True
        return func(*args, **kwargs)


def _can_nopython_and_has_rng(fn, args, kwargs):
    try:
        mode = FakeTensorMode()

        def map_to_fake(e):
            if isinstance(e, torch.Tensor):
                return mode.from_tensor(e)
            return e

        fake_args, fake_kwargs = tree_map(map_to_fake, (args, kwargs))
        with HasRngOp() as rng_mode, mode:
            with enable_python_dispatcher():
                fn(*fake_args, **fake_kwargs)
    except (DataDependentOutputException, DynamicOutputShapeException):
        return False, rng_mode.has_rng_op

    return True, rng_mode.has_rng_op


def _get_test_keys(device_type, dtype, op):
    op_name = op.name
    if op.variant_test_name:
        op_name += f".{op.variant_test_name}"
    return [
        (device_type, dtype, op_name),
        (None, dtype, op_name),
        (None, None, op_name),
        (device_type, dtype, op.name),
        (None, dtype, op.name),
        (None, None, op.name),
    ]


def _run_and_compare(test_case, device, dtype, op, check_backward=False):
    device_type = torch.device(device).type
    test_keys = _get_test_keys(device_type, dtype, op)
    if any(key in EXCLUDE_SET for key in test_keys):
        test_case.skipTest(f"{op.name} in {dtype} excluded")

    torch._dynamo.reset()

    func = op.get_op()

    def fn(*args, **kwargs):
        return func(*args, **kwargs)

    requires_grad = (
        check_backward
        and op.supports_autograd
        and dtype in op.supported_backward_dtypes(device_type)
        and dtype != torch.complex32
    )

    samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)
    if not ALL_SAMPLES:
        if isinstance(samples, (list, tuple)):
            samples = [samples[0]]
        else:
            samples = [next(samples)]

    for sample_input in samples:
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs

        nopython, has_rng_op = _can_nopython_and_has_rng(fn, args, kwargs)

        torch.manual_seed(0)
        expected = func(*args, **kwargs)

        compiled = torch.compile(fn, backend="inductor", fullgraph=nopython)
        torch.manual_seed(0)
        actual = compiled(*args, **kwargs)

        if has_rng_op:
            expected_flat = (
                expected if isinstance(expected, (list, tuple)) else [expected]
            )
            actual_flat = (
                actual if isinstance(actual, (list, tuple)) else [actual]
            )
            for e, a in zip(expected_flat, actual_flat):
                if isinstance(e, torch.Tensor):
                    test_case.assertEqual(e.shape, a.shape)
                    test_case.assertEqual(e.dtype, a.dtype)
                    test_case.assertEqual(e.device.type, a.device.type)
        else:
            test_case.assertEqual(
                actual,
                expected,
                rtol=0,
                atol=0,
                equal_nan=True,
                exact_dtype=True,
            )

        if not requires_grad:
            continue

        # Compare gradients through backward pass
        expected_flat = pytree.tree_leaves(expected)
        actual_flat = pytree.tree_leaves(actual)
        grad_inputs = [
            torch.randn_like(t) if isinstance(t, torch.Tensor) and t.is_floating_point() else None
            for t in expected_flat
        ]
        grad_inputs = [g for g in grad_inputs if g is not None]
        expected_tensors = [t for t in expected_flat if isinstance(t, torch.Tensor) and t.requires_grad]
        actual_tensors = [t for t in actual_flat if isinstance(t, torch.Tensor) and t.requires_grad]

        if not expected_tensors:
            continue

        # Need differentiable outputs to backprop through
        expected_loss = sum(
            t.sum() for t in expected_flat
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad
        )
        actual_loss = sum(
            t.sum() for t in actual_flat
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad
        )

        if not isinstance(expected_loss, torch.Tensor):
            continue

        # Collect leaf tensors that need grad
        leaf_tensors = [
            a for a in pytree.tree_leaves((args, kwargs))
            if isinstance(a, torch.Tensor) and a.requires_grad
        ]
        if not leaf_tensors:
            continue

        expected_grads = torch.autograd.grad(
            expected_loss, leaf_tensors, retain_graph=True, allow_unused=True
        )

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="inductor", fullgraph=nopython)
        torch.manual_seed(0)
        actual = compiled(*args, **kwargs)
        actual_flat = pytree.tree_leaves(actual)
        actual_loss = sum(
            t.sum() for t in actual_flat
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad
        )
        actual_grads = torch.autograd.grad(
            actual_loss, leaf_tensors, retain_graph=True, allow_unused=True
        )

        for i, (eg, ag) in enumerate(zip(expected_grads, actual_grads)):
            if eg is None and ag is None:
                continue
            test_case.assertEqual(
                ag,
                eg,
                rtol=0,
                atol=0,
                equal_nan=True,
                exact_dtype=True,
                msg=f"Gradient mismatch for input {i} of {op.name}",
            )


@unMarkDynamoStrictTest
class TestInductorLoweringCorrectness(TestCase):
    longMessage = True

    def tearDown(self):
        torch._dynamo.reset()

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @suppress_warnings
    @ops(
        _lowering_test_ops,
        dtypes=OpDTypes.supported,
        allowed_dtypes=[f16, f32, i32, i64, b8],
    )
    @torch._inductor.config.patch(_EAGER_NUMERICS_CONFIG)
    def test_lowering(self, device, dtype, op):
        _run_and_compare(self, device, dtype, op)

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @suppress_warnings
    @ops(
        _decomp_only_test_ops,
        dtypes=OpDTypes.supported,
        allowed_dtypes=[f16, f32, i32, i64, b8],
    )
    @torch._inductor.config.patch(_EAGER_NUMERICS_CONFIG_WITH_FALLBACKS)
    def test_full_pipeline(self, device, dtype, op):
        _run_and_compare(self, device, dtype, op)

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @suppress_warnings
    @ops(
        _lowering_test_ops,
        dtypes=OpDTypes.supported,
        allowed_dtypes=[f16, f32, i32, i64, b8],
    )
    @torch._inductor.config.patch(_EAGER_NUMERICS_CONFIG)
    def test_lowering_backward(self, device, dtype, op):
        _run_and_compare(self, device, dtype, op, check_backward=True)

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @suppress_warnings
    @ops(
        _decomp_only_test_ops,
        dtypes=OpDTypes.supported,
        allowed_dtypes=[f16, f32, i32, i64, b8],
    )
    @torch._inductor.config.patch(_EAGER_NUMERICS_CONFIG_WITH_FALLBACKS)
    def test_full_pipeline_backward(self, device, dtype, op):
        _run_and_compare(self, device, dtype, op, check_backward=True)


instantiate_device_type_tests(TestInductorLoweringCorrectness, globals())

if __name__ == "__main__":
    run_tests()
