# Owner(s): ["module: inductor"]

"""
Test that inductor's decomposition table is numerically correct using OpInfo.

test_decomp.py validates the global decomposition_table. Inductor uses a different
table (torch._inductor.decomposition.select_decomp_table()) that:
- Overrides some decompositions (e.g. silu, cat, bmm, pad)
- Excludes decompositions for ops that inductor lowers directly
- Has conditional decompositions based on device type and tensor properties

This test cross-references every decomposition in the inductor table against the
native ATen implementation, using OpInfo sample inputs for coverage.
"""

import torch
import torch._inductor.decomposition
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfCrossRef,
    suppress_warnings,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten


def overload_to_aten_name(op):
    return op._schema.name.split("::")[1]


def any_unsupported(args, kwargs):
    def test_unsupported(t):
        if type(t) is Tensor or type(t) is torch.nn.Parameter:
            return any(
                [
                    t.is_sparse_csr,
                    t.is_sparse,
                    t.is_mkldnn,
                    t.is_quantized,
                    t.is_nested,
                    torch._is_functional_tensor(t),
                ]
            )
        elif torch.overrides.is_tensor_like(t):
            return True
        return False

    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    return any(test_unsupported(x) for x in flat_args)


# Build the inductor decomposition table
_inductor_table = torch._inductor.decomposition.select_decomp_table()

_inductor_decomp_names = {
    overload_to_aten_name(k)
    for k in _inductor_table
    if isinstance(k, torch._ops.OpOverload)
}

_inductor_decomp_test_ops = [
    op
    for op in op_db
    if op.aten_name in _inductor_decomp_names
    or op.aten_backward_name in _inductor_decomp_names
]

# Default tolerances per dtype (same as test_decomp.py)
_dtype_precisions = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}

# Per-op tolerance overrides: (dtype, aten_op) -> (rtol, atol)
# Matches entries from test_decomp.py plus inductor-specific cases.
_op_tolerances = {
    (torch.float32, aten.native_layer_norm.default): (1e-3, 1e-3),
    (torch.float64, aten.native_layer_norm.default): (1e-6, 1e-6),
    (torch.float32, aten.grid_sampler_2d.default): (7e-6, 3e-5),
    (torch.float32, aten.mv.default): (1e-5, 3e-5),
    (torch.complex64, aten.mv.default): (5e-5, 5e-5),
    (torch.float64, aten.upsample_bicubic2d.vec): (1e-5, 5e-4),
    (torch.float64, aten.upsample_bicubic2d.default): (1e-5, 5e-4),
    # The decomposition computes in int64 so there's an off-by-one for integer
    # linspace/logspace.  See https://github.com/pytorch/pytorch/issues/81996
    (torch.int8, aten.linspace.default): (0, 1),
    (torch.uint8, aten.linspace.default): (0, 1),
    (torch.int16, aten.linspace.default): (0, 1),
    (torch.int32, aten.linspace.default): (0, 1),
    (torch.int64, aten.linspace.default): (0, 1),
    (torch.int8, aten.linspace.Tensor_Tensor): (0, 1),
    (torch.uint8, aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int16, aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int32, aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int64, aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int8, aten.linspace.Tensor_Scalar): (0, 1),
    (torch.uint8, aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int16, aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int32, aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int64, aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int8, aten.linspace.Scalar_Tensor): (0, 1),
    (torch.uint8, aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int16, aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int32, aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int64, aten.linspace.Scalar_Tensor): (0, 1),
    # logspace has the same integer rounding issue as linspace
    (torch.int8, aten.logspace.default): (0, 1),
    (torch.uint8, aten.logspace.default): (0, 1),
    (torch.int16, aten.logspace.default): (0, 1),
    (torch.int32, aten.logspace.default): (0, 1),
    (torch.int64, aten.logspace.default): (0, 1),
    (torch.int8, aten.logspace.Tensor_Tensor): (0, 1),
    (torch.uint8, aten.logspace.Tensor_Tensor): (0, 1),
    (torch.int16, aten.logspace.Tensor_Tensor): (0, 1),
    (torch.int32, aten.logspace.Tensor_Tensor): (0, 1),
    (torch.int64, aten.logspace.Tensor_Tensor): (0, 1),
    (torch.int8, aten.logspace.Tensor_Scalar): (0, 1),
    (torch.uint8, aten.logspace.Tensor_Scalar): (0, 1),
    (torch.int16, aten.logspace.Tensor_Scalar): (0, 1),
    (torch.int32, aten.logspace.Tensor_Scalar): (0, 1),
    (torch.int64, aten.logspace.Tensor_Scalar): (0, 1),
    (torch.int8, aten.logspace.Scalar_Tensor): (0, 1),
    (torch.uint8, aten.logspace.Scalar_Tensor): (0, 1),
    (torch.int16, aten.logspace.Scalar_Tensor): (0, 1),
    (torch.int32, aten.logspace.Scalar_Tensor): (0, 1),
    (torch.int64, aten.logspace.Scalar_Tensor): (0, 1),
}

# Tolerance for fp16/bf16 relative-check mode (distance from fp64 baseline)
_relative_tol_table = {
    (torch.bfloat16, aten.native_layer_norm.default): 1e-5,
    (torch.float16, aten.native_layer_norm.default): 1e-5,
    (torch.float16, aten.native_layer_norm_backward.default): 1e-3,
    (torch.bfloat16, aten.native_layer_norm_backward.default): 2e-2,
    (torch.bfloat16, aten.native_batch_norm.default): 1e-5,
    (torch.float16, aten.native_batch_norm.default): 1e-5,
    (torch.bfloat16, aten._native_batch_norm_legit.default): 1e-5,
    (torch.bfloat16, aten._native_batch_norm_legit.no_stats): 1e-5,
    (torch.float16, aten._native_batch_norm_legit.default): 1e-5,
    (torch.float16, aten._native_batch_norm_legit.no_stats): 1e-5,
    (torch.bfloat16, aten.linalg_vector_norm.default): 1e-4,
    (torch.float16, aten.linalg_vector_norm.default): 1e-4,
    (torch.bfloat16, aten.var_mean.correction): 5e-7,
    (torch.float16, aten.var_mean.correction): 5e-7,
    (torch.bfloat16, aten.var_mean.dim): 5e-7,
    (torch.float16, aten.var_mean.dim): 5e-7,
    (torch.float16, aten.nll_loss_forward.default): 1e-2,
    (torch.bfloat16, aten.nll_loss_forward.default): 1e-1,
    (torch.float16, aten.nll_loss2d_forward.default): 1e-2,
    (torch.bfloat16, aten.nll_loss2d_forward.default): 2e-1,
    (torch.float16, aten.hardswish.default): 2e-7,
    (torch.bfloat16, aten.hardswish.default): 2e-7,
    (torch.float16, aten.mv.default): 1e-5,
    (torch.bfloat16, aten.mv.default): 1e-5,
    (torch.float16, aten._batch_norm_with_update.default): 5e-7,
    (torch.bfloat16, aten._batch_norm_with_update.default): 5e-7,
    # mm decomp on complex types expands to real arithmetic with different
    # accumulation order, causing large divergence from native complex mm.
    (torch.float16, aten.mm.default): 1e-2,
    (torch.bfloat16, aten.mm.default): 1e-1,
}

# Ops where cross-referencing is not meaningful
EXCLUDE_SET = {
    (None, None, "special.ndtr"),
    (None, None, "new_empty"),
    (None, None, "empty_like"),
    (None, None, "empty"),
    # empty_permuted returns uninitialized memory
    (None, None, "empty_permuted"),
    (None, None, "item"),
    (None, None, "zero_"),
    (None, None, "nn.functional.relu6"),
    (None, None, "nn.functional.rrelu"),
    (None, None, "meshgrid"),
    (None, None, "nn.functional.hardshrink"),
    (None, None, "nn.functional.softshrink"),
    (None, None, "diag"),
    (None, None, "norm"),
    (None, None, "native_batch_norm"),
    (None, None, "_upsample_bilinear2d_aa"),
    (None, None, "empty_strided"),
    (None, None, "bernoulli"),
    ("cpu", torch.bfloat16, "_softmax_backward_data"),
    ("cuda", torch.bfloat16, "nn.functional.bilinear"),
    ("cpu", torch.float16, "signal.windows.exponential"),
    ("cpu", torch.float16, "signal.windows.gaussian"),
    ("cpu", torch.float16, "signal.windows.cosine"),
    # mm decomp for complex types uses real arithmetic with different
    # accumulation order, causing expected precision divergence
    (None, torch.complex64, "corrcoef"),
    (None, torch.complex128, "corrcoef"),
    (None, torch.complex64, "cov"),
    (None, torch.complex128, "cov"),
}


def _upcast(x, dtype=torch.float64):
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        return x.to(dtype=dtype)
    elif isinstance(x, torch.dtype) and x in [
        torch.float16,
        torch.bfloat16,
        torch.float,
    ]:
        return dtype
    return x


class InductorDecompCrossRefMode(TorchDispatchMode):
    """Cross-references inductor decompositions against native ATen ops.

    For each ATen op found in the inductor decomp table, runs both the native
    implementation and the decomposition, then checks numerical agreement.
    For fp16/bf16, uses fp64 as a reference baseline (same strategy as
    DecompCrossRefMode in test_decomp.py).
    """

    def __init__(self, test_case, decomp_table, dtype):
        self.test_case = test_case
        self.decomp_table = decomp_table
        self.test_dtype = dtype
        self.called = set()
        self.decomposed = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.called.add(func)

        in_place = func.name()[-1] == "_"
        ignored_ops = {
            aten.detach.default,
            aten.empty.memory_format,
            aten.empty_like.default,
            aten.new_empty.default,
            aten.empty_strided.default,
            aten.new_empty_strided.default,
            aten.randn.default,
            aten.native_dropout.default,
        }

        if (
            func not in self.decomp_table
            or func in ignored_ops
            or torch.Tag.nondeterministic_seeded in func.tags
            or any_unsupported(args, kwargs)
            or in_place
        ):
            return func(*args, **kwargs)

        self.decomposed.add(func)
        decomposition = self.decomp_table[func]

        decomp_out = decomposition(*args, **kwargs)

        # Some inductor decompositions are conditional (e.g. device-specific)
        # and return NotImplemented to fall back to the native op.
        if decomp_out is NotImplemented:
            return func(*args, **kwargs)

        # Run the real op after the decomposition so we can detect if the
        # decomposition mutated inputs.
        real_out_unflat = func(*args, **kwargs)

        decomp_flat = pytree.tree_leaves(decomp_out)
        real_flat = pytree.tree_leaves(real_out_unflat)

        if len(real_flat) != len(decomp_flat):
            raise AssertionError(
                f"Inductor decomp output count mismatch for {func}: "
                f"{len(real_flat)} != {len(decomp_flat)}"
            )

        do_relative_check = self.test_dtype in (torch.float16, torch.bfloat16)

        if do_relative_check:
            device_arg = kwargs.get("device", None)

            def upcast(x):
                if (isinstance(x, Tensor) and x.device.type == "mps") or (
                    device_arg and torch.device(device_arg).type == "mps"
                ):
                    return _upcast(x, dtype=torch.float32)
                return _upcast(x, dtype=torch.float64)

            real_out_ref = pytree.tree_leaves(
                func(
                    *pytree.tree_map(upcast, args),
                    **pytree.tree_map(upcast, kwargs),
                )
            )
            for i, (orig, decomp_val, ref) in enumerate(
                zip(real_flat, decomp_flat, real_out_ref)
            ):
                if not isinstance(orig, Tensor):
                    if orig != decomp_val:
                        raise AssertionError(
                            f"Value mismatch for {func}: {orig} != {decomp_val}"
                        )
                    continue
                if orig.dtype != decomp_val.dtype:
                    raise AssertionError(
                        f"dtype mismatch for {func} output {i}: "
                        f"{orig.dtype} != {decomp_val.dtype}"
                    )
                if ref.is_floating_point() and orig.numel() > 0:
                    orig_diff = (orig - ref).abs().max()
                    decomp_diff = (decomp_val - ref).abs().max()
                    atol = _relative_tol_table.get(
                        (self.test_dtype, func), 1e-7
                    )
                    if decomp_diff > orig_diff + atol:
                        raise AssertionError(
                            f"Inductor decomp for {func.__name__} on output {i} "
                            f"is further from fp64 than native. "
                            f"Native max diff: {orig_diff}, "
                            f"Decomp max diff: {decomp_diff}, atol: {atol}"
                        )
                else:
                    self.test_case.assertEqual(orig, decomp_val, msg=str(func))
        else:
            for orig, decomp_val in zip(real_flat, decomp_flat):
                if not isinstance(orig, Tensor):
                    if type(orig) is not type(decomp_val):
                        raise AssertionError(
                            f"Type mismatch for {func}: "
                            f"{type(orig)} != {type(decomp_val)}"
                        )
                    if orig != decomp_val:
                        raise AssertionError(
                            f"Value mismatch for {func}: {orig} != {decomp_val}"
                        )
                    continue
                self.test_case.assertEqual(
                    orig.dtype,
                    decomp_val.dtype,
                    msg=f"dtype mismatch for {func}",
                )
                if (decomp_val.dtype, func) in _op_tolerances:
                    rtol, atol = _op_tolerances[(decomp_val.dtype, func)]
                else:
                    rtol, atol = _dtype_precisions.get(orig.dtype, (0, 0))
                self.test_case.assertEqual(
                    orig,
                    decomp_val,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Inductor decomp numerically incorrect for {func}",
                )

        return real_out_unflat


@unMarkDynamoStrictTest
class TestInductorDecompCorrectness(TestCase):
    longMessage = True

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_inductor_decomp_test_ops)
    def test_inductor_decomp(self, device, dtype, op):
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported for cross-ref")

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        func = op.get_op()
        aten_name = op.decomp_aten_name or op.aten_name

        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            with (
                InductorDecompCrossRefMode(
                    self, _inductor_table, dtype
                ) as mode,
                enable_python_dispatcher(),
            ):
                func(*args, **kwargs)


instantiate_device_type_tests(TestInductorDecompCorrectness, globals())

if __name__ == "__main__":
    run_tests()
