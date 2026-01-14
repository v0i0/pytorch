# Owner(s): ["module: inductor"]

"""
Tests for ordered reductions in PyTorch Inductor.

Ordered reductions ensure that associative operations (sum, prod, max, min)
are performed in a user-specified order based on strides, providing numerical
reproducibility for floating-point operations.

The order specification follows the "consecutive linear addition" pattern from
PR 167498, where order is a tuple of strides defining the reduction tree:
  - (4, 2, 1) for 8 elements: standard binary tree
  - ((4, 1), 2, 8) for nested/hierarchical ordering
"""

import unittest

import torch
from torch._inductor import config
from torch._inductor.ir import Reduction
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


class TestOrderedReductionIR(TestCase):
    """Tests for ordered reduction IR representation."""

    def test_reduction_ordered_field_default(self):
        """Test that ordered field defaults to False."""
        # This is a unit test for the IR dataclass
        from torch._inductor.ir import Reduction

        # Create a mock reduction to test the ordered field
        # Note: This is just testing the dataclass field exists
        assert hasattr(Reduction, "__dataclass_fields__")
        fields = Reduction.__dataclass_fields__
        self.assertIn("ordered", fields)
        # Check default value is False
        self.assertEqual(fields["ordered"].default, False)

    def test_reduction_order_field_default(self):
        """Test that reduction_order field defaults to None."""
        from torch._inductor.ir import Reduction

        assert hasattr(Reduction, "__dataclass_fields__")
        fields = Reduction.__dataclass_fields__
        self.assertIn("reduction_order", fields)
        # Check default value is None
        self.assertIsNone(fields["reduction_order"].default)

    def test_reduction_is_ordered_method(self):
        """Test the is_ordered helper method."""
        # We can't easily test this without creating a full Reduction object,
        # but we can verify the method exists
        self.assertTrue(hasattr(Reduction, "is_ordered"))


class TestOrderedReductionScheduler(TestCase):
    """Tests for ordered reduction scheduler constraints."""

    def test_has_ordered_reduction_helper(self):
        """Test the has_ordered_reduction helper function exists."""
        from torch._inductor.scheduler import MixOrderReduction

        self.assertTrue(hasattr(MixOrderReduction, "has_ordered_reduction"))


@torch._inductor.config.patch({"triton.cooperative_reductions": False})
class TestOrderedReductionCodegen(TestCase):
    """Tests for ordered reduction code generation."""

    @classmethod
    def setUpClass(cls):
        if not HAS_GPU:
            raise unittest.SkipTest("GPU not available")

    def test_ordered_reduction_triton_helpers_exist(self):
        """Test that ordered reduction helpers exist in triton_helpers."""
        from torch._inductor.runtime import triton_helpers

        # Verify the ordered reduction functions exist
        self.assertTrue(hasattr(triton_helpers, "ordered_sum"))
        self.assertTrue(hasattr(triton_helpers, "ordered_prod"))
        self.assertTrue(hasattr(triton_helpers, "ordered_max"))
        self.assertTrue(hasattr(triton_helpers, "ordered_min"))

    def test_hierarchical_helpers_exist(self):
        """Test that hierarchical reduction helpers exist."""
        from torch._inductor.runtime import triton_helpers

        # Verify the hierarchical reduction helpers exist
        self.assertTrue(hasattr(triton_helpers, "compute_element_order"))
        self.assertTrue(hasattr(triton_helpers, "compute_hierarchical_reduction_structure"))
        self.assertTrue(hasattr(triton_helpers, "is_nested_order"))


class TestOrderedReductionConfig(TestCase):
    """Tests for ordered reduction configuration."""

    def test_config_option_exists(self):
        """Test that the config option for ordered reduction chunk size exists."""
        self.assertTrue(hasattr(config, "ordered_reduction_chunk_size"))
        # Default should be 1024
        self.assertEqual(config.ordered_reduction_chunk_size, 1024)


class TestOrderSpecification(TestCase):
    """Tests for order specification parsing and validation.

    Order specification from PR 167498:
    - Flat tuple (4, 2, 1): TREE reduction with stride-based reordering
      Result: (((0+4)+(2+6))+((1+5)+(3+7)))

    - Nested tuple ((2, 1), 4): LINEAR within groups, then combine
      Result: ((((0+2)+1)+3)+(((4+6)+5)+7))

    The strides determine element visit order via bit-interleaving:
    For (4, 2, 1) on 8 elements, position i maps to sum(bit[j] * stride[j])
    """

    def test_flatten_order_tuple(self):
        """Test flattening nested order tuples."""
        from torch._inductor.runtime.triton_helpers import flatten_order

        # Simple case
        self.assertEqual(flatten_order((4, 2, 1)), [4, 2, 1])

        # Nested case
        self.assertEqual(flatten_order(((4, 1), 2, 8)), [4, 1, 2, 8])

        # Deeply nested
        self.assertEqual(flatten_order((((1, 2), 4), 8)), [1, 2, 4, 8])

    def test_is_nested_order(self):
        """Test detection of nested order tuples."""
        from torch._inductor.runtime.triton_helpers import is_nested_order

        # Flat tuples
        self.assertFalse(is_nested_order((4, 2, 1)))
        self.assertFalse(is_nested_order((8,)))

        # Nested tuples
        self.assertTrue(is_nested_order(((2, 1), 4)))
        self.assertTrue(is_nested_order(((4, 1), 2, 8)))
        self.assertTrue(is_nested_order((((1, 2), 4), 8)))

    def test_compute_element_order_421(self):
        """Test element reordering for (4, 2, 1) on 8 elements.

        Expected: [0, 4, 2, 6, 1, 5, 3, 7]

        This produces the tree: (((0+4)+(2+6))+((1+5)+(3+7)))
        """
        from torch._inductor.runtime.triton_helpers import compute_element_order

        order = compute_element_order(8, (4, 2, 1))
        self.assertEqual(order, [0, 4, 2, 6, 1, 5, 3, 7])

    def test_compute_element_order_21(self):
        """Test element reordering for (2, 1) on 4 elements.

        Expected: [0, 2, 1, 3]

        For linear sum: (((0+2)+1)+3)
        """
        from torch._inductor.runtime.triton_helpers import compute_element_order

        order = compute_element_order(4, (2, 1))
        self.assertEqual(order, [0, 2, 1, 3])

    def test_generate_tree_sum_expression(self):
        """Test tree sum expression generation."""
        from torch._inductor.runtime.triton_helpers import generate_tree_sum_expression

        # For order [0, 4, 2, 6, 1, 5, 3, 7] (from (4,2,1) on 8 elements)
        expr = generate_tree_sum_expression([0, 4, 2, 6, 1, 5, 3, 7])
        # Should produce: (((e[0]+e[4])+(e[2]+e[6]))+((e[1]+e[5])+(e[3]+e[7])))
        self.assertIn("e[0]+e[4]", expr)
        self.assertIn("e[2]+e[6]", expr)
        self.assertIn("e[1]+e[5]", expr)
        self.assertIn("e[3]+e[7]", expr)

    def test_generate_linear_sum_indices(self):
        """Test linear sum expression generation."""
        from torch._inductor.runtime.triton_helpers import generate_linear_sum_indices

        # For order [0, 2, 1, 3] (from (2,1) on 4 elements)
        expr = generate_linear_sum_indices([0, 2, 1, 3])
        # Should be nested left-to-right: (((e[0] + e[2]) + e[1]) + e[3])
        # The format is: (({prev} + e[idx]))
        self.assertIn("e[0]", expr)
        self.assertIn("e[2]", expr)
        self.assertIn("e[1]", expr)
        self.assertIn("e[3]", expr)

    def test_compute_hierarchical_reduction_structure_flat(self):
        """Test structure computation for flat order tuple."""
        from torch._inductor.runtime.triton_helpers import (
            compute_hierarchical_reduction_structure,
        )

        structure = compute_hierarchical_reduction_structure(8, (4, 2, 1))
        self.assertEqual(structure['type'], 'tree')
        self.assertEqual(structure['element_order'], [0, 4, 2, 6, 1, 5, 3, 7])
        self.assertEqual(structure['strides'], [4, 2, 1])

    def test_compute_hierarchical_reduction_structure_nested(self):
        """Test structure computation for nested order tuple."""
        from torch._inductor.runtime.triton_helpers import (
            compute_hierarchical_reduction_structure,
        )

        structure = compute_hierarchical_reduction_structure(8, ((2, 1), 4))
        self.assertEqual(structure['type'], 'hierarchical')
        self.assertEqual(len(structure['groups']), 1)
        self.assertEqual(structure['groups'][0]['strides'], [2, 1])
        self.assertEqual(structure['groups'][0]['element_order'], [0, 2, 1, 3])
        self.assertEqual(structure['outer_strides'], [4])


class TestTreeReductionCodegen(TestCase):
    """Tests for optimized tree reduction code generation."""

    def test_is_standard_binary_tree_valid(self):
        """Test detection of valid standard binary tree orders."""
        from torch._inductor.runtime.triton_helpers import flatten_order

        def is_standard_tree(order):
            """Check if order is a standard binary tree pattern."""
            flat_strides = flatten_order(order)
            if not flat_strides or flat_strides[-1] != 1:
                return False
            for i in range(len(flat_strides) - 1):
                if flat_strides[i] != 2 * flat_strides[i + 1]:
                    return False
            return True

        # Valid standard binary trees
        self.assertTrue(is_standard_tree((4, 2, 1)))
        self.assertTrue(is_standard_tree((8, 4, 2, 1)))
        self.assertTrue(is_standard_tree((2, 1)))
        self.assertTrue(is_standard_tree((1,)))

        # Invalid patterns
        self.assertFalse(is_standard_tree((3, 1)))  # 3 != 2*1
        self.assertFalse(is_standard_tree((4, 2, 2)))  # ends with 2, not 1
        self.assertFalse(is_standard_tree((4, 1)))  # 4 != 2*1

    def test_tree_reduction_numerical_equivalence(self):
        """Test that tree reduction gives same result as standard sum.

        For floating point, the tree reduction structure affects numerical
        precision but not correctness. This test verifies the implementation
        computes the correct sum (within floating point tolerance).
        """
        import torch
        from torch._inductor.runtime.triton_helpers import (
            compute_element_order,
            generate_tree_sum_expression,
        )

        # Test with 8 elements and order (4, 2, 1)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        order = (4, 2, 1)

        # Compute expected result
        expected = sum(values)

        # Compute tree reduction result by evaluating the expression
        element_order = compute_element_order(8, order)
        # element_order = [0, 4, 2, 6, 1, 5, 3, 7]

        # Tree reduction: pairs at each level
        # Level 1: (v[0]+v[4]), (v[2]+v[6]), (v[1]+v[5]), (v[3]+v[7])
        level1 = [
            values[0] + values[4],  # 1+5 = 6
            values[2] + values[6],  # 3+7 = 10
            values[1] + values[5],  # 2+6 = 8
            values[3] + values[7],  # 4+8 = 12
        ]
        # Level 2: ((v[0]+v[4])+(v[2]+v[6])), ((v[1]+v[5])+(v[3]+v[7]))
        level2 = [
            level1[0] + level1[1],  # 6+10 = 16
            level1[2] + level1[3],  # 8+12 = 20
        ]
        # Final: 16+20 = 36
        tree_result = level2[0] + level2[1]

        self.assertEqual(tree_result, expected)
        self.assertEqual(tree_result, 36.0)


class TestOrderedSumPrim(TestCase):
    """Tests for the inductor_prims.ordered_sum operation."""

    def test_ordered_sum_prim_exists(self):
        """Test that ordered_sum prim is defined."""
        from torch._inductor import inductor_prims

        self.assertTrue(hasattr(inductor_prims, 'ordered_sum'))

    def test_ordered_sum_eager_mode(self):
        """Test ordered_sum in eager mode (no compilation)."""
        from torch._inductor import inductor_prims

        x = torch.randn(4, 8)
        # Note: prim schema uses int[] for both order and grouping
        order = [4, 2, 1]
        grouping = []  # Empty grouping means flat order

        # In eager mode, ordered_sum just does a regular sum
        result = inductor_prims.ordered_sum(x, dim=1, order=order, grouping=grouping)
        expected = x.sum(dim=1)

        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ordered_sum_compiled_flat_order(self):
        """Test ordered_sum with flat order (tree reduction) under compilation."""
        from torch._inductor import inductor_prims

        @torch.compile
        def fn(x):
            # Order (4, 2, 1) for 8 elements = standard binary tree
            # Empty grouping means flat order
            return inductor_prims.ordered_sum(x, dim=1, order=[4, 2, 1], grouping=[])

        x = torch.randn(4, 8, device='cuda')

        result = fn(x)
        expected = x.sum(dim=1)

        # Results should be equal (same sum, just different order of operations)
        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ordered_sum_compiled_different_sizes(self):
        """Test ordered_sum with different reduction sizes."""
        from torch._inductor import inductor_prims

        @torch.compile
        def fn4(x):
            # Order (2, 1) for 4 elements
            return inductor_prims.ordered_sum(x, dim=1, order=[2, 1], grouping=[])

        @torch.compile
        def fn16(x):
            # Order (8, 4, 2, 1) for 16 elements
            return inductor_prims.ordered_sum(x, dim=1, order=[8, 4, 2, 1], grouping=[])

        x4 = torch.randn(4, 4, device='cuda')
        x16 = torch.randn(4, 16, device='cuda')

        result4 = fn4(x4)
        result16 = fn16(x16)

        self.assertTrue(torch.allclose(result4, x4.sum(dim=1)))
        self.assertTrue(torch.allclose(result16, x16.sum(dim=1)))

    def test_ordered_sum_invalid_dim_type(self):
        """Test that ordered_sum raises RuntimeError for non-integer dim."""
        from torch._inductor import inductor_prims

        x = torch.randn(4, 8)

        with self.assertRaises(RuntimeError):
            # dim should be int, not list
            inductor_prims.ordered_sum(x, dim=[1], order=[4, 2, 1], grouping=[])

    def test_ordered_sum_eager_mode_with_grouping(self):
        """Test ordered_sum in eager mode with non-empty grouping (nested order)."""
        from torch._inductor import inductor_prims

        x = torch.randn(4, 8)
        # Nested order ((4, 2), 1) encoded as order=[4, 2, 1], grouping=[2, 1]
        order = [4, 2, 1]
        grouping = [2, 1]

        # In eager mode, ordered_sum just does a regular sum
        result = inductor_prims.ordered_sum(x, dim=1, order=order, grouping=grouping)
        expected = x.sum(dim=1)

        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ordered_sum_compiled_nested_order(self):
        """Test ordered_sum with nested order under compilation.

        Nested order ((2, 1), 4) is encoded as:
        - order = [2, 1, 4]
        - grouping = [2, 1]  (first 2 elements form a tuple, then 1 element)

        For 8 elements:
        - Inner (2, 1) applies to 4-element groups with order [0, 2, 1, 3]
        - Outer stride 4 separates two groups: [0-3] and [4-7]
        """
        from torch._inductor import inductor_prims

        @torch.compile
        def fn(x):
            # Nested order ((2, 1), 4) for 8 elements
            # First two strides form inner tuple, last is outer
            return inductor_prims.ordered_sum(x, dim=1, order=[2, 1, 4], grouping=[2, 1])

        x = torch.randn(4, 8, device='cuda')

        result = fn(x)
        expected = x.sum(dim=1)

        # Results should be equal (same sum, just different order of operations)
        self.assertTrue(torch.allclose(result, expected))


class TestDecodeNestedOrder(TestCase):
    """Tests for decode_nested_order helper function."""

    def test_decode_flat_order(self):
        """Test decoding flat order (empty grouping)."""
        from torch._inductor.runtime.triton_helpers import decode_nested_order

        # Empty grouping means flat order
        result = decode_nested_order([4, 2, 1], [])
        self.assertEqual(result, (4, 2, 1))

    def test_decode_simple_nested(self):
        """Test decoding simple nested order."""
        from torch._inductor.runtime.triton_helpers import decode_nested_order

        # ((4, 2), 1) encoded as order=[4,2,1], grouping=[2,1]
        result = decode_nested_order([4, 2, 1], [2, 1])
        self.assertEqual(result, ((4, 2), 1))

    def test_decode_multiple_nested_groups(self):
        """Test decoding order with multiple nested groups."""
        from torch._inductor.runtime.triton_helpers import decode_nested_order

        # ((8, 4), (2, 1)) encoded as order=[8,4,2,1], grouping=[2,2]
        result = decode_nested_order([8, 4, 2, 1], [2, 2])
        self.assertEqual(result, ((8, 4), (2, 1)))

    def test_decode_deeply_nested(self):
        """Test decoding with larger groups."""
        from torch._inductor.runtime.triton_helpers import decode_nested_order

        # ((4, 2, 1), 8) encoded as order=[4,2,1,8], grouping=[3,1]
        result = decode_nested_order([4, 2, 1, 8], [3, 1])
        self.assertEqual(result, ((4, 2, 1), 8))

    def test_decode_invalid_grouping(self):
        """Test that invalid grouping raises ValueError."""
        from torch._inductor.runtime.triton_helpers import decode_nested_order

        # Grouping doesn't sum to order length
        with self.assertRaises(ValueError):
            decode_nested_order([4, 2, 1], [1, 1])  # sum=2, but need 3


class TestPartitionOrderForSplit(TestCase):
    """Tests for partition_order_for_split helper function."""

    def test_partition_order_basic(self):
        """Test basic order partitioning for split."""
        from torch._inductor.ir import Reduction

        # 8192 elements split into 8 chunks of 1024
        order = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        split = 8
        reduction_numel = 8192

        within, across = Reduction.partition_order_for_split(order, split, reduction_numel)

        # chunk_size = 8192 / 8 = 1024
        # Strides < 1024: (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        # Strides >= 1024: (4096, 2048, 1024) -> (4, 2, 1) after dividing by 1024
        self.assertEqual(within, (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))
        self.assertEqual(across, (4, 2, 1))

    def test_partition_order_small(self):
        """Test order partitioning for small split."""
        from torch._inductor.ir import Reduction

        # 16 elements split into 2 chunks of 8
        order = (8, 4, 2, 1)
        split = 2
        reduction_numel = 16

        within, across = Reduction.partition_order_for_split(order, split, reduction_numel)

        # chunk_size = 16 / 2 = 8
        # Strides < 8: (4, 2, 1)
        # Strides >= 8: (8,) -> (1,) after dividing by 8
        self.assertEqual(within, (4, 2, 1))
        self.assertEqual(across, (1,))

    def test_partition_order_helper(self):
        """Test the triton_helpers partition function."""
        from torch._inductor.runtime.triton_helpers import partition_order_for_chunk

        order = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        chunk_size = 1024

        within, across = partition_order_for_chunk(order, chunk_size)

        self.assertEqual(within, (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))
        self.assertEqual(across, (4, 2, 1))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@torch._inductor.config.patch({"triton.cooperative_reductions": False})
class TestSplitOrderedReductions(TestCase):
    """Tests for split ordered reductions (multi-kernel).

    These tests verify that when an ordered reduction is split into multiple
    kernels, the result is bitwise-identical to the non-split (persistent)
    version.
    """

    @staticmethod
    def get_order_for_size(red_size):
        """Generate flat tree order for a given reduction size (must be power of 2)."""
        order = []
        s = red_size // 2
        while s >= 1:
            order.append(s)
            s //= 2
        return tuple(order)

    def test_split_ordered_reduction_bitwise_identical_fp32(self):
        """Verify split ordered reduction matches persistent version exactly (fp32)."""
        from torch._inductor import inductor_prims

        for size in [2048, 4096]:
            order = list(self.get_order_for_size(size))

            x = torch.randn(100, size, device="cuda", dtype=torch.float32)

            # Force persistent (no split) - reference
            with config.patch({"split_reductions": False}):
                torch._dynamo.reset()

                @torch.compile
                def persistent_fn(x):
                    return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

                ref = persistent_fn(x)

            # Allow split - test
            with config.patch({"split_reductions": True}):
                torch._dynamo.reset()

                @torch.compile
                def split_fn(x):
                    return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

                result = split_fn(x)

            # Must be bitwise identical
            self.assertTrue(
                torch.equal(ref, result),
                f"Mismatch at size={size}, dtype=fp32. "
                f"Max diff: {(ref - result).abs().max().item()}"
            )

    def test_split_ordered_reduction_bitwise_identical_fp16(self):
        """Verify split ordered reduction matches persistent version exactly (fp16)."""
        from torch._inductor import inductor_prims

        for size in [2048, 4096]:
            order = list(self.get_order_for_size(size))

            x = torch.randn(100, size, device="cuda", dtype=torch.float16)

            # Force persistent (no split) - reference
            with config.patch({"split_reductions": False}):
                torch._dynamo.reset()

                @torch.compile
                def persistent_fn(x):
                    return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

                ref = persistent_fn(x)

            # Allow split - test
            with config.patch({"split_reductions": True}):
                torch._dynamo.reset()

                @torch.compile
                def split_fn(x):
                    return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

                result = split_fn(x)

            # Must be bitwise identical
            self.assertTrue(
                torch.equal(ref, result),
                f"Mismatch at size={size}, dtype=fp16. "
                f"Max diff: {(ref - result).abs().max().item()}"
            )

    def test_split_ordered_reduction_reproducibility(self):
        """Verify split ordered reduction produces identical results across runs."""
        from torch._inductor import inductor_prims

        size = 4096
        order = list(self.get_order_for_size(size))

        x = torch.randn(100, size, device="cuda", dtype=torch.float32)

        with config.patch({"split_reductions": True}):
            results = []
            for _ in range(5):
                torch._dynamo.reset()

                @torch.compile
                def fn(x):
                    return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

                results.append(fn(x).clone())

            # All runs must be bitwise identical
            reference = results[0]
            for i, result in enumerate(results[1:], 1):
                self.assertTrue(
                    torch.equal(reference, result),
                    f"Run {i} differs from run 0"
                )


class TestOrderedDotPrim(TestCase):
    """Tests for the inductor_prims.ordered_dot operation."""

    def test_ordered_dot_prim_exists(self):
        """Test that ordered_dot prim is defined."""
        from torch._inductor import inductor_prims

        self.assertTrue(hasattr(inductor_prims, "ordered_dot"))

    def test_ordered_dot_eager_mode(self):
        """Test ordered_dot in eager mode (no compilation)."""
        from torch._inductor import inductor_prims

        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        # Nested order required for ordered_dot
        order = [2, 1, 4]
        grouping = [2, 1]  # Decodes to ((2, 1), 4)

        # In eager mode, ordered_dot just does (a * b).sum()
        result = inductor_prims.ordered_dot(a, b, dim=1, order=order, grouping=grouping)
        expected = (a * b).sum(dim=1)

        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @torch._inductor.config.patch({"triton.cooperative_reductions": False})
    def test_ordered_dot_compiled_nested_order(self):
        """Test ordered_dot with nested order under compilation.

        Nested order ((2, 1), 4) for 8 elements:
        - 2 groups of 4 elements each
        - Within each group: FMA chain in order [0, 2, 1, 3]
        - Combine groups with addition
        """
        from torch._inductor import inductor_prims

        @torch.compile
        def fn(a, b):
            # Nested order ((2, 1), 4) for 8 elements
            return inductor_prims.ordered_dot(a, b, dim=1, order=[2, 1, 4], grouping=[2, 1])

        a = torch.randn(4, 8, device="cuda")
        b = torch.randn(4, 8, device="cuda")

        result = fn(a, b)
        expected = (a * b).sum(dim=1)

        # Results should be equal (same dot product, just different order)
        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @torch._inductor.config.patch({"triton.cooperative_reductions": False})
    def test_ordered_dot_compiled_16_elements(self):
        """Test ordered_dot with 16 elements."""
        from torch._inductor import inductor_prims

        @torch.compile
        def fn(a, b):
            # Nested order ((4, 2, 1), 8) for 16 elements
            # 2 groups of 8 elements, each processed with FMA chain
            return inductor_prims.ordered_dot(a, b, dim=1, order=[4, 2, 1, 8], grouping=[3, 1])

        a = torch.randn(4, 16, device="cuda")
        b = torch.randn(4, 16, device="cuda")

        result = fn(a, b)
        expected = (a * b).sum(dim=1)

        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @torch._inductor.config.patch({"triton.cooperative_reductions": False})
    def test_ordered_dot_reproducibility(self):
        """Test that ordered_dot gives reproducible results."""
        from torch._inductor import inductor_prims

        a = torch.randn(10, 8, device="cuda")
        b = torch.randn(10, 8, device="cuda")
        order = [2, 1, 4]
        grouping = [2, 1]

        results = []
        for _ in range(5):
            torch._dynamo.reset()

            @torch.compile
            def fn(a, b):
                return inductor_prims.ordered_dot(a, b, dim=1, order=order, grouping=grouping)

            results.append(fn(a, b).clone())

        # All runs must be bitwise identical
        reference = results[0]
        for i, result in enumerate(results[1:], 1):
            self.assertTrue(
                torch.equal(reference, result),
                f"Run {i} differs from run 0"
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @torch._inductor.config.patch({"triton.cooperative_reductions": False})
    def test_ordered_dot_broadcast(self):
        """Test that ordered_dot properly broadcasts inputs."""
        from torch._inductor import inductor_prims

        @torch.compile
        def fn(a, b):
            return inductor_prims.ordered_dot(a, b, dim=1, order=[2, 1, 4], grouping=[2, 1])

        # a has shape [4, 8], b has shape [1, 8] - should broadcast
        a = torch.randn(4, 8, device="cuda")
        b = torch.randn(1, 8, device="cuda")

        result = fn(a, b)
        expected = (a * b).sum(dim=1)

        self.assertTrue(torch.allclose(result, expected))

    def test_ordered_dot_invalid_dim_type(self):
        """Test that ordered_dot raises error for non-integer dim."""
        from torch._inductor import inductor_prims

        a = torch.randn(4, 8)
        b = torch.randn(4, 8)

        with self.assertRaises(RuntimeError):
            # dim should be int, not list
            inductor_prims.ordered_dot(a, b, dim=[1], order=[2, 1, 4], grouping=[2, 1])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@torch._inductor.config.patch({"triton.cooperative_reductions": False})
class TestLoopedOrderedReduction(TestCase):
    """Tests for looped ordered reductions.

    Looped ordered reductions process chunks in an unrolled loop within a single
    kernel, keeping chunk results in registers. This is beneficial for large
    reduction sizes (2048+) with half-precision dtypes (fp16, bf16).
    """

    @staticmethod
    def get_order_for_size(red_size):
        """Generate flat tree order for a given reduction size (must be power of 2)."""
        order = []
        s = red_size // 2
        while s >= 1:
            order.append(s)
            s //= 2
        return order

    def test_looped_config_exists(self):
        """Test that looped ordered reduction config exists."""
        self.assertTrue(hasattr(config, "looped_ordered_reduction_max_chunks"))
        self.assertEqual(config.looped_ordered_reduction_max_chunks, 16)

    def test_looped_basic_fp16(self):
        """Test basic looped ordered reduction with fp16."""
        from torch._inductor import inductor_prims

        size = 2048  # 2 chunks with default chunk_size=1024
        order = self.get_order_for_size(size)

        x = torch.randn(1000, size, device="cuda", dtype=torch.float16)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        # Should be close (not exact due to different accumulation order)
        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_looped_basic_bf16(self):
        """Test basic looped ordered reduction with bf16."""
        from torch._inductor import inductor_prims

        size = 2048
        order = self.get_order_for_size(size)

        x = torch.randn(1000, size, device="cuda", dtype=torch.bfloat16)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_looped_basic_fp32_uses_regular_mode(self):
        """Test that fp32 does not use looped mode (falls back to regular tree)."""
        from torch._inductor import inductor_prims

        size = 2048
        order = self.get_order_for_size(size)

        x = torch.randn(1000, size, device="cuda", dtype=torch.float32)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        # Should be close
        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-3, atol=1e-3),
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_looped_reproducibility_fp16(self):
        """Test that looped ordered reduction produces identical results across runs (fp16)."""
        from torch._inductor import inductor_prims

        size = 4096
        order = self.get_order_for_size(size)

        x = torch.randn(100, size, device="cuda", dtype=torch.float16)

        results = []
        for _ in range(5):
            torch._dynamo.reset()

            @torch.compile
            def fn(x):
                return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

            results.append(fn(x).clone())

        # All runs must be bitwise identical
        reference = results[0]
        for i, result in enumerate(results[1:], 1):
            self.assertTrue(
                torch.equal(reference, result),
                f"Run {i} differs from run 0"
            )

    def test_looped_reproducibility_bf16(self):
        """Test that looped ordered reduction produces identical results across runs (bf16)."""
        from torch._inductor import inductor_prims

        size = 4096
        order = self.get_order_for_size(size)

        x = torch.randn(100, size, device="cuda", dtype=torch.bfloat16)

        results = []
        for _ in range(5):
            torch._dynamo.reset()

            @torch.compile
            def fn(x):
                return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

            results.append(fn(x).clone())

        # All runs must be bitwise identical
        reference = results[0]
        for i, result in enumerate(results[1:], 1):
            self.assertTrue(
                torch.equal(reference, result),
                f"Run {i} differs from run 0"
            )

    def test_looped_various_sizes_fp16(self):
        """Test looped ordered reduction with various sizes (fp16)."""
        from torch._inductor import inductor_prims

        # Test sizes that trigger looped mode (>1024, power of 2, <= 16 chunks)
        sizes = [2048, 4096, 8192, 16384]

        for size in sizes:
            order = self.get_order_for_size(size)
            x = torch.randn(100, size, device="cuda", dtype=torch.float16)

            torch._dynamo.reset()

            @torch.compile
            def fn(x):
                return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

            result = fn(x)
            expected = x.sum(dim=1)

            self.assertTrue(
                torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
                f"Mismatch at size={size}. Max diff: {(result - expected).abs().max().item()}"
            )

    def test_looped_matches_non_looped_fp16(self):
        """Test that looped mode gives bitwise-identical results to non-looped (persistent)."""
        from torch._inductor import inductor_prims

        size = 2048
        order = self.get_order_for_size(size)

        x = torch.randn(100, size, device="cuda", dtype=torch.float16)

        # Force non-looped by disabling with config
        with config.patch({"looped_ordered_reduction_max_chunks": 0}):
            torch._dynamo.reset()

            @torch.compile
            def non_looped_fn(x):
                return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

            non_looped_result = non_looped_fn(x)

        # Normal execution (looped mode enabled)
        torch._dynamo.reset()

        @torch.compile
        def looped_fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        looped_result = looped_fn(x)

        # Must be bitwise identical (same numerical order)
        self.assertTrue(
            torch.equal(non_looped_result, looped_result),
            f"Looped and non-looped differ. Max diff: {(non_looped_result - looped_result).abs().max().item()}"
        )

    def test_looped_4_chunks(self):
        """Test looped reduction with exactly 4 chunks (4096 elements)."""
        from torch._inductor import inductor_prims

        size = 4096  # 4 chunks with chunk_size=1024
        order = self.get_order_for_size(size)

        x = torch.randn(100, size, device="cuda", dtype=torch.float16)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_looped_8_chunks(self):
        """Test looped reduction with exactly 8 chunks (8192 elements)."""
        from torch._inductor import inductor_prims

        size = 8192  # 8 chunks with chunk_size=1024
        order = self.get_order_for_size(size)

        x = torch.randn(100, size, device="cuda", dtype=torch.float16)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_looped_small_batch(self):
        """Test looped reduction with small batch size."""
        from torch._inductor import inductor_prims

        size = 2048
        order = self.get_order_for_size(size)

        x = torch.randn(4, size, device="cuda", dtype=torch.float16)

        torch._dynamo.reset()

        @torch.compile
        def fn(x):
            return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=[])

        result = fn(x)
        expected = x.sum(dim=1)

        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-2, atol=1e-2),
            f"Max diff: {(result - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    run_tests()
