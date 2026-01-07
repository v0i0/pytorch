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


if __name__ == "__main__":
    run_tests()
