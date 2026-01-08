# Ordered Reduction Implementation Status

## Overview

This document summarizes the implementation status of **ordered reductions** in PyTorch Inductor, based on the design from PR 167498. Ordered reductions ensure associative operations (sum, prod, max, min) are performed in a user-specified order based on strides, providing numerical reproducibility for floating-point operations.

## Order Specification Semantics

The order is specified as a tuple of strides that defines the reduction tree structure:

### Flat Tuple: Tree Reduction
`(4, 2, 1)` for 8 elements produces a **binary tree** reduction:
- Elements reordered via bit-interleaving: `[0, 4, 2, 6, 1, 5, 3, 7]`
- Result: `(((0+4)+(2+6))+((1+5)+(3+7)))`

### Nested Tuple: Hierarchical (Linear within groups)
`((2, 1), 4)` for 8 elements produces **linear sums within groups**:
- Group 1: `[0,1,2,3]` reordered by `(2,1)` → `[0,2,1,3]` → `(((0+2)+1)+3)`
- Group 2: `[4,5,6,7]` reordered by `(2,1)` → `[4,6,5,7]` → `(((4+6)+5)+7)`
- Combined at stride 4: `((((0+2)+1)+3)+(((4+6)+5)+7))`

## Implementation Status

### Completed

| Component | File | Status |
|-----------|------|--------|
| **User-Facing API** | `torch/_inductor/inductor_prims.py` | ✅ Complete |
| - `inductor_prims.ordered_sum()` | | User-facing prim op |
| - Lowering | `torch/_inductor/lowering.py` | Creates ordered Reduction |
| IR Representation | `torch/_inductor/ir.py` | ✅ Complete |
| - `ordered` field on `Reduction` | | Added with default `False` |
| - `reduction_order` field | | Added with default `None` |
| - `reduction_grouping` field | | Added with default `None` (encodes nesting) |
| - `is_ordered()` method | | Added |
| - Propagation through `create()`, `create_multilayer()` | | Complete |
| Scheduler Constraints | `torch/_inductor/scheduler.py` | ✅ Complete |
| - `MixOrderReduction.has_ordered_reduction()` | | Added |
| - Fusion prevention for ordered reductions | | Implemented |
| Triton Codegen | `torch/_inductor/codegen/triton.py` | ✅ Complete |
| - `ordered_final_reduction()` | | Optimized tree codegen using `tl.reshape` + `tl.sum` |
| - `_generate_tree_reduction()` | | Binary tree codegen for flat orders |
| - `_generate_hierarchical_reduction()` | | Hierarchical codegen for nested orders |
| - Cooperative reduction check | | Prevents ordered + cooperative |
| - **No fallback** | | Throws `RuntimeError` for unsupported orders |
| Runtime Helpers | `torch/_inductor/runtime/triton_helpers.py` | ✅ Complete |
| - `compute_element_order()` | | Bit-interleaving algorithm |
| - `compute_hierarchical_reduction_structure()` | | Parses order tuple |
| - `generate_tree_sum_expression()` | | Generates tree expression |
| - `generate_linear_sum_indices()` | | Generates linear expression |
| - `is_nested_order()`, `flatten_order()` | | Helper utilities |
| - `decode_nested_order()` | | Decodes (order, grouping) to nested tuple |
| Configuration | `torch/_inductor/config.py` | ✅ Complete |
| - `ordered_reduction_chunk_size` | | Default 1024 |
| Tests | `test/inductor/test_ordered_reduction.py` | ✅ Complete |
| - IR field tests | | 3 tests |
| - Scheduler tests | | 1 test |
| - Codegen tests | | 2 tests |
| - Config tests | | 1 test |
| - Order specification tests | | 8 tests |
| - Tree codegen tests | | 2 tests |
| - `ordered_sum` prim tests | | 7 tests (including nested order) |
| - `decode_nested_order` tests | | 5 tests |

### Optimized Tree Codegen (Completed)

For flat order tuples like `(4, 2, 1)` that represent standard binary trees, the implementation generates optimized tree-structured code using Triton's `tl.reshape` and `tl.sum`:

```triton
# Optimized: Pairwise tree reduction for (4, 2, 1) on 8 elements
# Level 1: reshape [X,8]->[X,2,4], sum axis=1 -> [X,4]
tmp4 = tl.reshape(value, [XBLOCK, 2, 4])
tmp5 = tl.sum(tmp4, axis=1)  # pairs: (e0+e4), (e1+e5), (e2+e6), (e3+e7)
# Level 2: reshape [X,4]->[X,2,2], sum axis=1 -> [X,2]
tmp6 = tl.reshape(tmp5, [XBLOCK, 2, 2])
tmp7 = tl.sum(tmp6, axis=1)  # pairs: ((e0+e4)+(e2+e6)), ((e1+e5)+(e3+e7))
# Level 3: reshape [X,2]->[X,2,1], sum axis=1 -> [X,1]
tmp8 = tl.reshape(tmp7, [XBLOCK, 2, 1])
tmp9 = tl.sum(tmp8, axis=1)  # final sum
```

The implementation:
1. Checks if the order is a standard binary tree pattern (strides are `n//2, n//4, ..., 2, 1`)
2. If yes, generates optimized `tl.reshape` + `tl.sum` code at each tree level
3. If nested order, uses hierarchical reduction with `tl.split`

### Hierarchical Codegen (Completed)

For nested order tuples like `((2, 1), 4)` that represent hierarchical reductions, the implementation extracts individual elements using `tl.reshape` + `tl.split`, then sums them in the specified order:

```triton
# Hierarchical reduction for ((2, 1), 4) on 8 elements
# Step 1: Extract all 8 elements using binary splits
tmp4 = tl.reshape(tmp3, [XBLOCK, 4, 2])
tmp5, tmp6 = tl.split(tmp4)  # evens [0,2,4,6], odds [1,3,5,7]
tmp7 = tl.reshape(tmp5, [XBLOCK, 4])
tmp8 = tl.reshape(tmp6, [XBLOCK, 4])
# ... continue splitting until individual elements

# Step 2: Sum Group 1 (elements 0-3) in order [0, 2, 1, 3]
tmp39 = e0 + e2
tmp40 = tmp39 + e1
tmp41 = tmp40 + e3  # Group 1 result: (((e0+e2)+e1)+e3)

# Step 3: Sum Group 2 (elements 4-7) in order [4, 6, 5, 7]
tmp42 = e4 + e6
tmp43 = tmp42 + e5
tmp44 = tmp43 + e7  # Group 2 result: (((e4+e6)+e5)+e7)

# Step 4: Combine groups
tmp45 = tmp41 + tmp44  # Final result
```

The split-based approach:
1. Uses `tl.reshape([X, n], [X, n/2, 2])` + `tl.split()` to separate even/odd elements
2. Recursively splits until individual elements are extracted
3. Sums elements within each group following the inner element order
4. Combines group results

## Usage

### Via `inductor_prims.ordered_sum` (User-Facing API)

```python
from torch._inductor import inductor_prims

@torch.compile
def fn(x):
    # Tree reduction with flat order (4, 2, 1) for 8 elements
    return inductor_prims.ordered_sum(x, dim=1, order=[4, 2, 1], grouping=[])

@torch.compile
def fn_nested(x):
    # Hierarchical reduction with nested order ((2, 1), 4) for 8 elements
    # Encoded as: order=[2, 1, 4], grouping=[2, 1]
    # grouping=[2, 1] means: first 2 elements form tuple, then 1 element
    return inductor_prims.ordered_sum(x, dim=1, order=[2, 1, 4], grouping=[2, 1])

x = torch.randn(4, 8, device='cuda')
result = fn(x)  # Compiled with ordered tree reduction
result_nested = fn_nested(x)  # Compiled with hierarchical reduction
```

**Parameters:**
- `x`: Input tensor
- `dim`: Single dimension to reduce (int only)
- `order`: List of strides (flattened) defining the reduction order
- `grouping`: List specifying how elements in `order` form nested tuples
  - Empty list `[]` means flat order (standard binary tree)
  - `[2, 1]` means first 2 elements form tuple, then 1 element: `((a,b), c)`
  - `[2, 2]` means two pairs: `((a,b), (c,d))`

**Encoding Examples:**
| Nested Order | `order` | `grouping` |
|--------------|---------|------------|
| `(4, 2, 1)` (flat) | `[4, 2, 1]` | `[]` |
| `((4, 2), 1)` | `[4, 2, 1]` | `[2, 1]` |
| `((8, 4), (2, 1))` | `[8, 4, 2, 1]` | `[2, 2]` |
| `((2, 1), 4)` | `[2, 1, 4]` | `[2, 1]` |

**Error Handling:**
- Raises `RuntimeError` if the order cannot be achieved (non-standard binary tree pattern)
- Raises `ValueError` for invalid dim or order types

### Via Reduction.create (Internal API)

```python
from torch._inductor.ir import Reduction

# Tree reduction with order (4, 2, 1)
result = Reduction.create(
    device=torch.device("cuda"),
    dst_dtype=torch.float32,
    src_dtype=torch.float32,
    inner_fn=loader_fn,
    ranges=[batch_size],
    reduction_ranges=[8],
    reduction_type="sum",
    ordered=True,
    reduction_order=(4, 2, 1),
)

# Hierarchical reduction with linear within groups
result = Reduction.create(
    ...
    ordered=True,
    reduction_order=((2, 1), 4),
)
```

### Testing Helper Functions

```python
from torch._inductor.runtime.triton_helpers import (
    compute_element_order,
    generate_tree_sum_expression,
    is_nested_order,
    decode_nested_order,
)

# Compute element visit order
order = compute_element_order(8, (4, 2, 1))
# Returns: [0, 4, 2, 6, 1, 5, 3, 7]

# Generate tree expression
expr = generate_tree_sum_expression(order)
# Returns: (((e[0]+e[4])+(e[2]+e[6]))+((e[1]+e[5])+(e[3]+e[7])))

# Check if order is nested
is_nested_order(((2, 1), 4))  # True
is_nested_order((4, 2, 1))    # False

# Decode (order, grouping) back to nested tuple
decode_nested_order([2, 1, 4], [2, 1])  # Returns: ((2, 1), 4)
decode_nested_order([4, 2, 1], [])       # Returns: (4, 2, 1)
decode_nested_order([8, 4, 2, 1], [2, 2])  # Returns: ((8, 4), (2, 1))
```

## Files Modified

| File | Changes |
|------|---------|
| `torch/_inductor/inductor_prims.py` | Added `ordered_sum` prim op |
| `torch/_inductor/lowering.py` | Added lowering for `ordered_sum` |
| `torch/_inductor/ir.py` | Added `ordered`, `reduction_order` fields; updated create methods |
| `torch/_inductor/scheduler.py` | Added `has_ordered_reduction()`; fusion constraints |
| `torch/_inductor/codegen/triton.py` | Added `ordered_final_reduction()`, `_generate_tree_reduction()`, `_generate_hierarchical_reduction()` |
| `torch/_inductor/runtime/triton_helpers.py` | Added ordered reduction helper functions |
| `torch/_inductor/config.py` | Added `ordered_reduction_chunk_size` |
| `test/inductor/test_ordered_reduction.py` | Test file with 22 tests |
| `torch/_inductor/ops_handler.py` | Updated `reduction()` signature with ordered params |
| `torch/_inductor/loop_body.py` | Updated `reduction()` signature |
| `torch/_inductor/dependencies.py` | Updated `reduction()` signature |
| `torch/_inductor/bounds.py` | Updated `reduction()` signature |
| `torch/_inductor/shape_propagation.py` | Updated `reduction()` signature |
| `torch/_inductor/dtype_propagation.py` | Updated `reduction()` signature |
| `torch/_inductor/codegen/common.py` | Updated `reduction()` signatures |
| `torch/_inductor/codegen/cpp.py` | Updated `reduction()` signatures |
| `torch/_inductor/codegen/pallas.py` | Updated `reduction()` signature |
| `torch/_inductor/codegen/mps.py` | Updated `reduction()` signature |
| `torch/_inductor/codegen/halide.py` | Updated `reduction()` signature |
| `torch/_inductor/codegen/triton_split_scan.py` | Updated `reduction()` signature |

## Running Tests

```bash
python test/inductor/test_ordered_reduction.py
```

All 29 tests pass as of the current implementation.

## References

- PR 167498: Original design for consecutive linear addition / ordered reduction
- The order tuple encodes element visit order via bit-interleaving with strides
