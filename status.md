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
| IR Representation | `torch/_inductor/ir.py` | ✅ Complete |
| - `ordered` field on `Reduction` | | Added with default `False` |
| - `reduction_order` field | | Added with default `None` |
| - `is_ordered()` method | | Added |
| - Propagation through `create()`, `create_multilayer()` | | Complete |
| Scheduler Constraints | `torch/_inductor/scheduler.py` | ✅ Complete |
| - `MixOrderReduction.has_ordered_reduction()` | | Added |
| - Fusion prevention for ordered reductions | | Implemented |
| Triton Codegen | `torch/_inductor/codegen/triton.py` | ✅ Basic |
| - `ordered_final_reduction()` | | Uses `triton_helpers.ordered_sum` |
| - Cooperative reduction check | | Prevents ordered + cooperative |
| Runtime Helpers | `torch/_inductor/runtime/triton_helpers.py` | ✅ Complete |
| - `ordered_sum()`, `ordered_prod()` | | Sequential loop implementation |
| - `ordered_max()`, `ordered_min()` | | Delegates to `tl.max/min` |
| - `compute_element_order()` | | Bit-interleaving algorithm |
| - `compute_hierarchical_reduction_structure()` | | Parses order tuple |
| - `generate_tree_sum_expression()` | | Generates tree expression |
| - `generate_linear_sum_indices()` | | Generates linear expression |
| - `is_nested_order()`, `flatten_order()` | | Helper utilities |
| Configuration | `torch/_inductor/config.py` | ✅ Complete |
| - `ordered_reduction_chunk_size` | | Default 1024 |
| Tests | `test/inductor/test_ordered_reduction.py` | ✅ Complete |
| - IR field tests | | 3 tests |
| - Scheduler tests | | 1 test |
| - Codegen tests | | 2 tests |
| - Config tests | | 1 test |
| - Order specification tests | | 8 tests |

### TODO: Optimized Tree Codegen

The current implementation uses a sequential loop for ordered reductions:

```triton
# Current: Sequential loop (correct but not optimal for tree orders)
for i in tl.range(0, n):
    result = result + value[i]
```

For flat order tuples like `(4, 2, 1)`, the optimal implementation would generate tree-structured code:

```triton
# Optimal: Tree reduction for (4, 2, 1)
# Pairs: (e[0]+e[4]), (e[2]+e[6]), (e[1]+e[5]), (e[3]+e[7])
# Then: ((e[0]+e[4])+(e[2]+e[6])), ((e[1]+e[5])+(e[3]+e[7]))
# Final: result
```

The helper functions to compute the tree structure are complete:
- `compute_element_order(8, (4, 2, 1))` → `[0, 4, 2, 6, 1, 5, 3, 7]`
- `generate_tree_sum_expression([0, 4, 2, 6, 1, 5, 3, 7])` → `(((e[0]+e[4])+(e[2]+e[6]))+((e[1]+e[5])+(e[3]+e[7])))`

**Remaining work**: Update `ordered_final_reduction()` in `triton.py` to generate optimized `tl.reshape`-based code for flat order tuples.

## Usage

### Via Reduction.create

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
```

## Files Modified

| File | Changes |
|------|---------|
| `torch/_inductor/ir.py` | Added `ordered`, `reduction_order` fields; updated create methods |
| `torch/_inductor/scheduler.py` | Added `has_ordered_reduction()`; fusion constraints |
| `torch/_inductor/codegen/triton.py` | Added `ordered_final_reduction()` |
| `torch/_inductor/runtime/triton_helpers.py` | Added ordered reduction functions and helpers |
| `torch/_inductor/config.py` | Added `ordered_reduction_chunk_size` |
| `test/inductor/test_ordered_reduction.py` | New test file with 15 tests |

## Running Tests

```bash
python test/inductor/test_ordered_reduction.py
```

All 15 tests pass as of the current implementation.

## References

- PR 167498: Original design for consecutive linear addition / ordered reduction
- The order tuple encodes element visit order via bit-interleaving with strides
