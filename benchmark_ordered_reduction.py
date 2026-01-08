"""
Benchmark: Ordered Reduction vs Regular Reduction

Compares performance of inductor_prims.ordered_sum (with deterministic ordering)
versus standard torch.sum across various tensor sizes.

Usage:
    python benchmark_ordered_reduction.py
"""

import torch
from torch._inductor import inductor_prims


def benchmark_cuda(fn, x, num_iterations=1000):
    """Benchmark using CUDA events for accurate GPU timing."""
    for _ in range(50):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iterations):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iterations


def make_regular_sum():
    @torch.compile
    def fn(x):
        return x.sum(dim=1)
    return fn


def make_ordered_sum(order, grouping):
    @torch.compile
    def fn(x):
        return inductor_prims.ordered_sum(x, dim=1, order=order, grouping=grouping)
    return fn


def main():
    print("=" * 90)
    print("Benchmark: Ordered Reduction vs Regular Reduction")
    print("=" * 90)

    # Main comparison: Regular vs Ordered (flat tree)
    configs = [
        (1_000_000, 8, [4, 2, 1]),
        (1_000_000, 16, [8, 4, 2, 1]),
        (1_000_000, 32, [16, 8, 4, 2, 1]),
        (1_000_000, 64, [32, 16, 8, 4, 2, 1]),
        (1_000_000, 128, [64, 32, 16, 8, 4, 2, 1]),
        (1_000_000, 256, [128, 64, 32, 16, 8, 4, 2, 1]),
        (2_000_000, 64, [32, 16, 8, 4, 2, 1]),
        (4_000_000, 64, [32, 16, 8, 4, 2, 1]),
    ]

    print(
        f"\n{'Config':<18} {'MB':<10} {'Regular ms':<12} {'Ordered ms':<12} "
        f"{'Ratio':<8} {'Reg GB/s':<10} {'Ord GB/s':<10}"
    )
    print("-" * 90)

    for batch_size, red_size, order in configs:
        torch._dynamo.reset()
        x = torch.randn(batch_size, red_size, device="cuda", dtype=torch.float32)
        total_bytes = batch_size * red_size * 4

        reg_fn = make_regular_sum()
        ord_fn = make_ordered_sum(order, [])

        # Verify correctness
        reg_result = reg_fn(x)
        ord_result = ord_fn(x)
        assert torch.allclose(reg_result, ord_result, rtol=1e-3, atol=1e-3)

        reg_time = benchmark_cuda(reg_fn, x)
        ord_time = benchmark_cuda(ord_fn, x)

        ratio = ord_time / reg_time
        reg_gbps = (total_bytes / 1e9) / (reg_time / 1000)
        ord_gbps = (total_bytes / 1e9) / (ord_time / 1000)

        print(
            f"{batch_size}x{red_size:<10} {total_bytes/1e6:<10.1f} {reg_time:<12.4f} "
            f"{ord_time:<12.4f} {ratio:<8.2f}x {reg_gbps:<10.1f} {ord_gbps:<10.1f}"
        )

        del x
        torch.cuda.empty_cache()

    # Nested vs Flat comparison
    print(f"\n{'='*90}")
    print("Flat Order (Tree) vs Nested Order (Hierarchical) - 8 elements")
    print("=" * 90)

    nested_configs = [
        (2_000_000, 8, [4, 2, 1], [], "flat (4,2,1)"),
        (2_000_000, 8, [2, 1, 4], [2, 1], "nested ((2,1),4)"),
    ]

    print(f"\n{'Config':<18} {'Order':<25} {'Time ms':<12} {'GB/s':<10}")
    print("-" * 70)

    for batch_size, red_size, order, grouping, name in nested_configs:
        torch._dynamo.reset()
        x = torch.randn(batch_size, red_size, device="cuda", dtype=torch.float32)
        total_bytes = batch_size * red_size * 4

        fn = make_ordered_sum(order, grouping)
        result = fn(x)
        expected = x.sum(dim=1)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)

        time_ms = benchmark_cuda(fn, x)
        gbps = (total_bytes / 1e9) / (time_ms / 1000)

        print(f"{batch_size}x{red_size:<10} {name:<25} {time_ms:<12.4f} {gbps:<10.1f}")

        del x
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
