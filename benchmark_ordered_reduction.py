"""
Benchmark: Ordered Reduction vs Regular Reduction

Compares performance of inductor_prims.ordered_sum (with deterministic ordering)
versus standard torch.sum across various tensor sizes, data types, and configurations.

Usage:
    python benchmark_ordered_reduction.py
"""

import torch
import numpy as np
from torch._inductor import inductor_prims


def benchmark_cuda(fn, x, num_iterations=1000, num_warmup=50):
    """Benchmark using CUDA events for accurate GPU timing with statistics."""
    # Warmup
    for _ in range(num_warmup):
        fn(x)
    torch.cuda.synchronize()

    # Collect individual timings for statistical analysis
    timings = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings = np.array(timings)
    return {
        "mean": np.mean(timings),
        "std": np.std(timings),
        "min": np.min(timings),
        "max": np.max(timings),
        "median": np.median(timings),
    }


def verify_reproducibility(fn, x, num_runs=10):
    """Verify that ordered reduction produces identical results across runs."""
    results = []
    for _ in range(num_runs):
        results.append(fn(x).clone())

    # Check all results are bitwise identical
    reference = results[0]
    all_identical = all(torch.equal(r, reference) for r in results[1:])
    return all_identical, results


def get_order_for_size(red_size):
    """Generate flat tree order for a given reduction size (must be power of 2)."""
    order = []
    s = red_size // 2
    while s >= 1:
        order.append(s)
        s //= 2
    return order


def make_regular_sum(dim=1):
    @torch.compile
    def fn(x):
        return x.sum(dim=dim)
    return fn


def make_ordered_sum(order, grouping, dim=1):
    @torch.compile
    def fn(x):
        return inductor_prims.ordered_sum(x, dim=dim, order=order, grouping=grouping)
    return fn


def bytes_per_element(dtype):
    """Return bytes per element for a given dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    return 4


def format_bandwidth(gbps):
    """Format bandwidth with appropriate precision."""
    if gbps >= 1000:
        return f"{gbps:.0f}"
    elif gbps >= 100:
        return f"{gbps:.1f}"
    else:
        return f"{gbps:.2f}"


def run_main_benchmark():
    """Main comparison: Regular vs Ordered across sizes and dtypes."""
    print("=" * 120)
    print("SECTION 1: Regular Sum vs Ordered Sum (Flat Tree Order)")
    print("=" * 120)

    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    dtype_names = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}

    # Reduction sizes - extended to 8192 (persistent reduction is forced for ordered)
    reduction_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for dtype in dtypes:
        print(f"\n{'─' * 120}")
        print(f"Data Type: {dtype_names[dtype]}")
        print("─" * 120)
        print(
            f"{'Batch':<12} {'RedSize':<8} {'MB':<8} "
            f"{'Reg Mean':<10} {'Reg Std':<9} "
            f"{'Ord Mean':<10} {'Ord Std':<9} "
            f"{'Ratio':<7} "
            f"{'Reg In':<8} {'Ord In':<8} "
            f"{'Reg Out':<8} {'Ord Out':<8}"
        )
        print(
            f"{'':12} {'':8} {'':8} "
            f"{'(ms)':10} {'(ms)':9} "
            f"{'(ms)':10} {'(ms)':9} "
            f"{'':7} "
            f"{'GB/s':8} {'GB/s':8} "
            f"{'GB/s':8} {'GB/s':8}"
        )
        print("─" * 120)

        batch_size = 1_000_000
        elem_bytes = bytes_per_element(dtype)

        for red_size in reduction_sizes:
            torch._dynamo.reset()

            order = get_order_for_size(red_size)
            x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)

            input_bytes = batch_size * red_size * elem_bytes
            output_bytes = batch_size * elem_bytes

            reg_fn = make_regular_sum()
            ord_fn = make_ordered_sum(order, [])

            # Verify correctness
            reg_result = reg_fn(x)
            ord_result = ord_fn(x)
            rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
            atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
            assert torch.allclose(reg_result, ord_result, rtol=rtol, atol=atol), \
                f"Results differ for {dtype} with red_size={red_size}"

            # Benchmark with statistics
            reg_stats = benchmark_cuda(reg_fn, x, num_iterations=500)
            ord_stats = benchmark_cuda(ord_fn, x, num_iterations=500)

            ratio = ord_stats["mean"] / reg_stats["mean"]

            # Input bandwidth (reading input tensor)
            reg_in_gbps = (input_bytes / 1e9) / (reg_stats["mean"] / 1000)
            ord_in_gbps = (input_bytes / 1e9) / (ord_stats["mean"] / 1000)

            # Output bandwidth (writing output tensor)
            reg_out_gbps = (output_bytes / 1e9) / (reg_stats["mean"] / 1000)
            ord_out_gbps = (output_bytes / 1e9) / (ord_stats["mean"] / 1000)

            print(
                f"{batch_size:<12} {red_size:<8} {input_bytes/1e6:<8.1f} "
                f"{reg_stats['mean']:<10.4f} {reg_stats['std']:<9.4f} "
                f"{ord_stats['mean']:<10.4f} {ord_stats['std']:<9.4f} "
                f"{ratio:<7.2f}x "
                f"{format_bandwidth(reg_in_gbps):<8} {format_bandwidth(ord_in_gbps):<8} "
                f"{format_bandwidth(reg_out_gbps):<8} {format_bandwidth(ord_out_gbps):<8}"
            )

            del x, reg_result, ord_result
            torch.cuda.empty_cache()


def run_reproducibility_benchmark():
    """Test reproducibility of ordered vs regular reductions."""
    print("\n" + "=" * 120)
    print("SECTION 2: Reproducibility Verification")
    print("=" * 120)
    print("\nVerifying that ordered reductions produce bitwise-identical results across runs.")
    print("Regular reductions may vary due to non-deterministic floating-point accumulation order.\n")

    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    dtype_names = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}

    test_configs = [
        (100_000, 64),
        (100_000, 256),
        (100_000, 1024),
        (100_000, 4096),
        (100_000, 8192),
    ]

    print(f"{'Dtype':<8} {'Config':<16} {'Regular':<20} {'Ordered':<20} {'Reg Variance':<15} {'Ord Variance':<15}")
    print("─" * 100)

    num_repro_runs = 20

    for dtype in dtypes:
        for batch_size, red_size in test_configs:
            torch._dynamo.reset()

            order = get_order_for_size(red_size)
            x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)

            reg_fn = make_regular_sum()
            ord_fn = make_ordered_sum(order, [])

            # Warmup
            for _ in range(10):
                reg_fn(x)
                ord_fn(x)
            torch.cuda.synchronize()

            # Test reproducibility
            reg_reproducible, reg_results = verify_reproducibility(reg_fn, x, num_repro_runs)
            ord_reproducible, ord_results = verify_reproducibility(ord_fn, x, num_repro_runs)

            # Compute variance in results (convert to float64 for accurate variance)
            reg_stacked = torch.stack([r.to(torch.float64) for r in reg_results])
            ord_stacked = torch.stack([r.to(torch.float64) for r in ord_results])

            reg_var = reg_stacked.var(dim=0).mean().item()
            ord_var = ord_stacked.var(dim=0).mean().item()

            reg_status = "REPRODUCIBLE" if reg_reproducible else "VARIES"
            ord_status = "REPRODUCIBLE" if ord_reproducible else "VARIES"

            print(
                f"{dtype_names[dtype]:<8} {batch_size}x{red_size:<8} "
                f"{reg_status:<20} {ord_status:<20} "
                f"{reg_var:<15.2e} {ord_var:<15.2e}"
            )

            del x, reg_results, ord_results
            torch.cuda.empty_cache()


def run_variance_analysis():
    """Analyze timing variance for both methods."""
    print("\n" + "=" * 120)
    print("SECTION 3: Timing Variance Analysis")
    print("=" * 120)
    print("\nStatistical analysis of execution time stability (lower variance = more predictable performance).\n")

    dtype = torch.float32
    batch_size = 1_000_000

    reduction_sizes = [64, 256, 1024, 4096, 8192]

    print(f"{'RedSize':<10} {'Method':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'CV (%)':<10}")
    print("─" * 80)

    for red_size in reduction_sizes:
        torch._dynamo.reset()

        order = get_order_for_size(red_size)
        x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)

        reg_fn = make_regular_sum()
        ord_fn = make_ordered_sum(order, [])

        # Warmup
        for _ in range(50):
            reg_fn(x)
            ord_fn(x)
        torch.cuda.synchronize()

        reg_stats = benchmark_cuda(reg_fn, x, num_iterations=1000)
        ord_stats = benchmark_cuda(ord_fn, x, num_iterations=1000)

        # Coefficient of variation (CV) = std/mean * 100%
        reg_cv = (reg_stats["std"] / reg_stats["mean"]) * 100
        ord_cv = (ord_stats["std"] / ord_stats["mean"]) * 100

        print(
            f"{red_size:<10} {'Regular':<12} {reg_stats['mean']:<12.4f} {reg_stats['std']:<12.4f} "
            f"{reg_stats['min']:<12.4f} {reg_stats['max']:<12.4f} {reg_cv:<10.2f}"
        )
        print(
            f"{'':<10} {'Ordered':<12} {ord_stats['mean']:<12.4f} {ord_stats['std']:<12.4f} "
            f"{ord_stats['min']:<12.4f} {ord_stats['max']:<12.4f} {ord_cv:<10.2f}"
        )
        print()

        del x
        torch.cuda.empty_cache()


def run_nested_order_benchmark():
    """Benchmark various flat tree order configurations."""
    print("=" * 120)
    print("SECTION 4: Flat Tree Order Configurations")
    print("=" * 120)
    print("\nBenchmarking ordered reductions with different reduction sizes.\n")

    dtype = torch.float32
    batch_size = 2_000_000
    elem_bytes = bytes_per_element(dtype)

    # Flat order configurations only (nested orders have known codegen issues)
    nested_configs = [
        # 64 elements
        (64, [32, 16, 8, 4, 2, 1], [], "flat (32,16,8,4,2,1)"),
        # 256 elements
        (256, [128, 64, 32, 16, 8, 4, 2, 1], [], "flat (128,64,...)"),
        # 1024 elements
        (1024, get_order_for_size(1024), [], "flat (512,256,...)"),
        # 4096 elements
        (4096, get_order_for_size(4096), [], "flat (2048,1024,...)"),
        # 8192 elements
        (8192, get_order_for_size(8192), [], "flat (4096,2048,...)"),
    ]

    print(f"{'RedSize':<10} {'Order':<30} {'Mean (ms)':<12} {'Std (ms)':<10} {'In GB/s':<10} {'Out GB/s':<10}")
    print("─" * 90)

    current_size = None
    for red_size, order, grouping, name in nested_configs:
        if current_size != red_size:
            if current_size is not None:
                print()
            current_size = red_size

        torch._dynamo.reset()
        x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)

        input_bytes = batch_size * red_size * elem_bytes
        output_bytes = batch_size * elem_bytes

        fn = make_ordered_sum(order, grouping)

        # Verify correctness
        result = fn(x)
        expected = x.sum(dim=1)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)

        stats = benchmark_cuda(fn, x, num_iterations=500)

        in_gbps = (input_bytes / 1e9) / (stats["mean"] / 1000)
        out_gbps = (output_bytes / 1e9) / (stats["mean"] / 1000)

        print(
            f"{red_size:<10} {name:<30} {stats['mean']:<12.4f} {stats['std']:<10.4f} "
            f"{format_bandwidth(in_gbps):<10} {format_bandwidth(out_gbps):<10}"
        )

        del x, result, expected
        torch.cuda.empty_cache()


def run_scaling_benchmark():
    """Test scaling behavior with different batch sizes."""
    print("\n" + "=" * 120)
    print("SECTION 5: Batch Size Scaling")
    print("=" * 120)
    print("\nHow performance scales with batch size for a fixed reduction size.\n")

    dtype = torch.float32
    red_size = 1024
    order = get_order_for_size(red_size)
    elem_bytes = bytes_per_element(dtype)

    batch_sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000, 4_000_000]

    print(f"{'Batch':<12} {'MB':<10} {'Reg Mean':<12} {'Ord Mean':<12} {'Ratio':<8} {'Reg In GB/s':<12} {'Ord In GB/s':<12}")
    print("─" * 90)

    for batch_size in batch_sizes:
        torch._dynamo.reset()

        x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)
        input_bytes = batch_size * red_size * elem_bytes

        reg_fn = make_regular_sum()
        ord_fn = make_ordered_sum(order, [])

        reg_stats = benchmark_cuda(reg_fn, x, num_iterations=500)
        ord_stats = benchmark_cuda(ord_fn, x, num_iterations=500)

        ratio = ord_stats["mean"] / reg_stats["mean"]
        reg_gbps = (input_bytes / 1e9) / (reg_stats["mean"] / 1000)
        ord_gbps = (input_bytes / 1e9) / (ord_stats["mean"] / 1000)

        print(
            f"{batch_size:<12} {input_bytes/1e6:<10.1f} {reg_stats['mean']:<12.4f} "
            f"{ord_stats['mean']:<12.4f} {ratio:<8.2f}x {reg_gbps:<12.1f} {ord_gbps:<12.1f}"
        )

        del x
        torch.cuda.empty_cache()


def run_split_vs_persistent_benchmark():
    """Compare split vs persistent ordered reductions for large sizes."""
    from torch._inductor import config

    print("\n" + "=" * 120)
    print("SECTION 6: Split vs Persistent Ordered Reductions")
    print("=" * 120)
    print("\nComparing performance of split (multi-kernel) vs persistent (single-kernel) ordered reductions.\n")

    dtype = torch.float32
    batch_size = 500_000
    elem_bytes = bytes_per_element(dtype)

    # Test larger reduction sizes that benefit from splitting
    reduction_sizes = [2048, 4096, 8192]

    print(f"{'RedSize':<10} {'Persistent':<14} {'Split':<14} {'Speedup':<10} {'Bitwise OK':<12}")
    print("─" * 70)

    for red_size in reduction_sizes:
        order = get_order_for_size(red_size)
        x = torch.randn(batch_size, red_size, device="cuda", dtype=dtype)

        # Persistent (no split)
        with config.patch({"split_reductions": False}):
            torch._dynamo.reset()
            persistent_fn = make_ordered_sum(order, [])
            persistent_stats = benchmark_cuda(persistent_fn, x, num_iterations=200)
            persistent_result = persistent_fn(x)

        # Split (multi-kernel)
        with config.patch({"split_reductions": True}):
            torch._dynamo.reset()
            split_fn = make_ordered_sum(order, [])
            split_stats = benchmark_cuda(split_fn, x, num_iterations=200)
            split_result = split_fn(x)

        # Check bitwise identical
        bitwise_ok = torch.equal(persistent_result, split_result)

        speedup = persistent_stats["mean"] / split_stats["mean"]

        print(
            f"{red_size:<10} {persistent_stats['mean']:<14.4f} {split_stats['mean']:<14.4f} "
            f"{speedup:<10.2f}x {str(bitwise_ok):<12}"
        )

        del x, persistent_result, split_result
        torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 120)
    print(" " * 30 + "ORDERED REDUCTION BENCHMARK SUITE")
    print("=" * 120)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    run_main_benchmark()
    run_reproducibility_benchmark()
    run_variance_analysis()
    run_nested_order_benchmark()
    run_scaling_benchmark()
    run_split_vs_persistent_benchmark()

    print("\n" + "=" * 120)
    print("Benchmark complete!")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
