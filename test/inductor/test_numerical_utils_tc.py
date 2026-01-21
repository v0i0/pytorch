# Owner(s): ["module: inductor"]
"""
Test tensor core emulation for bit-exact matching against hardware captures.

Test vectors come from the MATLAB reference repository containing actual
inputs and outputs captured from real NVIDIA GPUs. Each test vector is a
dot product: d = a @ b + c

Test vectors location: Set TENSOR_CORE_TEST_DATA_DIR environment variable.

Random GPU tests automatically detect the current GPU and run random tests
comparing emulation against actual GPU computation.
"""

import os
import struct
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
from torch._inductor._numerical_utils_tc import (
    A100TC,
    B200TC,
    bits_to_float32,
    float32_to_bits,
    get_fp_format,
    H100TC,
    H200TC,
    round_to_format,
    V100TC,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


TENSOR_CORE_TEST_DATA_DIR_ENV = "TENSOR_CORE_TEST_DATA_DIR"


def get_validation_dir() -> Optional[Path]:
    """Get the validation data directory from environment variable."""
    env_path = os.environ.get(TENSOR_CORE_TEST_DATA_DIR_ENV)
    if env_path:
        path = Path(env_path)
        if path.is_dir():
            return path
    return None


VALIDATION_DIR = get_validation_dir()


def hex_to_float32(hex_str: str) -> float:
    """Convert hex string (e.g., '3f800000') to float32."""
    bits = int(hex_str, 16)
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def binary_to_float32(bin_str: str) -> float:
    """Convert 32-bit binary string to float32."""
    bits = int(bin_str, 2)
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def float32_to_bits_scalar(x: float) -> int:
    """Convert float32 to its bit representation."""
    return struct.unpack(">I", struct.pack(">f", x))[0]


def read_hex_float_file(filepath: Path) -> list[list[float]]:
    """Read file with hex floats, return list of rows."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hex_vals = line.split()
            floats = [hex_to_float32(h) for h in hex_vals]
            rows.append(floats)
    return rows


def read_binary_float_file(filepath: Path) -> list[float]:
    """Read file with binary IEEE 754 strings, return list of floats."""
    values = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(binary_to_float32(line))
    return values


class HardwareTestConfig:
    """Configuration for a hardware validation test."""

    def __init__(
        self,
        gpu_name: str,
        input_format: str,
        output_format: str,
        k_size: int,
        tc_func: Callable,
        informat_arg: str,
        outformat_arg: str,
    ):
        self.gpu_name = gpu_name
        self.input_format = input_format
        self.output_format = output_format
        self.k_size = k_size
        self.tc_func = tc_func
        self.informat_arg = informat_arg
        self.outformat_arg = outformat_arg


TEST_CONFIGS: dict[str, HardwareTestConfig] = {
    "H100_e5m2_fp32": HardwareTestConfig(
        "H100", "E5M2", "fp32", 32, H100TC, "fp8-e5m2", "binary32"
    ),
    "H100_e4m3_fp32": HardwareTestConfig(
        "H100", "E4M3", "fp32", 32, H100TC, "fp8-e4m3", "binary32"
    ),
    "B200_e5m2_fp32": HardwareTestConfig(
        "B200", "E5M2", "fp32", 32, B200TC, "fp8-e5m2", "binary32"
    ),
    "B200_e4m3_fp32": HardwareTestConfig(
        "B200", "E4M3", "fp32", 32, B200TC, "fp8-e4m3", "binary32"
    ),
    "V100_fp16_fp32": HardwareTestConfig(
        "V100", "fp16", "fp32", 4, V100TC, "binary16", "binary32"
    ),
    "V100_fp16_fp16": HardwareTestConfig(
        "V100", "fp16", "fp16", 4, V100TC, "binary16", "binary16"
    ),
    "A100_fp16_fp32": HardwareTestConfig(
        "A100", "fp16", "fp32", 8, A100TC, "binary16", "binary32"
    ),
    "A100_fp16_fp16": HardwareTestConfig(
        "A100", "fp16", "fp16", 8, A100TC, "binary16", "binary16"
    ),
    "A100_bf16_fp32": HardwareTestConfig(
        "A100", "bf16", "fp32", 8, A100TC, "bfloat16", "binary32"
    ),
    "A100_tf32_fp32": HardwareTestConfig(
        "A100", "tf32", "fp32", 4, A100TC, "tf32", "binary32"
    ),
    "H100_fp16_fp32": HardwareTestConfig(
        "H100", "fp16", "fp32", 16, H100TC, "binary16", "binary32"
    ),
    "H100_bf16_fp32": HardwareTestConfig(
        "H100", "bf16", "fp32", 16, H100TC, "bfloat16", "binary32"
    ),
    "H100_tf32_fp32": HardwareTestConfig(
        "H100", "tf32", "fp32", 8, H100TC, "tf32", "binary32"
    ),
    "B200_fp16_fp32": HardwareTestConfig(
        "B200", "fp16", "fp32", 16, B200TC, "binary16", "binary32"
    ),
    "B200_bf16_fp32": HardwareTestConfig(
        "B200", "bf16", "fp32", 16, B200TC, "bfloat16", "binary32"
    ),
    "B200_tf32_fp32": HardwareTestConfig(
        "B200", "tf32", "fp32", 8, B200TC, "tf32", "binary32"
    ),
    "H200_fp16_fp32": HardwareTestConfig(
        "H200", "fp16", "fp32", 16, H200TC, "binary16", "binary32"
    ),
}


def get_data_paths(config: HardwareTestConfig) -> tuple[Optional[Path], ...]:
    """Get paths to test data files for a configuration."""
    if VALIDATION_DIR is None:
        return None, None, None, None

    base_dir = VALIDATION_DIR / config.gpu_name / config.input_format
    in_suffix = config.input_format
    out_suffix = config.output_format

    a_path = base_dir / f"a_{config.gpu_name}_{in_suffix}.txt"
    b_path = base_dir / f"b_{config.gpu_name}_{in_suffix}.txt"
    c_path = base_dir / f"c_{config.gpu_name}_{out_suffix}.txt"
    d_path = base_dir / f"d_{config.gpu_name}_{out_suffix}.txt"

    if not all(p.exists() for p in [a_path, b_path, c_path, d_path]):
        return None, None, None, None

    return a_path, b_path, c_path, d_path


def run_validation_test(
    config: HardwareTestConfig, max_tests: Optional[int] = None
) -> tuple[int, int, list[int]]:
    """Run validation test for a GPU configuration."""
    a_path, b_path, c_path, d_path = get_data_paths(config)

    if a_path is None:
        return 0, 0, []

    a_data = read_hex_float_file(a_path)
    b_data = read_hex_float_file(b_path)
    d_expected = read_binary_float_file(d_path)

    is_fp8 = config.input_format in ("E5M2", "E4M3")
    is_h100_h200 = config.gpu_name in ("H100", "H200")
    if is_fp8 and is_h100_h200:
        c_data = [0.0] * len(a_data)
    else:
        c_data = read_binary_float_file(c_path)

    total = len(a_data)
    if max_tests is not None:
        total = min(total, max_tests)

    passed = 0
    failed_indices = []

    for i in range(total):
        a_row = a_data[i]
        b_row = b_data[i]
        c_val = c_data[i]
        d_exp = d_expected[i]

        a_tensor = torch.tensor([a_row], dtype=torch.float32)
        b_tensor = torch.tensor([[x] for x in b_row], dtype=torch.float32)
        c_tensor = torch.tensor([[c_val]], dtype=torch.float32)

        if config.tc_func == V100TC:
            result = config.tc_func(
                1.0, a_tensor, b_tensor, 1.0, c_tensor, config.outformat_arg
            )
        else:
            result = config.tc_func(
                1.0,
                a_tensor,
                b_tensor,
                1.0,
                c_tensor,
                config.informat_arg,
                config.outformat_arg,
            )

        d_actual = result[0, 0].item()
        actual_bits = float32_to_bits_scalar(d_actual)
        expected_bits = float32_to_bits_scalar(d_exp)

        if actual_bits == expected_bits:
            passed += 1
        else:
            failed_indices.append(i)

    return passed, total, failed_indices


def get_current_gpu_model() -> Optional[str]:
    """Detect the current GPU and return model name if supported."""
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_name(0).lower()

    if "v100" in gpu_name:
        return "V100"
    elif "a100" in gpu_name:
        return "A100"
    elif "h100" in gpu_name:
        return "H100"
    elif "h200" in gpu_name:
        return "H200"
    elif "b200" in gpu_name or "b100" in gpu_name or "blackwell" in gpu_name:
        return "B200"
    return None


def get_gpu_supported_formats(gpu_model: str) -> list[tuple[str, str, Callable]]:
    """Get the supported input/output format combinations for a GPU model."""
    if gpu_model == "V100":
        return [
            ("binary16", "binary32", V100TC),
            ("binary16", "binary16", V100TC),
        ]
    elif gpu_model == "A100":
        return [
            ("binary16", "binary32", A100TC),
            ("binary16", "binary16", A100TC),
            ("bfloat16", "binary32", A100TC),
            ("tf32", "binary32", A100TC),
        ]
    elif gpu_model in ("H100", "H200"):
        tc_func = H100TC if gpu_model == "H100" else H200TC
        return [
            ("binary16", "binary32", tc_func),
            ("bfloat16", "binary32", tc_func),
            ("tf32", "binary32", tc_func),
            ("fp8-e5m2", "binary32", tc_func),
            ("fp8-e4m3", "binary32", tc_func),
        ]
    elif gpu_model == "B200":
        return [
            ("binary16", "binary32", B200TC),
            ("bfloat16", "binary32", B200TC),
            ("tf32", "binary32", B200TC),
            ("fp8-e5m2", "binary32", B200TC),
            ("fp8-e4m3", "binary32", B200TC),
        ]
    return []


def run_gpu_matmul(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, informat: str, outformat: str
) -> torch.Tensor:
    """Run matrix multiplication on GPU with appropriate dtype conversions."""
    device = torch.device("cuda")

    if informat in ("binary16", "fp16", "half"):
        A_gpu = A.to(device=device, dtype=torch.float16)
        B_gpu = B.to(device=device, dtype=torch.float16)
    elif informat in ("bfloat16", "bf16"):
        A_gpu = A.to(device=device, dtype=torch.bfloat16)
        B_gpu = B.to(device=device, dtype=torch.bfloat16)
    elif informat in ("tf32", "tensorfloat32"):
        A_gpu = A.to(device=device, dtype=torch.float32)
        B_gpu = B.to(device=device, dtype=torch.float32)
    elif informat in ("fp8-e5m2", "e5m2", "fp8-e4m3", "e4m3"):
        if informat in ("fp8-e5m2", "e5m2") and hasattr(torch, "float8_e5m2"):
            A_fp8 = A.to(device=device, dtype=torch.float8_e5m2)
            B_fp8 = B.to(device=device, dtype=torch.float8_e5m2)
            A_gpu = A_fp8.to(torch.float16)
            B_gpu = B_fp8.to(torch.float16)
        elif informat in ("fp8-e4m3", "e4m3") and hasattr(torch, "float8_e4m3fn"):
            A_fp8 = A.to(device=device, dtype=torch.float8_e4m3fn)
            B_fp8 = B.to(device=device, dtype=torch.float8_e4m3fn)
            A_gpu = A_fp8.to(torch.float16)
            B_gpu = B_fp8.to(torch.float16)
        else:
            A_gpu = A.to(device=device, dtype=torch.float16)
            B_gpu = B.to(device=device, dtype=torch.float16)
    else:
        A_gpu = A.to(device=device, dtype=torch.float32)
        B_gpu = B.to(device=device, dtype=torch.float32)

    C_gpu = (
        C.to(device=device, dtype=torch.float32)
        if C is not None
        else torch.zeros(A.shape[0], B.shape[1], device=device, dtype=torch.float32)
    )

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = informat in ("tf32", "tensorfloat32")

    try:
        if A_gpu.dtype in (torch.float16, torch.bfloat16):
            result = torch.matmul(A_gpu, B_gpu).float() + C_gpu
        else:
            result = torch.matmul(A_gpu, B_gpu) + C_gpu
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    return result.cpu().to(torch.float32)


@unittest.skipIf(
    VALIDATION_DIR is None,
    f"Set {TENSOR_CORE_TEST_DATA_DIR_ENV} environment variable",
)
class TestTensorCoreEmulation(TestCase):
    """Test cases for tensor core emulation against hardware captures."""

    def _run_config_test(self, config_name: str):
        config = TEST_CONFIGS.get(config_name)
        self.assertIsNotNone(config, f"Config {config_name} not found")

        paths = get_data_paths(config)
        if paths[0] is None:
            self.skipTest(f"Test data not found for {config_name}")

        passed, total, _ = run_validation_test(config)
        self.assertEqual(passed, total, f"{config_name}: {passed}/{total} bit-exact")

    def test_v100_fp16_fp32(self):
        self._run_config_test("V100_fp16_fp32")

    def test_a100_fp16_fp32(self):
        self._run_config_test("A100_fp16_fp32")

    def test_a100_bf16_fp32(self):
        self._run_config_test("A100_bf16_fp32")

    def test_a100_tf32_fp32(self):
        self._run_config_test("A100_tf32_fp32")

    def test_h100_fp16_fp32(self):
        self._run_config_test("H100_fp16_fp32")

    def test_h100_bf16_fp32(self):
        self._run_config_test("H100_bf16_fp32")

    def test_h100_tf32_fp32(self):
        self._run_config_test("H100_tf32_fp32")

    def test_h100_e5m2_fp32(self):
        self._run_config_test("H100_e5m2_fp32")

    def test_h100_e4m3_fp32(self):
        self._run_config_test("H100_e4m3_fp32")

    def test_b200_fp16_fp32(self):
        self._run_config_test("B200_fp16_fp32")

    def test_b200_bf16_fp32(self):
        self._run_config_test("B200_bf16_fp32")

    def test_b200_tf32_fp32(self):
        self._run_config_test("B200_tf32_fp32")

    def test_b200_e5m2_fp32(self):
        self._run_config_test("B200_e5m2_fp32")

    def test_b200_e4m3_fp32(self):
        self._run_config_test("B200_e4m3_fp32")

    def test_h200_fp16_fp32(self):
        self._run_config_test("H200_fp16_fp32")


class TestTensorCoreUnit(TestCase):
    """Unit tests for tensor core emulation functions."""

    def test_simple_dot_product(self):
        A = torch.tensor([[0.5, 0.25, 0.125, 0.0625]], dtype=torch.float32)
        B = torch.tensor([[0.5], [0.25], [0.125], [0.0625]], dtype=torch.float32)
        result = V100TC(1.0, A, B, 0.0, None, "binary32")
        expected = 0.5 * 0.5 + 0.25 * 0.25 + 0.125 * 0.125 + 0.0625 * 0.0625
        self.assertAlmostEqual(result.item(), expected, places=7)

    def test_with_accumulator(self):
        A = torch.tensor([[0.5, 0.25, 0.125, 0.0625]], dtype=torch.float32)
        B = torch.tensor([[0.5], [0.25], [0.125], [0.0625]], dtype=torch.float32)
        C = torch.tensor([[1.0]], dtype=torch.float32)
        result = V100TC(1.0, A, B, 1.0, C, "binary32")
        expected = 0.5 * 0.5 + 0.25 * 0.25 + 0.125 * 0.125 + 0.0625 * 0.0625 + 1.0
        self.assertAlmostEqual(result.item(), expected, places=7)

    def test_negative_values(self):
        A = torch.tensor([[-0.5, 0.25]], dtype=torch.float32)
        B = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
        result = V100TC(1.0, A, B, 0.0, None, "binary32")
        self.assertAlmostEqual(result.item(), -0.3125, places=7)

    def test_matrix_multiply_2x2(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        result = V100TC(1.0, A, B, 0.0, None, "binary32")
        expected = A @ B
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_fp_format(self):
        fp32 = get_fp_format("binary32")
        self.assertEqual(fp32.exp_bits, 8)
        self.assertEqual(fp32.man_bits, 23)
        self.assertEqual(fp32.bias, 127)

        fp16 = get_fp_format("binary16")
        self.assertEqual(fp16.exp_bits, 5)
        self.assertEqual(fp16.man_bits, 10)
        self.assertEqual(fp16.bias, 15)

        bf16 = get_fp_format("bfloat16")
        self.assertEqual(bf16.exp_bits, 8)
        self.assertEqual(bf16.man_bits, 7)

    def test_round_to_format(self):
        x = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        x_fp16 = round_to_format(x, "binary16")
        self.assertEqual(x_fp16.dtype, torch.float32)
        x_bf16 = round_to_format(x, "bfloat16")
        self.assertEqual(x_bf16.dtype, torch.float32)

    def test_float32_bits_conversion(self):
        x = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
        bits = float32_to_bits(x)
        self.assertEqual(bits[0].item(), 0x3F800000)
        self.assertEqual(bits[1].item() & 0xFFFFFFFF, 0xBF800000)
        self.assertEqual(bits[2].item(), 0x3F000000)
        x_back = bits_to_float32(bits)
        torch.testing.assert_close(x, x_back)


@unittest.skipIf(not HAS_GPU, "No GPU available")
@unittest.skipIf(get_current_gpu_model() is None, "GPU not in supported list")
class TestTensorCoreRandomGPU(TestCase):
    """Random tests comparing emulation against actual GPU computation."""

    NUM_RANDOM_TESTS = 100

    K_SIZES = {
        "V100": {"binary16": 4},
        "A100": {"binary16": 8, "bfloat16": 8, "tf32": 4},
        "H100": {
            "binary16": 16,
            "bfloat16": 16,
            "tf32": 8,
            "fp8-e5m2": 32,
            "fp8-e4m3": 32,
        },
        "H200": {
            "binary16": 16,
            "bfloat16": 16,
            "tf32": 8,
            "fp8-e5m2": 32,
            "fp8-e4m3": 32,
        },
        "B200": {
            "binary16": 16,
            "bfloat16": 16,
            "tf32": 8,
            "fp8-e5m2": 32,
            "fp8-e4m3": 32,
        },
    }

    def setUp(self):
        super().setUp()
        self.gpu_model = get_current_gpu_model()
        self.supported_formats = get_gpu_supported_formats(self.gpu_model)

    def _get_k_size(self, informat: str) -> int:
        gpu_sizes = self.K_SIZES.get(self.gpu_model, {})
        return gpu_sizes.get(informat, 8)

    def _generate_random_inputs(
        self, K: int, informat: str, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)

        if informat in ("fp8-e5m2", "e5m2"):
            scale = 0.5
        elif informat in ("fp8-e4m3", "e4m3"):
            scale = 0.25
        elif informat in ("binary16", "fp16", "half", "bfloat16", "bf16"):
            scale = 2.0
        else:
            scale = 10.0

        A = (torch.rand(1, K) * 2 - 1) * scale
        B = (torch.rand(K, 1) * 2 - 1) * scale
        C = (torch.rand(1, 1) * 2 - 1) * scale
        return A.float(), B.float(), C.float()

    def _run_random_tests(
        self, informat: str, outformat: str, tc_func: Callable
    ) -> tuple[int, int, float, int]:
        K = self._get_k_size(informat)
        num_close = 0
        total = 0
        max_rel_error = 0.0
        num_nan_inf = 0

        if informat in ("fp8-e5m2", "e5m2", "fp8-e4m3", "e4m3"):
            rel_tol = 0.5
        elif informat in ("bfloat16", "bf16"):
            rel_tol = 0.05
        elif informat in ("binary16", "fp16", "half"):
            rel_tol = 0.01
        elif informat in ("tf32", "tensorfloat32"):
            rel_tol = 0.02
        else:
            rel_tol = 0.001

        for seed in range(self.NUM_RANDOM_TESTS):
            A, B, C = self._generate_random_inputs(K, informat, seed)

            if tc_func == V100TC:
                emu_result = tc_func(1.0, A, B, 1.0, C, outformat)
            else:
                emu_result = tc_func(1.0, A, B, 1.0, C, informat, outformat)

            gpu_result = run_gpu_matmul(A, B, C, informat, outformat)

            emu_val = emu_result[0, 0].item()
            gpu_val = gpu_result[0, 0].item()

            if not (
                torch.isfinite(emu_result).all() and torch.isfinite(gpu_result).all()
            ):
                num_nan_inf += 1
                total += 1
                continue

            if abs(gpu_val) > 1e-10:
                rel_err = abs(emu_val - gpu_val) / abs(gpu_val)
            else:
                rel_err = abs(emu_val - gpu_val)

            max_rel_error = max(max_rel_error, rel_err)
            if rel_err <= rel_tol:
                num_close += 1
            total += 1

        return num_close, total, max_rel_error, num_nan_inf

    def test_random_fp16(self):
        formats = [
            (inf, outf, fn)
            for inf, outf, fn in self.supported_formats
            if inf in ("binary16", "fp16", "half")
        ]
        if not formats:
            self.skipTest(f"FP16 not supported on {self.gpu_model}")

        informat, outformat, tc_func = formats[0]
        num_close, total, max_err, num_nan = self._run_random_tests(
            informat, outformat, tc_func
        )
        accuracy = num_close / total if total > 0 else 0
        self.assertGreater(
            accuracy,
            0.9,
            f"FP16: {num_close}/{total} within tolerance ({accuracy:.1%})",
        )
        self.assertEqual(num_nan, 0, f"FP16: {num_nan} NaN/Inf results")

    def test_random_bf16(self):
        formats = [
            (inf, outf, fn)
            for inf, outf, fn in self.supported_formats
            if inf in ("bfloat16", "bf16")
        ]
        if not formats:
            self.skipTest(f"BF16 not supported on {self.gpu_model}")

        informat, outformat, tc_func = formats[0]
        num_close, total, max_err, num_nan = self._run_random_tests(
            informat, outformat, tc_func
        )
        accuracy = num_close / total if total > 0 else 0
        self.assertGreater(
            accuracy,
            0.9,
            f"BF16: {num_close}/{total} within tolerance ({accuracy:.1%})",
        )
        self.assertEqual(num_nan, 0, f"BF16: {num_nan} NaN/Inf results")

    def test_random_tf32(self):
        formats = [
            (inf, outf, fn)
            for inf, outf, fn in self.supported_formats
            if inf in ("tf32", "tensorfloat32")
        ]
        if not formats:
            self.skipTest(f"TF32 not supported on {self.gpu_model}")

        informat, outformat, tc_func = formats[0]
        num_close, total, max_err, num_nan = self._run_random_tests(
            informat, outformat, tc_func
        )
        accuracy = num_close / total if total > 0 else 0
        self.assertGreater(
            accuracy,
            0.9,
            f"TF32: {num_close}/{total} within tolerance ({accuracy:.1%})",
        )
        self.assertEqual(num_nan, 0, f"TF32: {num_nan} NaN/Inf results")

    def test_random_fp8_e5m2(self):
        formats = [
            (inf, outf, fn)
            for inf, outf, fn in self.supported_formats
            if inf in ("fp8-e5m2", "e5m2")
        ]
        if not formats:
            self.skipTest(f"FP8-E5M2 not supported on {self.gpu_model}")
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("PyTorch does not support FP8 types")

        informat, outformat, tc_func = formats[0]
        num_close, total, max_err, num_nan = self._run_random_tests(
            informat, outformat, tc_func
        )
        accuracy = num_close / total if total > 0 else 0
        self.assertGreater(
            accuracy,
            0.7,
            f"FP8-E5M2: {num_close}/{total} within tolerance ({accuracy:.1%})",
        )

    def test_random_fp8_e4m3(self):
        formats = [
            (inf, outf, fn)
            for inf, outf, fn in self.supported_formats
            if inf in ("fp8-e4m3", "e4m3")
        ]
        if not formats:
            self.skipTest(f"FP8-E4M3 not supported on {self.gpu_model}")
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("PyTorch does not support FP8 types")

        informat, outformat, tc_func = formats[0]
        num_close, total, max_err, num_nan = self._run_random_tests(
            informat, outformat, tc_func
        )
        accuracy = num_close / total if total > 0 else 0
        self.assertGreater(
            accuracy,
            0.7,
            f"FP8-E4M3: {num_close}/{total} within tolerance ({accuracy:.1%})",
        )


if __name__ == "__main__":
    run_tests()
