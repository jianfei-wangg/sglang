#!/usr/bin/env python3

import time

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil


# Signed FP4 E2M1 lookup table: maps 4-bit FP4 code (0-15) directly to float value.
# Indices 0-7 = positive values, indices 8-15 = negative values (sign bit = bit 3).
# This eliminates separate sign extraction + 7 nested tl.where + conditional negate.
FP4_SIGNED_LUT = None  # lazily initialized on first use


def _get_fp4_lut():
    global FP4_SIGNED_LUT
    if FP4_SIGNED_LUT is None:
        FP4_SIGNED_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device="cuda",
        )
    return FP4_SIGNED_LUT


@triton.jit
def fp4_dequant_triton_kernel(
    quant_ptr,   # [B, H, D_HALF]  packed uint8
    scale_ptr,   # [B, H*D/16]     uint8 scale exponents
    output_ptr,  # [B, H, D]       bf16 output
    lut_ptr,     # [16]            float32 signed FP4 LUT
    D_HALF: tl.constexpr,
    D_OVER_16: tl.constexpr,      # D // 16, scale groups per head
    D_HALF_OVER_16: tl.constexpr, # D_HALF // 16, lower/upper scale offset
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for FP4 dequantization.
    Grid: (B, H, num_blocks) — uses native 3D grid to avoid runtime div/mod.
    Optimizations vs baseline:
      1. Signed FP4 LUT: 1 gather replaces 7 nested tl.where + sign handling (×2)
      2. Scale index decomposition: bit-shift instead of large multiply + divide
      3. Correct num_warps matching BLOCK_SIZE
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    offset = tl.arange(0, BLOCK_SIZE)
    elem_offset = block_idx * BLOCK_SIZE + offset
    data_mask = elem_offset < D_HALF

    # --- Load packed uint8 values from [B, H, D_HALF] (contiguous) ---
    D: tl.constexpr = D_HALF * 2
    H = tl.num_programs(1)  # grid dim 1 = H
    quant_idx = batch_idx * (H * D_HALF) + head_idx * D_HALF + elem_offset
    packed_vals = tl.load(quant_ptr + quant_idx, mask=data_mask, other=0)

    # --- LUT-based dequant: 1 gather replaces 7× tl.where + sign extraction ---
    lower_nibble = (packed_vals & 0x0F).to(tl.int32)
    upper_nibble = ((packed_vals >> 4) & 0x0F).to(tl.int32)
    lower_float_vals = tl.load(lut_ptr + lower_nibble, mask=data_mask, other=0.0)
    upper_float_vals = tl.load(lut_ptr + upper_nibble, mask=data_mask, other=0.0)

    # --- Scale factor indices (decomposed to avoid large runtime multiply) ---
    # (head_idx * D + x) // 16  =  head_idx * D_OVER_16 + x >> 4
    # Uses compile-time D_OVER_16 and bit-shift instead of runtime division.
    scale_base = batch_idx * (H * D_OVER_16) + head_idx * D_OVER_16
    elem_block_idx = elem_offset >> 4  # elem_offset // 16

    lower_scale_idx = scale_base + elem_block_idx
    upper_scale_idx = scale_base + D_HALF_OVER_16 + elem_block_idx

    scale_lower = tl.exp2(
        tl.load(scale_ptr + lower_scale_idx, mask=data_mask, other=127).to(tl.float32) - 127.0
    )
    scale_upper = tl.exp2(
        tl.load(scale_ptr + upper_scale_idx, mask=data_mask, other=127).to(tl.float32) - 127.0
    )

    # --- Store to output [B, H, D] with correct D stride ---
    output_base = batch_idx * (H * D) + head_idx * D
    tl.store(output_ptr + output_base + elem_offset,
             lower_float_vals * scale_lower, mask=data_mask)
    tl.store(output_ptr + output_base + D_HALF + elem_offset,
             upper_float_vals * scale_upper, mask=data_mask)


@triton.jit
def fp4_dequant_flat_kernel(
    quant_ptr,   # [B*H*D_HALF] flattened packed uint8
    scale_ptr,   # [B*H*D/16]   flattened scale exponents
    output_ptr,  # [B*H*D]      flattened bf16 output
    lut_ptr,     # [16]         signed FP4 LUT
    TOTAL_PACKED,  # B * H * D_HALF
    D_HALF: tl.constexpr,
    D: tl.constexpr,
    D_OVER_16: tl.constexpr,
    D_HALF_OVER_16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flat 1D-tiling kernel for FP4 dequantization — optimal for large B.
    Tiles across ALL B*H*D_HALF packed elements with large BLOCK_SIZE.
    Uses constexpr D_HALF for compiler-optimized div/mod decomposition.
    """
    block_idx = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    flat_idx = block_idx * BLOCK_SIZE + offset
    data_mask = flat_idx < TOTAL_PACKED

    # Decompose flat packed index → (bh_idx, elem_offset)
    # D_HALF is constexpr, so compiler optimizes to multiply-shift (no real division).
    bh_idx = flat_idx // D_HALF
    elem_offset = flat_idx - bh_idx * D_HALF   # faster than flat_idx % D_HALF

    # Load packed uint8 (flat, perfectly coalesced)
    packed_vals = tl.load(quant_ptr + flat_idx, mask=data_mask, other=0)

    # LUT-based dequant
    lower_nibble = (packed_vals & 0x0F).to(tl.int32)
    upper_nibble = ((packed_vals >> 4) & 0x0F).to(tl.int32)
    lower_float_vals = tl.load(lut_ptr + lower_nibble, mask=data_mask, other=0.0)
    upper_float_vals = tl.load(lut_ptr + upper_nibble, mask=data_mask, other=0.0)

    # Scale indices: bh_idx * D_OVER_16 + elem_offset >> 4
    # Works for any H because (b*H+h)*D_OVER_16 = b*(H*D_OVER_16) + h*D_OVER_16
    scale_base = bh_idx * D_OVER_16
    elem_block_idx = elem_offset >> 4
    scale_lower = tl.exp2(
        tl.load(scale_ptr + scale_base + elem_block_idx, mask=data_mask, other=127).to(tl.float32) - 127.0
    )
    scale_upper = tl.exp2(
        tl.load(scale_ptr + scale_base + D_HALF_OVER_16 + elem_block_idx, mask=data_mask, other=127).to(tl.float32) - 127.0
    )

    # Output: bh_idx * D + elem_offset (lower), + D_HALF (upper)
    output_base = bh_idx * D + elem_offset
    tl.store(output_ptr + output_base,
             lower_float_vals * scale_lower, mask=data_mask)
    tl.store(output_ptr + output_base + D_HALF,
             upper_float_vals * scale_upper, mask=data_mask)


# Threshold: if total blocks in 3D grid > this, switch to flat kernel
_FLAT_KERNEL_BLOCK_THRESHOLD = 8192


def fp4_dequant_triton(
    quant_tensor: torch.Tensor,
    scale_factors: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Triton-based FP4 dequantization wrapper.

    Automatically selects between:
      - 3D grid kernel: optimal for small-medium B (low per-block overhead)
      - Flat 1D kernel: optimal for large B (fewer blocks, 100% thread utilization)

    Args:
        quant_tensor: Quantized tensor of shape [B, M, N/2]
        scale_factors: Scale factors of shape [B, M*N/16]
        dtype: Target dtype for output

    Returns:
        Dequantized tensor of shape [B, M, N]
    """
    b, h, d_half = quant_tensor.shape
    d = d_half * 2

    output = torch.empty(b, h, d, dtype=dtype, device=quant_tensor.device)
    lut = _get_fp4_lut()

    d_over_16 = d // 16
    d_half_over_16 = d_half // 16

    # Estimate number of blocks for 3D kernel
    block_size_3d = 1 << (d_half - 1).bit_length()
    block_size_3d = max(block_size_3d, 32)
    num_blocks_3d = b * h * ((d_half + block_size_3d - 1) // block_size_3d)

    if num_blocks_3d <= _FLAT_KERNEL_BLOCK_THRESHOLD:
        # --- 3D grid: each block handles one (b, h) pair ---
        BLOCK_SIZE = block_size_3d
        num_d_blocks = (d_half + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_warps = max(1, min(BLOCK_SIZE // 32, 16))

        fp4_dequant_triton_kernel[(b, h, num_d_blocks)](
            quant_tensor, scale_factors, output, lut,
            d_half, d_over_16, d_half_over_16,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    else:
        # --- Flat 1D grid: tiles across all B*H*D_HALF elements ---
        # Large BLOCK_SIZE (4096) maximizes per-block work and memory throughput.
        BLOCK_SIZE = 4096
        total_packed = b * h * d_half
        num_blocks = (total_packed + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_warps = 32  # 32 warps × 32 threads = 1024 threads; each handles 4 elements

        fp4_dequant_flat_kernel[(num_blocks,)](
            quant_tensor, scale_factors, output, lut,
            total_packed, d_half, d, d_over_16, d_half_over_16,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

    return output


def calculate_accuracy_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """Calculate accuracy metrics between original and reconstructed tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()

    # PSNR calculation
    max_val = torch.max(torch.abs(original)).item()
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

    # Relative error
    rel_error = torch.mean(
        torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)
    ).item()

    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "Relative Error": rel_error}


def run_benchmark(m, n, k, num_runs=100) -> dict[str, dict[str, float]]:
    """Run FP8 vs KVFP4 quantization benchmark and return metrics."""
    tensor_bf16 = torch.randn(m, n, k, dtype=torch.bfloat16, device="cuda")

    # --- FP8 ---
    for _ in range(3):  # warmup
        _ = tensor_bf16 * 2
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fp8_quant_time = (time.time() - start) / num_runs

    start = time.time()
    for _ in range(num_runs):
        tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    fp8_dequant_time = (time.time() - start) / num_runs

    fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)

    # --- KVFP4 ---
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)

    start = time.time()
    for _ in range(num_runs):
        tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    torch.cuda.synchronize()
    fp4_quant_time = (time.time() - start) / num_runs

    start = time.time()
    for _ in range(num_runs):
        tensor_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
            tensor_fp4, scale_factors
        )
    torch.cuda.synchronize()
    fp4_dequant_time = (time.time() - start) / num_runs

    fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)

    return {
        "fp8": {
            "quant_time": fp8_quant_time,
            "dequant_time": fp8_dequant_time,
            **fp8_metrics,
        },
        "fp4": {
            "quant_time": fp4_quant_time,
            "dequant_time": fp4_dequant_time,
            **fp4_metrics,
        },
    }


# default tensor shapes (m, n, k)
# [M, 1, 576]: DeepSeekR1-FP4 MLA
# [M, 8, 64]: gpt-oss-20b MHA
MNK_FACTORS = [
    (64, 1, 576),
    (512, 1, 576),
    (1024, 1, 576),
    (4096, 1, 576),
    # (2868672, 1, 576),
    (64, 8, 64),
    (512, 8, 64),
    (1024, 8, 64),
    (4096, 8, 64),
    # (2868672, 8, 64),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
def test_kvfp4_quant_dequant(m, n, k):
    """Benchmark FP8 vs KVFP4 for predefined tensor shapes."""
    print(f"\n=== Running benchmark for tensor shape: [{m}, {n}, {k}] ===")
    results = run_benchmark(m, n, k)

    print("FP8:", results["fp8"])
    print("FP4:", results["fp4"])

    # Basic assertions to make sure metrics are reasonable
    assert results["fp4"]["MSE"] < 1.0
    assert results["fp8"]["MSE"] < 1.0


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
def test_triton_vs_pytorch_dequant_accuracy(m, n, k):
    """Compare accuracy between Triton kernel and PyTorch implementation."""
    print(f"\n=== Comparing Triton vs PyTorch for shape: [{m}, {n}, {k}] ===")
    
    # Generate random tensor and quantize it
    tensor_bf16 = torch.randn(m, n, k, dtype=torch.bfloat16, device="cuda")
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    
    # Dequantize using PyTorch implementation
    dequant_pytorch = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)
    
    # Dequantize using Triton kernel
    dequant_triton = fp4_dequant_triton(tensor_fp4, scale_factors, dtype=torch.bfloat16)
    
    # Calculate accuracy metrics between the two implementations
    metrics = calculate_accuracy_metrics(dequant_pytorch, dequant_triton)
    
    print("Triton vs PyTorch metrics:")
    print(f"  MSE: {metrics['MSE']:.2e}")
    print(f"  MAE: {metrics['MAE']:.2e}")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  Relative Error: {metrics['Relative Error']:.2e}")
    
    # Check that implementations are very close (allowing for minor numerical differences)
    assert metrics["MSE"] < 1e-6, f"MSE too high: {metrics['MSE']}"
    assert metrics["MAE"] < 1e-3, f"MAE too high: {metrics['MAE']}"
    
    # Also compare against original tensor
    pytorch_vs_orig = calculate_accuracy_metrics(tensor_bf16, dequant_pytorch)
    triton_vs_orig = calculate_accuracy_metrics(tensor_bf16, dequant_triton)
    
    print("\nPyTorch vs Original:")
    print(f"  MSE: {pytorch_vs_orig['MSE']:.2e}")
    print(f"  MAE: {pytorch_vs_orig['MAE']:.2e}")
    
    print("\nTriton vs Original:")
    print(f"  MSE: {triton_vs_orig['MSE']:.2e}")
    print(f"  MAE: {triton_vs_orig['MAE']:.2e}")
    
    # Both should have similar accuracy vs original
    mse_diff = abs(pytorch_vs_orig["MSE"] - triton_vs_orig["MSE"])
    assert mse_diff < 1e-4, f"MSE difference too large: {mse_diff}"


@pytest.mark.parametrize("m,n,k", [(10000, 8, 128)])
def test_triton_vs_pytorch_dequant_performance(m, n, k):
    """Compare performance between Triton kernel and PyTorch implementation."""
    print(f"\n=== Performance comparison for shape: [{m}, {n}, {k}] ===")
    
    tensor_bf16 = torch.randn(m, n, k, dtype=torch.bfloat16, device="cuda")
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    
    num_runs = 100
    
    # Warmup
    for _ in range(3):
        _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)
        _ = fp4_dequant_triton(tensor_fp4, scale_factors)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch implementation
    start = time.time()
    for _ in range(num_runs):
        _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs
    
    # Benchmark Triton kernel
    start = time.time()
    for _ in range(num_runs):
        _ = fp4_dequant_triton(tensor_fp4, scale_factors)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs
    
    print(f"PyTorch dequant time: {pytorch_time*1000:.4f} ms")
    print(f"Triton dequant time: {triton_time*1000:.4f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    
    assert triton_time > 0 and pytorch_time > 0
