#!/usr/bin/env python3
"""
Speed benchmark to compare performance between BF16 (baseline), INT4, and FP4 quantized decode attention for MHA and GQA.

This test evaluates the speed of different KV cache quantization schemes:
- BF16: Full precision baseline
- INT4: 4-bit integer quantization with per-head scale and zero-point
- FP4: 4-bit floating-point quantization with block-based scaling

Configuration: MHA with num_q_heads=64, num_kv_heads=64 (MQA ratio=1)
               GQA with num_q_heads=64, num_kv_heads=8 (GQA ratio=8)
               TP8 is considered as (heads//8) per rank.
"""

import numpy as np
import torch

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_quantized,
)
from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil
from sglang.test.test_decode_attention_fp4_precision import setup_gqa_inputs, quantize_kv_to_fp4, quantize_kv_to_int4

def benchmark_kernel(
    fn,
    *args,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    **kwargs,
):
    """
    Benchmark a kernel function with warmup and multiple runs.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for the function
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        **kwargs: Keyword arguments for the function (excluding warmup_runs and benchmark_runs)

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup_runs):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_runs):
        fn(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / benchmark_runs

    return elapsed_ms


def calculate_tflops(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    time_ms: float,
):
    """
    Calculate TFLOPS for decode attention.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        time_ms: Time in milliseconds

    Returns:
        TFLOPS
    """
    # Decode attention: Q @ K^T and (QK^T) @ V
    # Q: [batch_size, num_q_heads, head_dim]
    # K: [batch_size * seq_len, num_kv_heads, head_dim]
    # V: [batch_size * seq_len, num_kv_heads, head_dim]
    # O: [batch_size, num_q_heads, head_dim]

    # QK^T: batch_size * num_q_heads * seq_len * head_dim
    # OV: batch_size * num_q_heads * head_dim * seq_len
    total_ops = (
        2 * batch_size * num_q_heads * seq_len * head_dim
    )  # QK^T and OV matmuls
    tflops = (total_ops * 1e-12) / (time_ms * 1e-3)
    return tflops


if PYTEST_AVAILABLE:

    @pytest.mark.parametrize(
        "batch_size,seq_len, num_kv_heads, head_dim",
        [
            (1, 786, 64, 128),
            (4, 2048, 64, 128),
            (8, 314, 64, 128),
            (16, 237, 64, 128),
            (1, 786, 8, 128),
            (4, 2048, 8, 128),
            (8, 314, 8, 128),
            (16, 237, 8, 128),
        ],
    )
    def test_decode_attention_fp4_speed_gqa(
        batch_size, seq_len, num_kv_heads, head_dim
    ):
        """
        Test speed comparison between BF16, INT4, and FP4 decode attention for MHA and GQA.

        Configuration:
            - num_q_heads: 64
            - num_kv_heads: 64 for MHA, 8 for GQA
            - GQA ratio: 64/8 = 8
        """
        _run_benchmark(batch_size, seq_len, num_kv_heads, head_dim)

else:

    def test_decode_attention_fp4_speed_gqa(
        batch_size, seq_len, num_kv_heads, head_dim
    ):
        """
        Test speed comparison between BF16, INT4, and FP4 decode attention for GQA.

        Configuration:
            - num_q_heads: 64
            - num_kv_heads: 64 for MHA, 8 for GQA
            - GQA ratio: 64/8 = 8
        """
        _run_benchmark(batch_size, seq_len, num_kv_heads, head_dim)


def _run_benchmark(batch_size, seq_len, num_kv_heads, head_dim):
    """
    Core benchmark logic for speed comparison between BF16, INT4, and FP4 decode attention for GQA.

    Configuration:
        - num_q_heads: 64
        - num_kv_heads: 64 for MHA, 8 for GQA
        - GQA ratio: 64/8 = 8
        - TP8 model with 8 heads per KV head
    """
    num_q_heads = 8  # 64 // 8
    num_kv_heads = num_kv_heads // 8  # TP8 deploy configuration
    max_kv_splits = 8

    print(f"\n{'='*80}")
    print(
        f"Benchmarking MHA and GQA Decode Attention: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}"
    )
    print(
        f"    num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, GQA_ratio={num_q_heads//num_kv_heads if num_kv_heads > 0 else 1}"
    )
    print(f"{'='*80}")

    # Setup inputs
    inputs = setup_gqa_inputs(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        max_kv_splits=max_kv_splits,
    )

    # Pre-quantize KV cache (quantization time not included in benchmark)
    print("\n[Preparing] Quantizing KV cache...")
    k_quant_int4, k_scales_int4, v_quant_int4, v_scales_int4 = quantize_kv_to_int4(
        inputs["k_buffer"], inputs["v_buffer"]
    )
    k_quant_fp4, k_scales_fp4, v_quant_fp4, v_scales_fp4 = quantize_kv_to_fp4(
        inputs["k_buffer"], inputs["v_buffer"]
    )
    torch.cuda.synchronize()
    print("    ✓ Quantization completed")

    # Benchmark BF16 decode attention (baseline)
    print("\n[1/3] Benchmarking BF16 decode attention (baseline)...")
    time_bf16 = benchmark_kernel(
        decode_attention_fwd,
        inputs["q"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["o_bf16"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["attn_logits_bf16"],
        inputs["attn_lse_bf16"],
        inputs["num_kv_splits"],
        inputs["max_kv_splits"],
        inputs["sm_scale"],
        warmup_runs=10,
        benchmark_runs=100,
    )
    tflops_bf16 = calculate_tflops(
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, time_bf16
    )
    print(f"    ✓ BF16: {time_bf16:.3f} ms, {tflops_bf16:.2f} TFLOPS")

    # Benchmark INT4 quantized decode attention
    print("\n[2/3] Benchmarking INT4 quantized decode attention...")
    time_int4 = benchmark_kernel(
        decode_attention_fwd_quantized,
        inputs["q"],
        k_quant_int4,
        v_quant_int4,
        k_scales_int4,
        v_scales_int4,
        inputs["o_int4"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["attn_logits_int4"],
        inputs["attn_lse_int4"],
        inputs["num_kv_splits"],
        inputs["max_kv_splits"],
        inputs["sm_scale"],
        kv_dtype="int4",
        warmup_runs=10,
        benchmark_runs=100,
    )
    tflops_int4 = calculate_tflops(
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, time_int4
    )
    print(f"    ✓ INT4: {time_int4:.3f} ms, {tflops_int4:.2f} TFLOPS")

    # Benchmark FP4 quantized decode attention
    print("\n[3/3] Benchmarking FP4 quantized decode attention...")
    time_fp4 = benchmark_kernel(
        decode_attention_fwd_quantized,
        inputs["q"],
        k_quant_fp4,
        v_quant_fp4,
        k_scales_fp4,
        v_scales_fp4,
        inputs["o_fp4"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["attn_logits_fp4"],
        inputs["attn_lse_fp4"],
        inputs["num_kv_splits"],
        inputs["max_kv_splits"],
        inputs["sm_scale"],
        kv_dtype=torch.float4_e2m1fn_x2,
        warmup_runs=10,
        benchmark_runs=100,
    )
    tflops_fp4 = calculate_tflops(
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, time_fp4
    )
    print(f"    ✓ FP4: {time_fp4:.3f} ms, {tflops_fp4:.2f} TFLOPS")

    # Calculate speedup
    print(f"\n{'='*80}")
    print("Speed Comparison Results")
    print(f"{'='*80}")

    speedup_int4_vs_bf16 = time_bf16 / time_int4
    speedup_fp4_vs_bf16 = time_bf16 / time_fp4
    speedup_fp4_vs_int4 = time_int4 / time_fp4

    # Print comparison table
    print(f"\n{'Method':<15} {'Time (ms)':<15} {'TFLOPS':<15} {'Speedup vs BF16':<20}")
    print(f"{'-'*80}")
    print(f"{'BF16':<15} {time_bf16:<15.3f} {tflops_bf16:<15.2f} {'1.00x (baseline)':<20}")
    print(
        f"{'INT4':<15} {time_int4:<15.3f} {tflops_int4:<15.2f} {speedup_int4_vs_bf16:<20.2f}x"
    )
    print(
        f"{'FP4':<15} {time_fp4:<15.3f} {tflops_fp4:<15.2f} {speedup_fp4_vs_bf16:<20.2f}x"
    )

    print(f"\n{'='*80}")
    print("Relative Performance")
    print(f"{'='*80}")
    print(f"INT4 vs BF16: {speedup_int4_vs_bf16:.2f}x faster")
    print(f"FP4 vs BF16: {speedup_fp4_vs_bf16:.2f}x faster")
    print(f"FP4 vs INT4: {speedup_fp4_vs_int4:.2f}x faster")

    if speedup_fp4_vs_int4 > 1.0:
        print(f"✓ FP4 is {speedup_fp4_vs_int4:.2f}x faster than INT4")
    elif speedup_fp4_vs_int4 < 1.0:
        print(f"✗ FP4 is {1.0/speedup_fp4_vs_int4:.2f}x slower than INT4")
    else:
        print("≈ FP4 and INT4 have similar speed")

    print(f"\n{'='*80}")
    print("✓ Speed benchmark completed!")
    print(f"{'='*80}\n")

    return {
        "bf16": {"time_ms": time_bf16, "tflops": tflops_bf16},
        "int4": {"time_ms": time_int4, "tflops": tflops_int4},
        "fp4": {"time_ms": time_fp4, "tflops": tflops_fp4},
        "speedup_int4_vs_bf16": speedup_int4_vs_bf16,
        "speedup_fp4_vs_bf16": speedup_fp4_vs_bf16,
        "speedup_fp4_vs_int4": speedup_fp4_vs_int4,
    }


if __name__ == "__main__":
    # Run with multiple configurations
    test_configs = [
        # (1, 786, 8, 128),
        # (4, 2048, 8, 128),
        # (8, 314, 8, 128),
        # (16, 237, 8, 128),
        # (20, 1024, 8, 128),
        # (40, 1024, 8, 128),
        (1, 10000, 8, 128),
        (1, 120000, 8, 128),
    ]

    print("\n" + "=" * 80)
    print("GQA Decode Attention Quantization Speed Benchmark")
    print(f"Configuration: num_q_heads=64, num_kv_heads=8 (GQA ratio=8)")
    print(f"Comparing: BF16 (baseline) vs INT4 vs FP4")
    print("=" * 80)

    all_passed = True
    all_results = []

    for batch_size, seq_len, num_kv_heads, head_dim in test_configs:
        try:
            results = _run_benchmark(batch_size, seq_len, num_kv_heads, head_dim)
            all_results.append(
                {
                    "config": (batch_size, seq_len, num_kv_heads, head_dim),
                    "results": results,
                }
            )
        except Exception as e:
            print("\n" + "=" * 80)
            print(
                f"✗ Benchmark failed for config (batch={batch_size}, seq_len={seq_len}, num_kv_heads={num_kv_heads}, head_dim={head_dim})"
            )
            print(f"Error: {e}")
            print("=" * 80 + "\n")
            all_passed = False
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Quantization Speed Comparison")
    print("=" * 80)

    if all_results:
        print(
            f"\n{'Config':<50} {'BF16 (ms)':<12} {'INT4 (ms)':<12} {'FP4 (ms)':<12} {'INT4 Speedup':<15} {'FP4 Speedup':<15}"
        )
        print("-" * 80)
        for item in all_results:
            config = item["config"]
            results = item["results"]
            config_str = f"({config[0]}, {config[1]}, {config[2]}, {config[3]})"
            bf16_time = results["bf16"]["time_ms"]
            int4_time = results["int4"]["time_ms"]
            fp4_time = results["fp4"]["time_ms"]
            int4_speedup = results["speedup_int4_vs_bf16"]
            fp4_speedup = results["speedup_fp4_vs_bf16"]
            print(
                f"{config_str:<50} {bf16_time:<12.3f} {int4_time:<12.3f} {fp4_time:<12.3f} {int4_speedup:<15.2f}x {fp4_speedup:<15.2f}x"
            )

        # Calculate average speedups
        avg_int4_speedup = np.mean([r["results"]["speedup_int4_vs_bf16"] for r in all_results])
        avg_fp4_speedup = np.mean([r["results"]["speedup_fp4_vs_bf16"] for r in all_results])
        avg_fp4_vs_int4 = np.mean([r["results"]["speedup_fp4_vs_int4"] for r in all_results])

        print(f"\n{'='*80}")
        print("Average Speedups Across All Configurations")
        print(f"{'='*80}")
        print(f"INT4 vs BF16: {avg_int4_speedup:.2f}x faster")
        print(f"FP4 vs BF16: {avg_fp4_speedup:.2f}x faster")
        print(f"FP4 vs INT4: {avg_fp4_vs_int4:.2f}x faster")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All benchmarks completed!")
    else:
        print("✗ Some benchmarks failed!")
    print("=" * 80 + "\n")

