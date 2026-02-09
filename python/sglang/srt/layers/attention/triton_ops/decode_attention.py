# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()

logger = logging.getLogger(__name__)


_MIN_BLOCK_KV = 32


# FP4 E2M1 lookup table (same as in kvfp4_tensor.py)
@triton.jit
def _get_e2m1_value(idx):
    """Get E2M1 float value from 3-bit magnitude index"""
    # E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    return tl.where(
        idx == 0,
        0.0,
        tl.where(
            idx == 1,
            0.5,
            tl.where(
                idx == 2,
                1.0,
                tl.where(
                    idx == 3,
                    1.5,
                    tl.where(
                        idx == 4,
                        2.0,
                        tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_kernel_stage1_quant_int8(
    Q,
    K_Buffer,  # Quantized INT8 [cache_size, num_heads, head_dim] uint8
    V_Buffer,  # Quantized INT8 [cache_size, num_heads, head_dim] uint8
    K_Scales_Zeros,  # [cache_size, num_heads, 2] float32, [..., 0]=scale, [..., 1]=zero
    V_Scales_Zeros,  # [cache_size, num_heads, 2] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load quantized K (uint8)
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k_quant = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0,
            )

            # Load scales and zeros for K
            offs_sz_k = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            k_scales = tl.load(
                K_Scales_Zeros + offs_sz_k + 0,
                mask=offs_n < split_kv_end,
                other=1.0,
            )
            k_zeros = tl.load(
                K_Scales_Zeros + offs_sz_k + 1,
                mask=offs_n < split_kv_end,
                other=0.0,
            )

            # Dequantize K: k = (k_quant - zeros) * scales
            # k_quant: [BLOCK_N, BLOCK_DMODEL], k_scales/k_zeros: [BLOCK_N]
            k = ((k_quant.to(tl.float32) - k_zeros[:, None]) * k_scales[:, None]).to(
                q.dtype
            )

            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # Load quantized V (uint8)
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v_quant = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0,
            )

            # Load scales and zeros for V
            offs_sz_v = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
            v_scales = tl.load(
                V_Scales_Zeros + offs_sz_v + 0,
                mask=offs_n < split_kv_end,
                other=1.0,
            )
            v_zeros = tl.load(
                V_Scales_Zeros + offs_sz_v + 1,
                mask=offs_n < split_kv_end,
                other=0.0,
            )

            # Dequantize V inline: v = (v_quant - zeros) * scales
            # v_quant: [BLOCK_N, BLOCK_DV], v_scales/v_zeros: [BLOCK_N]
            v = ((v_quant.to(tl.float32) - v_zeros[:, None]) * v_scales[:, None]).to(
                q.dtype
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_kernel_stage1_quant_int4(
    Q,
    K_Buffer,  # Quantized INT4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    V_Buffer,  # Quantized INT4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2] float32, [..., 0]=scale, [..., 1]=zero
    V_Scales_Zeros,  # [cache_size, num_heads, 2] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    # For INT4, work with first half and second half separately
    acc_first = tl.zeros([BLOCK_DV // 2], dtype=tl.float32)
    acc_second = tl.zeros([BLOCK_DV // 2], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Load q split into first half and second half for INT4
        offs_d_first = tl.arange(0, BLOCK_DMODEL // 2)
        offs_d_second = tl.arange(0, BLOCK_DMODEL // 2)
        mask_d_first = offs_d_first < (Lk // 2)
        mask_d_second = offs_d_second < (Lk - Lk // 2)

        off_q_first = cur_batch * stride_qbs + cur_head * stride_qh + offs_d_first
        off_q_second = (
            cur_batch * stride_qbs + cur_head * stride_qh + (Lk // 2) + offs_d_second
        )

        q_first = tl.load(Q + off_q_first, mask=mask_d_first, other=0.0)
        q_second = tl.load(Q + off_q_second, mask=mask_d_second, other=0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT4 K (uint8, 2 values per byte)
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 2)
            mask_d_packed = offs_d_packed < (Lk // 2)

            offs_buf_k_packed = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[None, :]
            )
            k_quant_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for K
            offs_sz_k = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            k_scales = tl.load(
                K_Scales_Zeros + offs_sz_k + 0,
                mask=offs_n < split_kv_end,
                other=1.0,
            )
            k_zeros = tl.load(
                K_Scales_Zeros + offs_sz_k + 1,
                mask=offs_n < split_kv_end,
                other=0.0,
            )

            # Dequantize INT4 K inline: unpack and dequantize
            # k_quant_packed: [BLOCK_N, BLOCK_DMODEL//2], k_scales/k_zeros: [BLOCK_N]
            # Unpack lower and upper 4 bits, then dequantize: (q - zero) * scale
            k_lower = (
                ((k_quant_packed & 0x0F).to(tl.float32) - k_zeros[:, None])
                * k_scales[:, None]
            ).to(q_first.dtype)
            k_upper = (
                (((k_quant_packed >> 4) & 0x0F).to(tl.float32) - k_zeros[:, None])
                * k_scales[:, None]
            ).to(q_first.dtype)
            # Compute QK in q.dtype for efficiency, then convert to float32 for accumulation
            qk = tl.sum(q_first[None, :] * k_lower, 1) + tl.sum(
                q_second[None, :] * k_upper, 1
            )
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # Load packed INT4 V
            offs_dv_packed = tl.arange(0, BLOCK_DV // 2)
            mask_dv_packed = offs_dv_packed < (Lv // 2)

            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv_packed[None, :]
            )
            v_quant_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for V
            offs_sz_v = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
            v_scales = tl.load(
                V_Scales_Zeros + offs_sz_v + 0,
                mask=offs_n < split_kv_end,
                other=1.0,
            )
            v_zeros = tl.load(
                V_Scales_Zeros + offs_sz_v + 1,
                mask=offs_n < split_kv_end,
                other=0.0,
            )

            # Dequantize INT4 V inline: unpack and dequantize
            # v_quant_packed: [BLOCK_N, BLOCK_DV//2], v_scales/v_zeros: [BLOCK_N]
            # Unpack lower and upper 4 bits, then dequantize: (q - zero) * scale
            v_lower = (
                ((v_quant_packed & 0x0F).to(tl.float32) - v_zeros[:, None])
                * v_scales[:, None]
            ).to(q_first.dtype)
            v_upper = (
                (((v_quant_packed >> 4) & 0x0F).to(tl.float32) - v_zeros[:, None])
                * v_scales[:, None]
            ).to(q_first.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)

            # Accumulate separately for first half and second half
            # Convert p to q.dtype for efficient multiplication with v
            acc_first *= re_scale
            acc_second *= re_scale
            acc_first += tl.sum(p[:, None] * v_lower, 0)
            acc_second += tl.sum(p[:, None] * v_upper, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # Store first half and second half separately
        # First half: indices [0, Lv//2)
        offs_dv_first = tl.arange(0, BLOCK_DV // 2)
        mask_dv_first = offs_dv_first < (Lv // 2)
        offs_mid_o_first = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_first
        )
        tl.store(
            Att_Out + offs_mid_o_first,
            acc_first / e_sum,
            mask=mask_dv_first,
        )

        # Second half: indices [Lv//2, Lv)
        offs_dv_second = tl.arange(0, BLOCK_DV // 2)
        mask_dv_second = (offs_dv_second + Lv // 2) < Lv
        offs_mid_o_second = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_second
            + Lv // 2
        )
        tl.store(
            Att_Out + offs_mid_o_second,
            acc_second / e_sum,
            mask=mask_dv_second,
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_kernel_stage1_quant_fp4(
    Q,
    K_Buffer,  # Quantized FP4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    V_Buffer,  # Quantized FP4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    K_Scales,  # [cache_size, num_kv_heads, head_dim//16] uint8 (exponent + 127)
    V_Scales,  # [cache_size, num_kv_heads, head_dim//16] uint8 (exponent + 127)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_scale_kbs,
    stride_scale_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    # For FP4, work with first half and second half separately
    acc_first = tl.zeros([BLOCK_DV // 2], dtype=tl.float32)
    acc_second = tl.zeros([BLOCK_DV // 2], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Load q split into first half and second half for FP4
        offs_d_first = tl.arange(0, BLOCK_DMODEL // 2)
        offs_d_second = tl.arange(0, BLOCK_DMODEL // 2)
        mask_d_first = offs_d_first < (Lk // 2)
        mask_d_second = offs_d_second < (Lk - Lk // 2)

        off_q_first = cur_batch * stride_qbs + cur_head * stride_qh + offs_d_first
        off_q_second = (
            cur_batch * stride_qbs + cur_head * stride_qh + (Lk // 2) + offs_d_second
        )

        q_first = tl.load(Q + off_q_first, mask=mask_d_first, other=0.0)
        q_second = tl.load(Q + off_q_second, mask=mask_d_second, other=0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed FP4 K (uint8, 2 values per byte)
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 2)
            mask_d_packed = offs_d_packed < (Lk // 2)

            offs_buf_k_packed = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[None, :]
            )
            k_quant_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                other=0,
            )

            # Load scales for K (block-wise, 16 elements per scale)
            # scales are stored as uint8 exponents (scale_exp + 127)
            # For each block of 16 elements in K (or 8 packed bytes), we have one scale
            # K_Scales shape: [cache_size, num_kv_heads, head_dim//16]
            # Similar to INT4: [token_idx, head_idx, block_idx]

            # Dequantize FP4 K inline: unpack, lookup, and apply block-wise scaling
            # k_quant_packed: [BLOCK_N, BLOCK_DMODEL//2]
            # Unpack lower and upper 4 bits
            k_packed_lower = k_quant_packed & 0x0F  # Lower nibble
            k_packed_upper = (k_quant_packed >> 4) & 0x0F  # Upper nibble

            # Extract sign (bit 3) and magnitude index (bits 0-2)
            k_sign_lower = (k_packed_lower & 0x08) != 0
            k_sign_upper = (k_packed_upper & 0x08) != 0
            k_mag_lower = k_packed_lower & 0x07
            k_mag_upper = k_packed_upper & 0x07

            # Lookup E2M1 values
            k_float_lower = _get_e2m1_value(k_mag_lower)
            k_float_upper = _get_e2m1_value(k_mag_upper)

            # Apply sign
            k_float_lower = tl.where(k_sign_lower, -k_float_lower, k_float_lower)
            k_float_upper = tl.where(k_sign_upper, -k_float_upper, k_float_upper)

            # Apply block-wise scaling for FP4
            # MHA kernel: data layout is [BLOCK_N, BLOCK_DMODEL//2]
            # With first-half/second-half packing:
            #   packed byte d: lower nibble = element[d], upper nibble = element[d + Lk//2]
            # Scale blocks cover 16 consecutive elements each:
            #   block b covers elements [16*b, 16*(b+1))
            # So lower nibble (element d) needs scale at block d//16
            # And upper nibble (element d+Lk//2) needs scale at block (d+Lk//2)//16
            block_idx_lower = offs_d_packed // 16  # [BLOCK_DMODEL//2]
            block_idx_upper = (offs_d_packed + Lk // 2) // 16  # [BLOCK_DMODEL//2]

            # Load scales for lower nibble (first half elements)
            offs_scale_k_lower = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_lower[None, :]  # [1, BLOCK_DMODEL//2]
            )  # [BLOCK_N, BLOCK_DMODEL//2]

            k_scale_exp_lower = (
                tl.load(
                    K_Scales + offs_scale_k_lower,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DMODEL//2]

            # Load scales for upper nibble (second half elements)
            offs_scale_k_upper = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_upper[None, :]  # [1, BLOCK_DMODEL//2]
            )  # [BLOCK_N, BLOCK_DMODEL//2]

            k_scale_exp_upper = (
                tl.load(
                    K_Scales + offs_scale_k_upper,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DMODEL//2]

            # Apply separate scales
            k_scale_lower = tl.exp2(k_scale_exp_lower)
            k_scale_upper = tl.exp2(k_scale_exp_upper)
            k_lower = (k_float_lower * k_scale_lower).to(q_first.dtype)
            k_upper = (k_float_upper * k_scale_upper).to(q_first.dtype)

            # Compute QK
            qk = tl.sum(q_first[None, :] * k_lower, 1) + tl.sum(
                q_second[None, :] * k_upper, 1
            )
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # Load packed FP4 V and dequantize similarly
            offs_dv_packed = tl.arange(0, BLOCK_DV // 2)
            mask_dv_packed = offs_dv_packed < (Lv // 2)

            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv_packed[None, :]
            )
            v_quant_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                other=0,
            )

            # Load scales for V
            # V_Scales shape: [cache_size, num_kv_heads, head_dim//16]
            # Similar to INT4: [token_idx, head_idx, block_idx]

            # Dequantize FP4 V
            v_packed_lower = v_quant_packed & 0x0F
            v_packed_upper = (v_quant_packed >> 4) & 0x0F

            v_sign_lower = (v_packed_lower & 0x08) != 0
            v_sign_upper = (v_packed_upper & 0x08) != 0
            v_mag_lower = v_packed_lower & 0x07
            v_mag_upper = v_packed_upper & 0x07

            v_float_lower = _get_e2m1_value(v_mag_lower)
            v_float_upper = _get_e2m1_value(v_mag_upper)

            v_float_lower = tl.where(v_sign_lower, -v_float_lower, v_float_lower)
            v_float_upper = tl.where(v_sign_upper, -v_float_upper, v_float_upper)

            # Apply block-wise scaling for V
            # With first-half/second-half packing:
            #   packed byte d: lower nibble = element[d], upper nibble = element[d + Lv//2]
            block_idx_v_lower = offs_dv_packed // 16  # [BLOCK_DV//2]
            block_idx_v_upper = (offs_dv_packed + Lv // 2) // 16  # [BLOCK_DV//2]

            # Load scales for lower nibble (first half elements)
            offs_scale_v_lower = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_v_lower[None, :]  # [1, BLOCK_DV//2]
            )  # [BLOCK_N, BLOCK_DV//2]

            v_scale_exp_lower = (
                tl.load(
                    V_Scales + offs_scale_v_lower,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DV//2]

            # Load scales for upper nibble (second half elements)
            offs_scale_v_upper = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_v_upper[None, :]  # [1, BLOCK_DV//2]
            )  # [BLOCK_N, BLOCK_DV//2]

            v_scale_exp_upper = (
                tl.load(
                    V_Scales + offs_scale_v_upper,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DV//2]

            # Apply separate scales
            v_scale_lower = tl.exp2(v_scale_exp_lower)
            v_scale_upper = tl.exp2(v_scale_exp_upper)
            v_lower = (v_float_lower * v_scale_lower).to(q_first.dtype)
            v_upper = (v_float_upper * v_scale_upper).to(q_first.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)

            # Accumulate separately for first half and second half
            acc_first *= re_scale
            acc_second *= re_scale
            acc_first += tl.sum(p[:, None] * v_lower, 0)
            acc_second += tl.sum(p[:, None] * v_upper, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # Store first half and second half separately
        offs_dv_first = tl.arange(0, BLOCK_DV // 2)
        mask_dv_first = offs_dv_first < (Lv // 2)
        offs_mid_o_first = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_first
        )
        tl.store(
            Att_Out + offs_mid_o_first,
            acc_first / e_sum,
            mask=mask_dv_first,
        )

        offs_dv_second = tl.arange(0, BLOCK_DV // 2)
        mask_dv_second = (offs_dv_second + Lv // 2) < Lv
        offs_mid_o_second = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_second
            + Lv // 2
        )
        tl.store(
            Att_Out + offs_mid_o_second,
            acc_second / e_sum,
            mask=mask_dv_second,
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_grouped_kernel_stage1_quant_int8(
    Q,
    K_Buffer,  # Quantized INT8 [cache_size, num_heads, head_dim] int8
    V_Buffer,  # Quantized INT8 [cache_size, num_heads, head_dim] int8
    K_Scales_Zeros,  # [cache_size, num_heads, 2] float32, [..., 0]=scale, [..., 1]=zero
    V_Scales_Zeros,  # [cache_size, num_heads, 2] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,  # scales_zeros stride for cache
    stride_sz_kh,  # scales_zeros stride for head
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load quantized K and dequantize
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k_quant = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0,
            )

            # Load K scales and zeros for dequantization
            offs_sz_k = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            k_scale = tl.load(
                K_Scales_Zeros + offs_sz_k + 0, mask=offs_n < split_kv_end, other=1.0
            )
            k_zero = tl.load(
                K_Scales_Zeros + offs_sz_k + 1, mask=offs_n < split_kv_end, other=0.0
            )

            # Dequantize K: k = (k_quant - zero) * scale
            # k_quant shape: [BLOCK_DMODEL, BLOCK_N] (transposed), k_scale/k_zero: [BLOCK_N]
            k = ((k_quant.to(tl.float32) - k_zero[None, :]) * k_scale[None, :]).to(
                q.dtype
            )

            # q: [BLOCK_H, BLOCK_DMODEL], k: [BLOCK_DMODEL, BLOCK_N]
            # Compute qk = q @ k
            qk = tl.dot(q, k)
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[None, :]
                )
                kpe_quant = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dpe[None, :]),
                    other=0,
                )
                # Dequantize DPE: kpe = (kpe_quant - zero) * scale
                # kpe_quant shape: [BLOCK_N, BLOCK_DPE], k_scale/k_zero: [BLOCK_N]
                kpe = (
                    (kpe_quant.to(tl.float32) - k_zero[:, None]) * k_scale[:, None]
                ).to(qpe.dtype)
                qk += tl.dot(qpe, kpe)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # Load quantized V and dequantize
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v_quant = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0,
            )

            # Load V scales and zeros for dequantization
            offs_sz_v = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            v_scale = tl.load(
                V_Scales_Zeros + offs_sz_v + 0, mask=offs_n < split_kv_end, other=1.0
            )
            v_zero = tl.load(
                V_Scales_Zeros + offs_sz_v + 1, mask=offs_n < split_kv_end, other=0.0
            )

            # Dequantize V: v = (v_quant - zero) * scale
            # v_quant shape: [BLOCK_N, BLOCK_DV], v_scale/v_zero: [BLOCK_N]
            v = ((v_quant.to(tl.float32) - v_zero[:, None]) * v_scale[:, None]).to(
                q.dtype
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _fwd_grouped_kernel_stage1_quant_int4(
    Q,
    K_Buffer,  # Quantized INT4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    V_Buffer,  # Quantized INT4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2] float32, [..., 0]=scale, [..., 1]=zero
    V_Scales_Zeros,  # [cache_size, num_heads, 2] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,  # scales_zeros stride for cache
    stride_sz_kh,  # scales_zeros stride for head
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # Use separate accumulators for first and second halves (INT4 unpacks to two halves)
    acc_first = tl.zeros([BLOCK_H, BLOCK_DV // 2], dtype=tl.float32)
    acc_second = tl.zeros([BLOCK_H, BLOCK_DV // 2], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Split Q into first and second halves for INT4 dot product
        offs_d_first = tl.arange(0, BLOCK_DMODEL // 2)
        offs_d_second = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)
        mask_d_first = offs_d_first < (Lk // 2)
        mask_d_second = offs_d_second < Lk

        offs_q_first = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d_first[None, :]
        )
        offs_q_second = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d_second[None, :]
        )

        q_first = tl.load(
            Q + offs_q_first,
            mask=(mask_h[:, None]) & (mask_d_first[None, :]),
            other=0.0,
        )
        q_second = tl.load(
            Q + offs_q_second,
            mask=(mask_h[:, None]) & (mask_d_second[None, :]),
            other=0.0,
        )

        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT4 K in transposed format for efficient dot product
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 2)
            mask_d_packed = offs_d_packed < (Lk // 2)

            offs_buf_k_packed = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[:, None]
            )
            k_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                other=0,
            )

            # Load K scales and zeros for dequantization
            offs_sz_k = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            k_scale = tl.load(
                K_Scales_Zeros + offs_sz_k + 0, mask=offs_n < split_kv_end, other=1.0
            )
            k_zero = tl.load(
                K_Scales_Zeros + offs_sz_k + 1, mask=offs_n < split_kv_end, other=0.0
            )

            # Dequantize INT4 K inline: unpack and dequantize
            # k_packed shape: [BLOCK_DMODEL//2, BLOCK_N] (transposed), k_scale/k_zero: [BLOCK_N]
            # Unpack lower and upper 4 bits, then dequantize: (q - zero) * scale
            k_lower = (
                ((k_packed & 0x0F).to(tl.float32) - k_zero[None, :]) * k_scale[None, :]
            ).to(q_first.dtype)
            k_upper = (
                (((k_packed >> 4) & 0x0F).to(tl.float32) - k_zero[None, :])
                * k_scale[None, :]
            ).to(q_first.dtype)

            # Compute QK for both halves
            # q_first: [BLOCK_H, BLOCK_DMODEL//2], k_lower: [BLOCK_DMODEL//2, BLOCK_N]
            # qk = q_first @ k_lower + q_second @ k_upper
            qk = tl.dot(q_first, k_lower) + tl.dot(q_second, k_upper)

            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe_quant = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0,
                )
                # Dequantize DPE: kpe = (kpe_quant - zero) * scale
                # kpe_quant shape: [BLOCK_DPE, BLOCK_N] (transposed), k_scale/k_zero: [BLOCK_N]
                kpe = (
                    (kpe_quant.to(tl.float32) - k_zero[None, :]) * k_scale[None, :]
                ).to(qpe.dtype)
                qk += tl.dot(qpe, kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # Load packed INT4 V and dequantize
            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + tl.arange(0, BLOCK_DV // 2)[None, :]
            )
            v_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end)
                & (tl.arange(0, BLOCK_DV // 2)[None, :] < (Lv // 2)),
                other=0,
            )

            # Load V scales and zeros for dequantization
            offs_sz_v = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
            v_scale = tl.load(
                V_Scales_Zeros + offs_sz_v + 0, mask=offs_n < split_kv_end, other=1.0
            )
            v_zero = tl.load(
                V_Scales_Zeros + offs_sz_v + 1, mask=offs_n < split_kv_end, other=0.0
            )

            # Dequantize INT4 V inline: unpack and dequantize
            # v_packed shape: [BLOCK_N, BLOCK_DV//2], v_scale/v_zero: [BLOCK_N]
            # Unpack lower and upper 4 bits, then dequantize: (q - zero) * scale
            v_lower = (
                ((v_packed & 0x0F).to(tl.float32) - v_zero[:, None]) * v_scale[:, None]
            ).to(q_first.dtype)
            v_upper = (
                (((v_packed >> 4) & 0x0F).to(tl.float32) - v_zero[:, None])
                * v_scale[:, None]
            ).to(q_first.dtype)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Scale existing accumulators
            acc_first *= re_scale[:, None]
            acc_second *= re_scale[:, None]

            # Accumulate attention-weighted V for both halves
            acc_first += tl.dot(p.to(v_lower.dtype), v_lower)
            acc_second += tl.dot(p.to(v_upper.dtype), v_upper)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store first half and second half separately
        # First half: indices [0, Lv//2)
        offs_dv_first = tl.arange(0, BLOCK_DV // 2)
        mask_dv_first = offs_dv_first < (Lv // 2)
        offs_mid_o_first = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_first[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o_first,
            acc_first / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_first[None, :]),
        )

        # Second half: indices [Lv//2, Lv)
        offs_dv_second = tl.arange(0, BLOCK_DV // 2)
        mask_dv_second = (offs_dv_second + Lv // 2) < Lv
        offs_mid_o_second = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + (offs_dv_second + Lv // 2)[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o_second,
            acc_second / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_second[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _fwd_grouped_kernel_stage1_quant_fp4(
    Q,
    K_Buffer,  # Quantized FP4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    V_Buffer,  # Quantized FP4 [cache_size, num_heads, head_dim//2] uint8 (packed)
    K_Scales,  # [cache_size, num_kv_heads, head_dim//16] uint8 (exponent + 127)
    V_Scales,  # [cache_size, num_kv_heads, head_dim//16] uint8 (exponent + 127)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_scale_kbs,
    stride_scale_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # Use separate accumulators for first and second halves (FP4 unpacks to two halves)
    acc_first = tl.zeros([BLOCK_H, BLOCK_DV // 2], dtype=tl.float32)
    acc_second = tl.zeros([BLOCK_H, BLOCK_DV // 2], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Split Q into first and second halves
        # K/V packing should be: packed byte d contains element[d] (lower) and element[d+64] (upper)
        offs_d_first = tl.arange(0, BLOCK_DMODEL // 2)
        offs_d_second = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)
        mask_d_first = offs_d_first < (Lk // 2)
        mask_d_second = offs_d_second < Lk

        offs_q_first = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d_first[None, :]
        )
        offs_q_second = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d_second[None, :]
        )

        q_first = tl.load(
            Q + offs_q_first,
            mask=(mask_h[:, None]) & (mask_d_first[None, :]),
            other=0.0,
        )
        q_second = tl.load(
            Q + offs_q_second,
            mask=(mask_h[:, None]) & (mask_d_second[None, :]),
            other=0.0,
        )

        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed FP4 K in transposed format for efficient dot product
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 2)
            mask_d_packed = offs_d_packed < (Lk // 2)

            offs_buf_k_packed = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[:, None]
            )
            k_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                other=0,
            )

            # Load scales for K (block-wise, 16 elements per scale)
            # K_Scales shape: [cache_size, num_kv_heads, head_dim//16]
            # Similar to INT4: [token_idx, head_idx, block_idx]

            # Dequantize FP4 K: unpack, lookup E2M1, and apply block-wise scaling
            k_packed_lower = k_packed & 0x0F
            k_packed_upper = (k_packed >> 4) & 0x0F

            k_sign_lower = (k_packed_lower & 0x08) != 0
            k_sign_upper = (k_packed_upper & 0x08) != 0
            k_mag_lower = k_packed_lower & 0x07
            k_mag_upper = k_packed_upper & 0x07

            k_float_lower = _get_e2m1_value(k_mag_lower)
            k_float_upper = _get_e2m1_value(k_mag_upper)

            k_float_lower = tl.where(k_sign_lower, -k_float_lower, k_float_lower)
            k_float_upper = tl.where(k_sign_upper, -k_float_upper, k_float_upper)

            # Apply block-wise scaling for FP4
            # GQA kernel: data layout is transposed [BLOCK_DMODEL//2, BLOCK_N]
            # With first-half/second-half packing:
            #   packed byte d: lower nibble = element[d], upper nibble = element[d + Lk//2]
            # Scale blocks cover 16 consecutive elements each:
            #   block b covers elements [16*b, 16*(b+1))
            # So lower nibble (element d) needs scale at block d//16
            # And upper nibble (element d+Lk//2) needs scale at block (d+Lk//2)//16
            block_idx_lower = offs_d_packed // 16  # [BLOCK_DMODEL//2]
            block_idx_upper = (offs_d_packed + Lk // 2) // 16  # [BLOCK_DMODEL//2]

            # Load scales for lower nibble (first half elements)
            offs_scale_k_lower = (
                kv_loc[None, :] * stride_scale_kbs  # [1, BLOCK_N]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_lower[:, None]  # [BLOCK_DMODEL//2, 1]
            )  # [BLOCK_DMODEL//2, BLOCK_N]

            k_scale_exp_lower = (
                tl.load(
                    K_Scales + offs_scale_k_lower,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_DMODEL//2, BLOCK_N]

            # Load scales for upper nibble (second half elements)
            offs_scale_k_upper = (
                kv_loc[None, :] * stride_scale_kbs  # [1, BLOCK_N]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_upper[:, None]  # [BLOCK_DMODEL//2, 1]
            )  # [BLOCK_DMODEL//2, BLOCK_N]

            k_scale_exp_upper = (
                tl.load(
                    K_Scales + offs_scale_k_upper,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_DMODEL//2, BLOCK_N]

            k_scale_lower = tl.exp2(k_scale_exp_lower)
            k_scale_upper = tl.exp2(k_scale_exp_upper)
            k_lower = (k_float_lower * k_scale_lower).to(q_first.dtype)
            k_upper = (k_float_upper * k_scale_upper).to(q_first.dtype)

            # Compute QK for both halves
            qk = tl.dot(q_first, k_lower) + tl.dot(q_second, k_upper)

            if BLOCK_DPE > 0:
                # TODO: FP4 DPE support needs proper implementation for packed buffers
                # For now, use lower scale as approximation (DPE only active for Lk=576/288)
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe_quant = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0,
                )
                # For DPE: similar FP4 dequantization
                kpe_sign = (kpe_quant & 0x08) != 0
                kpe_mag = kpe_quant & 0x07
                kpe_float = _get_e2m1_value(kpe_mag)
                kpe_float = tl.where(kpe_sign, -kpe_float, kpe_float)
                # Apply lower scale as approximation for DPE
                kpe = (kpe_float * k_scale_lower).to(qpe.dtype)
                qk += tl.dot(qpe, kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # Load packed FP4 V and dequantize
            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + tl.arange(0, BLOCK_DV // 2)[None, :]
            )
            v_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end)
                & (tl.arange(0, BLOCK_DV // 2)[None, :] < (Lv // 2)),
                other=0,
            )

            # Load scales for V
            # V_Scales shape: [cache_size, num_kv_heads, head_dim//16]
            # Similar to INT4: [token_idx, head_idx, block_idx]

            # Dequantize FP4 V
            v_packed_lower = v_packed & 0x0F
            v_packed_upper = (v_packed >> 4) & 0x0F

            v_sign_lower = (v_packed_lower & 0x08) != 0
            v_sign_upper = (v_packed_upper & 0x08) != 0
            v_mag_lower = v_packed_lower & 0x07
            v_mag_upper = v_packed_upper & 0x07

            v_float_lower = _get_e2m1_value(v_mag_lower)
            v_float_upper = _get_e2m1_value(v_mag_upper)

            v_float_lower = tl.where(v_sign_lower, -v_float_lower, v_float_lower)
            v_float_upper = tl.where(v_sign_upper, -v_float_upper, v_float_upper)

            # Apply block-wise scaling for V
            # With first-half/second-half packing:
            #   packed byte d: lower nibble = element[d], upper nibble = element[d + Lv//2]
            offs_dv_packed = tl.arange(0, BLOCK_DV // 2)
            block_idx_v_lower = offs_dv_packed // 16  # [BLOCK_DV//2]
            block_idx_v_upper = (offs_dv_packed + Lv // 2) // 16  # [BLOCK_DV//2]

            # Load scales for lower nibble (first half elements)
            offs_scale_v_lower = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_v_lower[None, :]  # [1, BLOCK_DV//2]
            )  # [BLOCK_N, BLOCK_DV//2]

            v_scale_exp_lower = (
                tl.load(
                    V_Scales + offs_scale_v_lower,
                    mask=(offs_n[:, None] < split_kv_end)
                    & (offs_dv_packed[None, :] < (Lv // 2)),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DV//2]

            # Load scales for upper nibble (second half elements)
            offs_scale_v_upper = (
                kv_loc[:, None] * stride_scale_kbs  # [BLOCK_N, 1]
                + cur_kv_head * stride_scale_kh  # scalar
                + block_idx_v_upper[None, :]  # [1, BLOCK_DV//2]
            )  # [BLOCK_N, BLOCK_DV//2]

            v_scale_exp_upper = (
                tl.load(
                    V_Scales + offs_scale_v_upper,
                    mask=(offs_n[:, None] < split_kv_end)
                    & ((offs_dv_packed[None, :] + Lv // 2) < Lv),
                    other=127,
                ).to(tl.float32)
                - 127.0
            )  # [BLOCK_N, BLOCK_DV//2]

            v_scale_lower = tl.exp2(v_scale_exp_lower)
            v_scale_upper = tl.exp2(v_scale_exp_upper)
            v_lower = (v_float_lower * v_scale_lower).to(q_first.dtype)
            v_upper = (v_float_upper * v_scale_upper).to(q_first.dtype)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Scale existing accumulators
            acc_first *= re_scale[:, None]
            acc_second *= re_scale[:, None]

            # Accumulate attention-weighted V for both halves
            acc_first += tl.dot(p.to(v_lower.dtype), v_lower)
            acc_second += tl.dot(p.to(v_upper.dtype), v_upper)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store first half and second half separately
        offs_dv_first = tl.arange(0, BLOCK_DV // 2)
        mask_dv_first = offs_dv_first < (Lv // 2)
        offs_mid_o_first = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_first[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o_first,
            acc_first / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_first[None, :]),
        )

        offs_dv_second = tl.arange(0, BLOCK_DV // 2)
        mask_dv_second = (offs_dv_second + Lv // 2) < Lv
        offs_mid_o_second = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + (offs_dv_second + Lv // 2)[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o_second,
            acc_second / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_second[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


def _decode_att_m_fwd_quant_int8(
    q,
    k_buffer,  # Quantized INT8
    v_buffer,  # Quantized INT8
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    INT8 quantized KV cache attention wrapper.
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1_quant_int8[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


def _decode_att_m_fwd_quant_int4(
    q,
    k_buffer,  # Quantized INT4 (packed)
    v_buffer,  # Quantized INT4 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    INT4 quantized KV cache attention wrapper.
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    # For INT4, the buffer stores packed values (head_dim//2)
    # But we need to work with the actual head_dim
    Lk = k_buffer.shape[-1] * 2  # Unpack to get real dimension
    Lv = v_buffer.shape[-1] * 2

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1_quant_int4[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


def _decode_att_m_fwd_quant_fp4(
    q,
    k_buffer,  # Quantized FP4 (packed)
    v_buffer,  # Quantized FP4 (packed)
    k_scales,  # Scales for K (uint8 exponents)
    v_scales,  # Scales for V (uint8 exponents)
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    FP4 E2M1 quantized KV cache attention wrapper.
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    # For FP4, the buffer stores packed values (head_dim//2)
    # But we need to work with the actual head_dim
    Lk = k_buffer.shape[-1] * 2  # Unpack to get real dimension
    Lv = v_buffer.shape[-1] * 2

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    if len(k_scales.shape) == 2:
        k_scales = k_scales.reshape(k_scales.shape[0], head_num, -1)
        v_scales = v_scales.reshape(v_scales.shape[0], head_num, -1)

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1_quant_fp4[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales,
        v_scales,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales.stride(0),
        k_scales.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


def _decode_grouped_att_m_fwd_quant_int8(
    q,
    k_buffer,  # Quantized INT8
    v_buffer,  # Quantized INT8
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1_quant_int8[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


def _decode_grouped_att_m_fwd_quant_fp4(
    q,
    k_buffer,  # Quantized FP4 (packed)
    v_buffer,  # Quantized FP4 (packed)
    k_scales,  # Scales for K (uint8 exponents)
    v_scales,  # Scales for V (uint8 exponents)
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    FP4 E2M1 quantized KV cache attention wrapper for grouped query attention.
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 32
    # For FP4, k_buffer is packed, so actual Lk is 2x the last dimension
    Lk = k_buffer.shape[-1] * 2
    Lv = v_buffer.shape[-1] * 2

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_head_num = k_buffer.shape[1]
    kv_group_num = q.shape[1] // kv_head_num
    if len(k_scales.shape) == 2:
        k_scales = k_scales.reshape(k_scales.shape[0], kv_head_num, -1)
        v_scales = v_scales.reshape(v_scales.shape[0], kv_head_num, -1)

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    # print(f"*************fp4 quant kernel arguments:**************")
    # print(
    # f"    q: {q.shape}, k_buffer: {k_buffer.shape}, v_buffer: {v_buffer.shape}, k_scales: {k_scales.shape}, v_scales: {v_scales.shape}"
    # )
    # print(
    # f"    kv_indptr: {kv_indptr.shape}, kv_indices: {kv_indices.shape}, att_out: {att_out.shape}, att_lse: {att_lse.shape}"
    # )
    # print(
    # f"    num_kv_splits: {num_kv_splits}, max_kv_splits: {max_kv_splits}, sm_scale: {sm_scale}"
    # )
    # print(f"    logit_cap: {logit_cap}, xai_temperature_len: {xai_temperature_len}")
    # print(f"    Lk: {Lk}, Lv: {Lv}")
    # print(
    # f"    BLOCK: {BLOCK}, BLOCK_DMODEL: {BLOCK_DMODEL}, BLOCK_DPE: {BLOCK_DPE}, BLOCK_DV: {BLOCK_DV}, BLOCK_N: {BLOCK}, BLOCK_H: {BLOCK_H}, MIN_BLOCK_KV: {_MIN_BLOCK_KV}"
    # )
    # print(f"    num_stages: {num_stages}, num_warps: {4}")
    # print(f"    extra_kargs: {extra_kargs}")
    # print(f"*******************************************************")
    _fwd_grouped_kernel_stage1_quant_fp4[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales,
        v_scales,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales.stride(0),
        k_scales.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


def _decode_grouped_att_m_fwd_quant_int4(
    q,
    k_buffer,  # Quantized INT4 (packed)
    v_buffer,  # Quantized INT4 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 32
    # For INT4, k_buffer is packed, so actual Lk is 2x the last dimension
    Lk = k_buffer.shape[-1] * 2
    Lv = v_buffer.shape[-1] * 2

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    # print(f"*************int4 quant kernel arguments:**************")
    # print(
    # f"    q: {q.shape}, k_buffer: {k_buffer.shape}, v_buffer: {v_buffer.shape}, k_scales_zeros: {k_scales_zeros.shape}, v_scales_zeros: {v_scales_zeros.shape}"
    # )
    # print(
    # f"    kv_indptr: {kv_indptr.shape}, kv_indices: {kv_indices.shape}, att_out: {att_out.shape}, att_lse: {att_lse.shape}"
    # )
    # print(
    # f"    num_kv_splits: {num_kv_splits}, max_kv_splits: {max_kv_splits}, sm_scale: {sm_scale}"
    # )
    # print(f"    logit_cap: {logit_cap}, xai_temperature_len: {xai_temperature_len}")
    # print(f"    Lk: {Lk}, Lv: {Lv}")
    # print(
    # f"    BLOCK: {BLOCK}, BLOCK_DMODEL: {BLOCK_DMODEL}, BLOCK_DPE: {BLOCK_DPE}, BLOCK_DV: {BLOCK_DV}, BLOCK_N: {BLOCK}, BLOCK_H: {BLOCK_H}, MIN_BLOCK_KV: {_MIN_BLOCK_KV}"
    # )
    # print(f"    num_stages: {num_stages}, num_warps: {4}")
    # print(f"    extra_kargs: {extra_kargs}")
    # print(f"*******************************************************")

    _fwd_grouped_kernel_stage1_quant_int4[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


def decode_attention_fwd_normal_quant(
    q,
    k_buffer,  # Quantized INT4/INT8
    v_buffer,  # Quantized INT4/INT8
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_dtype,  # "int4" or "int8"
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """
    Normal (MHA) attention forward with quantized (INT4/INT8/FP4) KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.

    Args:
        kv_dtype: Type of quantization - "int4", "int8", or "fp4"
    """
    # Stage 1: Compute attention scores and accumulate values
    if kv_dtype == "int8":
        _decode_att_m_fwd_quant_int8(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For INT8, v_buffer has correct dimension
        v_buf_for_stage2 = v_buffer
    elif kv_dtype == "int4":
        _decode_att_m_fwd_quant_int4(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For INT4, v_buffer is packed (half size), but stage2 needs full dimension
        # Create a dummy tensor with correct shape for stage2 to extract Lv
        v_buf_for_stage2 = o  # o has the correct output dimension
    elif kv_dtype == "fp4" or kv_dtype == torch.float4_e2m1fn_x2:
        _decode_att_m_fwd_quant_fp4(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,  # For FP4, these are scale exponents
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For FP4, v_buffer is packed (half size), but stage2 needs full dimension
        v_buf_for_stage2 = o
    else:
        raise ValueError(
            f"Unsupported kv_dtype: {kv_dtype}. Must be 'int4', 'int8', or 'fp4'."
        )

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buf_for_stage2,  # Use the correct dimension buffer
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


def decode_attention_fwd_grouped_quant(
    q,
    k_buffer,  # Quantized INT4/INT8/FP4
    v_buffer,  # Quantized INT4/INT8/FP4
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_dtype,  # "int4" or "int8" or "fp4"
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """
    Grouped (GQA/MQA) attention forward with quantized (INT4/INT8/FP4) KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.

    Args:
        kv_dtype: Type of quantization - "int4", "int8", or "fp4" or "torch.float4_e2m1fn_x2"
    """
    # Stage 1: Compute attention scores and accumulate values
    if kv_dtype == "int8":
        _decode_grouped_att_m_fwd_quant_int8(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For INT8, v_buffer has correct dimension
        v_buf_for_stage2 = v_buffer
    elif kv_dtype == "int4":
        _decode_grouped_att_m_fwd_quant_int4(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For INT4, v_buffer is packed (half size), but stage2 needs full dimension
        # Create a dummy tensor with correct shape for stage2 to extract Lv
        v_buf_for_stage2 = o  # o has the correct output dimension
    elif kv_dtype == "fp4" or kv_dtype == torch.float4_e2m1fn_x2:
        _decode_grouped_att_m_fwd_quant_fp4(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,  # For FP4, these are scale exponents
            v_scales_zeros,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
        # For FP4, v_buffer is packed (half size), but stage2 needs full dimension
        v_buf_for_stage2 = o
    else:
        raise ValueError(
            f"Unsupported kv_dtype: {kv_dtype}. Must be 'int4', 'int8', or 'fp4'."
        )

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buf_for_stage2,  # Use the correct dimension buffer
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


def decode_attention_fwd_quantized(
    q,
    k_buffer,  # Quantized INT4/INT8/FP4
    v_buffer,  # Quantized INT4/INT8/FP4
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_dtype,  # "int4" or "int8" or "fp4"
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """
    Attention forward with quantized (INT4/INT8/FP4) KV cache.
    Dispatches to the appropriate kernel based on attention type (MHA vs GQA/MQA).

    Args:
        kv_dtype: Type of quantization - "int4", "int8", or "fp4"
    """
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal_quant(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            kv_dtype,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped_quant(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            kv_dtype,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
