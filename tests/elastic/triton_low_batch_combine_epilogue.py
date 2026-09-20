#!/usr/bin/env python3
"""Triton low-batch replacement for the DeepEP v2 combine epilogue.

This is specialized for the Kimi K3 decode shape used by the repro:

    local hidden = 3584 BF16 values
    top-k        = 16
    one token    = one token slot per received Top-K route

The receive buffer is the DeepEP non-expanded layout. It is interpreted as
[topk_slot, token_in_slot, token_record] where each token record contains:

    hidden:       hidden BF16 values
    top-k indices: topk int32 values
    top-k weights: topk float32 values

The kernel only reads the hidden portion and uses combined_topk_idx to skip
invalid/padded routes. It assigns one Triton program to each
(output_token, hidden_slice) pair. For HIDDEN=3584 and BLOCK_H=512 this is
seven one-warp programs per output token.

This file is intentionally independent of the DeepEP C++ extension. It is
useful for an isolated correctness/performance comparison and can be captured
inside a CUDA graph by the caller.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


HIDDEN = 3584
TOPK = 16
BLOCK_H = 512
NUM_HIDDEN_STAGES = (HIDDEN + BLOCK_H - 1) // BLOCK_H
NUM_MAX_TOKENS_PER_RANK = 1
TMA_ALIGNMENT_BYTES = 128


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def token_record_bytes(
    hidden: int = HIDDEN,
    topk: int = TOPK,
) -> int:
    """Return the DeepEP v2 non-expanded token-record size in bytes."""

    hidden_bytes = _align(hidden * 2, TMA_ALIGNMENT_BYTES)
    metadata_bytes = _align(topk * (4 + 4), TMA_ALIGNMENT_BYTES)
    return hidden_bytes + metadata_bytes


@triton.jit
def _low_batch_combine_kernel(
    recv_ptr,
    combined_topk_idx_ptr,
    output_ptr,
    num_combined_tokens,
    hidden,
    slot_stride_bf16,
    token_stride_bf16,
    BLOCK_H: tl.constexpr,
    TOPK: tl.constexpr,
):
    task = tl.program_id(0)
    num_stages = tl.cdiv(hidden, BLOCK_H)
    token_idx = task // num_stages
    stage_idx = task % num_stages

    hidden_offsets = stage_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden

    reduced = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for route_idx in tl.static_range(TOPK):
        route_idx_value = tl.load(
            combined_topk_idx_ptr + token_idx * TOPK + route_idx,
            mask=token_idx < num_combined_tokens,
            other=-1,
        )
        route_valid = route_idx_value >= 0
        route_ptr = (
            recv_ptr
            + route_idx * slot_stride_bf16
            + token_idx * token_stride_bf16
            + hidden_offsets
        )
        reduced += tl.load(
            route_ptr,
            mask=hidden_mask & route_valid,
            other=0.0,
        ).to(tl.float32)

    tl.store(
        output_ptr + token_idx * hidden + hidden_offsets,
        reduced.to(tl.bfloat16),
        mask=hidden_mask & (token_idx < num_combined_tokens),
    )


def low_batch_combine(
    recv_buffer: torch.Tensor,
    combined_topk_idx: torch.Tensor,
    *,
    num_combined_tokens: int | None = None,
    hidden: int = HIDDEN,
    topk: int = TOPK,
    num_max_tokens_per_rank: int = NUM_MAX_TOKENS_PER_RANK,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the low-batch Triton combine.

    recv_buffer must be a contiguous raw DeepEP receive allocation, either as
    torch.uint8 or torch.bfloat16. The hidden data for route r and token t
    starts at:

        recv_buffer_bf16[r * slot_stride_bf16 + t * token_stride_bf16]

    combined_topk_idx has shape [num_combined_tokens, topk]. A negative entry
    marks a padded route. In the non-expanded layout, route r maps directly
    to receive-buffer slot r.
    """

    if not recv_buffer.is_cuda or not combined_topk_idx.is_cuda:
        raise ValueError("recv_buffer and combined_topk_idx must be CUDA tensors")
    if not recv_buffer.is_contiguous() or not combined_topk_idx.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if recv_buffer.dtype not in (torch.uint8, torch.bfloat16):
        raise TypeError("recv_buffer must be torch.uint8 or torch.bfloat16")
    if combined_topk_idx.ndim != 2 or combined_topk_idx.shape[1] != topk:
        raise ValueError("combined_topk_idx must have shape [tokens, topk]")
    if hidden != HIDDEN or topk != TOPK:
        raise ValueError("this prototype is specialized for hidden=3584, topk=16")
    if hidden % BLOCK_H != 0:
        raise ValueError("hidden must be divisible by BLOCK_H")
    if num_max_tokens_per_rank < 1:
        raise ValueError("num_max_tokens_per_rank must be positive")

    if recv_buffer.dtype == torch.uint8:
        if recv_buffer.numel() % 2:
            raise ValueError("uint8 recv_buffer must contain an even byte count")
        recv_bf16 = recv_buffer.view(torch.bfloat16)
    else:
        recv_bf16 = recv_buffer

    if num_combined_tokens is None:
        num_combined_tokens = int(combined_topk_idx.shape[0])
    if not 0 < num_combined_tokens <= combined_topk_idx.shape[0]:
        raise ValueError("invalid num_combined_tokens")

    record_bytes = token_record_bytes(hidden, topk)
    token_stride_bf16 = record_bytes // 2
    slot_stride_bf16 = num_max_tokens_per_rank * token_stride_bf16

    if output is None:
        output = torch.empty(
            (num_combined_tokens, hidden),
            dtype=torch.bfloat16,
            device=recv_buffer.device,
        )
    elif (
        output.shape != (num_combined_tokens, hidden)
        or output.dtype != torch.bfloat16
        or output.device != recv_buffer.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with shape [tokens, 3584]")
    grid = (num_combined_tokens * NUM_HIDDEN_STAGES,)
    _low_batch_combine_kernel[grid](
        recv_bf16,
        combined_topk_idx,
        output,
        num_combined_tokens,
        hidden,
        slot_stride_bf16,
        token_stride_bf16,
        BLOCK_H=BLOCK_H,
        TOPK=topk,
        num_warps=1,
    )
    return output


def _reference(
    recv_buffer: torch.Tensor,
    combined_topk_idx: torch.Tensor,
    *,
    num_combined_tokens: int,
    hidden: int = HIDDEN,
    topk: int = TOPK,
    num_max_tokens_per_rank: int = NUM_MAX_TOKENS_PER_RANK,
) -> torch.Tensor:
    """Reference implementation for the standalone smoke test."""

    raw = recv_buffer.view(torch.bfloat16)
    record_bf16 = token_record_bytes(hidden, topk) // 2
    slot_stride_bf16 = num_max_tokens_per_rank * record_bf16
    output = torch.zeros(
        (num_combined_tokens, hidden),
        dtype=torch.float32,
        device=recv_buffer.device,
    )
    for token_idx in range(num_combined_tokens):
        for route_idx in range(topk):
            if int(combined_topk_idx[token_idx, route_idx]) >= 0:
                begin = route_idx * slot_stride_bf16 + token_idx * record_bf16
                output[token_idx] += raw[begin : begin + hidden].float()
    return output.to(torch.bfloat16)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 1
    record_bytes = token_record_bytes()
    recv = torch.zeros(
        (TOPK * NUM_MAX_TOKENS_PER_RANK * record_bytes,),
        dtype=torch.uint8,
        device=device,
    )
    recv_bf16 = recv.view(torch.bfloat16)
    record_bf16 = record_bytes // 2
    for route_idx in range(TOPK):
        begin = route_idx * record_bf16
        recv_bf16[begin : begin + HIDDEN].fill_(route_idx + 1)

    topk_idx = torch.arange(TOPK, device=device, dtype=torch.int32).view(1, TOPK)
    topk_idx[0, -1] = -1

    result = low_batch_combine(recv, topk_idx, num_combined_tokens=num_tokens)
    expected = _reference(recv, topk_idx, num_combined_tokens=num_tokens)
    torch.testing.assert_close(result, expected)
    torch.cuda.synchronize()
    print(
        f"ok: hidden={HIDDEN} topk={TOPK} stages={NUM_HIDDEN_STAGES} "
        f"record_bytes={record_bytes} "
        f"max_error={(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    main()
