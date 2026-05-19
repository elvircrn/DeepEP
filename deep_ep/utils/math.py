import torch
from typing import Tuple


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def safe_div(a, b) -> float:
    try:
        return a / b
    except ZeroDivisionError as e:
        if a == 0:
            return 0
        else:
            raise


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@torch.compile(dynamic=True)
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)


@torch.compile(dynamic=True)
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.shape[0], -1, 128)
    x_scales = x_scales.view(x_fp8.shape[0], -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()


@torch.compile(dynamic=True)
def per_token_cast_to_nvfp4(x: torch.Tensor, use_fp8_sf: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align(n, 16)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 16)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    x_scaled = (x_padded_view * (6.0 / x_amax.unsqueeze(2))).float()
    x_e2m1 = _float_to_e2m1_packed(x_scaled.view(m, aligned_n))[:, :n // 2]
    scales = (x_amax / 6.0).view(m, -1)
    if use_fp8_sf:
        scales = _pack_fp8_scales(scales)
    return x_e2m1.contiguous(), scales


def _pack_fp8_scales(scales: torch.Tensor) -> torch.Tensor:
    m, num_scales = scales.shape
    aligned_num = align(num_scales, 4)
    if aligned_num > num_scales:
        scales = torch.nn.functional.pad(scales, (0, aligned_num - num_scales), mode='constant', value=0)
    scales_fp8 = scales.to(torch.float8_e4m3fn)
    scales_u8 = scales_fp8.view(torch.uint8).to(torch.int32)
    scales_packed = scales_u8.view(m, -1, 4)
    result = scales_packed[:, :, 0] | (scales_packed[:, :, 1] << 8) | \
             (scales_packed[:, :, 2] << 16) | (scales_packed[:, :, 3] << 24)
    return result[:, :num_scales // 4].contiguous() if num_scales < aligned_num else result


def _float_to_e2m1_packed(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.shape[1] % 2 == 0
    m, n = x.shape
    abs_x = x.abs()
    sign = (x < 0).to(torch.uint8)
    abs_x_log2 = torch.log2(abs_x.clamp(min=2**-10))
    exp = torch.floor(abs_x_log2 + 1).clamp(0, 3).to(torch.uint8)
    mant = ((abs_x_log2 + 1 - exp.float()) > 0.5).to(torch.uint8)
    nibbles = (sign << 3) | (exp << 1) | mant
    even = nibbles[:, 0::2]
    odd = nibbles[:, 1::2]
    return (even | (odd << 4)).to(torch.uint8)


@torch.compile(dynamic=True)
def per_token_cast_back_nvfp4(x_nvfp4: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    if x_nvfp4.numel() == 0:
        return torch.empty(x_nvfp4.shape[0], x_nvfp4.shape[1] * 2,
                           dtype=torch.bfloat16, device=x_nvfp4.device)
    assert x_nvfp4.dim() == 2
    m, packed_n = x_nvfp4.shape
    n = packed_n * 2
    num_scale_groups = n // 16
    if x_scales.dtype == torch.int32:
        x_scales = _unpack_fp8_scales(x_scales, num_scale_groups)
    aligned_n = align(n, 16)
    x_unpacked = _e2m1_packed_to_float(x_nvfp4)
    if aligned_n > n:
        x_unpacked = torch.nn.functional.pad(x_unpacked, (0, aligned_n - n), mode='constant', value=0)
    x_grouped = x_unpacked.view(m, -1, 16)
    x_scales = x_scales.view(m, -1, 1)
    return (x_grouped * x_scales).view(m, aligned_n).to(torch.bfloat16)[:, :n].contiguous()


def _unpack_fp8_scales(packed: torch.Tensor, num_scales: int) -> torch.Tensor:
    m = packed.shape[0]
    b0 = packed & 0xFF
    b1 = (packed >> 8) & 0xFF
    b2 = (packed >> 16) & 0xFF
    b3 = (packed >> 24) & 0xFF
    unpacked = torch.stack([b0, b1, b2, b3], dim=-1).view(m, -1)[:, :num_scales]
    return unpacked.to(torch.uint8).view(torch.float8_e4m3fn).float()


def _e2m1_packed_to_float(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    m, packed_n = x.shape
    x_int = x.to(torch.int32)
    lo = x_int & 0x0F
    hi = (x_int >> 4) & 0x0F
    nibbles = torch.stack([lo, hi], dim=-1).view(m, packed_n * 2)
    sign = ((nibbles >> 3) & 1).float()
    exp = ((nibbles >> 1) & 3).float()
    mant = (nibbles & 1).float()
    vals = torch.where(exp == 0, mant * 0.5, torch.pow(2.0, exp - 1) * (1.0 + mant * 0.5))
    return vals * (1 - 2 * sign)


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def hash_tensor(t: torch.Tensor) -> int:
    return t.view(torch.int).sum().item()


def hash_tensors(*tensors) -> int:
    value = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            value ^= hash_tensors(*t)
        elif t is not None and isinstance(t, torch.Tensor):
            value ^= hash_tensor(t)
    return value


def count_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
