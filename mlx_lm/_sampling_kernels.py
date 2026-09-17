"""Stable BF16 radix sort and top-p masking with MLX's scan rounding order."""

import mlx.core as mx

from . import sample_utils

HEADER = """
inline uint ordered_key(ushort raw) {
    uint magnitude = uint(raw) & 0x7fffu;
    if (magnitude > 0x7f80u) return 0xffffu; // NaNs last, stable
    if (magnitude < 0x80u) return 0x8000u; // MLX Metal comparisons flush BF16 subnormals to signed zero
    return (uint(raw) & 0x8000u) ? ((~uint(raw)) & 0xffffu) : (uint(raw) ^ 0x8000u);
}
"""

HIST = mx.fast.metal_kernel(
    name="maple_radix_hist_rank",
    input_names=["bits", "ids"],
    output_names=["counts", "ranks"],
    header=HEADER,
    source="""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint lane = tid % 32u;
    uint warp = tid / 32u;
    uint i = group * TG + tid;
    bool valid = i < N;
    uint original = valid ? (FIRST ? i : uint(ids[i])) : 0u;
    uint digit = valid ? ((ordered_key(bits[original]) >> SHIFT) & 255u) : 0u;
    threadgroup ushort hist[TG / 32u * 256u];
    for (uint j = tid; j < TG / 32u * 256u; j += TG) hist[j] = 0;
    uint matches = uint((simd_vote::vote_t)simd_ballot(valid));
    for (uint b = 0; b < 8; ++b) {
        bool bit = (digit & (1u << b)) != 0u;
        uint votes = uint((simd_vote::vote_t)simd_ballot(bit));
        matches &= bit ? votes : ~votes;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && lane == ctz(matches)) hist[warp * 256u + digit] = popcount(matches);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        uint rank = popcount(matches & ((1u << lane) - 1u));
        for (uint w = 0; w < warp; ++w) rank += hist[w * 256u + digit];
        ranks[i] = ushort(rank);
    }
    for (uint digit_out = tid; digit_out < 256u; digit_out += TG) {
        uint total = 0;
        for (uint w = 0; w < TG / 32u; ++w) total += hist[w * 256u + digit_out];
        counts[digit_out * BLOCKS + group] = total;
    }
    """,
)

PREFIX = mx.fast.metal_kernel(
    name="maple_radix_prefix",
    input_names=["counts"],
    output_names=["prefix", "totals"],
    source="""
    uint tid = thread_position_in_threadgroup.x;
    uint digit = threadgroup_position_in_grid.y;
    uint lane = tid % 32u;
    uint warp = tid / 32u;
    uint start = tid * VALUES;
    uint values[VALUES];
    uint sum = 0;
    for (uint j = 0; j < VALUES; ++j) {
        values[j] = start + j < BLOCKS ? counts[digit * BLOCKS + start + j] : 0;
        sum += values[j];
    }
    uint offset = simd_prefix_exclusive_sum(sum);
    threadgroup uint warp_totals[8];
    if (lane == 31u) warp_totals[warp] = offset + sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 0; w < warp; ++w) offset += warp_totals[w];
    if (tid == 255u) totals[digit] = offset + sum;
    for (uint j = 0; j < VALUES; ++j) {
        if (start + j < BLOCKS) prefix[digit * BLOCKS + start + j] = offset;
        offset += values[j];
    }
    """,
)

SCATTER = mx.fast.metal_kernel(
    name="maple_radix_scatter",
    input_names=["bits", "ids", "ranks", "prefix", "totals"],
    output_names=["out"],
    header=HEADER,
    source="""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup uint base[256];
    if (tid < 32u) {
        uint values[8];
        uint sum = 0;
        for (uint j = 0; j < 8; ++j) {
            values[j] = totals[tid * 8u + j];
            sum += values[j];
        }
        uint start = simd_prefix_exclusive_sum(sum);
        for (uint j = 0; j < 8; ++j) {
            base[tid * 8u + j] = start;
            start += values[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i = thread_position_in_grid.x;
    if (i >= N) return;
    uint original = FIRST ? i : uint(ids[i]);
    uint digit = (ordered_key(bits[original]) >> SHIFT) & 255u;
    uint pos = base[digit] + prefix[digit * BLOCKS + i / TG] + ranks[i];
    out[pos] = original;
    """,
)


def _radix_argsort(x):
    tg = 512
    n = x.size
    bits = x.view(mx.uint16)
    blocks = (n + tg - 1) // tg
    ids = bits
    for shift in (0, 8):
        constants = [
            ("N", n),
            ("TG", tg),
            ("BLOCKS", blocks),
            ("FIRST", shift == 0),
            ("SHIFT", shift),
        ]
        counts, ranks = HIST(
            inputs=[bits, ids],
            template=constants,
            grid=(blocks * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(256, blocks), (n,)],
            output_dtypes=[mx.uint16, mx.uint16],
        )
        prefix, totals = PREFIX(
            inputs=[counts],
            template=[("BLOCKS", blocks), ("VALUES", (blocks + 255) // 256)],
            grid=(256, 256, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[counts.shape, (256,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        ids = SCATTER(
            inputs=[bits, ids, ranks, prefix, totals],
            template=constants,
            grid=((n + 255) // 256 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[mx.uint32],
        )[0]
    return ids.reshape(x.shape)


# Scan arithmetic follows MLX 0.32's contiguous_scan (Apple, 2023-2024, MIT).
# Each 4096-value chunk is independent until the final carry additions.
CDF_PARTIAL = mx.fast.metal_kernel(
    name="maple_top_p_partial",
    input_names=["logprobs", "indices"],
    output_names=["local", "thread_prefix", "group_prefix", "summary"],
    source="""
    uint tid = thread_position_in_threadgroup.x;
    uint chunk = threadgroup_position_in_grid.x;
    uint lane = tid % 32u;
    uint sg = tid / 32u;
    uint start = chunk * 4096u + tid * 4u;
    T values[4];
    for (uint j = 0; j < 4; ++j) {
        uint i = start + j;
        values[j] = i < N ? T(metal::precise::exp(logprobs[indices[i]])) : T(0);
    }
    for (uint j = 1; j < 4; ++j) values[j] = T(values[j] + values[j - 1]);
    T prev = simd_prefix_exclusive_sum(values[3]);
    threadgroup T group_sums[32];
    if (lane == 31u) group_sums[sg] = T(prev + values[3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) group_sums[lane] = simd_prefix_exclusive_sum(group_sums[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 4; ++j) if (start + j < N) local[start + j] = values[j];
    thread_prefix[chunk * 1024u + tid] = prev;
    if (lane == 0u) group_prefix[chunk * 32u + sg] = group_sums[sg];
    if (tid == 1023u) {
        summary[chunk * 3u] = values[3];
        summary[chunk * 3u + 1u] = group_sums[31];
        summary[chunk * 3u + 2u] = prev;
    }
    """,
)

CDF_MASK = mx.fast.metal_kernel(
    name="maple_top_p_scan_mask",
    input_names=[
        "local",
        "thread_prefix",
        "group_prefix",
        "summary",
        "logprobs",
        "indices",
        "threshold",
    ],
    output_names=["out"],
    source="""
    uint tid = thread_position_in_threadgroup.x;
    uint chunk = threadgroup_position_in_grid.x;
    T carry = T(0);
    if (tid % 32u == 0u) {
        for (uint c = 0; c < chunk; ++c) {
            carry = T(summary[c * 3u] + carry);
            carry = T(carry + summary[c * 3u + 1u]);
            carry = T(carry + summary[c * 3u + 2u]);
        }
    }
    carry = simd_broadcast(carry, 0);
    T prev = thread_prefix[chunk * 1024u + tid];
    T group = group_prefix[chunk * 32u + tid / 32u];
    for (uint j = 0; j < 4; ++j) {
        uint i = chunk * 4096u + tid * 4u + j;
        if (i < N) {
            T val = T(local[i] + carry);
            val = T(val + group);
            val = T(val + prev);
            uint original = indices[i];
            out[original] = val > threshold[0] ? logprobs[original] : T(-INFINITY);
        }
    }
    """,
)


@mx.compile
def _apply_top_p_radix(logprobs, top_p):
    indices = _radix_argsort(logprobs)
    # Match stock MLX's BF16 coercion of the Python threshold.
    threshold = mx.array([1 - top_p], dtype=logprobs.dtype)
    chunks = (logprobs.size + 4095) // 4096
    launch = dict(
        template=[("N", logprobs.size), ("T", logprobs.dtype)],
        grid=(chunks * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
    )
    partials = CDF_PARTIAL(
        inputs=[logprobs, indices],
        output_shapes=[logprobs.shape, (chunks, 1024), (chunks, 32), (chunks, 3)],
        output_dtypes=[logprobs.dtype] * 4,
        **launch,
    )
    return CDF_MASK(
        inputs=partials + [logprobs, indices, threshold],
        output_shapes=[logprobs.shape],
        output_dtypes=[logprobs.dtype],
        **launch,
    )[0]


_ready = None


def _probe():
    try:
        n = 151936
        # Include all BF16 bit patterns, signed zeros, subnormals, NaNs, and
        # repeated values. Multiplication permutes the uint16 keys bijectively.
        bits = (mx.arange(n, dtype=mx.uint32) * 40503).astype(mx.uint16)
        x = bits.view(mx.bfloat16)[None]
        if not bool(mx.array_equal(_radix_argsort(x), mx.argsort(x))):
            return False
        # Private keys leave the sampler's random-number stream untouched.
        x = mx.random.normal((1, n), key=mx.random.key(17)).astype(mx.bfloat16)
        sparse = mx.where(mx.arange(n)[None] % 9 == 0, x, -float("inf"))
        for values in (x, sparse):
            logprobs = values - mx.logsumexp(values, axis=-1, keepdims=True)
            if not bool(
                mx.array_equal(
                    _apply_top_p_radix(logprobs, 0.95),
                    sample_utils.apply_top_p(logprobs, 0.95),
                )
            ):
                return False
        return True
    except Exception:
        return False


def apply_top_p_fast(logprobs, top_p):
    """Use the verified shape only; latch the stock fallback after a failed probe."""
    global _ready
    if (
        mx.__version__ != "0.32.0"
        or mx.default_device() != mx.gpu
        or logprobs.dtype != mx.bfloat16
        or logprobs.shape != (1, 151936)
    ):
        return sample_utils.apply_top_p(logprobs, top_p)
    if _ready is None:
        _ready = _probe()
    return (
        _apply_top_p_radix(logprobs, top_p)
        if _ready
        else sample_utils.apply_top_p(logprobs, top_p)
    )
