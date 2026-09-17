# Copyright © 2026 DeepGrove AI.

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

# The native extension uses the MLX 0.32.0 C++ ABI.
_maple_native = None
if mx.__version__ == "0.32.0":
    try:
        from mlx_lm_maple import _maple_native
    except ImportError:
        try:
            from mlx_lm import _maple_native
        except ImportError:
            pass

# Absolute imports so this file also works standalone when shipped inside a
# checkpoint and loaded via the config's `model_file` (trust_remote_code).
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache as _KVCache
from mlx_lm.models.cache import RotatingKVCache as _RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

# SwiGLU clamp for the MoE experts only (the dense MapleMLP is unclamped);
# part of the trained forward pass, not an optional guard.
MLP_CLAMP = 7.0


@partial(mx.compile, shapeless=True)
def clamped_swiglu(gate, x):
    # Python floats, not 0-d arrays, so bf16 activations stay bf16.
    return nn.silu(mx.minimum(gate, MLP_CLAMP)) * mx.clip(x, -MLP_CLAMP, MLP_CLAMP)


def _matches(fast, reference):
    return _exact_result(fast, reference) is not None


def _exact_result(fast, reference, *, bitwise=False):
    """Probe an array or tuple of arrays; return None on mismatch or failure."""
    try:
        got, want = fast(), reference()
        mx.eval(got, want)
        actual = (got,) if isinstance(got, mx.array) else got
        expected = (want,) if isinstance(want, mx.array) else want
        if len(actual) == len(expected) and all(
            g.shape == w.shape
            and g.dtype == w.dtype
            and bool(
                mx.array_equal(
                    g.view(mx.uint8) if bitwise else g,
                    w.view(mx.uint8) if bitwise else w,
                )
            )
            for g, w in zip(actual, expected)
        ):
            return got
    except Exception:
        pass
    return None


class MapleRMSNorm(nn.Module):
    """RMSNorm with the weight multiply in float32.

    The reference rounds only the finished product; mx.fast.rms_norm rounds
    the normalized activation first (~1% per element). Float32 inputs to the
    same kernel reproduce the reference bit-for-bit.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32), self.eps
        ).astype(x.dtype)


@lru_cache(maxsize=None)
def _make_add_rms_norm_kernel(eps, aggregate=False):
    """Residual/RMSNorm with optional ordered, BF16-rounded expert reduction."""
    residual = "float v = (float)x[j] + (float)r[j];"
    store_h = "h_out[j] = vb;"
    store_hn = "hn_out[j] = (T_)((float)w[j] * (hb[i] * scale));"
    if aggregate:
        residual = """
            float agg = 0.0f;
            {
                #pragma clang fp contract(off)
                for (uint e = 0; e < 8; ++e) {
                    agg = agg + (float)r[e * N + j] * scores[e];
                }
            }
            float v = (float)x[j] + (float)(T_)agg;
        """
        guard = "if (j / (N / 8u) == threadgroup_position_in_grid.x)"
        store_h = f"{guard} {{ {store_h} }}"
        store_hn = f"{guard} {{ {store_hn} }}"
    source = """
        uint tid = thread_position_in_threadgroup.x;
        constexpr uint N = DIM;
        constexpr uint PT = 4u;
        float hb[PT];
        float ss = 0.0f;
        for (uint i = 0; i < PT; ++i) {
            uint j = tid * PT + i;
            RESIDUAL
            T_ vb = (T_)v;              // one rounding, same as a bf16 add
            STORE_H
            hb[i] = (float)vb;          // norm sees the rounded stream
            ss += hb[i] * hb[i];
        }
        ss = simd_sum(ss);
        // Match MLX 0.32.0 rms_single_row: four values per lane,
        // then the same two-level SIMD reduction and precise reciprocal root.
        threadgroup float sums[32];
        threadgroup float inv_mean[1];
        uint sg = tid / 32u;
        uint lane = tid % 32u;
        if (sg == 0u) sums[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) sums[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum(sums[lane]);
            if (lane == 0u) inv_mean[0] = metal::precise::rsqrt(tot / (float)N + EPS_);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = inv_mean[0];
        for (uint i = 0; i < PT; ++i) {
            uint j = tid * PT + i;
            STORE_HN
        }
    """
    source = (
        source.replace("RESIDUAL", residual)
        .replace("STORE_HN", store_hn)
        .replace("STORE_H", store_h)
        .replace("EPS_", f"{eps:.10e}f")
    )
    tag = f"{eps:.3e}".replace(".", "_").replace("-", "m").replace("+", "p")
    return mx.fast.metal_kernel(
        name=f"maple_{'aggregate_' if aggregate else ''}add_rms_norm_{tag}",
        input_names=["x", "r", "scores", "w"] if aggregate else ["x", "r", "w"],
        output_names=["h_out", "hn_out"],
        source=source,
    )


def _add_rms_norm(h, r, w, eps):
    return _make_add_rms_norm_kernel(eps)(
        inputs=[h.reshape(-1), r.reshape(-1), w],
        template=[("T_", h.dtype), ("DIM", h.shape[-1])],
        grid=(h.shape[-1] // 4, 1, 1),
        threadgroup=(h.shape[-1] // 4, 1, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[h.dtype, h.dtype],
    )


def _add_rms_norm_ok(dim, dtype, w, eps):
    if mx.__version__ != "0.32.0" or dim % 128 or dim > 4096 or dtype != w.dtype:
        return False
    x = mx.random.normal((1, 1, dim), key=mx.random.key(0)).astype(dtype)
    r = mx.random.normal((1, 1, dim), key=mx.random.key(1)).astype(dtype)
    return _matches(
        lambda: _add_rms_norm(x, r, w, eps),
        lambda: (
            x + r,
            mx.fast.rms_norm(
                (x + r).astype(mx.float32), w.astype(mx.float32), eps
            ).astype(dtype),
        ),
    )


@mx.compile
def _aggregate_add_rms_norm(h, r, scores, w, eps):
    return _make_add_rms_norm_kernel(eps, aggregate=True)(
        inputs=[h.reshape(-1), r.reshape(-1), scores.reshape(-1), w],
        template=[("T_", h.dtype), ("DIM", h.shape[-1])],
        grid=(8 * h.shape[-1] // 4, 1, 1),
        threadgroup=(h.shape[-1] // 4, 1, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[h.dtype, h.dtype],
    )


def _aggregate_add_rms_norm_ok(dim, dtype, w, eps):
    """Require bitwise agreement with the unfused operations on live weights."""
    if mx.__version__ != "0.32.0" or dim % 128 or dim > 4096 or dtype != w.dtype:
        return False
    try:
        for seed in range(4):
            h = mx.random.normal((1, 1, dim), key=mx.random.key(seed)).astype(dtype)
            y = mx.random.normal((1, 1, 8, dim), key=mx.random.key(10 + seed)).astype(
                dtype
            )
            scores = mx.softmax(
                mx.random.normal((1, 1, 8), key=mx.random.key(20 + seed))
            )
            got = _aggregate_add_rms_norm(h, y, scores, w, eps)
            residual = h + aggregate_expert_outputs(y, scores)
            want = (
                residual,
                mx.fast.rms_norm(
                    residual.astype(mx.float32), w.astype(mx.float32), eps
                ).astype(dtype),
            )
            mx.eval(got, want)
            if not all(bool(mx.array_equal(a, b)) for a, b in zip(got, want)):
                return False
    except Exception:
        return False
    return True


# Inlined rather than imported from switch_layers: those helpers are private
# (underscore-prefixed), and this file must keep loading against whatever
# mlx-lm a user has installed when it ships inside a checkpoint.
def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "maple"
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 140000
    vocab_size: int = 151936
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    use_qk_norm: bool = True
    use_bias: bool = False
    tie_word_embeddings: bool = False
    # FlashHead metadata written by `mlx_lm.ternary --flash-head`. The exact
    # lm_head is the default; opt in to the approximate fast head with
    # mlx_lm.load(..., model_config={"use_flash_head": True}).
    flash_head: Optional[dict] = None
    use_flash_head: bool = False
    # Populated from the checkpoint's config; sanitize() reads group_size from
    # it to expand row-scale (`row_alpha`) ternary tensors.
    quantization: Optional[dict] = None

    def __post_init__(self):
        # Single source of truth for per-layer attention types: attention
        # (RoPE/NoPE), masks, and caches all read this resolved list.
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


_rope_inv_freq_kernel = mx.fast.metal_kernel(
    name="maple_rope_inv_freq",
    input_names=["log_base"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < HALF) out[i] = metal::exp2(-(float(i) / float(HALF)) * log_base[0]);
    """,
)


def _make_qk_norm_rope_kernel():
    """Fused per-head RMSNorm + partial RoPE for single-token decode.

    One dispatch replaces q_norm, k_norm and two rope calls. One simdgroup per
    head: normalize head_dim values, scale by the head's norm weight, and
    rotate the first ROPE_DIM dims (non-traditional pairing i, i+R/2) at the
    given position. NoPE layers pass ROPE_DIM=0.
    """
    source = r"""
        uint head = thread_position_in_grid.y;
        uint lane = thread_position_in_grid.x;

        constexpr int per_lane = HEAD_DIM / 32;
        const device T_* xh = x + head * HEAD_DIM;
        device T_* oh = out + head * HEAD_DIM;

        if (head >= (uint)NQK) {               // v heads: passthrough copy
            for (int i = 0; i < per_lane; ++i) {
                int j = lane * per_lane + i;
                oh[j] = xh[j];
            }
            return;
        }
        const device T_* wh = head < (uint)NQ ? qw : kw;

        float ss = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            float v = (float)xh[lane * per_lane + i];
            ss += v * v;
        }
        ss = simd_sum(ss);
        float pos = pos_eps[0];
        float eps = pos_eps[1];
        float scale = metal::precise::rsqrt(ss / HEAD_DIM + eps);

        for (int i = 0; i < per_lane; ++i) {
            int j = lane * per_lane + i;
            float v = (float)(T_)((float)wh[j] * ((float)xh[j] * scale));
            if (ROPE_DIM > 0 && j < ROPE_DIM) {
                constexpr int rhalf = ROPE_DIM > 0 ? ROPE_DIM / 2 : 1;
                int p = j < rhalf ? j : j - rhalf;
                float inv = metal::exp2(-(float(p) / float(rhalf)) * pos_eps[2]);
                float theta = pos * inv;
                float c = metal::fast::cos(theta);
                float s = metal::fast::sin(theta);
                int j2 = j < rhalf ? j + rhalf : j - rhalf;
                float u = (float)(T_)((float)wh[j2] * ((float)xh[j2] * scale));
                v = j < rhalf ? (v * c - u * s) : (u * s + v * c);
            }
            oh[j] = (T_)v;
        }
    """
    return mx.fast.metal_kernel(
        name="maple_qk_norm_rope",
        input_names=["x", "qw", "kw", "pos_eps"],
        output_names=["out"],
        source=source,
    )


_qk_norm_rope_kernel = _make_qk_norm_rope_kernel()


# Affine dot-product evaluation order follows MLX v0.32.0 quantized.h
# (Copyright Apple Inc., MIT license). Keep its fp32 accumulation and bf16
# rounding boundaries, including those within the fused SwiGLU activation.
_ROW_QDOT_HEADER = r"""
float maple_qdot(const device uchar* w, const thread float* x, float scale, float bias, float sum) {
    float accum = 0;
    for (int i=0;i<4;++i) {
        accum += (x[4*i]*(w[i]&0x03) + x[4*i+1]*(w[i]&0x0c) +
                  x[4*i+2]*(w[i]&0x30) + x[4*i+3]*(w[i]&0xc0));
    }
    return scale*accum+sum*bias;
}
"""

_row_qmv_kernel = mx.fast.metal_kernel(
    name="maple_row_qmv_exact",
    input_names=["x", "weight", "scales", "biases", "ids"],
    output_names=["out"],
    header=_ROW_QDOT_HEADER,
    source=r"""
    uint lid=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    uint slot=threadgroup_position_in_grid.y;
    uint expert=GATHER?ids[slot]:0;
    uint row0=threadgroup_position_in_grid.x*8+sg*4;
    const device uchar* w=(const device uchar*)weight+expert*N*(K/4);
    const device bfloat16_t* sc=scales+expert*N*(GROUPED?K/128:1);
    const device bfloat16_t* bi=biases+expert*N*(GROUPED?K/128:1);
    const device bfloat16_t* xv=x+(SPLIT?slot*K:0);
    float result[4]={0};
    for(int k=0;k<K;k+=512) {
        const device bfloat16_t* xx=xv+k+lid*16;
        float xt[16],sum=0;
        for(int i=0;i<16;i+=4) {
            sum+=xx[i]+xx[i+1]+xx[i+2]+xx[i+3];
            xt[i]=xx[i];xt[i+1]=xx[i+1]/4.0f;
            xt[i+2]=xx[i+2]/16.0f;xt[i+3]=xx[i+3]/64.0f;
        }
        for(int r=0;r<4;++r) {
            uint row=row0+r;
            uint si=GROUPED?row*(K/128)+(k+lid*16)/128:row;
            result[r]+=maple_qdot(w+row*(K/4)+k/4+lid*4,xt,(float)sc[si],(float)bi[si],sum);
        }
    }
    for(int r=0;r<4;++r) {
        result[r]=simd_sum(result[r]);
        if(lid==0)out[slot*N+row0+r]=(bfloat16_t)result[r];
    }
""",
)

_row_up_kernel = mx.fast.metal_kernel(
    name="maple_row_up_swiglu_exact",
    input_names=["x", "weight", "scales", "biases", "ids"],
    output_names=["out"],
    header=_ROW_QDOT_HEADER,
    source=r"""
    constexpr int R = 4, SGS = 2;
    uint lid=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup;
    uint slot=threadgroup_position_in_grid.y;
    uint expert=ids[slot];
    uint first=threadgroup_position_in_grid.x*(SGS*R)+sg*R;
    const device uchar* w=(const device uchar*)weight+expert*1024*512;
    const device bfloat16_t* sc=scales+expert*1024*(GROUPED?16:1);
    const device bfloat16_t* bi=biases+expert*1024*(GROUPED?16:1);
    float up[R]={0},gate[R]={0};
    for(int k=0;k<2048;k+=512) {
        const device bfloat16_t* xx=x+k+lid*16;
        float xt[16];
        float sum=0;
        for(int i=0;i<16;i+=4) {
            sum+=xx[i]+xx[i+1]+xx[i+2]+xx[i+3];
            xt[i]=xx[i];xt[i+1]=xx[i+1]/4.0f;
            xt[i+2]=xx[i+2]/16.0f;xt[i+3]=xx[i+3]/64.0f;
        }
        for(int r=0;r<R;++r) {
            uint row=first+r;
            uint wi=row*512+k/4+lid*4;
            uint si=GROUPED?row*16+(k+lid*16)/128:row;
            up[r]+=maple_qdot(w+wi,xt,(float)sc[si],(float)bi[si],sum);
            gate[r]+=maple_qdot(w+wi+512*512,xt,(float)sc[si+512*(GROUPED?16:1)],(float)bi[si+512*(GROUPED?16:1)],sum);
        }
    }
    for(int r=0;r<R;++r) {
        up[r]=simd_sum(up[r]);gate[r]=simd_sum(gate[r]);
        if(lid==0) {
            bfloat16_t u=(bfloat16_t)up[r],g=(bfloat16_t)gate[r];
            g=g<(bfloat16_t)7.0f?g:(bfloat16_t)7.0f;
            u=u>(bfloat16_t)(-7.0f)?u:(bfloat16_t)(-7.0f);
            u=u<(bfloat16_t)7.0f?u:(bfloat16_t)7.0f;
            auto t=1/(1+metal::exp(metal::abs(g)));
            bfloat16_t sig=(g<0)?t:1-t;
            bfloat16_t silu=g*sig;
            out[slot*512+first+r]=silu*u;
        }
    }
""",
)


def _row_quantized_metadata(p):
    """Use compact row metadata with snapshots, live group metadata without."""
    if (
        mx.__version__ != "0.32.0"
        or type(p) not in (nn.QuantizedLinear, QuantizedSwitchLinear)
        or p.bits != 2
        or p.group_size != 128
        or p.mode != "affine"
        or p.weight.dtype != mx.uint32
        or p.scales.dtype != mx.bfloat16
        or p.get("biases") is None
        or p.biases.dtype != mx.bfloat16
    ):
        return None
    native = hasattr(_maple_native, "ArraySnapshot")
    sources = (p.weight, p.scales, p.biases) if native else None
    state = p.get("_maple_row_state")
    if (
        native
        and state is not None
        and state.get("sources") is not None
        and state["sources"].matches(sources)
    ):
        return state if state["supported"] else None
    k, n = p.scales.shape[-1] * 128, p.weight.shape[-2]
    if (
        k not in (512, 2048)
        or n % 8
        or p.weight.shape[-1] != k // 16
        or p.scales.shape != (*p.weight.shape[:-1], k // 128)
        or p.biases.shape != p.scales.shape
    ):
        return None
    if not native:
        if (
            state is None
            or state.get("sources") is not None
            or (state.get("k"), state.get("n")) != (k, n)
        ):
            state = dict(
                k=k, n=n, qmv_ok=None, zero=mx.array([0], mx.uint32), sources=None
            )
            p["_maple_row_state"] = state
        # Live arrays need no snapshot: no derived weight values are cached.
        state.update(scales=p.scales, biases=p.biases)
        return state
    state = {
        "sources": _maple_native.ArraySnapshot(sources),
        "supported": False,
        "qmv_ok": None,
    }
    p["_maple_row_state"] = state
    sc, bi = p.scales[..., :1], p.biases[..., :1]
    same = mx.all(p.scales.view(mx.uint16) == sc.view(mx.uint16)) & mx.all(
        p.biases.view(mx.uint16) == bi.view(mx.uint16)
    )
    mx.eval(same)
    if not bool(same):
        return None
    sc, bi = mx.contiguous(sc[..., 0]), mx.contiguous(bi[..., 0])
    zero = mx.array([0], mx.uint32)
    mx.eval(sc, bi, zero)
    state.update(supported=True, scales=sc, biases=bi, zero=zero, k=k, n=n)
    return state


@mx.compile
def _row_qmv_arrays(x, weight, scales, biases, ids, gather, split):
    """Compile dispatch construction; changing arrays stay explicit inputs."""
    k, n = weight.shape[-1] * 16, weight.shape[-2]
    shape = (8, 1, n) if gather else (*x.shape[:-1], n)
    return _row_qmv_kernel(
        inputs=[x, weight, scales, biases, ids],
        template=[
            ("K", k),
            ("N", n),
            ("GATHER", gather),
            ("SPLIT", split),
            ("GROUPED", scales.ndim == weight.ndim),
        ],
        grid=(n // 4 * 32, 8 if gather else 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
    )[0]


@mx.compile
def _row_experts_arrays(x, indices, uw, us, ub, dw, ds, db):
    ids = indices.reshape(-1).astype(mx.uint32)
    y = _row_up_kernel(
        inputs=[x, uw, us, ub, ids],
        template=[("GROUPED", us.ndim == uw.ndim)],
        grid=(4096, 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(8, 1, 512)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return _row_qmv_arrays(y, dw, ds, db, ids, True, True).reshape(1, 1, 8, 2048)


def _decode_projection(x, p):
    """Exact, runtime-probed row-metadata GEMV for a single bf16 vector."""
    if x.ndim != 3 or x.shape[:2] != (1, 1) or x.dtype != mx.bfloat16:
        return p(x)
    if type(p) is not nn.QuantizedLinear:
        return p(x)
    state = _row_quantized_metadata(p)
    if state is None or x.shape[-1] != state["k"] or state["qmv_ok"] is False:
        return p(x)

    def fast():
        y = _row_qmv_arrays(
            x, p.weight, state["scales"], state["biases"], state["zero"], False, False
        )
        return y + p.bias if "bias" in p else y

    if state["qmv_ok"] is None:
        got = _exact_result(fast, lambda: p(x))
        state["qmv_ok"] = got is not None
        return got if state["qmv_ok"] else p(x)
    return fast()


class MapleAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        # q/k/v are stored fused (one matmul per step); sanitize() concatenates
        # the checkpoint's split projections.
        self.qkv_proj = nn.Linear(
            args.hidden_size,
            (args.num_attention_heads + 2 * args.num_key_value_heads) * self.head_dim,
            bias=args.use_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.q_norm = MapleRMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = MapleRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._eps = args.rms_norm_eps
        self._rope_base = args.rope_theta
        self._qk_w = None
        self._qk_sources = None
        self._inv_freq = None
        self._fused_qk = None  # None = unprobed, then True/False
        self._native_qkv = None

        # Maple applies RoPE only on sliding-window layers; full-attention
        # layers use no positional encoding (NoPE).
        self.use_rope = args.layer_types[layer_idx] == "sliding_attention"
        if self.use_rope:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
            self.rope = initialize_rope(
                rope_dim,
                args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def _ensure_qk_state(self):
        sources = (self.q_norm.weight, self.k_norm.weight)
        if self._qk_sources is None or not self._qk_sources.matches(sources):
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            self._qk_w = mx.contiguous(
                mx.concatenate(
                    [
                        mx.broadcast_to(self.q_norm.weight[None], (n_q, self.head_dim)),
                        mx.broadcast_to(
                            self.k_norm.weight[None], (n_kv, self.head_dim)
                        ),
                    ]
                )
            )
            if self.use_rope:
                half = self.rope.dims // 2
                self._inv_freq = _rope_inv_freq_kernel(
                    inputs=[mx.array([math.log2(self._rope_base)], mx.float32)],
                    template=[("HALF", half)],
                    grid=(half, 1, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(half,)],
                    output_dtypes=[mx.float32],
                )[0]
            else:
                self._inv_freq = mx.ones((1,), dtype=mx.float32)
            if hasattr(_maple_native, "ArraySnapshot"):
                mx.eval(self._qk_w, self._inv_freq)
                self._qk_sources = _maple_native.ArraySnapshot(sources)

    def _qk_fused(self, qkv, offset):
        """Normalize/rotate Q and K; copy any trailing V heads unchanged."""
        # cache.offset is a Python int for a plain cache but an mx.array for
        # the batched caches; coerce before constructing the kernel input.
        pos_eps = mx.array(
            [float(offset), self._eps, math.log2(self._rope_base)], dtype=mx.float32
        )
        return _qk_norm_rope_kernel(
            inputs=[qkv, self.q_norm.weight, self.k_norm.weight, pos_eps],
            template=[
                ("T_", qkv.dtype),
                ("HEAD_DIM", self.head_dim),
                ("ROPE_DIM", self.rope.dims if self.use_rope else 0),
                ("NQ", self.num_attention_heads),
                ("NQK", self.num_attention_heads + self.num_key_value_heads),
            ],
            grid=(32, qkv.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[qkv.shape],
            output_dtypes=[qkv.dtype],
        )[0]

    def _qk_supported(self, dtype):
        return (
            mx.__version__ == "0.32.0"
            and self.head_dim == 128
            and dtype == mx.bfloat16
            and self.q_norm.weight.dtype == dtype
            and self.k_norm.weight.dtype == dtype
            and (
                not self.use_rope
                or (
                    type(self.rope) is nn.RoPE
                    and self.rope.scale == 1.0
                    and not self.rope.traditional
                )
            )
        )

    def _probe_native_qkv(self):
        if not (
            _maple_native is not None
            and getattr(_maple_native, "arithmetic_version", None) == 3
            and self.num_attention_heads == 16
            and self.num_key_value_heads == 4
            and self._qk_supported(mx.bfloat16)
            and (not self.use_rope or self.rope.dims == 64)
        ):
            return False
        try:
            self._ensure_qk_state()
            for offset in (7, 613):
                x = mx.random.normal((24, 128), key=mx.random.key(offset)).astype(
                    mx.bfloat16
                )
                cache = mx.zeros((2048 + 8 * 16 * 128,), mx.bfloat16)
                got = _maple_native.prepare_qkv(
                    x,
                    self._qk_w,
                    self._inv_freq,
                    cache,
                    offset,
                    7,
                    64 if self.use_rope else 0,
                    self._eps,
                )
                ref = mx.concatenate([self._qk_reference(x[:20], offset), x[20:]])
                wanted = mx.zeros((1, 8, 16, 128), cache.dtype)
                wanted[:, :, 7:8, :] = ref[16:].reshape(1, 8, 1, 128)
                mx.eval(got, ref, wanted)
                if not (
                    bool(mx.array_equal(got[:2048], ref[:16].reshape(-1)))
                    and bool(mx.array_equal(got[2048:], wanted.reshape(-1)))
                ):
                    return False
        except Exception:
            return False
        return True

    def _qk_reference(self, qk, offset):
        """The same result from stock ops: fallback, and the yardstick the
        fused kernel is checked against."""
        n_q = self.num_attention_heads
        q = self.q_norm(qk[None, :n_q, None, :])
        k = self.k_norm(qk[None, n_q:, None, :])
        if self.use_rope:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        return mx.concatenate([q, k], axis=1).reshape(qk.shape)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = _decode_projection(x, self.qkv_proj) if B == L == 1 else self.qkv_proj(x)

        if (
            B == 1
            and L == 1
            and self.use_qk_norm
            and self._qk_supported(qkv.dtype)
            and cache is not None
            and hasattr(cache, "native_compatible")
        ):
            if self._native_qkv is None:
                self._native_qkv = self._probe_native_qkv()
            if self._native_qkv and (
                cache._native_buffer is not None or cache.native_compatible()
            ):
                queries, keys, values = cache.update_native(qkv, self)
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask
                )
                return _decode_projection(
                    output.transpose(0, 2, 1, 3).reshape(1, 1, -1), self.o_proj
                )

        if B == 1 and L == 1 and self.use_qk_norm and self._qk_supported(qkv.dtype):
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            qk_size = (n_q + n_kv) * self.head_dim
            qk = qkv.reshape(-1)[:qk_size].reshape(n_q + n_kv, self.head_dim)
            if self._fused_qk is None:
                # A nonzero position, so a broken rotation cannot pass.
                self._fused_qk = _matches(
                    lambda: (self._qk_fused(qk, 7),),
                    lambda: (self._qk_reference(qk, 7),),
                )
            offset = cache.offset if cache is not None else 0
            out = (self._qk_fused if self._fused_qk else self._qk_reference)(qk, offset)
            queries = out[:n_q].reshape(1, n_q, 1, self.head_dim)
            keys = out[n_q:].reshape(1, n_kv, 1, self.head_dim)
            values = qkv.reshape(-1)[qk_size:].reshape(1, n_kv, 1, self.head_dim)
        else:
            q_size = self.num_attention_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

            queries = q.reshape(B, L, self.num_attention_heads, self.head_dim)
            keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
            values = v.reshape(B, L, self.num_key_value_heads, self.head_dim)

            if self.use_qk_norm:
                queries = self.q_norm(queries)
                keys = self.k_norm(keys)

            queries = queries.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            values = values.transpose(0, 2, 1, 3)

            if self.use_rope:
                offset = cache.offset if cache is not None else 0
                queries = self.rope(queries, offset=offset)
                keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return (
            _decode_projection(output, self.o_proj)
            if B == L == 1
            else self.o_proj(output)
        )


class MapleMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        # Dense / shared-expert MLP: no clamp; only the MoE experts clamp.
        # Unused at first_k_dense_replace=0 with no shared experts, but keep
        # it faithful.
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def group_expert_select(gates, top_k):
    # Maple routes with a plain softmax over all experts followed by top-k
    # selection and renormalization, computed in float32.
    scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(scores, inds, axis=-1)
    scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, scores


class MapleGate(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.hidden_size = args.hidden_size
        # Kept as a raw parameter (not nn.Linear) so quantization never
        # touches it. The matmul accumulates in float32 and selection runs on
        # float32 scores.
        self.weight = mx.zeros((args.num_experts, args.hidden_size))

    def __call__(self, x):
        # Preserve FP32 routing: rounding near-tied logits to BF16 changes picks.
        gates = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        inds, scores = group_expert_select(gates, self.top_k)
        if self.top_k == 8:
            inds = inds.astype(mx.uint32 if x.size == self.hidden_size else mx.int32)
        return inds, scores


@partial(mx.compile, shapeless=True)
def aggregate_expert_outputs(expert_outputs, scores):
    # Combined in float32, rounded once at the end (reference `moe_infer`).
    return (
        (expert_outputs.astype(mx.float32) * scores[..., None])
        .sum(axis=-2)
        .astype(expert_outputs.dtype)
    )


class MapleSwitchGLU(nn.Module):
    """SwitchGLU with the up and gate projections fused into one gather
    matmul; sanitize() concatenates the checkpoint's split tensors."""

    def __init__(self, input_dims, hidden_dims, num_experts, bias=False):
        super().__init__()
        self.up_gate_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self._decode_row_pair = None
        self._decode_row_ok = None

    def _decode_row_experts(self, x, indices, up, down):
        return _row_experts_arrays(
            x,
            indices,
            self.up_gate_proj.weight,
            up["scales"],
            up["biases"],
            self.down_proj.weight,
            down["scales"],
            down["biases"],
        )

    def __call__(self, x, indices):
        p, d = self.up_gate_proj, self.down_proj
        if (
            x.dtype == mx.bfloat16
            and x.shape == (1, 1, 2048)
            and indices.shape == (1, 1, 8)
            and indices.dtype in (mx.int32, mx.uint32)
            and type(p) is QuantizedSwitchLinear
            and type(d) is QuantizedSwitchLinear
            and p.weight.shape[-2:] == (1024, 128)
            and d.weight.shape[-2:] == (2048, 32)
            and p.weight.shape[0] == d.weight.shape[0]
            and "bias" not in p
            and "bias" not in d
        ):
            up, down = _row_quantized_metadata(p), _row_quantized_metadata(d)
            if up is not None and down is not None:
                pair = (up, down)
                if self._decode_row_pair is None or any(
                    a is not b for a, b in zip(pair, self._decode_row_pair)
                ):
                    self._decode_row_pair, self._decode_row_ok = pair, None
                if self._decode_row_ok is None:
                    got = _exact_result(
                        lambda: self._decode_row_experts(x, indices, up, down),
                        lambda: self._call(x, indices),
                    )
                    self._decode_row_ok = got is not None
                    if self._decode_row_ok:
                        return got
                elif self._decode_row_ok:
                    return self._decode_row_experts(x, indices, up, down)
        return self._call(x, indices)

    def _call(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up, x_gate = mx.split(
            self.up_gate_proj(x, idx, sorted_indices=do_sort), 2, axis=-1
        )
        x = self.down_proj(clamped_swiglu(x_gate, x_up), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MapleSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MapleGate(args)
        self.switch_mlp = MapleSwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        return aggregate_expert_outputs(y, scores)


class MapleDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MapleAttention(args, layer_idx)
        self.mlp = (
            MapleSparseMoeBlock(args)
            if layer_idx >= args.first_k_dense_replace
            else MapleMLP(args)
        )
        self.input_layernorm = MapleRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# Match MLX 0.32.0 GEMV and softmax/reduction arithmetic (Apple Inc., MIT).
_router_gemv_kernel = mx.fast.metal_kernel(
    name="maple_router_gemv",
    input_names=["x", "mat"],
    output_names=["logits"],
    source="""
        constexpr int K = 2048;
        constexpr int TM = 4;
        constexpr int TN = 4;
        uint lane = thread_position_in_threadgroup.x % 32;
        uint sg = thread_position_in_threadgroup.x / 32;
        uint row = threadgroup_position_in_grid.x * 16 + sg * TM;
        float result[TM] = {0};
        for (int base = 0; base < K; base += 128) {
            float v[TN];
            #pragma clang loop unroll(full)
            for (int n = 0; n < TN; n++)
                v[n] = float(x[base + lane * TN + n]);
            #pragma clang loop unroll(full)
            for (int m = 0; m < TM; m++) {
                #pragma clang loop unroll(full)
                for (int n = 0; n < TN; n++) {
                    float w = float(mat[(row + m) * K + base + lane * TN + n]);
                    result[m] += w * v[n];
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < TM; m++) {
            #pragma clang loop unroll(full)
            for (ushort offset = 16; offset >= 1; offset >>= 1)
                result[m] += simd_shuffle_down(result[m], offset);
            if (lane == 0) logits[row + m] = result[m];
        }
    """,
)

_router_select_kernel = mx.fast.metal_kernel(
    name="maple_router_select",
    input_names=["logits"],
    output_names=["indices", "scores"],
    source="""
        #pragma clang fp contract(off)
        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32;
        uint sg = tid / 32;
        threadgroup float maxima[32], sums[32], probs[256];
        float vals[4];
        if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0.0f; }
        float vmax = tid < 64 ? -MAXFLOAT : -INFINITY;
        for (int j = 0; j < 4; j++) {
            vals[j] = tid < 64 ? logits[tid * 4 + j] : -INFINITY;
            vmax = vmax < vals[j] ? vals[j] : vmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vmax = simd_max(vmax);
        if (lane == 0) maxima[sg] = vmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            vmax = simd_max(maxima[lane]);
            if (lane == 0) maxima[0] = vmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vmax = maxima[0];
        float sum = 0.0f;
        for (int j = 0; j < 4; j++) {
            vals[j] = tid < 64 ? metal::fast::exp(vals[j] - vmax) : 0.0f;
            sum += vals[j];
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            sum = simd_sum(sums[lane]);
            if (lane == 0) sums[0] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = 1.0f / sums[0];
        if (tid < 64)
            for (int j = 0; j < 4; j++) probs[tid * 4 + j] = vals[j] * inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float candidates[8];
            uint selected[8];
            for (int j = 0; j < 8; j++) {
                float p = probs[lane * 8 + j];
                candidates[j] = metal::isnan(p) ? INFINITY : p;
            }
            for (int pick = 7; pick >= 0; pick--) {
                float best = -INFINITY;
                for (int j = 0; j < 8; j++) best = metal::max(best, candidates[j]);
                best = simd_max(best);
                int winner = -1;
                for (int j = 0; j < 8; j++)
                    if (candidates[j] == best) winner = int(lane * 8 + j);
                winner = simd_max(winner);
                selected[pick] = uint(winner);
                for (int j = 0; j < 8; j++)
                    if (lane * 8 + j == uint(winner)) candidates[j] = -INFINITY;
            }
            float denom = 0.0f;
            for (int j = 0; j < 8; j++) denom = probs[selected[j]] + denom;
            denom = denom + 1e-20f;
            if (lane < 8) {
                indices[lane] = selected[lane];
                scores[lane] = probs[selected[lane]] / denom;
            }
        }
    """,
)


@mx.compile
def _norm_router_arrays(h, r, norm_weight, router_weight, eps, portable=False):
    h, hn = _add_rms_norm(h, r, norm_weight, eps)
    if portable:
        logits = hn.astype(mx.float32) @ router_weight.astype(mx.float32).T
    else:
        logits = _router_gemv_kernel(
            inputs=[hn, router_weight],
            grid=(2048, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, 1, 256)],
            output_dtypes=[mx.float32],
        )[0]
    indices, scores = _router_select_kernel(
        inputs=[logits],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 8), (1, 1, 8)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return h, hn, logits, indices, scores


def _decode_norm_router(h, r, norm, gate):
    def fallback():
        hh, hn = _add_rms_norm(h, r, norm.weight, norm.eps)
        indices, scores = gate(hn)
        return hh, hn, indices, scores

    if (
        mx.__version__ != "0.32.0"
        or type(norm) is not MapleRMSNorm
        or type(gate) is not MapleGate
        or h.shape != (1, 1, 2048)
        or r.shape != h.shape
        or h.dtype != mx.bfloat16
        or r.dtype != mx.bfloat16
        or norm.weight.shape != (2048,)
        or norm.weight.dtype != mx.bfloat16
        or gate.weight.shape != (256, 2048)
        or gate.weight.dtype != mx.bfloat16
        or gate.top_k != 8
        or not math.isfinite(norm.eps)
        or norm.eps <= 0
    ):
        return fallback()
    native = hasattr(_maple_native, "ArraySnapshot")
    sources = (norm.weight, gate.weight)
    state = gate.get("_maple_norm_router_state")
    if (
        state is None
        or state["eps"] != norm.eps
        or native != (state["sources"] is not None)
        or (native and not state["sources"].matches(sources))
    ):
        state = {
            "sources": _maple_native.ArraySnapshot(sources) if native else None,
            "eps": norm.eps,
            "ok": None,
        }
        gate["_maple_norm_router_state"] = state
    if state["ok"] is False:
        return fallback()

    def fast():
        return _norm_router_arrays(h, r, norm.weight, gate.weight, norm.eps, not native)

    if state["ok"] is None:

        def reference():
            hh = h + r
            hn = norm(hh)
            logits = hn.astype(mx.float32) @ gate.weight.astype(mx.float32).T
            indices, scores = group_expert_select(logits, gate.top_k)
            return hh, hn, logits, indices.astype(mx.uint32), scores

        got = _exact_result(fast, reference, bitwise=True)
        state["ok"] = got is not None
        if not state["ok"]:
            return fallback()
    else:
        got = fast()
    return got[0], got[1], got[3], got[4]


class MapleModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MapleDecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = MapleRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.layer_types = args.layer_types
        self.window_size = args.sliding_window
        self.swa_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )
        self.ga_idx = (
            self.layer_types.index("full_attention")
            if "full_attention" in self.layer_types
            else None
        )
        self._fused_add_norm = None  # None = unprobed, then True/False
        self._zero = None
        self._fused_aggregate_norm = None

    def _decode_fused(self, h, cache, full_mask, swa_mask):
        """Decode loop with residual adds folded into the norms.

        Carries (h, r) instead of adding r back each step, so every
        add+norm pair is one dispatch. Identical arithmetic: the kernel
        rounds the sum once (as the bf16 add did) and norms the rounded
        stream with an fp32 weight multiply.
        """
        if self._zero is None:
            self._zero = mx.zeros(h.shape, h.dtype)
            mx.eval(self._zero)
        r = self._zero  # x + 0 is exact in bf16
        if self._fused_aggregate_norm is None:
            self._fused_aggregate_norm = (
                h.dtype == mx.bfloat16
                and h.shape == (1, 1, 2048)
                and bool(self.layers)
                and all(
                    isinstance(l.mlp, MapleSparseMoeBlock)
                    and l.mlp.gate.top_k == 8
                    and l.input_layernorm.weight.dtype == h.dtype
                    and l.post_attention_layernorm.weight.dtype == h.dtype
                    for l in self.layers
                )
                and _aggregate_add_rms_norm_ok(
                    h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                )
            )
        if self._fused_aggregate_norm:
            ln = self.layers[0].input_layernorm
            h, hn = _add_rms_norm(h, r, ln.weight, ln.eps)
            for i, (layer, c, layer_type) in enumerate(
                zip(self.layers, cache, self.layer_types)
            ):
                mask = full_mask if layer_type == "full_attention" else swa_mask
                r = layer.self_attn(hn, mask, c)
                ln = layer.post_attention_layernorm
                h, hn, inds, scores = _decode_norm_router(h, r, ln, layer.mlp.gate)
                y = layer.mlp.switch_mlp(hn, inds)
                ln = (
                    self.layers[i + 1].input_layernorm
                    if i + 1 < len(self.layers)
                    else self.norm
                )
                h, hn = _aggregate_add_rms_norm(h, y, scores, ln.weight, ln.eps)
            return hn
        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            ln = layer.input_layernorm
            h, hn = _add_rms_norm(h, r, ln.weight, ln.eps)
            r = layer.self_attn(hn, mask, c)
            ln = layer.post_attention_layernorm
            h, hn = _add_rms_norm(h, r, ln.weight, ln.eps)
            r = layer.mlp(hn)
        return _add_rms_norm(h, r, self.norm.weight, self.norm.eps)[1]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        swa_mask = None
        if self.ga_idx is not None:
            full_mask = create_attention_mask(h, cache[self.ga_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.window_size
            )

        if h.size == h.shape[-1]:
            if self._fused_add_norm is None:
                self._fused_add_norm = _add_rms_norm_ok(
                    h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                )
            if self._fused_add_norm:
                return self._decode_fused(h, cache, full_mask, swa_mask)

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            h = layer(h, mask, c)
            if inputs.shape[-1] > 2048:
                # Bound large prefill graphs for MLX 0.32 correctness.
                mx.eval(h)

        return self.norm(h)


_flash_scatter_kernel = mx.fast.metal_kernel(
    name="maple_flash_scatter",
    input_names=["top", "token_map", "logits", "force_ids", "force_logits"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < COUNT) {
            int token = token_map[top[i / CLUSTER] * CLUSTER + i % CLUSTER];
            bool forced = false;
            for (uint j = 0; j < FORCED; ++j) forced |= token == force_ids[j];
            if (!forced) out[token] = logits[i];
        }
        if (i < FORCED) {
            bool duplicate = false;
            for (uint j = 0; j < i; ++j) duplicate |= force_ids[j] == force_ids[i];
            if (!duplicate) out[force_ids[i]] = force_logits[i];
        }
    """,
)


def _flash_scatter(top, token_map, logits, force_ids, force_logits, vocab_size):
    return _flash_scatter_kernel(
        inputs=[top, token_map, logits, force_ids, force_logits],
        template=[
            ("COUNT", logits.size),
            ("CLUSTER", token_map.shape[1]),
            ("FORCED", force_ids.size),
        ],
        grid=(max(logits.size, force_ids.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, vocab_size)],
        output_dtypes=[logits.dtype],
        init_value=-float("inf"),
    )[0]


# Exact bf16 top-k set: two radix histograms, with MLX's stable tie order.
# One threadgroup owns selection and compaction; no cross-group synchronization.
_head_select_kernel = mx.fast.metal_kernel(
    name="maple_head_select",
    input_names=["x"],
    output_names=["out"],
    header=r"""
    inline uint head_key(bfloat16_t x) {
        uint raw=as_type<ushort>(x), magnitude=raw&0x7fffu;
        if(magnitude>0x7f80u) return 0xffffu;
        if(magnitude<0x80u) return 0x8000u;
        return (raw&0x8000u)?((~raw)&0xffffu):(raw^0x8000u);
    }
    inline uint head_prefix(uint value,uint tid,threadgroup uint* groups) {
        uint lane=tid%32u,sg=tid/32u;
        uint prefix=simd_prefix_exclusive_sum(value);
        if(lane==31u) groups[sg]=prefix+value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint w=0;w<sg;++w) prefix+=groups[w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return prefix;
    }
    """,
    source=r"""
    uint tid=thread_position_in_threadgroup.x;
    constexpr uint VALUES=(N+1023u)/1024u;
    uint keys[VALUES];
    threadgroup atomic_uint hist[256];
    threadgroup uint groups[32],cut[5];
    if(tid<256u) atomic_store_explicit(hist+tid,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<VALUES;++j) {
        uint i=tid*VALUES+j;
        keys[j]=i<N?head_key(x[i]):0u;
        uint digit=keys[j]>>8;
        uint matches=uint((simd_vote::vote_t)simd_ballot(i<N));
        for(uint b=0;b<8;++b) {
            bool bit=(digit&(1u<<b))!=0u;
            uint votes=uint((simd_vote::vote_t)simd_ballot(bit));
            matches &= bit?votes:~votes;
        }
        if(i<N && tid%32u==ctz(matches))
            atomic_fetch_add_explicit(hist+digit,popcount(matches),memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count=tid<256u?atomic_load_explicit(hist+tid,memory_order_relaxed):0u;
    uint prefix=head_prefix(count,tid,groups);
    if(tid<256u && prefix<=N-K && prefix+count>N-K) {cut[0]=tid;cut[1]=prefix;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<256u) atomic_store_explicit(hist+tid,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<VALUES;++j) {
        if(tid*VALUES+j<N && (keys[j]>>8)==cut[0])
            atomic_fetch_add_explicit(hist+(keys[j]&255u),1u,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    count=tid<256u?atomic_load_explicit(hist+tid,memory_order_relaxed):0u;
    prefix=head_prefix(count,tid,groups);
    uint rank=N-K-cut[1];
    if(tid<256u && prefix<=rank && prefix+count>rank) {
        cut[2]=(cut[0]<<8)|tid;
        cut[3]=N-(cut[1]+prefix+count);
        cut[4]=count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint greater=0u,equal=0u;
    for(uint j=0;j<VALUES;++j) if(tid*VALUES+j<N) {
        greater+=keys[j]>cut[2]; equal+=keys[j]==cut[2];
    }
    uint greater_prefix=head_prefix(greater,tid,groups);
    uint equal_prefix=head_prefix(equal,tid,groups);
    uint skip=cut[4]-(K-cut[3]);
    for(uint j=0;j<VALUES;++j) {
        uint i=tid*VALUES+j;
        if(i>=N) break;
        if(keys[j]>cut[2]) out[greater_prefix++]=i;
        else if(keys[j]==cut[2]) {
            if(equal_prefix>=skip) out[cut[3]+equal_prefix-skip]=i;
            equal_prefix++;
        }
    }
    """,
)


def _head_select(x, k):
    return _head_select_kernel(
        inputs=[x],
        template=[("N", x.size), ("K", k)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, k)],
        output_dtypes=[mx.uint32],
    )[0]


class FlashHead(nn.Module):
    """Two-phase approximate lm_head for single-stream decode.

    Phase one scores quantized cluster centroids of the vocabulary; phase two
    computes exact logits only for the tokens of the top ``n_probes`` clusters
    (plus a fixed set of forced control tokens such as EOS). All other logits
    are -inf, so greedy decoding is exact whenever the true argmax lies in the
    probed clusters. Prefill and batched calls use the exact lm_head.

    Reference: FlashHead — Efficient Drop-in Replacement for the
    Classification Head in Language Model Inference.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        meta = args.flash_head
        if not meta.get("scaled_centroids"):
            raise ValueError(
                "FlashHead metadata predates scaled centroids; regenerate with "
                "`python -m mlx_lm.ternary <checkpoint> --flash-head-only`."
            )
        n_clusters = meta["n_clusters"]
        cluster_size = meta["cluster_size"]
        # Default matches the converter's `--probes` default; every generated
        # checkpoint records the value explicitly.
        self.n_probes = min(meta.get("n_probes", 512), n_clusters)
        self.head_group_size = meta.get("head_group_size", 64)
        self.head_bits = meta.get("head_bits", 4)
        # The converter stores scaled centroids for a single scoring matmul.
        self.centroids = nn.QuantizedLinear(
            args.hidden_size,
            n_clusters,
            bias=False,
            group_size=meta.get("group_size", 64),
            bits=meta.get("bits", 4),
        )
        self.token_map = mx.zeros((n_clusters, cluster_size), dtype=mx.int32)
        # Cluster-ordered copy of the quantized lm_head: subset logits are one
        # gather_qmm over the probed 32-row blocks, with no per-step gather.
        # It is a row-permutation of lm_head by token_map and nothing more, so
        # it is derived rather than stored: Model.sanitize rebuilds it at load.
        hidden = args.hidden_size
        self.head = {
            "weight": mx.zeros(
                (n_clusters, cluster_size, hidden * self.head_bits // 32),
                dtype=mx.uint32,
            ),
            "scales": mx.zeros(
                (n_clusters, cluster_size, hidden // self.head_group_size),
                dtype=mx.bfloat16,
            ),
            "biases": mx.zeros(
                (n_clusters, cluster_size, hidden // self.head_group_size),
                dtype=mx.bfloat16,
            ),
        }
        self._force_ids = mx.array(meta.get("force_tokens", []), dtype=mx.int32)
        self._scatter = None
        self._select = None

    def _scatter_reference(self, top, logits, force_logits, vocab_size):
        oids = self.token_map[top[0]].reshape(-1)
        if self._force_ids.size:
            oids = mx.concatenate([oids, self._force_ids])
            logits = mx.concatenate([logits, force_logits])
        full = mx.full((1, 1, vocab_size), float("-inf"), dtype=logits.dtype)
        full[0, 0, oids] = logits
        return full

    def __call__(self, h: mx.array, lm_head: nn.Module) -> mx.array:
        hv = h[:, -1, :]
        scores = self.centroids(hv)
        reference = lambda: mx.argpartition(scores, kth=-self.n_probes, axis=-1)[
            ..., -self.n_probes :
        ]
        if (
            mx.__version__ == "0.32.0"
            and mx.default_device() == mx.gpu
            and scores.dtype == mx.bfloat16
            and scores.ndim == 2
            and scores.shape[0] == 1
            and 0 < self.n_probes <= scores.size <= 8192
        ):
            if self._select is None:
                self._select = _matches(
                    lambda: (mx.sort(_head_select(scores, self.n_probes)),),
                    lambda: (mx.sort(reference()),),
                )
            top = _head_select(scores, self.n_probes) if self._select else reference()
        else:
            top = reference()

        logits = mx.gather_qmm(
            hv.reshape(1, 1, 1, 1, -1),
            self.head["weight"],
            self.head["scales"],
            self.head["biases"],
            rhs_indices=top[:, None, :],
            transpose=True,
            group_size=self.head_group_size,
            bits=self.head_bits,
        ).reshape(-1)

        force_logits = logits[:0]
        if self._force_ids.size:
            force_logits = mx.gather_qmm(
                hv.reshape(1, 1, -1),
                lm_head.weight.reshape(lm_head.weight.shape[0], 1, -1),
                lm_head.scales.reshape(lm_head.weight.shape[0], 1, -1),
                lm_head.biases.reshape(lm_head.weight.shape[0], 1, -1),
                rhs_indices=self._force_ids,
                transpose=True,
                group_size=lm_head.group_size,
                bits=lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            ).reshape(-1)
        vocab_size = lm_head.weight.shape[0]
        if mx.__version__ == "0.32.0" and self._force_ids.size <= 8:
            fast = lambda: _flash_scatter(
                top, self.token_map, logits, self._force_ids, force_logits, vocab_size
            )
            if self._scatter is None:
                self._scatter = _matches(
                    lambda: (fast(),),
                    lambda: (
                        self._scatter_reference(top, logits, force_logits, vocab_size),
                    ),
                )
            if self._scatter:
                return fast()
        return self._scatter_reference(top, logits, force_logits, vocab_size)


class _FusedKVBase:
    """Native decode buffer; stock caches handle growth, prefill, and rotation."""

    # Defaults also support MLX's from_state(), which skips __init__.
    _native_buffer = None
    _native_capacity = 0
    _keys = _values = None

    def native_compatible(self):
        if self._keys is None:
            return self._values is None
        return all(
            a is not None
            and a.dtype == mx.bfloat16
            and a.ndim == 4
            and a.shape[:2] == (1, 4)
            and a.shape[3] == 128
            for a in (self._keys, self._values)
        )

    def _native_kv(self):
        return self._native_buffer[2048:].reshape(1, 8, self._native_capacity, 128)

    def update_native(self, qkv, attn):
        rotating = isinstance(self, _RotatingKVCache)
        capacity = self._native_capacity
        if (
            self._native_buffer is None
            or (self.offset >= capacity and (not rotating or capacity < self.max_size))
            or (rotating and capacity > self.max_size)
        ):
            # Use stock arithmetic and cache allocation at transitions. Packing
            # Q/K/V into one output lets subsequent writes obey MLX donation.
            qkv = qkv.reshape(24, 128)
            qk = attn._qk_reference(qkv[:20], self.offset)
            self.update_and_fetch(
                qk[16:].reshape(1, 4, 1, 128), qkv[20:].reshape(1, 4, 1, 128)
            )
            self._native_capacity = self._keys.shape[2]
            self._native_buffer = mx.concatenate(
                [qk[:16].reshape(-1), self._keys.reshape(-1), self._values.reshape(-1)]
            )
            self._keys = self._values = None
        else:
            attn._ensure_qk_state()
            if rotating and self._idx == self.max_size:
                self._idx = self.keep
            self._native_buffer = _maple_native.prepare_qkv(
                qkv,
                attn._qk_w,
                attn._inv_freq,
                self._native_buffer,
                self.offset,
                self._idx if rotating else self.offset,
                64 if attn.use_rope else 0,
                attn._eps,
            )
            self.offset += 1
            if rotating:
                self._idx += 1
        length = min(self.offset, self._native_capacity)
        queries = self._native_buffer[:2048].reshape(1, 16, 1, 128)
        kv = self._native_kv()
        return queries, kv[:, :4, :length, :], kv[:, 4:, :length, :]

    @property
    def keys(self):
        return (
            self._native_kv()[:, :4] if self._native_buffer is not None else self._keys
        )

    @keys.setter
    def keys(self, value):
        self._unfuse()
        self._keys = value

    @property
    def values(self):
        return (
            self._native_kv()[:, 4:]
            if self._native_buffer is not None
            else self._values
        )

    @values.setter
    def values(self, value):
        self._unfuse()
        self._values = value

    def _unfuse(self):
        if self._native_buffer is not None:
            kv = self._native_kv()
            self._keys, self._values = kv[:, :4], kv[:, 4:]
            self._native_buffer = None

    def update_and_fetch(self, keys, values):
        self._unfuse()
        keys, values = super().update_and_fetch(keys, values)
        # Stock ring updates can return the mutable cache array wrappers.
        # Separate views keep retained outputs stable across later assignments.
        return keys[:], values[:]


# Stock names keep saved caches readable by MLX's public cache loader.
class KVCache(_FusedKVBase, _KVCache):
    pass


class RotatingKVCache(_FusedKVBase, _RotatingKVCache):
    pass


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MapleModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.flash_head and args.use_flash_head and not args.tie_word_embeddings:
            self.lm_head_flash = FlashHead(args)
        else:
            self.lm_head_flash = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        if (
            self.lm_head_flash is not None
            and out.shape[0] == 1
            and out.shape[1] == 1
            and isinstance(self.lm_head, nn.QuantizedLinear)
            and getattr(self.lm_head, "mode", "affine") == "affine"
        ):
            return self.lm_head_flash(out, self.lm_head)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            # Drop the head entirely (weight + quantization scales/biases).
            weights = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}

        # FlashHead disabled (e.g. model_config={"flash_head": None}): drop its
        # tensors so checkpoints that carry them still load.
        if self.lm_head_flash is None:
            weights = {
                k: v for k, v in weights.items() if not k.startswith("lm_head_flash.")
            }
        else:
            # Folded into the centroid rows at generation time; older shards
            # still carry the tensor.
            weights.pop("lm_head_flash.cluster_scale", None)
            # `lm_head_flash.head.*` is lm_head permuted by token_map (see
            # mlx_lm.ternary.generate_flash_head), so it is pure redundancy on
            # disk. Checkpoints may ship it or omit it; reconcile both here.
            if "lm_head_flash.head.weight" not in weights:
                token_map = weights["lm_head_flash.token_map"]
                order = token_map.reshape(-1)
                for k in ("weight", "scales", "biases"):
                    weights[f"lm_head_flash.head.{k}"] = weights[f"lm_head.{k}"][
                        order
                    ].reshape(*token_map.shape, -1)

        # Ternary tensors carry one scale per output row, so checkpoints store
        # it once as `row_alpha` and omit biases entirely (bias == -scale).
        # Expand here so everything downstream — fusion below, and mlx's own
        # quantized kernels — sees the per-group layout. Checkpoints written
        # with `--group-scales` have no row_alpha and pass straight through.
        row_alpha_keys = [k for k in weights if k.endswith(".row_alpha")]
        if row_alpha_keys:
            group_size = (self.args.quantization or {}).get("group_size", 128)
            for key in row_alpha_keys:
                alpha = weights.pop(key)
                prefix = key[: -len(".row_alpha")]
                packed = weights.get(f"{prefix}.weight")
                if packed is None:
                    continue
                # 2-bit packing stores 16 codes per uint32 word.
                n_groups = (packed.shape[-1] * 16) // group_size
                scales = mx.contiguous(
                    mx.broadcast_to(alpha[..., None], (*alpha.shape, n_groups))
                )
                weights[f"{prefix}.scales"] = scales
                weights[f"{prefix}.biases"] = -scales

        # Stack per-expert weights from the Hugging Face layout into the
        # SwitchGLU layout. Already-converted checkpoints pass through.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases", "bias"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            # Fuse split projections: q/k/v -> qkv_proj (rows), MoE up/gate ->
            # up_gate_proj (per-expert rows). Row-wise quantized tensors
            # (weight/scales/biases) concatenate losslessly along the output
            # axis.
            for suffix in ["weight", "scales", "biases", "bias"]:
                qkv = [
                    f"{prefix}.self_attn.{p}.{suffix}"
                    for p in ("q_proj", "k_proj", "v_proj")
                ]
                if qkv[0] in weights:
                    weights[f"{prefix}.self_attn.qkv_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(k) for k in qkv], axis=0
                    )
                up = f"{prefix}.mlp.switch_mlp.up_proj.{suffix}"
                gate = f"{prefix}.mlp.switch_mlp.gate_proj.{suffix}"
                if up in weights:
                    weights[f"{prefix}.mlp.switch_mlp.up_gate_proj.{suffix}"] = (
                        mx.concatenate([weights.pop(up), weights.pop(gate)], axis=1)
                    )

        return weights

    def make_cache(self):
        native = (
            getattr(_maple_native, "arithmetic_version", None) == 3
            and self.args.num_attention_heads == 16
            and self.args.num_key_value_heads == 4
            and self.args.head_dim == 128
        )
        caches = []
        for layer_type in self.model.layer_types:
            if layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window)
                    if native
                    else _RotatingKVCache(max_size=self.args.sliding_window)
                )
            else:
                caches.append(KVCache() if native else _KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("lm_head") or "word_embeddings" in path:
                return {"group_size": 64, "bits": 4}
            return True

        return predicate
