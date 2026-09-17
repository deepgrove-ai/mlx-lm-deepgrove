# Copyright © 2026 DeepGrove AI.

"""Maple kernel arithmetic, fallbacks, and checkpoint regressions."""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models import maple


def _args(**kw):
    base = dict(
        num_hidden_layers=2,
        num_experts=64,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        vocab_size=1024,
        layer_types=["sliding_attention", "full_attention"],
    )
    base.update(kw)
    return maple.ModelArgs(**base)


class TestMapleKernels(unittest.TestCase):
    def test_router_logits_are_float32(self):
        """Routing must not round the expert logits to bf16.

        With 256 experts the top-k boundary is routinely a near tie, and
        rounding logits of this magnitude to bf16 (spacing ~0.5 near 100)
        perturbs the renormalized scores by percent, not ulps. Check routing
        against a float64 computation.
        """
        args = _args(num_experts=256)
        gate = maple.MapleGate(args)
        mx.random.seed(0)
        # Logits land around 100, where bf16 has ~0.5 resolution.
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05 + 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)

        x = (mx.random.normal((1, 1, args.hidden_size)) * 0.2 + 1.0).astype(mx.bfloat16)
        w64 = np.array(gate.weight.astype(mx.float32), dtype=np.float64)
        x64 = np.array(x.astype(mx.float32), dtype=np.float64).reshape(-1)
        logits = w64 @ x64
        self.assertGreater(np.abs(logits).max(), 20.0, "test needs large logits")
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.sort(np.argsort(-p)[: args.num_experts_per_tok])

        inds, scores = gate(x)
        mx.eval(inds, scores)
        self.assertEqual(scores.dtype, mx.float32, "scores must stay float32")
        self.assertEqual(inds.dtype, mx.uint32)
        span_ids, _ = gate(mx.broadcast_to(x, (1, 3, args.hidden_size)))
        self.assertEqual(span_ids.dtype, mx.int32)
        order = mx.argsort(inds.reshape(-1))
        np.testing.assert_array_equal(inds.reshape(-1)[order], top)
        np.testing.assert_allclose(
            scores.reshape(-1)[order],
            p[top] / p[top].sum(),
            rtol=1e-3,
            atol=0,
            equal_nan=False,
        )

    def test_add_rms_norm_matches_reference(self):
        """Fused residual add + RMSNorm vs the stock two-step path."""
        args = _args()
        dim, eps = args.hidden_size, args.rms_norm_eps
        w = (mx.random.normal((dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
        x = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        r = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        if not maple._add_rms_norm_ok(dim, mx.bfloat16, w, eps):
            self.skipTest("fused add+norm disabled on this build")

        h, hn = maple._add_rms_norm(x, r, w, eps)
        mx.eval(h, hn)
        # The residual stream must be rounded exactly once, like a bf16 add.
        self.assertTrue(mx.array_equal(h, x + r), "residual add is not bit-exact")
        ref = mx.fast.rms_norm(
            (x + r).astype(mx.float32), w.astype(mx.float32), eps
        ).astype(mx.bfloat16)
        self.assertTrue(mx.array_equal(hn, ref))

    def test_qk_norm_rope_matches_reference(self):
        """Fused per-head norm + partial RoPE vs q_norm/k_norm + mx.fast.rope,
        on both the RoPE and the NoPE layer type."""
        for layer_type in ("sliding_attention", "full_attention"):
            attn = maple.MapleAttention(
                _args(num_hidden_layers=1, layer_types=[layer_type]), 0
            )
            attn.q_norm.weight = mx.linspace(0.5, 1.5, 128).astype(mx.bfloat16)
            attn.k_norm.weight = mx.linspace(1.5, 0.5, 128).astype(mx.bfloat16)
            for seed in range(8):
                if seed:
                    for norm, key in ((attn.q_norm, seed), (attn.k_norm, seed + 8)):
                        norm.weight = (
                            mx.random.normal((128,), key=mx.random.key(key)) * 0.1 + 1
                        ).astype(mx.bfloat16)
                qk = mx.random.normal((20, 128), key=mx.random.key(seed)).astype(
                    mx.bfloat16
                )
                for position in (0, 1, 7, 511, 512, 613, 8192, 139999):
                    got = attn._qk_fused(qk, position)
                    self.assertEqual(got.dtype, mx.bfloat16)
                    self.assertTrue(
                        mx.array_equal(got, attn._qk_reference(qk, position)),
                        (layer_type, seed, position),
                    )

    def test_probe_rejects_a_mismatched_kernel(self):
        """The self-check must latch the fallback rather than ship garbage.

        Norm weights in a dtype the kernel was not templated for are the
        realistic case: the kernel reinterprets the buffer and the output is
        silently wrong, so only a value check catches it.
        """
        args = _args(num_hidden_layers=1, layer_types=["sliding_attention"])
        attn = maple.MapleAttention(args, 0)  # q_norm/k_norm default to float32
        qk = (mx.random.normal((20, args.head_dim)) * 0.5).astype(mx.bfloat16)
        mx.eval(attn.parameters(), qk)
        self.assertFalse(
            maple._matches(
                lambda: (attn._qk_fused(qk, 7),),
                lambda: (attn._qk_reference(qk, 7),),
            )
        )

    def test_decode_matches_with_fast_paths_off(self):
        """A decode step through the fused kernels must equal the stock path."""
        mx.random.seed(0)
        args = _args()
        model = maple.Model(args)
        for layer in model.layers:
            layer.mlp.gate.weight = (
                mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
            )
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        def decode(fused):
            cache = model.make_cache()
            model(mx.array([[3, 1, 4, 1, 5]]), cache=cache)
            model.model._fused_add_norm = fused
            model.model._fused_aggregate_norm = fused
            for layer in model.layers:
                layer.self_attn._fused_qk = fused
                layer.self_attn._native_qkv = fused
            out = model(mx.array([[9]]), cache=cache)
            mx.eval(out)
            return np.array(out.astype(mx.float32)).reshape(-1)

        # `None` leaves the probes to decide, i.e. exactly what a user gets.
        fast = decode(None)
        if not model.model._fused_add_norm:
            self.skipTest("fused decode disabled on this build")
        self.assertTrue(
            all(l.self_attn._fused_qk or l.self_attn._native_qkv for l in model.layers)
        )
        slow = decode(False)
        self.assertTrue(np.array_equal(fast, slow))

    def test_experts_clamp_both_swiglu_branches(self):
        """silu(min(gate, 7)) * clip(up, -7, 7), in the activation dtype."""
        gate = mx.array([[-3.0, 0.5, 9.0, 40.0]], dtype=mx.bfloat16)
        up = mx.array([[100.0, -50.0, 2.0, -0.25]], dtype=mx.bfloat16)
        got = maple.clamped_swiglu(gate, up)
        mx.eval(got)
        self.assertEqual(got.dtype, mx.bfloat16, "clamp must not promote to f32")

        g = np.array(gate.astype(mx.float32))
        u = np.array(up.astype(mx.float32))
        g = np.minimum(g, maple.MLP_CLAMP)
        u = np.clip(u, -maple.MLP_CLAMP, maple.MLP_CLAMP)
        want = (g / (1 + np.exp(-g))) * u
        np.testing.assert_allclose(
            got.astype(mx.float32), want, rtol=8e-3, atol=1e-8, equal_nan=False
        )

    def test_row_and_group_scale_layouts_agree(self):
        """A row_alpha checkpoint must expand to the per-group tensors."""
        args = _args(quantization={"group_size": 128, "bits": 2})
        model = maple.Model(args)

        n, k, groups = 256, 2048, 2048 // 128
        alpha = mx.abs(mx.random.normal((n,))).astype(mx.bfloat16) + 0.01
        packed = mx.random.randint(0, 2**31, (n, k // 16)).astype(mx.uint32)
        # o_proj is not part of the q/k/v fusion, so it exercises the
        # row_alpha expansion on its own.
        prefix = "model.layers.0.self_attn.o_proj"

        rows = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.row_alpha": alpha}
        )
        scales = mx.broadcast_to(alpha[:, None], (n, groups))
        mx.eval(rows, scales)

        self.assertTrue(mx.array_equal(rows[f"{prefix}.scales"], scales))
        self.assertTrue(mx.array_equal(rows[f"{prefix}.biases"], -scales))
        # `--group-scales` checkpoints must pass through untouched.
        grouped = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.scales": scales}
        )
        self.assertTrue(mx.array_equal(grouped[f"{prefix}.scales"], scales))

    def test_aggregate_add_norm_preserves_rounding(self):
        """Keep the ordered FP32 reduction and both BF16 rounding points."""
        dim, eps = 2048, 1e-6
        w = mx.linspace(0.5, 1.5, dim).astype(mx.bfloat16)
        if not maple._aggregate_add_rms_norm_ok(dim, mx.bfloat16, w, eps):
            self.skipTest("exact fused aggregation disabled on this build")
        for seed in range(8):
            eps = 1e-6 if seed % 2 else 1e-4
            w[0] = seed + 0.5
            y = mx.random.normal((1, 1, 8, dim), key=mx.random.key(seed)).astype(
                mx.bfloat16
            )
            scores = mx.softmax(
                mx.random.normal((1, 1, 8), key=mx.random.key(32 + seed)) * 3
            )
            reduced = maple.aggregate_expert_outputs(y, scores)
            # Near cancellation makes the residual rounding consequential.
            h = -reduced + mx.array(0.03125, dtype=mx.bfloat16)
            got = maple._aggregate_add_rms_norm(h, y, scores, w, eps)
            want = maple._add_rms_norm(h, reduced, w, eps)
            mx.eval(got, want)
            for a, b in zip(got, want):
                self.assertTrue(mx.array_equal(a, b), f"seed {seed}: rounding changed")

    def test_aggregate_probe_rejects_bad_output(self):
        """A broken kernel must select the portable fallback."""
        dim = 2048
        w = mx.ones((dim,), dtype=mx.bfloat16)
        wrong = mx.zeros((1, 1, dim), dtype=mx.bfloat16)
        with patch.object(
            maple, "_aggregate_add_rms_norm", return_value=(wrong, wrong)
        ):
            self.assertFalse(
                maple._aggregate_add_rms_norm_ok(dim, mx.bfloat16, w, 1e-6)
            )


@unittest.skipUnless(
    mx.__version__ == "0.32.0",
    "router arithmetic targets MLX 0.32.0",
)
class TestNormRouter(unittest.TestCase):
    def setUp(self):
        self.norm = maple.MapleRMSNorm(2048)
        self.norm.weight = mx.linspace(0.5, 1.5, 2048).astype(mx.bfloat16)
        self.gate = maple.MapleGate(maple.ModelArgs())
        self.gate.weight = (
            mx.random.normal((256, 2048), key=mx.random.key(731)) * 0.02
        ).astype(mx.bfloat16)
        self.x = mx.random.normal((1, 1, 2048), key=mx.random.key(732)).astype(
            mx.bfloat16
        )
        self.r = -self.x * mx.array(0.25, mx.bfloat16)

    def stock(self, x=None):
        h = (self.x if x is None else x) + self.r
        hn = self.norm(h)
        logits = hn.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        ids, scores = maple.group_expert_select(logits, 8)
        return h, hn, logits, ids.astype(mx.uint32), scores

    def kernel(self, x=None):
        return maple._norm_router_arrays(
            self.x if x is None else x,
            self.r,
            self.norm.weight,
            self.gate.weight,
            self.norm.eps,
        )

    def assert_bits_equal(self, got, ref):
        self.assertEqual(len(got), len(ref))
        mx.eval(got, ref)
        for a, b in zip(got, ref):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            dtype = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
            self.assertTrue(mx.array_equal(a.view(dtype), b.view(dtype)))

    def check_model_helper(self):
        ref = self.stock()
        self.assert_bits_equal(
            maple._decode_norm_router(self.x, self.r, self.norm, self.gate),
            (ref[0], ref[1], ref[3], ref[4]),
        )

    def test_all_stages_and_retained_outputs_match_stock(self):
        retained = self.kernel()
        reference = self.stock()
        self.assert_bits_equal(retained, reference)
        inputs_before = [a.view(mx.uint16).tolist() for a in (self.x, self.r)]
        for scale in (0.0, 0.1, 1.0, 20.0):
            x = self.x * mx.array(scale, mx.bfloat16)
            self.assert_bits_equal(self.kernel(x), self.stock(x))
        self.assert_bits_equal(retained, reference)
        self.assertEqual(
            inputs_before, [a.view(mx.uint16).tolist() for a in (self.x, self.r)]
        )

    def test_routing_preserves_rounded_ties_and_underflow(self):
        self.x = mx.ones_like(self.x)
        self.r = mx.zeros_like(self.r)
        self.norm.weight = mx.ones_like(self.norm.weight)
        rows = [mx.zeros((256,)), mx.linspace(1e-13, 0, 256)]
        rows += [
            mx.where(mx.arange(256) < 9, 0.01, 0),
            mx.where(mx.arange(256) == 0, 1.0, 0),
        ]
        for row in rows:
            self.gate.weight = mx.contiguous(
                mx.broadcast_to(row[:, None], (256, 2048)).astype(mx.bfloat16)
            )
            self.assert_bits_equal(self.kernel(), self.stock())

    @unittest.skipUnless(
        hasattr(maple._maple_native, "ArraySnapshot"),
        "native metadata tracker is not built",
    )
    def test_reprobes_inplace_parameters_and_epsilon(self):
        self.check_model_helper()
        for parameter in (self.norm.weight, self.gate.weight):
            state = self.gate["_maple_norm_router_state"]
            self.assertTrue(state["ok"])
            parameter[0] = parameter[0] * mx.array(1.125, mx.bfloat16)
            self.check_model_helper()
            self.assertIsNot(state, self.gate["_maple_norm_router_state"])
        state = self.gate["_maple_norm_router_state"]
        self.norm.eps = 1e-4
        self.check_model_helper()
        self.assertIsNot(state, self.gate["_maple_norm_router_state"])

    @unittest.skipUnless(
        hasattr(maple._maple_native, "ArraySnapshot"),
        "native metadata tracker is not built",
    )
    def test_missing_capability_and_failed_probe_use_fallback(self):
        with patch.object(maple, "_maple_native", None), patch.object(
            maple, "_norm_router_arrays", side_effect=AssertionError
        ):
            self.check_model_helper()
        bad = list(self.kernel())
        bad[-1] = mx.zeros_like(bad[-1])
        with patch.object(maple, "_norm_router_arrays", return_value=bad):
            self.check_model_helper()
        self.assertFalse(self.gate["_maple_norm_router_state"]["ok"])
        with patch.object(maple, "_norm_router_arrays", side_effect=AssertionError):
            self.check_model_helper()

    def test_noncontiguous_weights_and_changed_dtype(self):
        self.check_model_helper()
        self.gate.weight = mx.contiguous(self.gate.weight.T).T
        self.check_model_helper()
        if hasattr(maple._maple_native, "ArraySnapshot"):
            self.assertFalse(self.gate["_maple_norm_router_state"]["ok"])
        self.gate.weight = self.gate.weight.astype(mx.float32)
        self.check_model_helper()

    def test_portable_router_reads_changed_parameters(self):
        with patch.object(maple, "_maple_native", None):
            self.check_model_helper()
            self.norm.weight[0] = mx.array(0.125, mx.bfloat16)
            self.gate.weight[0] = mx.zeros_like(self.gate.weight[0])
            self.check_model_helper()
            self.gate.weight = mx.contiguous(self.gate.weight.T).T
            self.norm.eps = 1e-4
            self.check_model_helper()
            self.assertTrue(self.gate._maple_norm_router_state["ok"])


@unittest.skipUnless(
    hasattr(maple._maple_native, "ArraySnapshot"),
    "native metadata tracker is not built",
)
class TestMapleRowKernels(unittest.TestCase):
    @staticmethod
    def uniform_metadata(p):
        # Synthetic row-constant affine tensors, with unequal row scales and
        # nonzero biases. Every packed 2-bit code remains valid.
        shape = (*p.scales.shape[:-1], 1)
        sc = (0.01 + mx.random.uniform(shape=shape) * 0.04).astype(mx.bfloat16)
        bi = (-sc * 1.5).astype(mx.bfloat16)
        p.scales = mx.contiguous(mx.broadcast_to(sc, p.scales.shape))
        p.biases = mx.contiguous(mx.broadcast_to(bi, p.biases.shape))
        mx.eval(p.parameters())

    def projection(self, k=512, bias=False):
        p = nn.QuantizedLinear(k, 24, bias=bias, group_size=128, bits=2)
        p.set_dtype(mx.bfloat16)
        self.uniform_metadata(p)
        if bias:
            p.bias = mx.linspace(-0.25, 0.5, 24).astype(mx.bfloat16)
        return p

    def test_projection_exact_for_strides_bias_and_accumulation_lengths(self):
        for k, bias in product((512, 2048), (False, True)):
            p = self.projection(k, bias)
            for scale in (0.1, 3.0, 20.0):
                with self.subTest(k=k, bias=bias, scale=scale):
                    x = (mx.random.normal((1, 1, k * 2)) * scale).astype(mx.bfloat16)[
                        ..., ::2
                    ]
                    got, want = maple._decode_projection(x, p), p(x)
                    mx.eval(got, want)
                    self.assertTrue(p._maple_row_state["qmv_ok"])
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(mx.array_equal(got, want))

    def test_projection_fallback_and_metadata_invalidation(self):
        p = self.projection()
        x = mx.ones((1, 1, 512), mx.bfloat16)
        mx.eval(maple._decode_projection(x, p))
        old = p._maple_row_state
        # A changed affine tensor must invalidate the compact copy. A single
        # differing group is enough to reject the row-constant specialization.
        p.scales = p.scales.at[0, 1].add(mx.array(0.5, mx.bfloat16))
        self.assertIsNone(maple._row_quantized_metadata(p))
        self.assertIsNot(old, p._maple_row_state)
        self.assertTrue(mx.array_equal(maple._decode_projection(x, p), p(x)))
        for other in (x.astype(mx.float32), mx.broadcast_to(x, (1, 3, 512))):
            self.assertTrue(
                mx.array_equal(maple._decode_projection(other, p), p(other))
            )
        p4 = nn.QuantizedLinear(512, 24, group_size=128, bits=4)
        p4.set_dtype(mx.bfloat16)
        self.assertIsNone(maple._row_quantized_metadata(p4))
        self.assertTrue(mx.array_equal(maple._decode_projection(x, p4), p4(x)))

    def test_failed_projection_probe_keeps_stock_result(self):
        p = self.projection()
        x = mx.ones((1, 1, 512), mx.bfloat16)
        with patch.object(
            maple, "_row_qmv_arrays", return_value=mx.zeros((1, 1, 24), mx.bfloat16)
        ):
            got = maple._decode_projection(x, p)
        self.assertFalse(p._maple_row_state["qmv_ok"])
        self.assertTrue(mx.array_equal(got, p(x)))
        with patch.object(
            maple,
            "_row_qmv_arrays",
            side_effect=AssertionError("retried failed kernel"),
        ):
            self.assertTrue(mx.array_equal(maple._decode_projection(x, p), p(x)))

    def test_inplace_metadata_mutation_invalidates_compact_copy(self):
        for name in ("scales", "biases"):
            with self.subTest(name=name):
                p = self.projection()
                x = mx.ones((1, 1, 512), mx.bfloat16)
                mx.eval(maple._decode_projection(x, p))
                metadata = getattr(p, name)
                metadata[0, 1] = mx.array(0.5, mx.bfloat16)
                self.assertIs(metadata, getattr(p, name))
                self.assertIsNone(maple._row_quantized_metadata(p))
                self.assertTrue(mx.array_equal(maple._decode_projection(x, p), p(x)))

    def test_compiled_projection_keeps_parameters_and_inputs_dynamic(self):
        # Shared shapes reuse a compiled dispatch graph across layers. Its
        # arrays must still come from each call, including after mutation.
        projections = [self.projection(), self.projection()]
        for step in range(6):
            p = projections[step % 2]
            if step == 2:
                p.weight[0] = mx.zeros_like(p.weight[0])
            if step == 3:
                p.scales[0] = mx.full_like(p.scales[0], 0.125)
            x = (mx.random.normal((1, 1, 512)) * (step + 1)).astype(mx.bfloat16)
            got, want = maple._decode_projection(x, p), p(x)
            mx.eval(got, want)
            self.assertTrue(p._maple_row_state["qmv_ok"])
            self.assertTrue(mx.array_equal(got, want))

    def experts(self):
        block = maple.MapleSwitchGLU(2048, 512, 8)
        block.set_dtype(mx.bfloat16)
        for name in ("up_gate_proj", "down_proj"):
            p = getattr(block, name).to_quantized(group_size=128, bits=2)
            self.uniform_metadata(p)
            setattr(block, name, p)
        return block

    def test_experts_exact_for_repeated_ids_and_clamped_activations(self):
        block = self.experts()
        ids = mx.array([7, 2, 0, 2, 5, 1, 7, 4], mx.uint32).reshape(1, 1, 8)
        for scale in (0.1, 3.0, 20.0):
            x = (mx.random.normal((1, 1, 4096)) * scale).astype(mx.bfloat16)[..., ::2]
            got, want = block(x, ids), block._call(x, ids)
            mx.eval(got, want)
            self.assertTrue(block._decode_row_ok)
            self.assertTrue(mx.array_equal(got, want))
        # Additive bias requires the stock expert path.
        block.up_gate_proj.bias = mx.ones((8, 1024), mx.bfloat16)
        with patch.object(
            block, "_decode_row_experts", side_effect=AssertionError("biased fusion")
        ):
            self.assertTrue(mx.array_equal(block(x, ids), block._call(x, ids)))

    def test_failed_expert_probe_uses_stock_fallback(self):
        block = self.experts()
        ids = mx.arange(8, dtype=mx.uint32).reshape(1, 1, 8)
        x = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)
        with patch.object(
            block,
            "_decode_row_experts",
            return_value=mx.zeros((1, 1, 8, 2048), mx.bfloat16),
        ):
            got = block(x, ids)
        self.assertFalse(block._decode_row_ok)
        self.assertTrue(mx.array_equal(got, block._call(x, ids)))
        with patch.object(
            block, "_decode_row_experts", side_effect=AssertionError("retried")
        ):
            self.assertTrue(mx.array_equal(block(x, ids), block._call(x, ids)))

    def test_unsupported_expert_inputs_use_stock_fallback(self):
        block = self.experts()
        for batch, length, top_k, dtype in (
            (1, 1, 4, mx.bfloat16),
            (1, 3, 8, mx.bfloat16),
            (2, 1, 8, mx.bfloat16),
            (1, 1, 8, mx.float32),
        ):
            x = mx.ones((batch, length, 2048), dtype)
            ids = mx.broadcast_to(mx.arange(top_k), (batch, length, top_k))
            with patch.object(
                block, "_decode_row_experts", side_effect=AssertionError("unsupported")
            ):
                self.assertTrue(mx.array_equal(block(x, ids), block._call(x, ids)))


@unittest.skipUnless(mx.__version__ == "0.32.0", "targets MLX 0.32.0")
class TestMaplePortableKernels(unittest.TestCase):
    def test_live_group_metadata_after_mutation_and_layout_changes(self):
        with patch.object(maple, "_maple_native", None):
            for k, bias in product((512, 2048), (False, True)):
                p = nn.QuantizedLinear(k, 24, bias=bias, group_size=128, bits=2)
                p.set_dtype(mx.bfloat16)
                for name in (None, "weight", "scales", "biases"):
                    if name:
                        value = getattr(p, name)
                        value[0, 1] = 0
                        setattr(p, name, mx.contiguous(value.T).T)
                    x = mx.random.normal((1, 1, k * 2)).astype(mx.bfloat16)[..., ::2]
                    got, want = maple._decode_projection(x, p), p(x)
                    self.assertTrue(mx.array_equal(got, want))
                    self.assertTrue(p._maple_row_state["qmv_ok"])
                    self.assertIsNone(p._maple_row_state["sources"])

    def test_live_expert_metadata_and_repeated_ids(self):
        block = maple.MapleSwitchGLU(2048, 512, 8)
        block.set_dtype(mx.bfloat16)
        for name in ("up_gate_proj", "down_proj"):
            p = getattr(block, name).to_quantized(group_size=128, bits=2)
            setattr(block, name, p)
        ids = mx.array([7, 2, 0, 2, 5, 1, 7, 4], mx.uint32).reshape(1, 1, 8)
        with patch.object(maple, "_maple_native", None):
            for scale in (0.1, 3.0, 20.0):
                x = (mx.random.normal((1, 1, 4096)) * scale).astype(mx.bfloat16)
                x = x[..., ::2]
                self.assertTrue(mx.array_equal(block(x, ids), block._call(x, ids)))
                self.assertTrue(block._decode_row_ok)
                block.up_gate_proj.scales[2, 0, 1] = scale
                block.down_proj.biases[7, 0, 2] = -scale
                block.down_proj.weight[0, 0] = 0


# Independent stock-MLX specification: forced tokens do not admit neighbors.
def flash_head_reference(head, h, lm_head):
    hv = h[:, -1, :]
    top = mx.argpartition(head.centroids(hv), kth=-head.n_probes, axis=-1)[
        ..., -head.n_probes :
    ]
    ids = head.token_map[top[0]].reshape(-1)
    logits = mx.gather_qmm(
        hv.reshape(1, 1, 1, 1, -1),
        head.head["weight"],
        head.head["scales"],
        head.head["biases"],
        rhs_indices=top[:, None, :],
        transpose=True,
        group_size=head.head_group_size,
        bits=head.head_bits,
    ).reshape(-1)
    if head._force_ids.size:
        forced = mx.quantized_matmul(
            hv,
            lm_head.weight[head._force_ids],
            scales=lm_head.scales[head._force_ids],
            biases=lm_head.biases[head._force_ids],
            transpose=True,
            group_size=lm_head.group_size,
            bits=lm_head.bits,
            mode=getattr(lm_head, "mode", "affine"),
        )[0]
        ids = mx.concatenate([ids, head._force_ids])
        logits = mx.concatenate([logits, forced])
    out = mx.full((1, 1, lm_head.weight.shape[0]), -float("inf"), logits.dtype)
    out[0, 0, ids] = logits
    return out


class TestFlashHead(unittest.TestCase):
    def test_selection_matches_mlx_for_ties_and_bfloat_bit_patterns(self):
        for n in (31, 1024, 4748, 8192):
            cases = [
                mx.random.normal((1, n)).astype(mx.bfloat16),
                mx.zeros((1, n), mx.bfloat16),
                (mx.arange(n) * 40503).astype(mx.uint16).view(mx.bfloat16)[None],
            ]
            for x, k in product(cases, (1, min(n, 512), n)):
                got = maple._head_select(x, k)
                want = mx.argpartition(x, kth=-k)[..., -k:]
                self.assertTrue(mx.array_equal(mx.sort(got), mx.sort(want)))

    def make_head(self, forced, probes=1):
        args = maple.ModelArgs(
            hidden_size=128,
            flash_head={
                "scaled_centroids": True,
                "n_clusters": 4,
                "cluster_size": 32,
                "n_probes": probes,
                "force_tokens": forced,
            },
        )
        head = maple.FlashHead(args)
        lm = nn.QuantizedLinear(128, 128, bias=False, group_size=64, bits=4)
        lm.set_dtype(mx.bfloat16)
        head.token_map = mx.arange(128, dtype=mx.int32).reshape(4, 32)
        head.head = {
            k: lm[k].reshape(4, 32, -1) for k in ("weight", "scales", "biases")
        }
        head.centroids = lambda h: mx.array([[4, 3, 2, 1]], mx.bfloat16)
        return head, lm

    def test_candidate_mask_and_logits_with_forced_tokens(self):
        for forced in ([], [63], [0], [0, 63, 127], [63, 63]):
            for probes in (1, 4):
                with self.subTest(forced=forced, probes=probes):
                    head, lm = self.make_head(forced, probes)
                    for seed in range(5):
                        h = mx.random.normal(
                            (1, 1, 256), key=mx.random.key(seed)
                        ).astype(mx.bfloat16)[..., ::2]
                        got, want = head(h, lm), flash_head_reference(head, h, lm)
                        mx.eval(got, want)
                        if mx.__version__ == "0.32.0" and mx.default_device() == mx.gpu:
                            self.assertTrue(head._scatter)
                        self.assertTrue(mx.array_equal(got, want))
                        expected = set(range(32 * probes)) | set(forced)
                        finite = mx.isfinite(got).reshape(-1).tolist()
                        self.assertEqual(
                            {i for i, value in enumerate(finite) if value}, expected
                        )

    def test_forced_rows_follow_inplace_weight_edits(self):
        head, lm = self.make_head([63])
        h = mx.ones((1, 1, 128), mx.bfloat16)
        mx.eval(head(h, lm))
        lm.scales[63] = mx.full(lm.scales[63].shape, 0.5, mx.bfloat16)
        got, want = head(h, lm), flash_head_reference(head, h, lm)
        mx.eval(got, want)
        self.assertTrue(mx.array_equal(got, want))

    def test_failed_scatter_probe_latches_portable_fallback(self):
        head, lm = self.make_head([63])
        h = mx.ones((1, 1, 128), mx.bfloat16)
        with patch.object(maple, "_flash_scatter", return_value=mx.zeros((1, 1, 128))):
            got = head(h, lm)
        self.assertFalse(head._scatter)
        self.assertTrue(mx.array_equal(got, flash_head_reference(head, h, lm)))
        with patch.object(
            maple, "_flash_scatter", side_effect=AssertionError("retried")
        ):
            self.assertTrue(
                mx.array_equal(head(h, lm), flash_head_reference(head, h, lm))
            )


class TestMaplePortability(unittest.TestCase):
    def setUp(self):
        device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, device)
        mx.random.seed(7)
        self.args = dict(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
            sliding_window=8,
            layer_types=["full_attention", "sliding_attention"],
        )
        self.model = maple.Model(maple.ModelArgs(**self.args))
        self.tokens = mx.array([[3, 8, 9, 4, 21, 17, 2, 7, 14, 5, 6, 10, 1]])

    def test_prefill_and_incremental_decode_agree_across_window_boundary(self):
        expected = self.model(self.tokens)
        cache = self.model.make_cache()
        outputs = [self.model(self.tokens[:, :4], cache)]
        for position in range(4, self.tokens.shape[1]):
            outputs.append(self.model(self.tokens[:, position : position + 1], cache))
        actual = mx.concatenate(outputs, axis=1)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual([c.offset for c in cache], [self.tokens.shape[1]] * 2)

    def test_batched_prefill_matches_individual_sequences(self):
        tokens = mx.concatenate([self.tokens, self.tokens[:, ::-1]], axis=0)
        actual = self.model(tokens)
        expected = mx.concatenate(
            [self.model(tokens[i : i + 1]) for i in range(2)], axis=0
        )
        self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_checkpoint_model_file_loads_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maple.py"
            shutil.copy2(maple.__file__, path)
            name = "_maple_checkpoint_portability_test"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
                standalone = module.Model(module.ModelArgs(**self.args))
                standalone.update(self.model.parameters())
                self.assertTrue(
                    mx.array_equal(standalone(self.tokens), self.model(self.tokens))
                )
            finally:
                del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
