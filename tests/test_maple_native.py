"""Exact QKV/cache equivalence, including ownership and prefill transitions."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.models import maple
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    load_prompt_cache,
    save_prompt_cache,
)


@unittest.skipIf(maple._maple_native is None, "optional native extension is not built")
class TestNativeMaple(unittest.TestCase):
    def test_snapshot_tracks_array_descriptors_and_inplace_assignment(self):
        a = mx.arange(16)
        snapshot = maple._maple_native.ArraySnapshot([a])
        self.assertTrue(snapshot.matches([a]))
        mx.eval(a)
        self.assertTrue(snapshot.matches([a]))
        self.assertFalse(snapshot.matches([]))
        a[0] = 0
        self.assertFalse(snapshot.matches([a]))

    def test_native_rebuilds_norm_weights_after_inplace_edit(self):
        attn = self.attention(True)
        x = mx.random.normal((24, 128), key=mx.random.key(8)).astype(mx.bfloat16)
        cache = maple.KVCache()
        mx.eval(cache.update_native(x, attn))
        attn.q_norm.weight[0] = mx.array(2.0, mx.bfloat16)
        attn.k_norm.weight = attn.k_norm.weight * 2
        q, k, v = cache.update_native(x, attn)
        ref = attn._qk_reference(x[:20], 1)
        self.assert_arrays_equal(q.reshape(16, 128), ref[:16])
        self.assert_arrays_equal(k[:, :, -1, :].reshape(4, 128), ref[16:])
        self.assert_arrays_equal(v[:, :, -1, :].reshape(4, 128), x[20:])

    def test_norm_dtype_change_after_warmup_uses_portable_path(self):
        attn = self.attention(True)
        attn.qkv_proj.set_dtype(mx.bfloat16)
        attn.o_proj.set_dtype(mx.bfloat16)
        x = mx.ones((1, 1, 2048), mx.bfloat16)
        mx.eval(attn(x, cache=maple.KVCache()))
        self.assertTrue(attn._native_qkv)
        attn.q_norm.weight = attn.q_norm.weight.astype(mx.float32) * 1.1
        cache = maple.KVCache()
        with patch.object(
            cache, "update_native", side_effect=AssertionError("unsupported dtype")
        ):
            got = attn(x, cache=cache)
        want = attn(x, cache=KVCache())
        self.assert_arrays_equal(got, want)

    def attention(self, rope):
        args = maple.ModelArgs(
            num_hidden_layers=1,
            layer_types=["sliding_attention" if rope else "full_attention"],
        )
        attn = maple.MapleAttention(args, 0)
        attn.q_norm.weight = mx.linspace(0.5, 1.5, 128).astype(mx.bfloat16)
        attn.k_norm.weight = mx.linspace(1.5, 0.5, 128).astype(mx.bfloat16)
        self.assertTrue(attn._probe_native_qkv(), "native path did not activate")
        return attn

    def assert_arrays_equal(self, a, b):
        mx.eval(a, b)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(mx.array_equal(a, b))

    def cache_pair(self, window, keep):
        if window is None:
            native, stock = maple.KVCache(), KVCache()
        else:
            native = maple.RotatingKVCache(window, keep)
            stock = RotatingKVCache(window, keep)
        native.step = stock.step = 4
        return native, stock

    def test_native_cache_matches_stock_across_transitions(self):
        for rope in (False, True):
            attn = self.attention(rope)
            for window, keep, prefill in (
                (None, 0, 0),
                (None, 0, 5),
                (8, 0, 0),
                (8, 0, 13),
                (8, 2, 5),
            ):
                with self.subTest(rope=rope, window=window, keep=keep, prefill=prefill):
                    native, stock = self.cache_pair(window, keep)

                    def append_many(count, seed):
                        k = mx.random.normal(
                            (1, 4, count, 128), key=mx.random.key(seed)
                        ).astype(mx.bfloat16)
                        v = -k
                        for a, b in zip(
                            native.update_and_fetch(k, v), stock.update_and_fetch(k, v)
                        ):
                            self.assert_arrays_equal(a, b)

                    if prefill:
                        append_many(prefill, 123)
                    retained = []
                    for i in range(26):
                        # Resume with a multi-token prefill after the ring has wrapped.
                        if i == 15:
                            append_many(3, 456)
                        if i == 5 and stock.is_trimmable():
                            self.assertEqual(native.trim(2), stock.trim(2))
                        x = mx.random.normal(
                            (24, 128), key=mx.random.key(i + 30)
                        ).astype(mx.bfloat16)
                        ref = attn._qk_fused(x, stock.offset)
                        q, k, v = native.update_native(x, attn)
                        a, b = stock.update_and_fetch(
                            ref[16:20].reshape(1, 4, 1, 128),
                            ref[20:].reshape(1, 4, 1, 128),
                        )
                        self.assert_arrays_equal(q, ref[:16].reshape(1, 16, 1, 128))
                        self.assert_arrays_equal(k, a)
                        self.assert_arrays_equal(v, b)
                        self.assertEqual(native.offset, stock.offset)
                        if window:
                            self.assertEqual(native._idx, stock._idx)
                        if i in (4, 7, 10):
                            # Retain the cache alone at step 7; also retain views otherwise.
                            snapshot = copy.copy(native)
                            arrays = [] if i == 7 else [*snapshot.state, q, k, v]
                            retained.append(
                                (
                                    snapshot,
                                    arrays,
                                    [
                                        np.array(t.astype(mx.float32))
                                        for t in arrays or stock.state
                                    ],
                                )
                            )
                        del q, k, v, a, b, ref
                    for snapshot, arrays, saved in retained:
                        for t, want in zip(arrays, saved):
                            np.testing.assert_array_equal(
                                np.array(t.astype(mx.float32)), want
                            )
                        for t, want in zip(snapshot.state, saved[:2]):
                            np.testing.assert_array_equal(
                                np.array(t.astype(mx.float32)), want
                            )
                    with tempfile.TemporaryDirectory() as directory:
                        path = str(Path(directory) / "cache.safetensors")
                        save_prompt_cache(path, [native])
                        restored = load_prompt_cache(path)[0]
                    self.assertEqual(restored.meta_state, native.meta_state)
                    x = mx.ones((24, 128), mx.bfloat16)
                    ref = attn._qk_fused(x, restored.offset)
                    _, k, v = native.update_native(x, attn)
                    a, b = restored.update_and_fetch(
                        ref[16:20].reshape(1, 4, 1, 128),
                        ref[20:].reshape(1, 4, 1, 128),
                    )
                    self.assert_arrays_equal(k, a)
                    self.assert_arrays_equal(v, b)

    def test_native_probe_rejects_wrong_output(self):
        attn = self.attention(True)
        with patch.object(
            maple._maple_native,
            "prepare_qkv",
            side_effect=lambda x, w, inv, kv, *args: kv,
        ):
            self.assertFalse(attn._probe_native_qkv())

    def test_native_probe_rejects_older_arithmetic(self):
        attn = self.attention(False)
        with patch.object(maple._maple_native, "arithmetic_version", 2):
            self.assertFalse(attn._probe_native_qkv())

    def test_unsupported_attention_does_not_reuse_a_warmed_fusion(self):
        for dtype, head_dim in ((mx.float32, 128), (mx.bfloat16, 64)):
            args = maple.ModelArgs(
                num_hidden_layers=1,
                head_dim=head_dim,
                layer_types=["sliding_attention"],
            )
            attn = maple.MapleAttention(args, 0)
            attn.set_dtype(dtype)
            attn._fused_qk = True
            attn._native_qkv = False
            x = mx.ones((1, 1, 2048), dtype)
            with patch.object(
                attn,
                "_qk_fused",
                side_effect=AssertionError("unsupported fused dtype/layout"),
            ):
                a = attn(x, cache=maple.KVCache())
                b = attn(x, cache=KVCache())
            self.assert_arrays_equal(a, b)

    def test_alternating_native_and_stock_cache_writes(self):
        attn = self.attention(True)
        for window, keep in ((None, 0), (8, 0), (8, 2)):
            with self.subTest(window=window, keep=keep):
                fused, stock = self.cache_pair(window, keep)
                retained = []
                for i in range(26):
                    x = mx.random.normal((24, 128), key=mx.random.key(1000 + i)).astype(
                        mx.bfloat16
                    )
                    ref = attn._qk_fused(x, stock.offset)
                    k, v = stock.update_and_fetch(
                        ref[16:20].reshape(1, 4, 1, 128),
                        ref[20:].reshape(1, 4, 1, 128),
                    )
                    if i % 3:
                        _, a, b = fused.update_native(x, attn)
                    else:
                        a, b = fused.update_and_fetch(
                            ref[16:20].reshape(1, 4, 1, 128),
                            ref[20:].reshape(1, 4, 1, 128),
                        )
                    self.assert_arrays_equal(a, k)
                    self.assert_arrays_equal(b, v)
                    self.assertEqual(fused.offset, stock.offset)
                    if window:
                        self.assertEqual(fused._idx, stock._idx)
                    if i in (4, 9):
                        retained.extend(
                            (t, np.array(t.astype(mx.float32))) for t in (a, b)
                        )
                for t, want in retained:
                    np.testing.assert_array_equal(np.array(t.astype(mx.float32)), want)

    def test_float32_saved_cache_uses_portable_update(self):
        attn = self.attention(False)
        attn.set_dtype(mx.bfloat16)
        x = mx.ones((1, 1, 2048), mx.bfloat16)
        k = mx.random.normal((1, 4, 5, 128), key=mx.random.key(50))
        v = -k
        results = []
        for enabled in (False, True):
            cache = maple.KVCache()
            cache.update_and_fetch(k, v)
            self.assertFalse(cache.native_compatible())
            attn._native_qkv = enabled
            results.append(attn(x, cache=cache))
            self.assertIsNone(cache._native_buffer)
            self.assertEqual(cache.keys.dtype, mx.float32)
        self.assert_arrays_equal(*results)

    def test_unsupported_dtype_and_layout_rejected(self):
        attn = self.attention(False)
        kv = mx.zeros((2048 + 8 * 16 * 128,), mx.bfloat16)
        native = maple._maple_native.prepare_qkv
        for x, index in (
            (mx.ones((24, 128)), 0),
            (mx.ones((24, 128), mx.bfloat16), 16),
        ):
            with self.assertRaises(ValueError):
                native(x, attn._qk_w, attn._inv_freq, kv, 0, index, 0, attn._eps)


if __name__ == "__main__":
    unittest.main()
