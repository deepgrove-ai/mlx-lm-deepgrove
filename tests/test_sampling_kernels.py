"""Exact permutation, BF16 threshold semantics, and seeded sampler equivalence."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

SUPPORTED = mx.__version__ == "0.32.0" and mx.metal.is_available()
if SUPPORTED:
    from mlx_lm import _sampling_kernels as kernels
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    apply_xtc,
    categorical_sampling,
    make_sampler,
)


@unittest.skipUnless(SUPPORTED, "verified Metal runtime")
class TestSamplingKernels(unittest.TestCase):
    n = 151936

    def values(self, seed=0):
        x = (mx.random.normal((1, self.n), key=mx.random.key(seed)) * 3).astype(
            mx.bfloat16
        )
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def equal(self, a, b):
        mx.eval(a, b)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(mx.array_equal(a, b, equal_nan=True))

    def test_stable_sort_all_bf16_bits_and_partial_groups(self):
        rng = np.random.default_rng(53)
        for n in (1, 31, 33, 257, 4096, 151935, 151936, 151937):
            bits = np.resize(np.arange(65536, dtype=np.uint16), n)
            rng.shuffle(bits)
            x = mx.array(bits).view(mx.bfloat16)[None]
            with self.subTest(n=n):
                self.equal(kernels._radix_argsort(x), mx.argsort(x))
        # Large tie runs span SIMD and threadgroup boundaries, including zeros
        # and the subnormal BF16 values which MLX compares as zero on Metal.
        bits = mx.array(
            [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFF81], mx.uint16
        )
        x = mx.tile(bits, self.n // bits.size).view(mx.bfloat16)[None]
        self.equal(kernels._radix_argsort(x), mx.argsort(x))

    def test_top_p_masks_and_rounded_threshold_boundaries(self):
        x = self.values()
        sparse = mx.where(mx.arange(self.n)[None] % 9 == 0, x, -float("inf"))
        strided = mx.stack([x, -x], axis=-1).reshape(1, -1)[..., ::2]
        inputs = [
            x,
            sparse,
            strided,
            mx.full(x.shape, -12.0, mx.bfloat16),
            mx.full(x.shape, -float("inf"), mx.bfloat16),
            mx.full(x.shape, float("inf"), mx.bfloat16),
            mx.full(x.shape, float("nan"), mx.bfloat16),
        ]
        # Halfway and adjacent values around BF16 thresholds test scalar
        # coercion: comparing against a float32 threshold can change the mask.
        cdf = mx.cumsum(mx.take_along_axis(mx.exp(x), mx.argsort(x), axis=-1), axis=-1)
        mx.eval(cdf)
        boundary = float(cdf[0, self.n // 2])
        raw = int(mx.array([boundary], mx.bfloat16).view(mx.uint16)[0])
        upper = float(mx.array([raw + 1], mx.uint16).view(mx.bfloat16)[0])
        midpoint = (boundary + upper) / 2
        thresholds = [
            0.0,
            0.05,
            0.5,
            0.9,
            0.95,
            0.999,
            1.0,
            1 - boundary,
            1 - midpoint,
            1 - float(np.nextafter(midpoint, np.inf)),
            1 - float(np.nextafter(midpoint, -np.inf)),
        ]
        for i, values in enumerate(inputs):
            for p in thresholds:
                with self.subTest(input=i, top_p=p):
                    self.equal(
                        kernels._apply_top_p_radix(values, p), apply_top_p(values, p)
                    )

    def test_top_p_scan_chunk_boundaries_and_retained_outputs(self):
        for seed in (3, 17):
            x = self.values(seed)
            original = np.array(x.view(mx.uint16))
            cdf = mx.cumsum(
                mx.take_along_axis(mx.exp(x), mx.argsort(x), axis=-1), axis=-1
            )
            # Exercise the carry into every chunk at and just below its BF16
            # CDF value; changing the accumulation order can change these masks.
            raw = cdf[:, 4095::4096].view(mx.uint16).reshape(-1)
            thresholds = mx.concatenate([raw, raw - 1]).view(mx.bfloat16).tolist()
            retained = []
            for threshold in thresholds:
                p = 1 - threshold
                got = kernels._apply_top_p_radix(x, p)
                want = apply_top_p(x, p)
                self.equal(got.view(mx.uint16), want.view(mx.uint16))
                retained.append((got, want))
            for got, want in retained:
                self.equal(got.view(mx.uint16), want.view(mx.uint16))
            np.testing.assert_array_equal(np.array(x.view(mx.uint16)), original)

    def test_first_probe_preserves_rng_and_composed_sampler(self):
        x = self.values(4)
        for modifiers in (False, True):
            with patch.object(kernels, "_ready", None):
                fast = make_sampler(
                    0.8,
                    0.95,
                    min_p=0.02 if modifiers else 0.0,
                    top_k=64 if modifiers else 0,
                    xtc_probability=0.1 if modifiers else 0.0,
                    xtc_threshold=0.1,
                )
                for seed in range(8):
                    mx.random.seed(seed)
                    got = fast(x)
                    mx.eval(got)
                    mx.random.seed(seed)
                    y = apply_top_p(x, 0.95)
                    if modifiers:
                        y = apply_min_p(y, 0.02, 1)
                        y = apply_xtc(y, 0.1, 0.1, [])
                        y = apply_top_k(y, 64)
                    want = categorical_sampling(y, 0.8)
                    self.equal(got, want)
                self.assertTrue(kernels._ready)

    def test_failed_probe_latches_stock_fallback(self):
        x = self.values(7)
        with patch.object(
            kernels, "_radix_argsort", return_value=mx.zeros(x.shape, mx.uint32)
        ):
            self.assertFalse(kernels._probe())
        with patch.object(kernels, "_ready", None), patch.object(
            kernels, "_probe", return_value=False
        ) as probe:
            for _ in range(2):
                self.equal(kernels.apply_top_p_fast(x, 0.95), apply_top_p(x, 0.95))
            self.assertFalse(kernels._ready)
            self.assertEqual(probe.call_count, 1)

    def test_unsupported_calls_keep_stock_path(self):
        x = self.values(8)
        with patch.object(
            kernels, "_apply_top_p_radix", side_effect=AssertionError("unsupported")
        ):
            for values in (
                x.astype(mx.float32),
                x[:, :-1],
                x[0],
                mx.broadcast_to(x, (2, self.n)),
            ):
                self.equal(
                    kernels.apply_top_p_fast(values, 0.95), apply_top_p(values, 0.95)
                )
            with patch.object(mx, "__version__", "0.32.1"):
                self.equal(kernels.apply_top_p_fast(x, 0.95), apply_top_p(x, 0.95))
            with mx.stream(mx.cpu):
                self.equal(kernels.apply_top_p_fast(x, 0.95), apply_top_p(x, 0.95))


if __name__ == "__main__":
    unittest.main()
