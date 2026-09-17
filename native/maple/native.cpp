#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

static const char *shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void maple_copy_cache(const device ushort4* src [[buffer(0)]],
    device ushort4* dst [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}
kernel void maple_prepare_qkv(
    const device bfloat* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    const device float* inv_freq [[buffer(2)]],
    device bfloat* q [[buffer(3)]],
    device bfloat* kv [[buffer(4)]],
    constant float& pos [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant int& capacity [[buffer(7)]],
    constant int& write_pos [[buffer(8)]],
    constant int& rope_dim [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint head = gid.y;
    uint lane = gid.x;
    constexpr int HEAD_DIM = 128;
    constexpr int per_lane = 4;
    const device bfloat* xh = x + head * HEAD_DIM;
    device bfloat* oh = head < 16
        ? q + head * HEAD_DIM
        : kv + ((head - 16) * capacity + write_pos) * HEAD_DIM;
    if (head >= 20) {
        for (int i = 0; i < per_lane; ++i) {
            int j = lane * per_lane + i;
            oh[j] = xh[j];
        }
        return;
    }
    const device bfloat* wh = w + head * HEAD_DIM;
    float ss = 0.0f;
    for (int i = 0; i < per_lane; ++i) {
        float v = (float)xh[lane * per_lane + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    float scale = metal::precise::rsqrt(ss / HEAD_DIM + eps);
    for (int i = 0; i < per_lane; ++i) {
        int j = lane * per_lane + i;
        float v = (float)(bfloat)((float)wh[j] * ((float)xh[j] * scale));
        if (rope_dim > 0 && j < rope_dim) {
            int rhalf = rope_dim / 2;
            int p = j < rhalf ? j : j - rhalf;
            float theta = pos * inv_freq[p];
            float c = metal::fast::cos(theta);
            float s = metal::fast::sin(theta);
            int j2 = j < rhalf ? j + rhalf : j - rhalf;
            float u = (float)(bfloat)((float)wh[j2] * ((float)xh[j2] * scale));
            v = j < rhalf ? (v * c - u * s) : (u * s + v * c);
        }
        oh[j] = (bfloat)v;
    }
}
)METAL";

class PrepareQKV : public mx::Primitive {
public:
  PrepareQKV(mx::Stream s, int position, int write_pos, int rope_dim, float eps)
      : Primitive(s), position_(position), write_pos_(write_pos),
        rope_dim_(rope_dim), eps_(eps) {}
  const char *name() const override { return "MaplePrepareQKV"; }
  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("MaplePrepareQKV requires Metal");
  }
  void eval_gpu(const std::vector<mx::array> &in,
                std::vector<mx::array> &out) override {
    for (const auto &a : in) {
      if (!a.flags().row_contiguous)
        throw std::invalid_argument(
            "MaplePrepareQKV requires contiguous inputs");
    }
    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    // Honor MLX's copy-on-write/donation predicate. The general internal
    // copy_gpu helper is not exported by the wheel, so copy this contiguous
    // bf16 buffer with a bit-preserving vector kernel when it is shared.
    bool can_donate = in[3].is_donatable();
    auto lib = d.get_library("maple_native_qkv_v3",
                             [] { return std::string(shader); });
    auto &enc = mx::metal::get_command_encoder(s);
    if (can_donate) {
      out[0].copy_shared_buffer(in[3]);
    } else {
      out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
      auto copy = d.get_kernel("maple_copy_cache", lib);
      enc.set_compute_pipeline_state(copy);
      enc.set_input_array(in[3], 0);
      enc.set_output_array(out[0], 1);
      enc.dispatch_threads(MTL::Size(out[0].size() / 4, 1, 1),
                           MTL::Size(256, 1, 1));
    }
    auto kernel = d.get_kernel("maple_prepare_qkv", lib);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(in[0], 0);
    enc.set_input_array(in[1], 1);
    enc.set_input_array(in[2], 2);
    enc.set_output_array(out[0], 3);
    enc.set_output_array(out[0], 4, 2048 * sizeof(uint16_t));
    enc.set_bytes(float(position_), 5);
    enc.set_bytes(eps_, 6);
    enc.set_bytes(int((in[3].size() - 2048) / 1024), 7);
    enc.set_bytes(write_pos_, 8);
    enc.set_bytes(rope_dim_, 9);
    enc.dispatch_threads(MTL::Size(32, 24, 1), MTL::Size(32, 1, 1));
  }

private:
  int position_, write_pos_, rope_dim_;
  float eps_;
};

mx::array prepare_qkv(const mx::array &x, const mx::array &w,
                      const mx::array &inv, const mx::array &kv, int position,
                      int write_pos, int rope_dim, float eps,
                      mx::StreamOrDevice s = {}) {
  if (x.dtype() != mx::bfloat16 || w.dtype() != mx::bfloat16 ||
      kv.dtype() != mx::bfloat16 || inv.dtype() != mx::float32 ||
      x.size() != 3072 || w.size() != 2560 || kv.ndim() != 1 ||
      kv.size() <= 2048 || (kv.size() - 2048) % 1024 || write_pos < 0 ||
      write_pos >= (kv.size() - 2048) / 1024 ||
      (rope_dim != 0 && rope_dim != 64) || (rope_dim && inv.size() < 32)) {
    throw std::invalid_argument("Unsupported Maple QKV layout");
  }
  auto stream = mx::to_stream(s);
  return mx::array(
      kv.shape(), mx::bfloat16,
      std::make_shared<PrepareQKV>(stream, position, write_pos, rope_dim, eps),
      {x, w, inv, kv});
}

// Python __setitem__ preserves the wrapper's identity but replaces its MLX
// array descriptor. Keep descriptor owners alive to avoid pointer reuse, and
// compare IDs without evaluating or reading any GPU memory.
class ArraySnapshot {
public:
  explicit ArraySnapshot(std::vector<mx::array> arrays)
      : arrays_(std::move(arrays)) {}
  bool matches(const std::vector<mx::array> &arrays) const {
    if (arrays.size() != arrays_.size()) return false;
    for (size_t i = 0; i < arrays.size(); ++i)
      if (arrays[i].id() != arrays_[i].id()) return false;
    return true;
  }
private:
  std::vector<mx::array> arrays_;
};

NB_MODULE(_maple_native, m) {
  m.attr("arithmetic_version") = 3;
  nb::class_<ArraySnapshot>(m, "ArraySnapshot")
      .def(nb::init<std::vector<mx::array>>())
      .def("matches", &ArraySnapshot::matches);
  m.def("prepare_qkv", &prepare_qkv, "x"_a, "w"_a, "inv_freq"_a, "kv"_a,
        "position"_a, "write_pos"_a, "rope_dim"_a, "eps"_a, nb::kw_only(),
        "stream"_a = nb::none());
}
