#include "slmpt/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "slmpt/parallel.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace slmpt {
namespace {

// Vectorized dot product (falls back to a scalar loop everywhere else).
inline float dot_f32(const float* a, const float* b, int n) {
#if defined(__ARM_NEON)
  float32x4_t acc = vdupq_n_f32(0.f);
  int i = 0;
  for (; i + 4 <= n; i += 4) acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = vaddvq_f32(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#elif defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  float tmp[8];
  _mm256_storeu_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#else
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
#endif
}

// Four dot products at once: out[k] = dot(x, w[k]) for k = 0..3.
// Uses one independent accumulator per output so the FMA latency is hidden by
// instruction-level parallelism (a single accumulator chain runs ~4x slower),
// and loads x once per 4-wide chunk to feed all four outputs.
inline void dot4_f32(const float* x, const float* w0, const float* w1,
                     const float* w2, const float* w3, int n, float* out) {
#if defined(__ARM_NEON)
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t xv = vld1q_f32(x + i);
    a0 = vmlaq_f32(a0, xv, vld1q_f32(w0 + i));
    a1 = vmlaq_f32(a1, xv, vld1q_f32(w1 + i));
    a2 = vmlaq_f32(a2, xv, vld1q_f32(w2 + i));
    a3 = vmlaq_f32(a3, xv, vld1q_f32(w3 + i));
  }
  float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
  float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
  for (; i < n; ++i) {
    const float xv = x[i];
    s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
  }
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
#elif defined(__AVX2__)
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 xv = _mm256_loadu_ps(x + i);
    a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + i), a0);
    a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + i), a1);
    a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + i), a2);
    a3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + i), a3);
  }
  auto hsum = [](__m256 v) {
    float t[8]; _mm256_storeu_ps(t, v);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
  };
  float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
  for (; i < n; ++i) {
    const float xv = x[i];
    s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
  }
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
#else
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  for (int i = 0; i < n; ++i) {
    const float xv = x[i];
    s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
  }
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
#endif
}

// y += a * x
inline void saxpy(float* y, const float* x, float a, int n) {
#if defined(__ARM_NEON)
  const float32x4_t va = vdupq_n_f32(a);
  int i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(y + i, vmlaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
  for (; i < n; ++i) y[i] += a * x[i];
#elif defined(__AVX2__)
  const __m256 va = _mm256_set1_ps(a);
  int i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i,
                     _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += a * x[i];
#else
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
#endif
}

// y[rows,out] = x[rows,in] @ W[out,in]^T   (W row-major [out][in])
void linear_fwd(const float* x, const float* W, float* y, int rows, int in, int out) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * in;
      float* yr = y + static_cast<size_t>(r) * out;
      int o = 0;
      for (; o + 4 <= out; o += 4) {
        dot4_f32(xr, W + static_cast<size_t>(o + 0) * in,
                 W + static_cast<size_t>(o + 1) * in,
                 W + static_cast<size_t>(o + 2) * in,
                 W + static_cast<size_t>(o + 3) * in, in, yr + o);
      }
      for (; o < out; ++o) yr[o] = dot_f32(xr, W + static_cast<size_t>(o) * in, in);
    }
  });
}

// Accumulates into dx, dW (caller zeroes them once per step).
void linear_bwd(const float* x, const float* W, const float* dy, float* dx, float* dW,
                int rows, int in, int out) {
  if (dx) {
    parallel_for(rows, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const float* dyr = dy + static_cast<size_t>(r) * out;
        float* dxr = dx + static_cast<size_t>(r) * in;
        for (int i = 0; i < in; ++i) dxr[i] = 0.f;
        for (int o = 0; o < out; ++o) {
          const float g = dyr[o];
          if (g == 0.f) continue;
          const float* wo = W + static_cast<size_t>(o) * in;
          saxpy(dxr, wo, g, in);
        }
      }
    });
  }
  if (!dW) return;
  // Block four output rows together: x[r] is loaded once and reused by all
  // four saxpy updates, cutting x traffic by 4x.
  const int ob = (out + 3) / 4;
  parallel_for(ob, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
      const int o = b * 4;
      const int cnt = (out - o < 4) ? (out - o) : 4;
      float* dw[4] = {nullptr, nullptr, nullptr, nullptr};
      for (int k = 0; k < cnt; ++k) dw[k] = dW + static_cast<size_t>(o + k) * in;
      for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<size_t>(r) * in;
        const float* dyr = dy + static_cast<size_t>(r) * out;
        for (int k = 0; k < cnt; ++k) {
          const float g = dyr[o + k];
          if (g != 0.f) saxpy(dw[k], xr, g, in);
        }
      }
    }
  });
}

// RMSNorm: y = g * x / sqrt(mean(x^2) + eps)
void rmsnorm_fwd(const float* x, const float* g, float* y, float* rstd, int rows,
                 int C, float eps) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * C;
      float ms = 0.f;
      for (int i = 0; i < C; ++i) ms += xr[i] * xr[i];
      ms /= C;
      const float rs = 1.f / std::sqrt(ms + eps);
      rstd[r] = rs;
      float* yr = y + static_cast<size_t>(r) * C;
      for (int i = 0; i < C; ++i) yr[i] = g[i] * xr[i] * rs;
    }
  });
}

void rmsnorm_bwd(const float* x, const float* g, const float* dy, float* dx, float* dg,
                 const float* rstd, int rows, int C) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * C;
      const float* dyr = dy + static_cast<size_t>(r) * C;
      float* dxr = dx + static_cast<size_t>(r) * C;
      const float inv = rstd[r];
      float S = 0.f;
      for (int i = 0; i < C; ++i) S += g[i] * dyr[i] * xr[i];
      const float k = inv * inv * inv * S / C;
      for (int i = 0; i < C; ++i) dxr[i] = g[i] * dyr[i] * inv - xr[i] * k;
    }
  });
  if (dg) {
    for (int i = 0; i < C; ++i) {
      float sg = 0.f;
      for (int r = 0; r < rows; ++r)
        sg += dy[static_cast<size_t>(r) * C + i] * x[static_cast<size_t>(r) * C + i] *
              rstd[r];
      dg[i] += sg;
    }
  }
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float silu_grad(float x) {
  const float s = 1.f / (1.f + std::exp(-x));
  return s + x * s * (1.f - s);
}
void silu_fwd(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) y[i] = silu(x[i]);
}
void silu_bwd(const float* x, const float* dy, float* dx, int n) {
  for (int i = 0; i < n; ++i) dx[i] = dy[i] * silu_grad(x[i]);
}

}  // namespace

Model::Model(const Config& cfg) : cfg_(cfg), rng_(cfg.seed) {
  C_ = cfg.n_embd;
  L_ = cfg.n_layer;
  H_ = cfg.n_head;
  T_ = cfg.block;
  V_ = cfg.vocab;
  B_ = cfg.batch;
  hd_ = C_ / H_;
  if (C_ % H_ != 0 || hd_ % 2 != 0) {
    std::fprintf(stderr, "bad head config: n_embd=%d n_head=%d (head_dim must be even)\n",
                 C_, H_);
    std::exit(1);
  }
  if (cfg.mlp_hidden > 0) {
    Hd_ = cfg.mlp_hidden;
  } else {
    const int raw = (8 * C_) / 3;
    Hd_ = ((raw + 63) / 64) * 64;  // round up to a multiple of 64
    if (Hd_ < 64) Hd_ = 64;
  }
  build_rope();
  init_params();
  reallocate(B_);
}

void Model::build_rope() {
  const int half = hd_ / 2;
  cos_.assign(static_cast<size_t>(T_) * half, 0.f);
  sin_.assign(static_cast<size_t>(T_) * half, 0.f);
  for (int p = 0; p < T_; ++p) {
    for (int i = 0; i < half; ++i) {
      const float freq = 1.f / std::pow(cfg_.rope_theta, (2.f * i) / hd_);
      const float ang = p * freq;
      cos_[static_cast<size_t>(p) * half + i] = std::cos(ang);
      sin_[static_cast<size_t>(p) * half + i] = std::sin(ang);
    }
  }
}

void Model::init_params() {
  wte_ = add(static_cast<size_t>(V_) * C_);
  layers_.resize(L_);
  for (int l = 0; l < L_; ++l) {
    LayerWeights& w = layers_[l];
    w.ln1g = add(C_);
    w.Wq = add(static_cast<size_t>(C_) * C_);
    w.Wk = add(static_cast<size_t>(C_) * C_);
    w.Wv = add(static_cast<size_t>(C_) * C_);
    w.Wo = add(static_cast<size_t>(C_) * C_);
    w.ln2g = add(C_);
    w.W1 = add(static_cast<size_t>(Hd_) * C_);
    w.W3 = add(static_cast<size_t>(Hd_) * C_);
    w.W2 = add(static_cast<size_t>(C_) * Hd_);
  }
  lnfg_ = add(C_);

  // init: N(0, 0.02); RMSNorm gains = 1.
  for (Param* p : params_)
    for (float& x : p->data) x = 0.02f * rng_.normal();
  for (int l = 0; l < L_; ++l) {
    std::fill(layers_[l].ln1g->data.begin(), layers_[l].ln1g->data.end(), 1.f);
    std::fill(layers_[l].ln2g->data.begin(), layers_[l].ln2g->data.end(), 1.f);
  }
  std::fill(lnfg_->data.begin(), lnfg_->data.end(), 1.f);
}

void Model::reallocate(int batch) {
  B_ = batch;
  R_ = B_ * T_;
  acts_.resize(L_);
  const size_t rc = static_cast<size_t>(R_) * C_;
  const size_t rh = static_cast<size_t>(R_) * Hd_;
  const size_t rv = static_cast<size_t>(R_) * V_;
  x_.assign(rc, 0.f);
  o_.assign(rc, 0.f);
  xf_.assign(rc, 0.f);
  logits_.assign(rv, 0.f);
  dlogits_.assign(rv, 0.f);
  rstdf_.assign(R_, 0.f);
  for (int l = 0; l < L_; ++l) {
    LayerActs& A = acts_[l];
    A.xin.assign(rc, 0.f);
    A.a.assign(rc, 0.f);
    A.q.assign(rc, 0.f);
    A.k.assign(rc, 0.f);
    A.v.assign(rc, 0.f);
    A.ctx.assign(rc, 0.f);
    A.x1.assign(rc, 0.f);
    A.a2.assign(rc, 0.f);
    A.h1.assign(rh, 0.f);
    A.h2.assign(rh, 0.f);
    A.h3.assign(rh, 0.f);
    A.h4.assign(rh, 0.f);
    A.ln1_rstd.assign(R_, 0.f);
    A.ln2_rstd.assign(R_, 0.f);
    A.att.assign(static_cast<size_t>(B_) * H_ * T_ * T_, 0.f);
  }
  // KV cache + single-token inference scratch
  cache_k_.assign(static_cast<size_t>(L_) * T_ * C_, 0.f);
  cache_v_.assign(static_cast<size_t>(L_) * T_ * C_, 0.f);
  ic_x_.assign(C_, 0.f);
  ic_a_.assign(C_, 0.f);
  ic_q_.assign(C_, 0.f);
  ic_k_.assign(C_, 0.f);
  ic_v_.assign(C_, 0.f);
  ic_ctx_.assign(C_, 0.f);
  ic_o_.assign(C_, 0.f);
  ic_h1_.assign(Hd_, 0.f);
  ic_h2_.assign(Hd_, 0.f);
  ic_h3_.assign(Hd_, 0.f);
  ic_h4_.assign(Hd_, 0.f);
  ic_logits_.assign(V_, 0.f);
  ic_scores_.assign(T_, 0.f);
  cache_pos_ = 0;
}

void Model::rope_apply(float* x, int rows, int T) const {
  const int C = C_, H = H_, hd = hd_, half = hd / 2;
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int pos = r % T;
      const float* cs = cos_.data() + static_cast<size_t>(pos) * half;
      const float* sn = sin_.data() + static_cast<size_t>(pos) * half;
      for (int h = 0; h < H; ++h) {
        float* v = x + static_cast<size_t>(r) * C + h * hd;
        for (int i = 0; i < half; ++i) {
          const float a = v[2 * i], b = v[2 * i + 1];
          v[2 * i] = a * cs[i] - b * sn[i];
          v[2 * i + 1] = a * sn[i] + b * cs[i];
        }
      }
    }
  });
}

void Model::rope_apply_inv(float* x, int rows, int T) const {
  const int C = C_, H = H_, hd = hd_, half = hd / 2;
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int pos = r % T;
      const float* cs = cos_.data() + static_cast<size_t>(pos) * half;
      const float* sn = sin_.data() + static_cast<size_t>(pos) * half;
      for (int h = 0; h < H; ++h) {
        float* v = x + static_cast<size_t>(r) * C + h * hd;
        for (int i = 0; i < half; ++i) {
          const float a = v[2 * i], b = v[2 * i + 1];
          v[2 * i] = a * cs[i] + b * sn[i];
          v[2 * i + 1] = -a * sn[i] + b * cs[i];
        }
      }
    }
  });
}

void Model::rope_apply_one(float* x, int pos) const {
  const int H = H_, hd = hd_, half = hd / 2;
  const float* cs = cos_.data() + static_cast<size_t>(pos) * half;
  const float* sn = sin_.data() + static_cast<size_t>(pos) * half;
  for (int h = 0; h < H; ++h) {
    float* v = x + h * hd;
    for (int i = 0; i < half; ++i) {
      const float a = v[2 * i], b = v[2 * i + 1];
      v[2 * i] = a * cs[i] - b * sn[i];
      v[2 * i + 1] = a * sn[i] + b * cs[i];
    }
  }
}

void Model::forward(const int* toks, int B, int T) {
  curB_ = B;
  curT_ = T;
  curR_ = B * T;
  const int R = curR_, C = C_, Tc = T, Hd = Hd_;
  if (R > R_) {
    std::fprintf(stderr, "forward: R=%d exceeds capacity %d\n", R, R_);
    std::exit(1);
  }

  parallel_for(R, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* e = wte_->data.data() + static_cast<size_t>(toks[r]) * C;
      float* xr = x_.data() + static_cast<size_t>(r) * C;
      for (int c = 0; c < C; ++c) xr[c] = e[c];
    }
  });

  for (int l = 0; l < L_; ++l) {
    LayerWeights& w = layers_[l];
    LayerActs& A = acts_[l];

    std::copy(x_.begin(), x_.begin() + static_cast<size_t>(R) * C, A.xin.begin());
    rmsnorm_fwd(x_.data(), w.ln1g->data.data(), A.a.data(), A.ln1_rstd.data(), R, C,
                cfg_.rms_eps);
    linear_fwd(A.a.data(), w.Wq->data.data(), A.q.data(), R, C, C);
    linear_fwd(A.a.data(), w.Wk->data.data(), A.k.data(), R, C, C);
    linear_fwd(A.a.data(), w.Wv->data.data(), A.v.data(), R, C, C);
    rope_apply(A.q.data(), R, Tc);
    rope_apply(A.k.data(), R, Tc);
    attn_forward(A, B, Tc);
    linear_fwd(A.ctx.data(), w.Wo->data.data(), o_.data(), R, C, C);
    for (size_t i = 0; i < static_cast<size_t>(R) * C; ++i) x_[i] += o_[i];

    std::copy(x_.begin(), x_.begin() + static_cast<size_t>(R) * C, A.x1.begin());
    rmsnorm_fwd(x_.data(), w.ln2g->data.data(), A.a2.data(), A.ln2_rstd.data(), R, C,
                cfg_.rms_eps);
    linear_fwd(A.a2.data(), w.W1->data.data(), A.h1.data(), R, C, Hd);
    silu_fwd(A.h1.data(), A.h2.data(), R * Hd);
    linear_fwd(A.a2.data(), w.W3->data.data(), A.h3.data(), R, C, Hd);
    for (size_t i = 0; i < static_cast<size_t>(R) * Hd; ++i) A.h4[i] = A.h2[i] * A.h3[i];
    linear_fwd(A.h4.data(), w.W2->data.data(), o_.data(), R, Hd, C);
    for (size_t i = 0; i < static_cast<size_t>(R) * C; ++i) x_[i] += o_[i];
  }

  rmsnorm_fwd(x_.data(), lnfg_->data.data(), xf_.data(), rstdf_.data(), R, C,
              cfg_.rms_eps);
  linear_fwd(xf_.data(), wte_->data.data(), logits_.data(), R, C, V_);
}

void Model::attn_forward(LayerActs& A, int B, int T) {
  const float scale = 1.f / std::sqrt(static_cast<float>(hd_));
  const int C = C_, H = H_, hd = hd_;
  parallel_for(B * H, [&](int bh0, int bh1) {
    for (int bh = bh0; bh < bh1; ++bh) {
      const int b = bh / H, h = bh % H;
      float* pr = A.att.data() + static_cast<size_t>(bh) * T * T;
      for (int i = 0; i < T; ++i) {
        const int ri = b * T + i;
        const float* qi = A.q.data() + static_cast<size_t>(ri) * C + h * hd;
        float mx = -1e30f;
        for (int j = 0; j <= i; ++j) {
          const int rj = b * T + j;
          const float* kj = A.k.data() + static_cast<size_t>(rj) * C + h * hd;
          float s = dot_f32(qi, kj, hd);
          s *= scale;
          pr[i * T + j] = s;
          if (s > mx) mx = s;
        }
        float sum = 0.f;
        for (int j = 0; j <= i; ++j) {
          const float e = std::exp(pr[i * T + j] - mx);
          pr[i * T + j] = e;
          sum += e;
        }
        const float inv = 1.f / sum;
        for (int j = 0; j <= i; ++j) pr[i * T + j] *= inv;
        float* ci = A.ctx.data() + static_cast<size_t>(ri) * C + h * hd;
        for (int d = 0; d < hd; ++d) ci[d] = 0.f;
        for (int j = 0; j <= i; ++j) {
          const int rj = b * T + j;
          const float* vj = A.v.data() + static_cast<size_t>(rj) * C + h * hd;
          saxpy(ci, vj, pr[i * T + j], hd);
        }
      }
    }
  });
}

void Model::attn_backward(LayerActs& A, int B, int T, const float* dctx,
                          std::vector<float>& dq, std::vector<float>& dk,
                          std::vector<float>& dv) {
  const float scale = 1.f / std::sqrt(static_cast<float>(hd_));
  const int C = C_, H = H_, hd = hd_;
  std::fill(dq.begin(), dq.end(), 0.f);
  std::fill(dk.begin(), dk.end(), 0.f);
  std::fill(dv.begin(), dv.end(), 0.f);
  parallel_for(B * H, [&](int bh0, int bh1) {
    std::vector<float> dp(T);
    for (int bh = bh0; bh < bh1; ++bh) {
      const int b = bh / H, h = bh % H;
      const float* pr = A.att.data() + static_cast<size_t>(bh) * T * T;
      for (int i = 0; i < T; ++i) {
        const int ri = b * T + i;
        const float* dci = dctx + static_cast<size_t>(ri) * C + h * hd;
        const float* qi = A.q.data() + static_cast<size_t>(ri) * C + h * hd;
        for (int j = 0; j <= i; ++j) {
          const int rj = b * T + j;
          const float* vj = A.v.data() + static_cast<size_t>(rj) * C + h * hd;
          dp[j] = dot_f32(dci, vj, hd);
        }
        for (int j = 0; j <= i; ++j) {
          const int rj = b * T + j;
          float* dvj = dv.data() + static_cast<size_t>(rj) * C + h * hd;
          saxpy(dvj, dci, pr[i * T + j], hd);
        }
        float dot = 0.f;
        for (int j = 0; j <= i; ++j) dot += pr[i * T + j] * dp[j];
        float* dqi = dq.data() + static_cast<size_t>(ri) * C + h * hd;
        for (int j = 0; j <= i; ++j) {
          const float dsc = pr[i * T + j] * (dp[j] - dot) * scale;
          const int rj = b * T + j;
          const float* kj = A.k.data() + static_cast<size_t>(rj) * C + h * hd;
          float* dkj = dk.data() + static_cast<size_t>(rj) * C + h * hd;
          saxpy(dqi, kj, dsc, hd);
          saxpy(dkj, qi, dsc, hd);
        }
      }
    }
  });
}

double Model::cross_entropy(const int* tgts) {
  const int R = curR_, V = V_;
  double total = 0.0;
  parallel_for(R, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* lg = logits_.data() + static_cast<size_t>(r) * V;
      float* dl = dlogits_.data() + static_cast<size_t>(r) * V;
      float mx = -1e30f;
      for (int v = 0; v < V; ++v) mx = std::max(mx, lg[v]);
      float sum = 0.f;
      for (int v = 0; v < V; ++v) {
        const float e = std::exp(lg[v] - mx);
        dl[v] = e;
        sum += e;
      }
      const float inv = 1.f / sum;
      for (int v = 0; v < V; ++v) dl[v] *= inv;
      dl[tgts[r]] -= 1.f;
    }
  });
  for (int r = 0; r < R; ++r) {
    const float* lg = logits_.data() + static_cast<size_t>(r) * V;
    float mx = -1e30f;
    for (int v = 0; v < V; ++v) mx = std::max(mx, lg[v]);
    float sum = 0.f;
    for (int v = 0; v < V; ++v) sum += std::exp(lg[v] - mx);
    total += static_cast<double>(mx + std::log(sum)) - lg[tgts[r]];
  }
  const float invR = 1.f / R;
  for (float& d : dlogits_) d *= invR;
  return total / R;
}

void Model::backward(const int* toks) {
  const int R = curR_, C = C_, T = curT_, B = curB_, V = V_, Hd = Hd_;
  const size_t rc = static_cast<size_t>(R) * C;
  const size_t rh = static_cast<size_t>(R) * Hd;

  std::vector<float> dcur(rc, 0.f), tmp(rc, 0.f);
  linear_bwd(xf_.data(), wte_->data.data(), dlogits_.data(), dcur.data(),
             wte_->grad.data(), R, C, V);
  std::vector<float> dx(rc, 0.f);
  rmsnorm_bwd(x_.data(), lnfg_->data.data(), dcur.data(), dx.data(), lnfg_->grad.data(),
              rstdf_.data(), R, C);

  std::vector<float> dh4(rh), dh2(rh), dh3(rh), dh1(rh), da2(rc), dctx(rc), da(rc);
  std::vector<float> dq(rc), dk(rc), dv(rc);
  std::vector<float> da_q(rc), da_k(rc), da_v(rc), dxin(rc);

  for (int l = L_ - 1; l >= 0; --l) {
    LayerWeights& w = layers_[l];
    LayerActs& A = acts_[l];

    // x_after = x1 + o(mlp)  =>  grad wrt mlp output = dx
    linear_bwd(A.h4.data(), w.W2->data.data(), dx.data(), dh4.data(), w.W2->grad.data(),
               R, Hd, C);
    // h4 = h2 * h3
    for (size_t i = 0; i < rh; ++i) {
      dh2[i] = dh4[i] * A.h3[i];
      dh3[i] = dh4[i] * A.h2[i];
    }
    // h2 = silu(h1)
    silu_bwd(A.h1.data(), dh2.data(), dh1.data(), R * Hd);
    linear_bwd(A.a2.data(), w.W1->data.data(), dh1.data(), da2.data(), w.W1->grad.data(),
               R, C, Hd);
    // W3 contributes to the same a2 grad
    std::vector<float> da2b(rc, 0.f);
    linear_bwd(A.a2.data(), w.W3->data.data(), dh3.data(), da2b.data(),
               w.W3->grad.data(), R, C, Hd);
    for (size_t i = 0; i < rc; ++i) da2[i] += da2b[i];
    rmsnorm_bwd(A.x1.data(), w.ln2g->data.data(), da2.data(), tmp.data(),
                w.ln2g->grad.data(), A.ln2_rstd.data(), R, C);
    for (size_t i = 0; i < rc; ++i) dx[i] += tmp[i];  // now grad wrt x1

    // x1 = x0 + o(attn)
    linear_bwd(A.ctx.data(), w.Wo->data.data(), dx.data(), dctx.data(),
               w.Wo->grad.data(), R, C, C);
    attn_backward(A, B, T, dctx.data(), dq, dk, dv);
    // undo RoPE on the query/key gradients (attention used rotated q,k)
    rope_apply_inv(dq.data(), R, T);
    rope_apply_inv(dk.data(), R, T);
    linear_bwd(A.a.data(), w.Wq->data.data(), dq.data(), da_q.data(), w.Wq->grad.data(),
               R, C, C);
    linear_bwd(A.a.data(), w.Wk->data.data(), dk.data(), da_k.data(), w.Wk->grad.data(),
               R, C, C);
    linear_bwd(A.a.data(), w.Wv->data.data(), dv.data(), da_v.data(), w.Wv->grad.data(),
               R, C, C);
    for (size_t i = 0; i < rc; ++i) da[i] = da_q[i] + da_k[i] + da_v[i];
    rmsnorm_bwd(A.xin.data(), w.ln1g->data.data(), da.data(), dxin.data(),
                w.ln1g->grad.data(), A.ln1_rstd.data(), R, C);
    for (size_t i = 0; i < rc; ++i) dx[i] += dxin[i];  // grad wrt block input
  }

  // token embedding grads
  parallel_for(R, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int t = toks[r];
      const float* dxr = dx.data() + static_cast<size_t>(r) * C;
      float* gwte = wte_->grad.data() + static_cast<size_t>(t) * C;
      for (int c = 0; c < C; ++c) gwte[c] += dxr[c];
    }
  });
}

float Model::train_step(const int* toks, const int* tgts, int B, int T) {
  zero_grad();
  forward(toks, B, T);
  const double loss = cross_entropy(tgts);
  backward(toks);
  return static_cast<float>(loss);
}

double Model::compute_loss(const int* toks, const int* tgts, int B, int T) {
  forward(toks, B, T);
  return cross_entropy(tgts);
}

void Model::mean_hidden(std::vector<float>& out) const {
  out.assign(C_, 0.f);
  if (curR_ <= 0) return;
  for (int r = 0; r < curR_; ++r) {
    const float* xr = xf_.data() + static_cast<size_t>(r) * C_;
    for (int c = 0; c < C_; ++c) out[c] += xr[c];
  }
  const float inv = 1.f / curR_;
  for (int c = 0; c < C_; ++c) out[c] *= inv;
}

void Model::zero_grad() {
  for (Param* p : params_) std::fill(p->grad.begin(), p->grad.end(), 0.f);
}

// ---------------------------------------------------------------------------
// Incremental inference with a KV cache.
// ---------------------------------------------------------------------------
void Model::reset_cache() {
  cache_pos_ = 0;
  std::fill(cache_k_.begin(), cache_k_.end(), 0.f);
  std::fill(cache_v_.begin(), cache_v_.end(), 0.f);
}

const float* Model::forward_token(int token) {
  const int C = C_, H = H_, hd = hd_, Hd = Hd_;
  const int pos = cache_pos_;
  if (pos >= T_) {
    // Safety net: context exhausted. Callers should stop generating before
    // this point; if not, drop history and continue (never crash).
    reset_cache();
  }
  const int p = cache_pos_;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));

  const float* e = wte_->data.data() + static_cast<size_t>(token) * C;
  for (int c = 0; c < C; ++c) ic_x_[c] = e[c];

  for (int l = 0; l < L_; ++l) {
    LayerWeights& w = layers_[l];
    float rstd = 0.f;
    rmsnorm_fwd(ic_x_.data(), w.ln1g->data.data(), ic_a_.data(), &rstd, 1, C,
                cfg_.rms_eps);
    linear_fwd(ic_a_.data(), w.Wq->data.data(), ic_q_.data(), 1, C, C);
    linear_fwd(ic_a_.data(), w.Wk->data.data(), ic_k_.data(), 1, C, C);
    linear_fwd(ic_a_.data(), w.Wv->data.data(), ic_v_.data(), 1, C, C);
    rope_apply_one(ic_q_.data(), p);
    rope_apply_one(ic_k_.data(), p);

    float* ck = cache_k_.data() + (static_cast<size_t>(l) * T_ + p) * C;
    float* cv = cache_v_.data() + (static_cast<size_t>(l) * T_ + p) * C;
    std::memcpy(ck, ic_k_.data(), C * sizeof(float));
    std::memcpy(cv, ic_v_.data(), C * sizeof(float));

    for (int h = 0; h < H; ++h) {
      const float* qi = ic_q_.data() + h * hd;
      float* sc = ic_scores_.data();
      float mx = -1e30f;
      for (int j = 0; j <= p; ++j) {
        const float* kj = cache_k_.data() + (static_cast<size_t>(l) * T_ + j) * C + h * hd;
        float s = dot_f32(qi, kj, hd);
        s *= scale;
        sc[j] = s;
        if (s > mx) mx = s;
      }
      float sum = 0.f;
      for (int j = 0; j <= p; ++j) {
        const float ex = std::exp(sc[j] - mx);
        sc[j] = ex;
        sum += ex;
      }
      const float inv = 1.f / sum;
      float* ci = ic_ctx_.data() + h * hd;
      for (int d = 0; d < hd; ++d) ci[d] = 0.f;
      for (int j = 0; j <= p; ++j) {
        const float* vj = cache_v_.data() + (static_cast<size_t>(l) * T_ + j) * C + h * hd;
        const float pr = sc[j] * inv;
        for (int d = 0; d < hd; ++d) ci[d] += pr * vj[d];
      }
    }

    linear_fwd(ic_ctx_.data(), w.Wo->data.data(), ic_o_.data(), 1, C, C);
    for (int c = 0; c < C; ++c) ic_x_[c] += ic_o_[c];

    rmsnorm_fwd(ic_x_.data(), w.ln2g->data.data(), ic_a_.data(), &rstd, 1, C,
                cfg_.rms_eps);
    linear_fwd(ic_a_.data(), w.W1->data.data(), ic_h1_.data(), 1, C, Hd);
    silu_fwd(ic_h1_.data(), ic_h2_.data(), Hd);
    linear_fwd(ic_a_.data(), w.W3->data.data(), ic_h3_.data(), 1, C, Hd);
    for (int i = 0; i < Hd; ++i) ic_h4_[i] = ic_h2_[i] * ic_h3_[i];
    linear_fwd(ic_h4_.data(), w.W2->data.data(), ic_o_.data(), 1, Hd, C);
    for (int c = 0; c < C; ++c) ic_x_[c] += ic_o_[c];
  }

  float rstd = 0.f;
  rmsnorm_fwd(ic_x_.data(), lnfg_->data.data(), ic_a_.data(), &rstd, 1, C, cfg_.rms_eps);
  linear_fwd(ic_a_.data(), wte_->data.data(), ic_logits_.data(), 1, C, V_);
  cache_pos_++;
  return ic_logits_.data();
}

}  // namespace slmpt
