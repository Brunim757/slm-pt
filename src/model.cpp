#include "slmpt/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "slmpt/parallel.hpp"

namespace slmpt {
namespace {

constexpr float kLnEps = 1e-5f;

// y[rows,out] = x[rows,in] @ W[out,in]^T + b   (b may be null)
void linear_fwd(const float* x, const float* W, const float* b, float* y, int rows,
                int in, int out) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * in;
      float* yr = y + static_cast<size_t>(r) * out;
      for (int o = 0; o < out; ++o) {
        const float* wo = W + static_cast<size_t>(o) * in;
        float acc = b ? b[o] : 0.f;
        for (int i = 0; i < in; ++i) acc += xr[i] * wo[i];
        yr[o] = acc;
      }
    }
  });
}

// Accumulates into dx, dW, db (caller zeroes them once per step).
void linear_bwd(const float* x, const float* W, const float* dy, float* dx,
                float* dW, float* db, int rows, int in, int out) {
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
          for (int i = 0; i < in; ++i) dxr[i] += g * wo[i];
        }
      }
    });
  }
  if (!dW && !db) return;
  parallel_for(out, [&](int o0, int o1) {
    for (int o = o0; o < o1; ++o) {
      if (db) {
        float acc = 0.f;
        for (int r = 0; r < rows; ++r) acc += dy[static_cast<size_t>(r) * out + o];
        db[o] += acc;
      }
      if (dW) {
        float* dwo = dW + static_cast<size_t>(o) * in;
        for (int r = 0; r < rows; ++r) {
          const float g = dy[static_cast<size_t>(r) * out + o];
          if (g == 0.f) continue;
          const float* xr = x + static_cast<size_t>(r) * in;
          for (int i = 0; i < in; ++i) dwo[i] += g * xr[i];
        }
      }
    }
  });
}

void ln_fwd(const float* x, const float* g, const float* b, float* y, float* mean,
            float* rstd, int rows, int C) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * C;
      float m = 0.f;
      for (int i = 0; i < C; ++i) m += xr[i];
      m /= C;
      float var = 0.f;
      for (int i = 0; i < C; ++i) {
        const float d = xr[i] - m;
        var += d * d;
      }
      var /= C;
      const float rs = 1.f / std::sqrt(var + kLnEps);
      mean[r] = m;
      rstd[r] = rs;
      float* yr = y + static_cast<size_t>(r) * C;
      for (int i = 0; i < C; ++i) yr[i] = g[i] * (xr[i] - m) * rs + b[i];
    }
  });
}

void ln_bwd(const float* x, const float* g, const float* dy, float* dx, float* dg,
            float* db, const float* mean, const float* rstd, int rows, int C) {
  parallel_for(rows, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const float* xr = x + static_cast<size_t>(r) * C;
      const float* dyr = dy + static_cast<size_t>(r) * C;
      float* dxr = dx + static_cast<size_t>(r) * C;
      const float m = mean[r], rs = rstd[r];
      float s1 = 0.f, s2 = 0.f;
      for (int i = 0; i < C; ++i) {
        const float xn = (xr[i] - m) * rs;
        const float gd = g[i] * dyr[i];
        s1 += gd;
        s2 += gd * xn;
      }
      const float inv = 1.f / C;
      for (int i = 0; i < C; ++i) {
        const float xn = (xr[i] - m) * rs;
        const float gd = g[i] * dyr[i];
        dxr[i] = rs * (gd - s1 * inv - xn * s2 * inv);
      }
    }
  });
  if (dg || db) {
    for (int i = 0; i < C; ++i) {
      float sg = 0.f, sb = 0.f;
      for (int r = 0; r < rows; ++r) {
        const float xn = (x[static_cast<size_t>(r) * C + i] - mean[r]) * rstd[r];
        sg += dy[static_cast<size_t>(r) * C + i] * xn;
        sb += dy[static_cast<size_t>(r) * C + i];
      }
      if (dg) dg[i] += sg;
      if (db) db[i] += sb;
    }
  }
}

inline float gelu(float x) {
  return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
inline float gelu_grad(float x) {
  const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
  const float t = std::tanh(u);
  const float du = 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * du;
}
void gelu_fwd(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) y[i] = gelu(x[i]);
}
void gelu_bwd(const float* x, const float* dy, float* dx, int n) {
  for (int i = 0; i < n; ++i) dx[i] = dy[i] * gelu_grad(x[i]);
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
  if (C_ % H_ != 0) {
    std::fprintf(stderr, "n_embd (%d) must be divisible by n_head (%d)\n", C_, H_);
    std::exit(1);
  }
  init_params();
  reallocate(B_);
}

void Model::init_params() {
  wte_ = add(static_cast<size_t>(V_) * C_);
  wpe_ = add(static_cast<size_t>(T_) * C_);
  layers_.resize(L_);
  for (int l = 0; l < L_; ++l) {
    LayerWeights& w = layers_[l];
    w.ln1g = add(C_); w.ln1b = add(C_);
    w.Wq = add(static_cast<size_t>(C_) * C_); w.bq = add(C_);
    w.Wk = add(static_cast<size_t>(C_) * C_); w.bk = add(C_);
    w.Wv = add(static_cast<size_t>(C_) * C_); w.bv = add(C_);
    w.Wo = add(static_cast<size_t>(C_) * C_); w.bo = add(C_);
    w.ln2g = add(C_); w.ln2b = add(C_);
    w.fc1 = add(static_cast<size_t>(C_) * 4 * C_); w.b1 = add(4 * C_);
    w.fc2 = add(static_cast<size_t>(4) * C_ * C_); w.b2 = add(C_);
  }
  lnfg_ = add(C_); lnfb_ = add(C_);

  // init: N(0, 0.02), layernorm gains = 1.
  for (Param* p : params_)
    for (float& x : p->data) x = 0.02f * rng_.normal();
  for (int l = 0; l < L_; ++l) {
    std::fill(layers_[l].ln1g->data.begin(), layers_[l].ln1g->data.end(), 1.f);
    std::fill(layers_[l].ln2g->data.begin(), layers_[l].ln2g->data.end(), 1.f);
    std::fill(layers_[l].bq->data.begin(), layers_[l].bq->data.end(), 0.f);
    std::fill(layers_[l].bk->data.begin(), layers_[l].bk->data.end(), 0.f);
    std::fill(layers_[l].bv->data.begin(), layers_[l].bv->data.end(), 0.f);
    std::fill(layers_[l].bo->data.begin(), layers_[l].bo->data.end(), 0.f);
  }
  std::fill(lnfg_->data.begin(), lnfg_->data.end(), 1.f);
  std::fill(lnfb_->data.begin(), lnfb_->data.end(), 0.f);
}

void Model::reallocate(int batch) {
  B_ = batch;
  R_ = B_ * T_;
  acts_.resize(L_);
  const size_t rc = static_cast<size_t>(R_) * C_;
  const size_t rc4 = static_cast<size_t>(R_) * 4 * C_;
  const size_t rv = static_cast<size_t>(R_) * V_;
  x_.assign(rc, 0.f);
  o_.assign(rc, 0.f);
  xf_.assign(rc, 0.f);
  logits_.assign(rv, 0.f);
  dlogits_.assign(rv, 0.f);
  meanf_.assign(R_, 0.f);
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
    A.h1.assign(rc4, 0.f);
    A.h2.assign(rc4, 0.f);
    A.h3.assign(rc, 0.f);
    A.ln1_mean.assign(R_, 0.f);
    A.ln1_rstd.assign(R_, 0.f);
    A.ln2_mean.assign(R_, 0.f);
    A.ln2_rstd.assign(R_, 0.f);
    A.att.assign(static_cast<size_t>(B_) * H_ * T_ * T_, 0.f);
  }
}

void Model::forward(const int* toks, int B, int T) {
  curB_ = B;
  curT_ = T;
  curR_ = B * T;
  const int R = curR_, C = C_, Tc = T;
  if (R > R_) {
    std::fprintf(stderr, "forward: R=%d exceeds capacity %d\n", R, R_);
    std::exit(1);
  }

  parallel_for(R, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int t = toks[r];
      const int p = r % Tc;
      const float* e = wte_->data.data() + static_cast<size_t>(t) * C;
      const float* pe = wpe_->data.data() + static_cast<size_t>(p) * C;
      float* xr = x_.data() + static_cast<size_t>(r) * C;
      for (int c = 0; c < C; ++c) xr[c] = e[c] + pe[c];
    }
  });

  for (int l = 0; l < L_; ++l) {
    LayerWeights& w = layers_[l];
    LayerActs& A = acts_[l];

    std::copy(x_.begin(), x_.begin() + static_cast<size_t>(R) * C, A.xin.begin());
    ln_fwd(x_.data(), w.ln1g->data.data(), w.ln1b->data.data(), A.a.data(),
           A.ln1_mean.data(), A.ln1_rstd.data(), R, C);
    linear_fwd(A.a.data(), w.Wq->data.data(), w.bq->data.data(), A.q.data(), R, C, C);
    linear_fwd(A.a.data(), w.Wk->data.data(), w.bk->data.data(), A.k.data(), R, C, C);
    linear_fwd(A.a.data(), w.Wv->data.data(), w.bv->data.data(), A.v.data(), R, C, C);
    attn_forward(A, B, Tc);
    linear_fwd(A.ctx.data(), w.Wo->data.data(), w.bo->data.data(), o_.data(), R, C, C);
    for (size_t i = 0; i < static_cast<size_t>(R) * C; ++i) x_[i] += o_[i];

    std::copy(x_.begin(), x_.begin() + static_cast<size_t>(R) * C, A.x1.begin());
    ln_fwd(x_.data(), w.ln2g->data.data(), w.ln2b->data.data(), A.a2.data(),
           A.ln2_mean.data(), A.ln2_rstd.data(), R, C);
    linear_fwd(A.a2.data(), w.fc1->data.data(), w.b1->data.data(), A.h1.data(), R, C,
               4 * C);
    gelu_fwd(A.h1.data(), A.h2.data(), R * 4 * C);
    linear_fwd(A.h2.data(), w.fc2->data.data(), w.b2->data.data(), A.h3.data(), R, 4 * C,
               C);
    for (size_t i = 0; i < static_cast<size_t>(R) * C; ++i) x_[i] += A.h3[i];
  }

  ln_fwd(x_.data(), lnfg_->data.data(), lnfb_->data.data(), xf_.data(), meanf_.data(),
         rstdf_.data(), R, C);
  linear_fwd(xf_.data(), wte_->data.data(), nullptr, logits_.data(), R, C, V_);
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
          float s = 0.f;
          for (int d = 0; d < hd; ++d) s += qi[d] * kj[d];
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
          const float p = pr[i * T + j];
          for (int d = 0; d < hd; ++d) ci[d] += p * vj[d];
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
          float s = 0.f;
          for (int d = 0; d < hd; ++d) s += dci[d] * vj[d];
          dp[j] = s;
        }
        for (int j = 0; j <= i; ++j) {
          const int rj = b * T + j;
          float* dvj = dv.data() + static_cast<size_t>(rj) * C + h * hd;
          const float p = pr[i * T + j];
          for (int d = 0; d < hd; ++d) dvj[d] += p * dci[d];
        }
        float dot = 0.f;
        for (int j = 0; j <= i; ++j) dot += pr[i * T + j] * dp[j];
        for (int j = 0; j <= i; ++j) {
          const float dsc = pr[i * T + j] * (dp[j] - dot) * scale;
          const int rj = b * T + j;
          const float* kj = A.k.data() + static_cast<size_t>(rj) * C + h * hd;
          float* dqi = dq.data() + static_cast<size_t>(ri) * C + h * hd;
          float* dkj = dk.data() + static_cast<size_t>(rj) * C + h * hd;
          for (int d = 0; d < hd; ++d) {
            dqi[d] += dsc * kj[d];
            dkj[d] += dsc * qi[d];
          }
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
  const int R = curR_, C = C_, T = curT_, B = curB_, V = V_;
  const size_t rc = static_cast<size_t>(R) * C;
  const size_t rc4 = static_cast<size_t>(R) * 4 * C;

  std::vector<float> dcur(rc, 0.f), tmp(rc, 0.f);
  // grad wrt xf, via tied output projection
  linear_bwd(xf_.data(), wte_->data.data(), dlogits_.data(), dcur.data(),
             wte_->grad.data(), nullptr, R, C, V);
  // final layernorm (input x_, output xf_)
  std::vector<float> dx(rc, 0.f);
  ln_bwd(x_.data(), lnfg_->data.data(), dcur.data(), dx.data(), lnfg_->grad.data(),
         lnfb_->grad.data(), meanf_.data(), rstdf_.data(), R, C);

  std::vector<float> dh3(rc), dh2(rc4), dh1(rc4), da2(rc), dctx(rc), da(rc);
  std::vector<float> dq(rc), dk(rc), dv(rc);
  std::vector<float> da_q(rc), da_k(rc), da_v(rc), dxin(rc);

  for (int l = L_ - 1; l >= 0; --l) {
    LayerWeights& w = layers_[l];
    LayerActs& A = acts_[l];

    // x_after = x1 + h3  =>  grad wrt h3 = dx ; residual passthrough stays in dx
    std::copy(dx.begin(), dx.end(), dh3.begin());
    linear_bwd(A.h2.data(), w.fc2->data.data(), dh3.data(), dh2.data(),
               w.fc2->grad.data(), w.b2->grad.data(), R, 4 * C, C);
    gelu_bwd(A.h1.data(), dh2.data(), dh1.data(), R * 4 * C);
    linear_bwd(A.a2.data(), w.fc1->data.data(), dh1.data(), da2.data(),
               w.fc1->grad.data(), w.b1->grad.data(), R, C, 4 * C);
    ln_bwd(A.x1.data(), w.ln2g->data.data(), da2.data(), tmp.data(),
           w.ln2g->grad.data(), w.ln2b->grad.data(), A.ln2_mean.data(),
           A.ln2_rstd.data(), R, C);
    for (size_t i = 0; i < rc; ++i) dx[i] += tmp[i];  // now grad wrt x1

    // x1 = x0 + o , o = Wo*ctx + bo  =>  grad wrt o = dx
    linear_bwd(A.ctx.data(), w.Wo->data.data(), dx.data(), dctx.data(),
               w.Wo->grad.data(), w.bo->grad.data(), R, C, C);
    attn_backward(A, B, T, dctx.data(), dq, dk, dv);
    linear_bwd(A.a.data(), w.Wq->data.data(), dq.data(), da_q.data(), w.Wq->grad.data(),
               w.bq->grad.data(), R, C, C);
    linear_bwd(A.a.data(), w.Wk->data.data(), dk.data(), da_k.data(), w.Wk->grad.data(),
               w.bk->grad.data(), R, C, C);
    linear_bwd(A.a.data(), w.Wv->data.data(), dv.data(), da_v.data(), w.Wv->grad.data(),
               w.bv->grad.data(), R, C, C);
    for (size_t i = 0; i < rc; ++i) da[i] = da_q[i] + da_k[i] + da_v[i];
    ln_bwd(A.xin.data(), w.ln1g->data.data(), da.data(), dxin.data(),
           w.ln1g->grad.data(), w.ln1b->grad.data(), A.ln1_mean.data(),
           A.ln1_rstd.data(), R, C);
    for (size_t i = 0; i < rc; ++i) dx[i] += dxin[i];  // now grad wrt block input
  }

  // token + position embedding grads
  parallel_for(R, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int t = toks[r];
      const int p = r % T;
      const float* dxr = dx.data() + static_cast<size_t>(r) * C;
      float* gwte = wte_->grad.data() + static_cast<size_t>(t) * C;
      float* gwpe = wpe_->grad.data() + static_cast<size_t>(p) * C;
      for (int c = 0; c < C; ++c) {
        gwte[c] += dxr[c];
        gwpe[c] += dxr[c];
      }
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

void Model::zero_grad() {
  for (Param* p : params_) std::fill(p->grad.begin(), p->grad.end(), 0.f);
}

}  // namespace slmpt
