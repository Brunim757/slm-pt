#pragma once
#include <memory>
#include <vector>

#include "slmpt/config.hpp"
#include "slmpt/param.hpp"
#include "slmpt/rng.hpp"

namespace slmpt {

struct LayerWeights {
  Param *ln1g = nullptr, *ln1b = nullptr;
  Param *Wq = nullptr, *bq = nullptr;
  Param *Wk = nullptr, *bk = nullptr;
  Param *Wv = nullptr, *bv = nullptr;
  Param *Wo = nullptr, *bo = nullptr;
  Param *ln2g = nullptr, *ln2b = nullptr;
  Param *fc1 = nullptr, *b1 = nullptr;
  Param *fc2 = nullptr, *b2 = nullptr;
};

// Activations cached for the backward pass of one layer.
struct LayerActs {
  std::vector<float> xin, a, q, k, v, ctx, x1, a2, h1, h2, h3;
  std::vector<float> ln1_mean, ln1_rstd, ln2_mean, ln2_rstd;
  std::vector<float> att;  // [B*H*T*T] attention probabilities
};

// A small GPT (decoder-only transformer), byte-level, pre-norm, tied embeddings.
class Model {
 public:
  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  std::vector<Param*>& params() { return params_; }
  const std::vector<Param*>& params() const { return params_; }

  int vocab() const { return V_; }
  int block() const { return T_; }
  int capacity_rows() const { return R_; }

  // One optimization step: zero grads, forward, cross-entropy, backward.
  // Returns the mean cross-entropy loss. Gradients are left in params().
  float train_step(const int* toks, const int* tgts, int B, int T);

  // Forward only. Logits end up in last_logits().
  void forward(const int* toks, int B, int T);

  // Forward + cross-entropy without touching gradients (for grad checks).
  double compute_loss(const int* toks, const int* tgts, int B, int T);

  const float* last_logits() const {
    return logits_.data() + static_cast<size_t>(curR_ - 1) * V_;
  }

  void zero_grad();
  void reallocate(int batch);

 private:
  void init_params();
  double cross_entropy(const int* tgts);
  void backward(const int* toks);
  void attn_forward(LayerActs& A, int B, int T);
  void attn_backward(LayerActs& A, int B, int T, const float* dctx,
                     std::vector<float>& dq, std::vector<float>& dk,
                     std::vector<float>& dv);

  Config cfg_;
  int C_ = 0, L_ = 0, H_ = 0, T_ = 0, V_ = 0, B_ = 0, hd_ = 0, R_ = 0;
  int curB_ = 0, curT_ = 0, curR_ = 0;
  Rng rng_;

  std::vector<std::unique_ptr<Param>> storage_;
  std::vector<Param*> params_;
  Param *wte_ = nullptr, *wpe_ = nullptr, *lnfg_ = nullptr, *lnfb_ = nullptr;
  std::vector<LayerWeights> layers_;
  std::vector<LayerActs> acts_;

  std::vector<float> x_, o_, xf_, logits_, dlogits_, meanf_, rstdf_;

  Param* add(size_t n) {
    storage_.push_back(std::make_unique<Param>(n));
    params_.push_back(storage_.back().get());
    return storage_.back().get();
  }
};

}  // namespace slmpt
