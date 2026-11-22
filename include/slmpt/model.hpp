#pragma once
#include <memory>
#include <vector>

#include "slmpt/config.hpp"
#include "slmpt/param.hpp"
#include "slmpt/rng.hpp"

namespace slmpt {

struct LayerWeights {
  Param *ln1g = nullptr;                              // RMSNorm gain (attn)
  Param *Wq = nullptr, *Wk = nullptr, *Wv = nullptr;  // C x C
  Param *Wo = nullptr;                                // C x C
  Param *ln2g = nullptr;                              // RMSNorm gain (mlp)
  Param *W1 = nullptr, *W3 = nullptr;                 // gate/up  H x C  (SwiGLU)
  Param *W2 = nullptr;                                // down     C x H
};

// Activations cached for the backward pass of one layer.
struct LayerActs {
  std::vector<float> xin, a, q, k, v, ctx, x1, a2;
  std::vector<float> h1, h2, h3, h4;  // SwiGLU: gate pre, silu(gate), up, product
  std::vector<float> ln1_rstd, ln2_rstd;
  std::vector<float> att;  // [B*H*T*T] attention probabilities
};

// A small GPT (decoder-only transformer), pre-norm, RMSNorm + RoPE + SwiGLU,
// tied embeddings.
class Model {
 public:
  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  std::vector<Param*>& params() { return params_; }
  const std::vector<Param*>& params() const { return params_; }

  int vocab() const { return V_; }
  int block() const { return T_; }
  int capacity_rows() const { return R_; }
  int n_layer() const { return L_; }
  int n_head() const { return H_; }
  int head_dim() const { return hd_; }
  int hidden() const { return Hd_; }
  int cur_T() const { return curT_; }
  const std::vector<LayerActs>& acts() const { return acts_; }
  const Param* wte() const { return wte_; }

  // ---- training ----
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
  // Mean of the final hidden state over the current sequence (RAG embedding).
  void mean_hidden(std::vector<float>& out) const;

  // ---- incremental inference (KV cache) ----
  void reset_cache();
  int cache_pos() const { return cache_pos_; }
  // Feeds one token at the current cache position; returns logits [V].
  const float* forward_token(int token);

  void zero_grad();
  void reallocate(int batch);

 private:
  void init_params();
  void build_rope();
  double cross_entropy(const int* tgts);
  void backward(const int* toks);
  void attn_forward(LayerActs& A, int B, int T);
  void attn_backward(LayerActs& A, int B, int T, const float* dctx,
                     std::vector<float>& dq, std::vector<float>& dk,
                     std::vector<float>& dv);
  void rope_apply(float* x, int rows, int T) const;    // forward rotation
  void rope_apply_inv(float* x, int rows, int T) const;  // inverse (for grads)
  void rope_apply_one(float* x, int pos) const;        // single row, given pos

  Config cfg_;
  int C_ = 0, L_ = 0, H_ = 0, T_ = 0, V_ = 0, B_ = 0, hd_ = 0, R_ = 0, Hd_ = 0;
  int curB_ = 0, curT_ = 0, curR_ = 0;
  Rng rng_;

  std::vector<float> cos_, sin_;  // [T_ * hd_/2]

  std::vector<std::unique_ptr<Param>> storage_;
  std::vector<Param*> params_;
  Param *wte_ = nullptr, *lnfg_ = nullptr;
  std::vector<LayerWeights> layers_;
  std::vector<LayerActs> acts_;

  std::vector<float> x_, o_, xf_, logits_, dlogits_, rstdf_;

  // KV cache (single sequence) for incremental inference.
  int cache_pos_ = 0;
  std::vector<float> cache_k_, cache_v_;  // [L_ * T_ * C_]
  std::vector<float> ic_x_, ic_a_, ic_q_, ic_k_, ic_v_, ic_ctx_, ic_o_, ic_h1_,
      ic_h2_, ic_h3_, ic_h4_, ic_logits_, ic_scores_;

  Param* add(size_t n) {
    storage_.push_back(std::make_unique<Param>(n));
    params_.push_back(storage_.back().get());
    return storage_.back().get();
  }
};

}  // namespace slmpt
