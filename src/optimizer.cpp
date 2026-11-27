#include "slmpt/optimizer.hpp"

#include <cmath>

namespace slmpt {

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay,
             float grad_clip)
    : lr_(lr), b1_(beta1), b2_(beta2), eps_(eps), wd_(weight_decay), clip_(grad_clip) {}

void AdamW::step(std::vector<Param*>& params, int step) {
  double sumsq = 0.0;
  for (Param* p : params)
    for (float g : p->grad) sumsq += static_cast<double>(g) * g;
  grad_norm_ = static_cast<float>(std::sqrt(sumsq));

  float scale = 1.f;
  if (clip_ > 0.f && grad_norm_ > clip_) scale = clip_ / (grad_norm_ + 1e-6f);

  const float bc1 = 1.f - std::pow(b1_, static_cast<float>(step));
  const float bc2 = 1.f - std::pow(b2_, static_cast<float>(step));

  for (Param* p : params) {
    for (size_t i = 0; i < p->data.size(); ++i) {
      const float g = p->grad[i] * scale;
      p->m[i] = b1_ * p->m[i] + (1.f - b1_) * g;
      p->v[i] = b2_ * p->v[i] + (1.f - b2_) * g * g;
      const float mh = p->m[i] / bc1;
      const float vh = p->v[i] / bc2;
      const float upd = mh / (std::sqrt(vh) + eps_);
      const float wd = p->decay ? wd_ * p->data[i] : 0.f;
      p->data[i] -= lr_ * (upd + wd);
    }
  }
}

}  // namespace slmpt
