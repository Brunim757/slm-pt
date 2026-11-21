#pragma once
#include <vector>

#include "slmpt/param.hpp"

namespace slmpt {

// AdamW with decoupled weight decay and global grad-norm clipping.
class AdamW {
 public:
  AdamW(float lr, float beta1, float beta2, float eps, float weight_decay,
        float grad_clip);

  void step(std::vector<Param*>& params, int step);

  void set_lr(float lr) { lr_ = lr; }

  float last_grad_norm() const { return grad_norm_; }

 private:
  float lr_, b1_, b2_, eps_, wd_, clip_;
  float grad_norm_ = 0.f;
};

}  // namespace slmpt
