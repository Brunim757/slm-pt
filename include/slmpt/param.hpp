#pragma once
#include <cstddef>
#include <vector>

namespace slmpt {

// A tensor with its gradient and AdamW moments.
struct Param {
  std::vector<float> data, grad, m, v;

  explicit Param(size_t n = 0) : data(n, 0.f), grad(n, 0.f), m(n, 0.f), v(n, 0.f) {}

  size_t size() const { return data.size(); }
};

}  // namespace slmpt
