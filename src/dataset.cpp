#include "slmpt/dataset.hpp"

#include <cstdio>

namespace slmpt {

bool Dataset::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    std::fclose(f);
    return false;
  }
  data_.resize(static_cast<size_t>(n));
  const size_t rd = std::fread(data_.data(), 1, static_cast<size_t>(n), f);
  std::fclose(f);
  return rd == static_cast<size_t>(n);
}

void Dataset::random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                           std::vector<int>& tgts) const {
  toks.resize(static_cast<size_t>(B) * T);
  tgts.resize(static_cast<size_t>(B) * T);
  const int N = static_cast<int>(data_.size());
  const int hi = N - T - 1;
  const int span = hi > 0 ? hi : 1;
  for (int b = 0; b < B; ++b) {
    const int off = rng.below(span);
    for (int i = 0; i < T; ++i) {
      toks[static_cast<size_t>(b) * T + i] = static_cast<int>(data_[off + i]);
      tgts[static_cast<size_t>(b) * T + i] = static_cast<int>(data_[off + i + 1]);
    }
  }
}

}  // namespace slmpt
