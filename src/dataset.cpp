#include "slmpt/dataset.hpp"

#include <cstdio>
#include <string>

namespace slmpt {

bool Dataset::load_ids(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes <= 0 || bytes % 4 != 0) {
    std::fclose(f);
    return false;
  }
  const size_t n = static_cast<size_t>(bytes) / 4;
  std::vector<int32_t> buf(n);
  const size_t rd = std::fread(buf.data(), sizeof(int32_t), n, f);
  std::fclose(f);
  if (rd != n) return false;
  tokens_.assign(buf.begin(), buf.end());
  return true;
}

bool Dataset::load_text(const std::string& path, const BPETokenizer& tok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::string text;
  char b[1 << 16];
  size_t n;
  while ((n = std::fread(b, 1, sizeof(b), f)) > 0) text.append(b, n);
  std::fclose(f);
  if (text.empty()) return false;
  tokens_ = tok.encode(text);
  return !tokens_.empty();
}

int Dataset::max_id() const {
  int m = 0;
  for (int t : tokens_)
    if (t > m) m = t;
  return m;
}

void Dataset::random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                           std::vector<int>& tgts) const {
  toks.resize(static_cast<size_t>(B) * T);
  tgts.resize(static_cast<size_t>(B) * T);
  const int N = static_cast<int>(tokens_.size());
  const int hi = N - T - 1;
  const int span = hi > 0 ? hi : 1;
  for (int b = 0; b < B; ++b) {
    const int off = rng.below(span);
    for (int i = 0; i < T; ++i) {
      toks[static_cast<size_t>(b) * T + i] = tokens_[off + i];
      tgts[static_cast<size_t>(b) * T + i] = tokens_[off + i + 1];
    }
  }
}

}  // namespace slmpt
