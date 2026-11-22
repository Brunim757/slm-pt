#include "slmpt/eval.hpp"

#include <cmath>

namespace slmpt {

double eval_perplexity(Model& model, const std::vector<int>& tokens, size_t lo, size_t hi,
                       int block, int stride, double* mean_loss) {
  if (hi > tokens.size()) hi = tokens.size();
  if (stride <= 0) stride = block;
  double total = 0.0;
  long count = 0;
  const int* base = tokens.data();
  for (size_t off = lo; off + static_cast<size_t>(block) + 1 <= hi;
       off += static_cast<size_t>(stride)) {
    const double l = model.compute_loss(base + off, base + off + 1, 1, block);
    total += l * block;
    count += block;
  }
  const double avg = count > 0 ? total / count : 0.0;
  if (mean_loss) *mean_loss = avg;
  return std::exp(avg);
}

}  // namespace slmpt
