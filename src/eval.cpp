#include "slmpt/eval.hpp"

#include <cmath>
#include <vector>

namespace slmpt {

double eval_perplexity(Model& model, const Dataset& ds, size_t lo, size_t hi, int block,
                       int stride, double* mean_loss) {
  if (hi > ds.size()) hi = ds.size();
  if (stride <= 0) stride = block;
  double total = 0.0;
  long count = 0;
  // The dataset may be a uint16 mapping, so each window is copied into a small
  // int buffer (block + 1 ids) before being handed to the model.
  std::vector<int> win(static_cast<size_t>(block) + 1);
  for (size_t off = lo; off + static_cast<size_t>(block) + 1 <= hi;
       off += static_cast<size_t>(stride)) {
    for (int i = 0; i <= block; ++i) win[static_cast<size_t>(i)] = ds.at(off + i);
    const double l = model.compute_loss(win.data(), win.data() + 1, 1, block);
    total += l * block;
    count += block;
  }
  const double avg = count > 0 ? total / count : 0.0;
  if (mean_loss) *mean_loss = avg;
  return std::exp(avg);
}

}  // namespace slmpt
