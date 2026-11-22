#pragma once
#include <vector>

#include "slmpt/model.hpp"

namespace slmpt {

// Mean cross-entropy and perplexity over tokens[lo, hi) using non-overlapping
// windows of length `block` (stride defaults to block). Writes the mean loss to
// *mean_loss (if non-null) and returns exp(mean_loss).
double eval_perplexity(Model& model, const std::vector<int>& tokens, size_t lo, size_t hi,
                       int block, int stride, double* mean_loss);

}  // namespace slmpt
