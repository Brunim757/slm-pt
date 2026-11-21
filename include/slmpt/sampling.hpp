#pragma once
#include <string>

#include "slmpt/model.hpp"
#include "slmpt/rng.hpp"

namespace slmpt {

struct SampleParams {
  float temperature = 0.8f;
  int top_k = 40;
  int max_new = 240;
};

// Autoregressively generate text continuing `prompt`.
std::string generate(Model& model, const std::string& prompt, const SampleParams& sp,
                     Rng& rng);

}  // namespace slmpt
