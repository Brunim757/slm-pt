#pragma once
#include <string>

#include "slmpt/model.hpp"
#include "slmpt/rng.hpp"

namespace slmpt {

struct SampleParams {
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.95f;
  float repetition_penalty = 1.15f;
  int repetition_window = 96;
  std::string stop = "\n\n### Usuário:";
  int max_new = 240;
};

// Autoregressively generate text continuing `prompt`.
std::string generate(Model& model, const std::string& prompt, const SampleParams& sp,
                     Rng& rng);

}  // namespace slmpt
