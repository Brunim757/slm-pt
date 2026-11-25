#pragma once
#include <string>
#include <vector>

#include "slmpt/model.hpp"
#include "slmpt/rng.hpp"
#include "slmpt/tokenizer.hpp"

namespace slmpt {

struct SampleParams {
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.95f;
  float repetition_penalty = 1.15f;
  int repetition_window = 128;
  std::string stop = "<|end|>";
  // Extra sequences that also end the reply (stripped from the output). The
  // model is trained on multi-turn data, so after answering it may start a new
  // turn ("<|user|>") instead of emitting "<|end|>"; stopping on any turn
  // boundary keeps the reply to a single turn.
  std::vector<std::string> stop_any;
  int max_new = 240;
};

// Autoregressively generate text continuing `prompt`, using the model's KV cache.
std::string generate(Model& model, const BPETokenizer& tok, const std::string& prompt,
                     const SampleParams& sp, Rng& rng);

}  // namespace slmpt
