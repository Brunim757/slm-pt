#pragma once
#include <cstdint>
#include <string>

namespace slmpt {

// Everything needed to build + train + resume a model.
struct Config {
  // architecture
  int vocab = 256;    // byte-level
  int n_embd = 192;
  int n_layer = 4;
  int n_head = 6;
  int block = 192;    // context length
  int batch = 48;

  // optimizer
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;
  int warmup = 100;
  int max_steps = 1000000000;

  uint64_t seed = 1337;

  // io / loop
  std::string data = "data/corpus.txt";
  std::string out_dir = "runs";
  int save_every = 250;
  int sample_every = 250;
  double max_minutes = 0.0;  // 0 = no limit
  std::string resume;        // optional checkpoint to resume from
};

}  // namespace slmpt
