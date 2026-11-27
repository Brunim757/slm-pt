// Held-out evaluation: mean cross-entropy + perplexity on a validation split.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "slmpt/dataset.hpp"
#include "slmpt/eval.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/serialization.hpp"
#include "slmpt/tokenizer.hpp"

using namespace slmpt;

static const char* argval(int argc, char** argv, const char* key, const char* def) {
  const size_t kl = std::strlen(key);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0 && i + 1 < argc) return argv[i + 1];
    if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') return argv[i] + kl + 1;
  }
  return def;
}

int main(int argc, char** argv) {
  const std::string path = argval(argc, argv, "--model", "runs/model.bin");
  const std::string ids = argval(argc, argv, "--ids", "data/corpus.ids");
  const std::string data = argval(argc, argv, "--data", "data/corpus.txt");
  const double val_frac = std::atof(argval(argc, argv, "--val", "0.05"));
  const int threads = std::atoi(argval(argc, argv, "--threads", "0"));
  if (threads > 0) set_threads(threads);

  Config cfg;
  if (!peek_config(path, cfg)) {
    std::fprintf(stderr, "cannot read checkpoint: %s\n", path.c_str());
    return 1;
  }
  cfg.batch = 1;
  Model model(cfg);
  std::string blob;
  if (!load_checkpoint(model, path, &blob)) {
    std::fprintf(stderr, "cannot load checkpoint: %s\n", path.c_str());
    return 1;
  }
  BPETokenizer tok;
  if (!blob.empty() && tok.load_from_string(blob)) {
  } else if (!tok.load(cfg.tokenizer)) {
    tok.set_merges({});
  }

  Dataset ds;
  if (!ds.load_ids(ids) && !ds.load_text(data, tok)) {
    std::fprintf(stderr, "cannot load data (tried %s and %s)\n", ids.c_str(), data.c_str());
    return 1;
  }
  const size_t n = ds.size();
  const size_t val_start = n - static_cast<size_t>(n * val_frac);
  double mean_loss = 0.0;
  const double ppl = eval_perplexity(model, ds, val_start, n, cfg.block, cfg.block,
                                     &mean_loss);
  std::printf("model: %s (vocab %d, block %d, params", path.c_str(), cfg.vocab, cfg.block);
  long params = 0;
  for (Param* p : model.params()) params += (long)p->size();
  std::printf(" %.2fM)\n", params / 1e6);
  std::printf("val tokens: %zu (last %.1f%%)\n", n - val_start, val_frac * 100.0);
  std::printf("loss: %.4f   perplexity: %.2f\n", mean_loss, ppl);
  return 0;
}
