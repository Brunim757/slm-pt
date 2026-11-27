// Compress a checkpoint with group-wise int8/int4/int2 quantization and report
// size + held-out perplexity before/after.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "slmpt/dataset.hpp"
#include "slmpt/eval.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/quant.hpp"
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
  const std::string out = argval(argc, argv, "--out", "runs/model_q4.bin");
  const int bits = std::atoi(argval(argc, argv, "--bits", "4"));
  const int group = std::atoi(argval(argc, argv, "--group", "64"));
  const double val_frac = std::atof(argval(argc, argv, "--val", "0.02"));
  const long max_toks = std::atol(argval(argc, argv, "--max-tokens", "0"));
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
    std::fprintf(stderr, "cannot load data\n");
    return 1;
  }
  const size_t n = ds.size();
  const size_t val_start = n - static_cast<size_t>(n * val_frac);
  size_t val_end = n;
  if (max_toks > 0 && val_start + static_cast<size_t>(max_toks) < val_end)
    val_end = val_start + static_cast<size_t>(max_toks);

  double loss0 = 0.0;
  const double ppl0 = eval_perplexity(model, ds, val_start, val_end, cfg.block, cfg.block,
                                      &loss0);

  QuantStats st;
  if (!save_quantized(model, out, bits, group, blob, &st)) {
    std::fprintf(stderr, "quantize failed\n");
    return 1;
  }

  Model qmodel(cfg);
  std::string qblob;
  if (!load_quantized(out, qmodel, &qblob)) {
    std::fprintf(stderr, "cannot reload quantized file\n");
    return 1;
  }
  double loss1 = 0.0;
  const double ppl1 = eval_perplexity(qmodel, ds, val_start, val_end, cfg.block, cfg.block,
                                      &loss1);

  std::printf("== quantization report ==\n");
  std::printf("bits/group:        %d / %d   (tensors %d/%d quantized)\n", st.bits, st.group,
              st.quantized_tensors, st.total_tensors);
  std::printf("fp32 size:         %.2f MB\n", st.orig_bytes / 1048576.0);
  std::printf("compressed size:   %.2f MB\n", st.quant_bytes / 1048576.0);
  std::printf("compression:       %.2fx smaller\n", st.ratio());
  std::printf("perplexity fp32:   %.3f (loss %.4f)\n", ppl0, loss0);
  std::printf("perplexity int%d:   %.3f (loss %.4f)\n", bits, ppl1, loss1);
  std::printf("quality delta:     %+.2f%% perplexity\n",
              100.0 * (ppl1 - ppl0) / (ppl0 > 0 ? ppl0 : 1.0));
  return 0;
}
