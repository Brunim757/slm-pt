#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "slmpt/config.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/quant.hpp"
#include "slmpt/rag.hpp"
#include "slmpt/sampling.hpp"
#include "slmpt/serialization.hpp"
#include "slmpt/tokenizer.hpp"

using namespace slmpt;

// Identity, matching the system line used when building the corpus (first
// variant), prepended to every prompt so the model always knows who it is.
static const char* kSystem =
    "<|system|>\nVocê é o KizuSmall, uma IA pequena de terminal que fala português.\n";

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
  const float temp = std::atof(argval(argc, argv, "--temp", "0.8"));
  const int top_k = std::atoi(argval(argc, argv, "--top-k", "40"));
  const float top_p = std::atof(argval(argc, argv, "--top-p", "0.9"));
  const float rep = std::atof(argval(argc, argv, "--rep", "1.15"));
  const int max_new = std::atoi(argval(argc, argv, "--max-new", "240"));
  const int threads = std::atoi(argval(argc, argv, "--threads", "0"));
  if (threads > 0) set_threads(threads);

  Config cfg;
  if (!peek_config(path, cfg)) {
    std::fprintf(stderr, "cannot read checkpoint: %s\n", path.c_str());
    return 1;
  }
  cfg.batch = 1;  // inference
  Model model(cfg);
  std::string blob;
  const bool quantized = is_quantized(path);
  if (quantized ? !load_quantized(path, model, &blob) : !load_checkpoint(model, path, &blob)) {
    std::fprintf(stderr, "cannot load checkpoint: %s\n", path.c_str());
    return 1;
  }

  BPETokenizer tok;
  if (!blob.empty() && tok.load_from_string(blob)) {
    // embedded tokenizer
  } else if (!tok.load(cfg.tokenizer)) {
    tok.set_merges({});
    std::fprintf(stderr, "WARN: no tokenizer found, using raw bytes\n");
  }

  std::printf("KizuSmall chat (BPE, vocab %d) — digite e pressione Enter. /sair para sair.\n",
              tok.vocab_size());
  std::printf("modelo: %s (embd=%d layer=%d head=%d block=%d)\n\n", path.c_str(),
              cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block);

  // Fresh random seed per session by default, so the same question does not
  // always get the same answer. Override with --seed N for reproducibility.
  const std::string seed_str = argval(argc, argv, "--seed", "");
  const uint64_t seed =
      seed_str.empty()
          ? static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count())
          : std::strtoull(seed_str.c_str(), nullptr, 10);
  Rng rng(seed);
  SampleParams sp;
  sp.temperature = temp;
  sp.top_k = top_k;
  sp.top_p = top_p;
  sp.repetition_penalty = rep;
  sp.max_new = max_new;
  sp.stop = "<|end|>";
  // End the reply at the first turn boundary, so a model that keeps going after
  // the answer (starting a new "<|user|>" turn) doesn't ramble on.
  sp.stop_any = {"<|user|>", "<|system|>", "<|text|>", "<|assistant|>"};

  // Optional local retrieval (RAG): ground answers in your own files.
  RagIndex rag;
  bool use_rag = false;
  const std::string rag_path = argval(argc, argv, "--rag", "");
  const int rag_topk = std::atoi(argval(argc, argv, "--rag-topk", "3"));
  if (!rag_path.empty()) {
    if (load_index(rag_path, rag))
      use_rag = true;
    else
      std::fprintf(stderr, "WARN: cannot load rag index %s\n", rag_path.c_str());
  }

  std::string line;
  while (true) {
    std::printf("você> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line == "/sair" || line == "/exit") break;
    if (line.empty()) continue;

    std::string context;
    if (use_rag) {
      std::vector<float> qe;
      embed_text(model, tok, line, qe);
      std::vector<int> idx;
      std::vector<float> score;
      retrieve(rag, qe, rag_topk, idx, score);
      for (int i : idx) context += rag.texts[i] + "\n";
    }

    const std::string prompt = kSystem + std::string("<|user|>\n") + context + line +
                               "\n<|assistant|>\n";
    std::string resp = generate(model, tok, prompt, sp, rng);
    std::printf("ia> %s\n\n", resp.c_str());
    std::fflush(stdout);
  }
  return 0;
}
