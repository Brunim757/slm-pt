// Local RAG: build a retrieval index from your own text file, then search it.
//   slmpt_rag build --model runs/model.bin --in notes.txt --index runs/rag.bin
//   slmpt_rag query --model runs/model.bin --index runs/rag.bin --q "pergunta"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "slmpt/config.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/rag.hpp"
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
  const std::string cmd = argc > 1 ? argv[1] : "";
  const std::string model_path = argval(argc, argv, "--model", "runs/model.bin");
  const int threads = std::atoi(argval(argc, argv, "--threads", "0"));
  if (threads > 0) set_threads(threads);

  Config cfg;
  if (!peek_config(model_path, cfg)) {
    std::fprintf(stderr, "cannot read checkpoint: %s\n", model_path.c_str());
    return 1;
  }
  cfg.batch = 1;
  Model model(cfg);
  std::string blob;
  if (!load_checkpoint(model, model_path, &blob)) {
    std::fprintf(stderr, "cannot load checkpoint\n");
    return 1;
  }
  BPETokenizer tok;
  if (!blob.empty() && tok.load_from_string(blob)) {
  } else if (!tok.load(cfg.tokenizer)) {
    tok.set_merges({});
  }

  if (cmd == "build") {
    const std::string in = argval(argc, argv, "--in", "notes.txt");
    const std::string index_path = argval(argc, argv, "--index", "runs/rag.bin");
    RagIndex index;
    if (!build_index(model, tok, in, index)) {
      std::fprintf(stderr, "failed to build index from %s\n", in.c_str());
      return 1;
    }
    if (!save_index(index, index_path)) {
      std::fprintf(stderr, "failed to write %s\n", index_path.c_str());
      return 1;
    }
    std::printf("indexed %zu chunks (dim %d) -> %s\n", index.texts.size(), index.dim,
                index_path.c_str());
    return 0;
  }

  if (cmd == "query") {
    const std::string index_path = argval(argc, argv, "--index", "runs/rag.bin");
    const int topk = std::atoi(argval(argc, argv, "--topk", "3"));
    RagIndex index;
    if (!load_index(index_path, index)) {
      std::fprintf(stderr, "cannot load index: %s\n", index_path.c_str());
      return 1;
    }
    std::string q = argval(argc, argv, "--q", "");
    while (q.empty()) {
      std::printf("busca> ");
      std::fflush(stdout);
      if (!std::getline(std::cin, q) || q == "/sair") break;
    }
    std::vector<float> qe;
    embed_text(model, tok, q, qe);
    std::vector<int> idx;
    std::vector<float> score;
    retrieve(index, qe, topk, idx, score);
    for (size_t i = 0; i < idx.size(); ++i) {
      std::string t = index.texts[idx[i]];
      if (t.size() > 400) t = t.substr(0, 400) + "...";
      std::printf("[%.3f] %s\n\n", score[i], t.c_str());
    }
    return 0;
  }

  std::fprintf(stderr, "usage: %s build|query [...args]\n", argv[0]);
  return 1;
}
