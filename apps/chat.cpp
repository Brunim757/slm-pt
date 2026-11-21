#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "slmpt/config.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/sampling.hpp"
#include "slmpt/serialization.hpp"

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
  const float temp = std::atof(argval(argc, argv, "--temp", "0.8"));
  const int top_k = std::atoi(argval(argc, argv, "--top-k", "40"));
  const int max_new = std::atoi(argval(argc, argv, "--max-new", "240"));
  const int threads = std::atoi(
      argval(argc, argv, "--threads", "0"));
  if (threads > 0) set_threads(threads);

  Config cfg;
  if (!peek_config(path, cfg)) {
    std::fprintf(stderr, "cannot read checkpoint: %s\n", path.c_str());
    return 1;
  }
  cfg.batch = 1;  // inference
  Model model(cfg);
  if (!load_checkpoint(model, path)) {
    std::fprintf(stderr, "cannot load checkpoint: %s\n", path.c_str());
    return 1;
  }

  std::printf("slm-pt chat (byte-level) — digite e pressione Enter. /sair para sair.\n");
  std::printf("modelo: %s (embd=%d layer=%d head=%d block=%d)\n\n", path.c_str(),
              cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block);

  Rng rng(20240521);
  SampleParams sp;
  sp.temperature = temp;
  sp.top_k = top_k;
  sp.max_new = max_new;

  std::string line;
  while (true) {
    std::printf("você> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line == "/sair" || line == "/exit") break;
    if (line.empty()) continue;

    std::string prompt = "### Usuário:\n" + line + "\n\n### Assistente:\n";
    std::string resp = generate(model, prompt, sp, rng);
    const auto stop = resp.find("\n\n");
    if (stop != std::string::npos) resp = resp.substr(0, stop);
    std::printf("ia> %s\n\n", resp.c_str());
    std::fflush(stdout);
  }
  return 0;
}
