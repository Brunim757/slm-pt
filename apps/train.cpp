#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "slmpt/dataset.hpp"
#include "slmpt/model.hpp"
#include "slmpt/optimizer.hpp"
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
static int argi(int argc, char** argv, const char* key, int def) {
  const char* v = argval(argc, argv, key, nullptr);
  return v ? std::atoi(v) : def;
}
static float argf(int argc, char** argv, const char* key, float def) {
  const char* v = argval(argc, argv, key, nullptr);
  return v ? static_cast<float>(std::atof(v)) : def;
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  Config cfg;
  cfg.data = argval(argc, argv, "--data", cfg.data.c_str());
  cfg.out_dir = argval(argc, argv, "--out", cfg.out_dir.c_str());
  cfg.resume = argval(argc, argv, "--resume", "");
  cfg.n_embd = argi(argc, argv, "--embd", cfg.n_embd);
  cfg.n_layer = argi(argc, argv, "--layer", cfg.n_layer);
  cfg.n_head = argi(argc, argv, "--head", cfg.n_head);
  cfg.block = argi(argc, argv, "--block", cfg.block);
  cfg.batch = argi(argc, argv, "--batch", cfg.batch);
  cfg.max_steps = argi(argc, argv, "--steps", cfg.max_steps);
  cfg.warmup = argi(argc, argv, "--warmup", cfg.warmup);
  cfg.save_every = argi(argc, argv, "--save-every", cfg.save_every);
  cfg.sample_every = argi(argc, argv, "--sample-every", cfg.sample_every);
  cfg.lr = argf(argc, argv, "--lr", cfg.lr);
  cfg.weight_decay = argf(argc, argv, "--wd", cfg.weight_decay);
  cfg.grad_clip = argf(argc, argv, "--clip", cfg.grad_clip);
  cfg.max_minutes = std::atof(argval(argc, argv, "--max-minutes", "0"));
  const int threads = argi(argc, argv, "--threads", (int)std::thread::hardware_concurrency());
  cfg.seed = (uint64_t)std::atoll(argval(argc, argv, "--seed", "1337"));

  set_threads(threads > 0 ? threads : 1);
  std::filesystem::create_directories(cfg.out_dir);

  Dataset ds;
  if (!ds.load(cfg.data)) {
    std::fprintf(stderr, "failed to load data: %s\n", cfg.data.c_str());
    return 1;
  }

  Model model(cfg);
  long params = 0;
  for (Param* p : model.params()) params += (long)p->size();
  std::printf("== slm-pt ==\n");
  std::printf("data: %s (%.2f MB)\n", cfg.data.c_str(), ds.size() / 1048576.0);
  std::printf("arch: embd=%d layer=%d head=%d block=%d batch=%d\n", cfg.n_embd,
              cfg.n_layer, cfg.n_head, cfg.block, cfg.batch);
  std::printf("params: %.2f M   threads: %d\n", params / 1e6, get_threads());

  if (!cfg.resume.empty()) {
    if (load_checkpoint(model, cfg.resume))
      std::printf("resumed from %s\n", cfg.resume.c_str());
    else
      std::printf("WARN: could not resume from %s (starting fresh)\n", cfg.resume.c_str());
  }

  AdamW opt(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, cfg.grad_clip);
  Rng rng(cfg.seed ^ 0xABCDEFull);

  std::vector<int> toks, tgts;
  const std::string ckpt = cfg.out_dir + "/model.bin";
  const std::string samples = cfg.out_dir + "/samples.txt";
  SampleParams sp;
  sp.temperature = 0.8f;
  sp.top_k = 40;
  sp.max_new = 220;
  const std::string probe = "### Usuário:\nOlá! Quem é você?\n\n### Assistente:\n";

  const double t0 = now_s();
  const double budget = cfg.max_minutes > 0 ? cfg.max_minutes * 60.0 : 1e18;
  double last_log = t0;
  long tokens_seen = 0;

  for (int step = 1; step <= cfg.max_steps; ++step) {
    ds.random_batch(rng, cfg.batch, cfg.block, toks, tgts);
    const float loss = model.train_step(toks.data(), tgts.data(), cfg.batch, cfg.block);

    float lr = cfg.lr;
    if (step <= cfg.warmup)
      lr *= static_cast<float>(step) / cfg.warmup;
    else if (cfg.max_steps <= 200000) {
      float p = static_cast<float>(step - cfg.warmup) / (cfg.max_steps - cfg.warmup + 1);
      p = std::min(1.f, p);
      lr *= 0.5f * (1.f + std::cos(3.14159265f * p));
    }
    opt.set_lr(lr);
    opt.step(model.params(), step);

    tokens_seen += (long)cfg.batch * cfg.block;
    const double t = now_s();
    if (step % 10 == 0 || step == 1) {
      const double dt = t - last_log;
      const double tps = dt > 0 ? (10.0 * cfg.batch * cfg.block) / dt : 0;
      std::printf("step %6d | loss %.4f | lr %.2e | gnorm %.2f | %.0f tok/s | %.1f min\n",
                  step, loss, lr, opt.last_grad_norm(), tps, (t - t0) / 60.0);
      std::fflush(stdout);
      last_log = t;
    }

    if (cfg.sample_every > 0 && step % cfg.sample_every == 0) {
      std::string text = generate(model, probe, sp, rng);
      auto pos = text.find("\n\n");
      if (pos != std::string::npos) text = text.substr(0, pos);
      FILE* f = std::fopen(samples.c_str(), "a");
      if (f) {
        std::fprintf(f, "=== step %d ===\n%s\n\n", step, text.c_str());
        std::fclose(f);
      }
      std::printf("  [sample] %s\n", text.c_str());
      std::fflush(stdout);
    }

    if (cfg.save_every > 0 && step % cfg.save_every == 0) {
      save_checkpoint(model, ckpt);
      std::printf("  [saved] %s\n", ckpt.c_str());
    }
    if ((t - t0) > budget) {
      std::printf("time budget reached at step %d\n", step);
      break;
    }
  }

  save_checkpoint(model, ckpt);
  std::string text = generate(model, probe, sp, rng);
  auto pos = text.find("\n\n");
  if (pos != std::string::npos) text = text.substr(0, pos);
  std::printf("\nfinal sample:\n%s\n", text.c_str());
  std::printf("saved checkpoint: %s\n", ckpt.c_str());
  return 0;
}
