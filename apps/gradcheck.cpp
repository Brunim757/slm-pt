// Numerical gradient check: compares analytic gradients from backward()
// against central finite differences of the loss (computed in double).
#include <cmath>
#include <cstdio>
#include <vector>

#include "slmpt/config.hpp"
#include "slmpt/model.hpp"
#include "slmpt/parallel.hpp"
#include "slmpt/rng.hpp"

using namespace slmpt;

int main() {
  Config c;
  c.vocab = 256;
  // Dims chosen as multiples of the widest SIMD register (32 floats) so the
  // finite-difference check exercises the vectorized GEMM kernels, not just the
  // scalar fallback.
  c.n_embd = 32;
  c.n_layer = 2;
  c.n_head = 4;
  c.block = 8;
  c.batch = 2;
  c.seed = 7;

  set_threads(1);
  Model model(c);
  const int B = c.batch, T = c.block;
  std::vector<int> toks(static_cast<size_t>(B) * T), tgts(static_cast<size_t>(B) * T);
  Rng rng(99);
  for (size_t i = 0; i < toks.size(); ++i) {
    toks[i] = rng.below(256);
    tgts[i] = rng.below(256);
  }

  const float loss = model.train_step(toks.data(), tgts.data(), B, T);
  std::printf("loss = %.5f\n", loss);

  double max_abs = 0.0, max_ratio = 0.0;
  int bad = 0, checked = 0;
  const int P = static_cast<int>(model.params().size());
  Rng pick(5);
  const float h = 5e-3f;

  for (int trial = 0; trial < 300; ++trial) {
    const int pi = pick.below(P);
    Param* p = model.params()[pi];
    if (p->size() == 0) continue;
    const int idx = pick.below(static_cast<int>(p->size()));
    const float orig = p->data[idx];
    const double analytic = p->grad[idx];

    p->data[idx] = orig + h;
    const float plus_val = p->data[idx];
    const double lp = model.compute_loss(toks.data(), tgts.data(), B, T);
    p->data[idx] = orig - h;
    const float minus_val = p->data[idx];
    const double lm = model.compute_loss(toks.data(), tgts.data(), B, T);
    p->data[idx] = orig;

    const double delta = static_cast<double>(plus_val) - static_cast<double>(minus_val);
    const double numeric = (lp - lm) / delta;
    const double abs_err = std::fabs(analytic - numeric);
    const double tol = 3e-4 + 3e-3 * std::fabs(analytic);

    checked++;
    if (abs_err > max_abs) max_abs = abs_err;
    if (tol > 0) max_ratio = std::max(max_ratio, abs_err / tol);
    if (abs_err > tol) {
      bad++;
      if (bad <= 10)
        std::printf("MISMATCH param %d elem %d: analytic=%.6e numeric=%.6e "
                    "abs_err=%.3e tol=%.3e\n",
                    pi, idx, analytic, numeric, abs_err, tol);
    }
  }

  std::printf("checked=%d  max_abs_err=%.3e  worst/tol=%.2f  bad=%d\n", checked,
              max_abs, max_ratio, bad);

  // KV-cache consistency: sequential forward_token must reproduce the full
  // (parallel) forward used in training.
  {
    std::vector<int> seq(T);
    for (int i = 0; i < T; ++i) seq[i] = (i * 97 + 13) % 256;
    model.forward(seq.data(), 1, T);
    const int V = model.vocab();
    std::vector<float> full(model.last_logits(), model.last_logits() + V);
    model.reset_cache();
    const float* lg = nullptr;
    for (int i = 0; i < T; ++i) lg = model.forward_token(seq[i]);
    double md = 0.0;
    for (int v = 0; v < V; ++v)
      md = std::max(md, static_cast<double>(std::fabs(full[v] - lg[v])));
    const bool ok = md < 1e-3;
    std::printf("kv-cache max|diff|=%.3e  =>  %s\n", md, ok ? "OK" : "FAIL");
    if (!ok) bad++;
  }

  std::printf("RESULT => %s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
