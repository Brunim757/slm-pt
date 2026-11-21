#include "slmpt/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>
#include <vector>

#include "slmpt/tokenizer.hpp"

namespace slmpt {

std::string generate(Model& model, const std::string& prompt, const SampleParams& sp,
                     Rng& rng) {
  ByteTokenizer tok;
  std::vector<int> ctx = tok.encode(prompt);
  if (ctx.empty()) ctx.push_back('\n');

  const int T = model.block();
  const int V = model.vocab();
  const int k = (sp.top_k > 0 && sp.top_k < V) ? sp.top_k : V;
  const float temp = sp.temperature > 0.f ? sp.temperature : 1.f;
  const float rep = sp.repetition_penalty > 1.f ? sp.repetition_penalty : 1.f;
  const float top_p = (sp.top_p > 0.f && sp.top_p < 1.f) ? sp.top_p : 1.f;

  const int prompt_len = static_cast<int>(ctx.size());
  std::vector<int> out;
  out.reserve(sp.max_new);
  std::vector<std::pair<float, int>> cand(V);

  for (int s = 0; s < sp.max_new; ++s) {
    const int n = std::min(static_cast<int>(ctx.size()), T);
    const int* window = ctx.data() + (ctx.size() - n);
    model.forward(window, 1, n);
    const float* lg = model.last_logits();

    // Repetition penalty over the recently generated tokens.
    std::unordered_set<int> recent;
    if (rep > 1.f) {
      const int w0 = std::max(0, static_cast<int>(ctx.size()) - sp.repetition_window);
      for (int i = w0; i < static_cast<int>(ctx.size()); ++i) recent.insert(ctx[i]);
    }

    for (int v = 0; v < V; ++v) {
      float l = lg[v];
      if (rep > 1.f && recent.count(v)) l = l > 0.f ? l / rep : l * rep;
      cand[v] = {l / temp, v};
    }
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        return a.first > b.first;
                      });

    const float mx = cand[0].first;
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
      cand[i].first = std::exp(cand[i].first - mx);
      sum += cand[i].first;
    }

    // Nucleus (top-p) truncation, then renormalize.
    int keep = k;
    if (top_p < 1.f) {
      float acc = 0.f;
      keep = k;
      for (int i = 0; i < k; ++i) {
        acc += cand[i].first / sum;
        if (acc >= top_p) {
          keep = i + 1;
          break;
        }
      }
    }
    float mass = 0.f;
    for (int i = 0; i < keep; ++i) mass += cand[i].first;

    float r = rng.uniform() * mass;
    int pick = cand[keep - 1].second;
    float acc = 0.f;
    for (int i = 0; i < keep; ++i) {
      acc += cand[i].first;
      if (r <= acc) {
        pick = cand[i].second;
        break;
      }
    }
    ctx.push_back(pick);
    out.push_back(pick);

    // Stop when the decoded continuation hits the stop sequence.
    if (!sp.stop.empty() && out.size() >= sp.stop.size()) {
      std::string tail = tok.decode(
          std::vector<int>(out.end() - static_cast<int>(sp.stop.size()), out.end()));
      if (tail == sp.stop) {
        out.resize(out.size() - sp.stop.size());
        break;
      }
    }
  }
  (void)prompt_len;
  return tok.decode(out);
}

}  // namespace slmpt
