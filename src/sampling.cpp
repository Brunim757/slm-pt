#include "slmpt/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>
#include <vector>

namespace slmpt {

std::string generate(Model& model, const BPETokenizer& tok, const std::string& prompt,
                     const SampleParams& sp, Rng& rng) {
  const int T = model.block();
  const int V = model.vocab();
  std::vector<int> ctx = tok.encode(prompt);
  if (ctx.empty()) ctx.push_back('\n');

  // Keep room to generate: cap the prompt to the context window.
  if (static_cast<int>(ctx.size()) >= T) {
    ctx.erase(ctx.begin(), ctx.end() - (T - 1));
  }
  const int stop_len = static_cast<int>(sp.stop.size());

  const int k = (sp.top_k > 0 && sp.top_k < V) ? sp.top_k : V;
  const float temp = sp.temperature > 0.f ? sp.temperature : 1.f;
  const float rep = sp.repetition_penalty > 1.f ? sp.repetition_penalty : 1.f;
  const float top_p = (sp.top_p > 0.f && sp.top_p < 1.f) ? sp.top_p : 1.f;

  model.reset_cache();
  const float* lg = nullptr;
  for (int id : ctx) lg = model.forward_token(id);

  std::vector<int> out;
  out.reserve(sp.max_new);
  std::vector<std::pair<float, int>> cand(V);
  std::vector<int> all = ctx;  // prompt + generated, for the repetition penalty

  for (int s = 0; s < sp.max_new; ++s) {
    if (model.cache_pos() >= T) break;

    std::unordered_set<int> recent;
    if (rep > 1.f) {
      const int w0 = std::max(0, static_cast<int>(all.size()) - sp.repetition_window);
      for (int i = w0; i < static_cast<int>(all.size()); ++i) recent.insert(all[i]);
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

    int keep = k;
    if (top_p < 1.f) {
      float acc = 0.f;
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

    out.push_back(pick);
    all.push_back(pick);

    // Stop when the decoded text ends with the stop sequence.
    if (stop_len > 0) {
      const std::string cur = tok.decode(out);
      if (cur.size() >= sp.stop.size() &&
          cur.compare(cur.size() - sp.stop.size(), sp.stop.size(), sp.stop) == 0) {
        return cur.substr(0, cur.size() - sp.stop.size());
      }
    }

    lg = model.forward_token(pick);
  }
  return tok.decode(out);
}

}  // namespace slmpt
