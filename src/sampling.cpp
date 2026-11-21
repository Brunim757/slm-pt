#include "slmpt/sampling.hpp"

#include <algorithm>
#include <cmath>
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

  std::vector<int> out;
  out.reserve(sp.max_new);
  std::vector<std::pair<float, int>> cand(V);

  for (int s = 0; s < sp.max_new; ++s) {
    const int n = std::min(static_cast<int>(ctx.size()), T);
    const int* window = ctx.data() + (ctx.size() - n);
    model.forward(window, 1, n);
    const float* lg = model.last_logits();

    for (int v = 0; v < V; ++v) cand[v] = {lg[v] / temp, v};
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
    float r = rng.uniform() * sum;
    int pick = cand[k - 1].second;
    float acc = 0.f;
    for (int i = 0; i < k; ++i) {
      acc += cand[i].first;
      if (r <= acc) {
        pick = cand[i].second;
        break;
      }
    }
    ctx.push_back(pick);
    out.push_back(pick);
  }
  return tok.decode(out);
}

}  // namespace slmpt
