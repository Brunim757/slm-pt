#include "slmpt/rag.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace slmpt {

namespace {
void normalize(std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  const float inv = s > 0 ? static_cast<float>(1.0 / std::sqrt(s)) : 0.f;
  for (float& x : v) x *= inv;
}
}  // namespace

void embed_text(Model& model, const BPETokenizer& tok, const std::string& text,
                std::vector<float>& out) {
  std::vector<int> ids = tok.encode(text);
  const int cap = model.block();
  if (static_cast<int>(ids.size()) > cap)
    ids.erase(ids.begin(), ids.end() - cap);
  if (ids.empty()) ids.push_back('\n');
  model.forward(ids.data(), 1, static_cast<int>(ids.size()));
  model.mean_hidden(out);
  normalize(out);
}

bool build_index(Model& model, const BPETokenizer& tok, const std::string& text_file,
                 RagIndex& index) {
  FILE* f = std::fopen(text_file.c_str(), "rb");
  if (!f) return false;
  std::string text;
  char b[1 << 16];
  size_t n;
  while ((n = std::fread(b, 1, sizeof(b), f)) > 0) text.append(b, n);
  std::fclose(f);

  index.texts.clear();
  index.emb.clear();

  // Split into paragraphs on blank lines; fall back to lines, then the whole file.
  std::vector<std::string> chunks;
  std::string cur;
  size_t i = 0;
  while (i < text.size()) {
    if (i + 1 < text.size() && text[i] == '\n' && text[i + 1] == '\n') {
      if (!cur.empty()) chunks.push_back(cur);
      cur.clear();
      i += 2;
    } else {
      cur.push_back(text[i]);
      ++i;
    }
  }
  if (!cur.empty()) chunks.push_back(cur);
  if (chunks.empty() && !text.empty()) chunks.push_back(text);

  for (const std::string& c : chunks) {
    std::vector<float> e;
    embed_text(model, tok, c, e);
    if (index.dim == 0) index.dim = static_cast<int>(e.size());
    index.texts.push_back(c);
    index.emb.push_back(std::move(e));
  }
  return !index.texts.empty();
}

bool save_index(const RagIndex& index, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const char magic[8] = {'S', 'L', 'M', 'R', 'A', 'G', '0', '1'};
  std::fwrite(magic, 1, 8, f);
  int32_t dim = index.dim;
  int64_t count = static_cast<int64_t>(index.texts.size());
  std::fwrite(&dim, sizeof(int32_t), 1, f);
  std::fwrite(&count, sizeof(int64_t), 1, f);
  for (size_t i = 0; i < index.texts.size(); ++i) {
    const int64_t tl = static_cast<int64_t>(index.texts[i].size());
    std::fwrite(&tl, sizeof(int64_t), 1, f);
    std::fwrite(index.texts[i].data(), 1, index.texts[i].size(), f);
    std::fwrite(index.emb[i].data(), sizeof(float), index.emb[i].size(), f);
  }
  std::fclose(f);
  return true;
}

bool load_index(const std::string& path, RagIndex& index) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "SLMRAG01", 8) != 0) {
    std::fclose(f);
    return false;
  }
  int32_t dim = 0;
  int64_t count = 0;
  std::fread(&dim, sizeof(int32_t), 1, f);
  std::fread(&count, sizeof(int64_t), 1, f);
  index.dim = dim;
  index.texts.assign(static_cast<size_t>(count), std::string());
  index.emb.assign(static_cast<size_t>(count), std::vector<float>(dim, 0.f));
  for (int64_t i = 0; i < count; ++i) {
    int64_t tl = 0;
    std::fread(&tl, sizeof(int64_t), 1, f);
    index.texts[i].resize(static_cast<size_t>(tl));
    if (tl > 0) std::fread(&index.texts[i][0], 1, static_cast<size_t>(tl), f);
    std::fread(index.emb[i].data(), sizeof(float), dim, f);
  }
  std::fclose(f);
  return true;
}

void retrieve(const RagIndex& index, const std::vector<float>& query, int topk,
              std::vector<int>& out_idx, std::vector<float>& out_score) {
  std::vector<std::pair<float, int>> scored;
  scored.reserve(index.emb.size());
  for (size_t i = 0; i < index.emb.size(); ++i) {
    const std::vector<float>& e = index.emb[i];
    float s = 0.f;
    const int d = std::min(static_cast<int>(e.size()), static_cast<int>(query.size()));
    for (int k = 0; k < d; ++k) s += e[k] * query[k];
    scored.emplace_back(s, static_cast<int>(i));
  }
  if (topk > static_cast<int>(scored.size())) topk = static_cast<int>(scored.size());
  std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                    });
  out_idx.clear();
  out_score.clear();
  for (int i = 0; i < topk; ++i) {
    out_idx.push_back(scored[i].second);
    out_score.push_back(scored[i].first);
  }
}

}  // namespace slmpt
