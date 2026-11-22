#include "slmpt/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace slmpt {

namespace {
inline uint64_t pair_key(int a, int b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}
}  // namespace

void BPETokenizer::rebuild() {
  id_bytes_.assign(kBase + merges_.size(), std::string());
  for (int i = 0; i < kBase; ++i) id_bytes_[i] = std::string(1, static_cast<char>(i));
  rank_.clear();
  rank_.reserve(merges_.size() * 2);
  for (size_t i = 0; i < merges_.size(); ++i) {
    const int a = merges_[i].first, b = merges_[i].second;
    const int id = kBase + static_cast<int>(i);
    id_bytes_[id] = id_bytes_[a] + id_bytes_[b];
    rank_[pair_key(a, b)] = static_cast<int>(i);
  }
}

void BPETokenizer::set_merges(std::vector<std::pair<int, int>> merges) {
  merges_ = std::move(merges);
  rebuild();
}

bool BPETokenizer::load_from_string(const std::string& blob) {
  std::istringstream in(blob);
  std::string magic;
  int version = 0;
  if (!(in >> magic >> version) || magic != "SLMPTBPE") return false;
  int vocab = 0, n_merge = 0;
  if (!(in >> magic >> vocab)) return false;   // "vocab" <V>
  if (!(in >> magic >> n_merge)) return false;  // "merges" <M>
  std::vector<std::pair<int, int>> merges;
  merges.reserve(n_merge);
  for (int i = 0; i < n_merge; ++i) {
    int a = 0, b = 0;
    if (!(in >> a >> b)) return false;
    merges.emplace_back(a, b);
  }
  if (vocab != kBase + n_merge) return false;
  set_merges(std::move(merges));
  return true;
}

bool BPETokenizer::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::string blob;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) blob.append(buf, n);
  std::fclose(f);
  return load_from_string(blob);
}

std::string BPETokenizer::to_string() const {
  std::string s;
  s.reserve(merges_.size() * 12 + 64);
  s += "SLMPTBPE 1\n";
  s += "vocab " + std::to_string(vocab_size()) + "\n";
  s += "merges " + std::to_string(merges_.size()) + "\n";
  for (const auto& m : merges_) {
    s += std::to_string(m.first);
    s += ' ';
    s += std::to_string(m.second);
    s += '\n';
  }
  return s;
}

bool BPETokenizer::save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const std::string s = to_string();
  const bool ok = std::fwrite(s.data(), 1, s.size(), f) == s.size();
  std::fclose(f);
  return ok;
}

std::vector<std::string> BPETokenizer::pretokenize(const std::string& text) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : text) {
    if (c == ' ') {
      if (!cur.empty()) out.push_back(cur);
      cur = " ";
    } else {
      cur.push_back(static_cast<char>(c));
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
  std::vector<int> result;
  const std::vector<std::string> words = pretokenize(text);
  std::vector<int> ids, next;
  for (const std::string& w : words) {
    ids.clear();
    for (unsigned char c : w) ids.push_back(static_cast<int>(c));
    while (ids.size() >= 2) {
      int best_rank = -1, best_i = -1;
      for (size_t i = 0; i + 1 < ids.size(); ++i) {
        auto it = rank_.find(pair_key(ids[i], ids[i + 1]));
        if (it != rank_.end() && (best_rank < 0 || it->second < best_rank)) {
          best_rank = it->second;
          best_i = static_cast<int>(i);
        }
      }
      if (best_rank < 0) break;
      const int a = ids[best_i], b = ids[best_i + 1];
      const int merged = kBase + best_rank;
      next.clear();
      next.reserve(ids.size());
      for (size_t i = 0; i < ids.size();) {
        if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
          next.push_back(merged);
          i += 2;
        } else {
          next.push_back(ids[i]);
          ++i;
        }
      }
      ids.swap(next);
    }
    result.insert(result.end(), ids.begin(), ids.end());
  }
  return result;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
  return decode(ids.data(), static_cast<int>(ids.size()));
}

std::string BPETokenizer::decode(const int* ids, int n) const {
  std::string s;
  for (int i = 0; i < n; ++i) {
    const int id = ids[i];
    if (id >= 0 && id < static_cast<int>(id_bytes_.size())) s += id_bytes_[id];
  }
  return s;
}

}  // namespace slmpt
