// Train a byte-level BPE tokenizer from scratch (no external libraries) and
// optionally emit the corpus as int32 token ids.
//
//   slmpt_bpe --in data/corpus.txt --out data/tokenizer.txt \
//             --ids data/corpus.ids --vocab 4096
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

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

static std::string read_file(const std::string& path, bool& ok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  ok = false;
  if (!f) return {};
  std::string s;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  ok = true;
  return s;
}

int main(int argc, char** argv) {
  const std::string in = argval(argc, argv, "--in", "data/corpus.txt");
  const std::string out = argval(argc, argv, "--out", "data/tokenizer.txt");
  const std::string ids_out = argval(argc, argv, "--ids", "");
  const int target_vocab = std::atoi(argval(argc, argv, "--vocab", "4096"));

  bool ok = false;
  const std::string text = read_file(in, ok);
  if (!ok) {
    std::fprintf(stderr, "cannot read %s\n", in.c_str());
    return 1;
  }

  // 1) word frequencies
  std::unordered_map<std::string, int> freq;
  for (const std::string& w : BPETokenizer::pretokenize(text)) freq[w]++;

  std::vector<std::vector<int>> words;
  words.reserve(freq.size());
  for (const auto& kv : freq) {
    std::vector<int> sym;
    sym.reserve(kv.first.size());
    for (unsigned char c : kv.first) sym.push_back(static_cast<int>(c));
    words.push_back(std::move(sym));
    // keep the weight in a parallel vector
  }
  std::vector<int> weight;
  weight.reserve(freq.size());
  for (const auto& kv : freq) weight.push_back(kv.second);

  std::fprintf(stderr, "corpus %.2f MB, %zu unique words, target vocab %d\n",
               text.size() / 1048576.0, words.size(), target_vocab);

  // 2) learn merges
  std::vector<std::pair<int, int>> merges;
  const int max_merges = target_vocab - BPETokenizer::kBase;
  const auto t0 = std::chrono::steady_clock::now();
  std::unordered_map<uint64_t, long long> counts;
  for (int m = 0; m < max_merges; ++m) {
    counts.clear();
    for (size_t w = 0; w < words.size(); ++w) {
      const std::vector<int>& s = words[w];
      const long long f = weight[w];
      for (size_t i = 0; i + 1 < s.size(); ++i) {
        const uint64_t key =
            (static_cast<uint64_t>(static_cast<uint32_t>(s[i])) << 32) |
            static_cast<uint32_t>(s[i + 1]);
        counts[key] += f;
      }
    }
    uint64_t best_key = 0;
    long long best = 1;
    for (const auto& kv : counts) {
      if (kv.second > best) {
        best = kv.second;
        best_key = kv.first;
      }
    }
    if (best <= 1) break;
    const int a = static_cast<int>(best_key >> 32);
    const int b = static_cast<int>(best_key & 0xffffffffu);
    const int merged = BPETokenizer::kBase + static_cast<int>(merges.size());
    merges.emplace_back(a, b);
    // apply everywhere
    for (auto& s : words) {
      std::vector<int> nxt;
      nxt.reserve(s.size());
      for (size_t i = 0; i < s.size();) {
        if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) {
          nxt.push_back(merged);
          i += 2;
        } else {
          nxt.push_back(s[i]);
          ++i;
        }
      }
      s.swap(nxt);
    }
    if ((m + 1) % 500 == 0) {
      const double sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "  merge %d/%d (pair freq %lld, %.0fs)\n", m + 1, max_merges,
                   best, sec);
    }
  }

  BPETokenizer tok;
  tok.set_merges(merges);
  if (!tok.save(out)) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  std::fprintf(stderr, "wrote %s (vocab %d, %d merges)\n", out.c_str(), tok.vocab_size(),
               tok.n_merges());

  // 3) encode the corpus to int32 ids
  if (!ids_out.empty()) {
    std::vector<int> ids = tok.encode(text);
    FILE* f = std::fopen(ids_out.c_str(), "wb");
    if (!f) {
      std::fprintf(stderr, "cannot write %s\n", ids_out.c_str());
      return 1;
    }
    std::vector<int32_t> buf(ids.begin(), ids.end());
    std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "wrote %s (%zu tokens, %.2f tokens/byte)\n", ids_out.c_str(),
                 ids.size(), ids.empty() ? 0.0 : double(ids.size()) / text.size());
    const std::string back = tok.decode(ids);
    std::fprintf(stderr, "roundtrip: %s\n", back == text ? "identical" : "MISMATCH");
    if (back != text) return 1;
  }
  return 0;
}
