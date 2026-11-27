// Train a byte-level BPE tokenizer from scratch (no external libraries) and
// optionally emit the corpus as uint16 token ids.
//
//   slmpt_bpe --in data/corpus.txt --out data/tokenizer.txt \
//             --ids data/corpus.ids --vocab 4096
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "slmpt/ids_format.hpp"
#include "slmpt/parallel.hpp"
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

// Encodes `text` to a uint16 ids file, in parallel and streaming: the corpus is
// cut into newline-aligned segments, each segment is split across threads at
// newline boundaries (which can never slice a special token), the pieces are
// encoded concurrently, then written in order.  Only one segment's pieces are
// ever resident, so even a multi-GB corpus encodes without a matching RAM peak.
static void write_ids_u16(const BPETokenizer& tok, const std::string& text,
                          const std::string& path, int nthreads) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  unsigned char hdr[idsfmt::kHeaderBytes] = {0};
  std::memcpy(hdr, idsfmt::kMagic, 8);
  const uint32_t dtype = idsfmt::kDtypeU16;
  std::memcpy(hdr + 8, &dtype, 4);
  std::fwrite(hdr, 1, sizeof(hdr), f);

  const size_t seg_target = 16u << 20;  // ~16 MB of text per segment
  const size_t N = text.size();
  uint64_t total = 0;
  for (size_t a = 0; a < N;) {
    size_t b = std::min(N, a + seg_target);
    if (b < N) {
      const size_t nl = text.find('\n', b);
      b = (nl == std::string::npos) ? N : nl + 1;
    }
    if (b <= a) b = std::min(N, a + seg_target);
    const size_t len = b - a;

    int k = nthreads < 1 ? 1 : nthreads;
    if (len < static_cast<size_t>(k) * 4096) k = std::max(1, static_cast<int>(len / 4096));

    std::vector<size_t> bnd(static_cast<size_t>(k) + 1);
    bnd[0] = a;
    bnd[static_cast<size_t>(k)] = b;
    for (int i = 1; i < k; ++i) {
      const size_t t = a + len * static_cast<size_t>(i) / static_cast<size_t>(k);
      size_t p = text.find('\n', t);
      if (p == std::string::npos || p >= b)
        p = b;
      else
        p += 1;
      if (p < bnd[static_cast<size_t>(i) - 1]) p = bnd[static_cast<size_t>(i) - 1];
      bnd[static_cast<size_t>(i)] = p;
    }

    std::vector<std::vector<int>> parts(static_cast<size_t>(k));
    parallel_for(k, [&](int p0, int p1) {
      for (int i = p0; i < p1; ++i)
        parts[static_cast<size_t>(i)] =
            tok.encode(text.substr(bnd[static_cast<size_t>(i)],
                                   bnd[static_cast<size_t>(i) + 1] -
                                       bnd[static_cast<size_t>(i)]));
    });
    for (int i = 0; i < k; ++i) {
      const std::vector<int>& ids = parts[static_cast<size_t>(i)];
      std::vector<uint16_t> w(ids.begin(), ids.end());
      if (!w.empty()) std::fwrite(w.data(), sizeof(uint16_t), w.size(), f);
      total += w.size();
      std::vector<int>().swap(parts[static_cast<size_t>(i)]);  // free immediately
    }
    a = b;
  }

  const uint32_t cnt = total > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(total);
  std::fseek(f, 12, SEEK_SET);
  std::fwrite(&cnt, sizeof(uint32_t), 1, f);
  std::fclose(f);
  std::fprintf(stderr, "wrote %s (%llu tokens, uint16, %d threads)\n", path.c_str(),
               static_cast<unsigned long long>(total), nthreads);
}

// Streams the ids file back and checks that decoding reproduces `text` exactly.
static bool verify_roundtrip(const BPETokenizer& tok, const std::string& text,
                             const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  if (std::fseek(f, idsfmt::kHeaderBytes, SEEK_SET) != 0) {
    std::fclose(f);
    return false;
  }
  std::vector<uint16_t> blk(1u << 20);
  std::vector<int> ids;
  size_t off = 0;
  bool identical = true;
  size_t got;
  while ((got = std::fread(blk.data(), sizeof(uint16_t), blk.size(), f)) > 0) {
    ids.assign(blk.begin(), blk.begin() + static_cast<long>(got));
    const std::string d = tok.decode(ids);
    if (d.size() > text.size() - off || text.compare(off, d.size(), d) != 0) {
      identical = false;
      break;
    }
    off += d.size();
  }
  std::fclose(f);
  return identical && off == text.size();
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

  // --load <tok>: reuse an existing tokenizer and only (re)encode the corpus.
  // This keeps the vocabulary byte-for-byte identical when the corpus grows, so
  // a checkpoint can keep training on more data without a tokenizer mismatch.
  const std::string load_tok = argval(argc, argv, "--load", "");
  BPETokenizer tok;
  if (!load_tok.empty()) {
    if (!tok.load(load_tok)) {
      std::fprintf(stderr, "cannot load tokenizer %s\n", load_tok.c_str());
      return 1;
    }
    std::fprintf(stderr, "loaded tokenizer %s (vocab %d, %d merges)\n",
                 load_tok.c_str(), tok.vocab_size(), tok.n_merges());
  } else {
  // 1) word frequencies.  Reserved control tokens are stripped first so that no
  //    merge is ever learned inside them — they must stay atomic.
  const std::vector<std::string>& specials = BPETokenizer::default_specials();
  std::string freq_text = text;
  for (const std::string& s : specials) {
    size_t p = 0;
    while ((p = freq_text.find(s, p)) != std::string::npos) {
      freq_text.replace(p, s.size(), " ");
      p += 1;
    }
  }
  std::unordered_map<std::string, int> freq;
  for (const std::string& w : BPETokenizer::pretokenize(freq_text)) freq[w]++;

  std::vector<std::pair<std::string, int>> wf(freq.begin(), freq.end());
  const int max_words = std::atoi(argval(argc, argv, "--max-words", "30000"));
  if (max_words > 0 && static_cast<int>(wf.size()) > max_words) {
    std::nth_element(wf.begin(), wf.begin() + max_words, wf.end(),
                     [](const std::pair<std::string, int>& a,
                        const std::pair<std::string, int>& b) {
                       return a.second > b.second;
                     });
    wf.resize(max_words);
  }

  std::vector<std::vector<int>> words;
  words.reserve(wf.size());
  std::vector<int> weight;
  weight.reserve(wf.size());
  for (const auto& kv : wf) {
    std::vector<int> sym;
    sym.reserve(kv.first.size());
    for (unsigned char c : kv.first) sym.push_back(static_cast<int>(c));
    words.push_back(std::move(sym));
    weight.push_back(kv.second);
  }

  std::fprintf(stderr, "corpus %.2f MB, %zu unique words (using %zu), target vocab %d\n",
               text.size() / 1048576.0, freq.size(), words.size(), target_vocab);

  // 2) learn merges (leaving room for the reserved special tokens)
  std::vector<std::pair<int, int>> merges;
  int max_merges = target_vocab - BPETokenizer::kBase -
                   static_cast<int>(specials.size());
  if (max_merges < 0) max_merges = 0;
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
      if (kv.second > best || (kv.second == best && kv.first < best_key)) {
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

  tok.set_merges(merges);
  tok.set_specials(BPETokenizer::default_specials());
  }  // end of training branch (skipped when --load is used)
  if (!tok.save(out)) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  std::fprintf(stderr, "wrote %s (vocab %d, %d merges)\n", out.c_str(), tok.vocab_size(),
               tok.n_merges());

  // 3) encode the corpus to uint16 ids (parallel, streaming)
  if (!ids_out.empty()) {
    int nthreads = std::atoi(argval(argc, argv, "--threads", "0"));
    if (nthreads <= 0)
      nthreads = static_cast<int>(std::thread::hardware_concurrency());
    if (nthreads <= 0) nthreads = 1;
    set_threads(nthreads);
    write_ids_u16(tok, text, ids_out, nthreads);
    const bool ok_rt = verify_roundtrip(tok, text, ids_out);
    std::fprintf(stderr, "roundtrip: %s\n", ok_rt ? "identical" : "MISMATCH");
    if (!ok_rt) return 1;
  }
  return 0;
}
