#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "slmpt/rng.hpp"
#include "slmpt/tokenizer.hpp"

namespace slmpt {

// Holds the whole corpus as token ids and samples fixed-length windows.
class Dataset {
 public:
  // Load pre-tokenized int32 little-endian ids (fast path, produced by slmpt_bpe).
  bool load_ids(const std::string& path);
  // Load raw text and encode it with the tokenizer.
  bool load_text(const std::string& path, const BPETokenizer& tok);

  size_t size() const { return tokens_.size(); }
  const std::vector<int>& tokens() const { return tokens_; }
  int max_id() const;

  // Fills toks/tgts with B*T ids: toks = window, tgts = window shifted by 1.
  void random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                    std::vector<int>& tgts) const;

 private:
  std::vector<int> tokens_;
};

}  // namespace slmpt
