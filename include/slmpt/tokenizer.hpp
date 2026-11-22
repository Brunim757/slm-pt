#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace slmpt {

// Byte-level BPE (GPT-2 style, no external deps).
//
// Base alphabet is the 256 raw bytes, so every UTF-8 sequence (Portuguese
// accents included) survives encode/decode untouched. Merges are learned from
// the corpus by `slmpt_bpe`; token id = 256 + merge_rank.
class BPETokenizer {
 public:
  static constexpr int kBase = 256;

  // Load from a tokenizer file / in-memory blob (same format as save()).
  bool load(const std::string& path);
  bool load_from_string(const std::string& blob);
  bool save(const std::string& path) const;
  std::string to_string() const;

  int vocab_size() const { return kBase + static_cast<int>(merges_.size()); }
  int n_merges() const { return static_cast<int>(merges_.size()); }

  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string decode(const int* ids, int n) const;

  // Splits text into the "words" the merges operate on (leading space attaches).
  static std::vector<std::string> pretokenize(const std::string& text);

  // Merge table access (used by the trainer to write the file).
  const std::vector<std::pair<int, int>>& merges() const { return merges_; }

  // Build from an explicit merge list.
  void set_merges(std::vector<std::pair<int, int>> merges);

 private:
  std::vector<std::pair<int, int>> merges_;
  std::vector<std::string> id_bytes_;
  std::unordered_map<uint64_t, int> rank_;
  void rebuild();
};

}  // namespace slmpt
