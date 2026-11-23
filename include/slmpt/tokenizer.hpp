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
//
// On top of the merges sit a few *reserved special tokens* (`<|user|>`,
// `<|assistant|>`, `<|end|>`, `<|text|>`).  They are assigned ids right after
// the merges, are stored in the tokenizer file, and `encode()` matches them
// atomically so a role marker is always exactly one token (never split into
// bytes/merges) — the way real chat models delimit turns.
class BPETokenizer {
 public:
  static constexpr int kBase = 256;

  static const std::vector<std::string>& default_specials();

  // Load from a tokenizer file / in-memory blob (same format as save()).
  bool load(const std::string& path);
  bool load_from_string(const std::string& blob);
  bool save(const std::string& path) const;
  std::string to_string() const;

  int vocab_size() const {
    return kBase + static_cast<int>(merges_.size()) + static_cast<int>(specials_.size());
  }
  int n_merges() const { return static_cast<int>(merges_.size()); }
  int n_specials() const { return static_cast<int>(specials_.size()); }
  // Id assigned to special token k (0-based), and the inverse (-1 if not special).
  int special_id(int k) const { return kBase + static_cast<int>(merges_.size()) + k; }
  int special_index(int id) const { return id - (kBase + static_cast<int>(merges_.size())); }

  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string decode(const int* ids, int n) const;

  // Splits text into the "words" the merges operate on (leading space attaches).
  static std::vector<std::string> pretokenize(const std::string& text);

  // Merge table access (used by the trainer to write the file).
  const std::vector<std::pair<int, int>>& merges() const { return merges_; }
  const std::vector<std::string>& specials() const { return specials_; }

  // Build from an explicit merge list.
  void set_merges(std::vector<std::pair<int, int>> merges);
  void set_specials(std::vector<std::string> specials);

 private:
  std::vector<std::pair<int, int>> merges_;
  std::vector<std::string> specials_;
  std::vector<std::string> id_bytes_;
  std::unordered_map<uint64_t, int> rank_;
  void rebuild();
  std::vector<int> encode_plain(const std::string& text) const;
};

}  // namespace slmpt
