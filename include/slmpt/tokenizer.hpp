#pragma once
#include <string>
#include <vector>

namespace slmpt {

// Byte-level tokenizer: every UTF-8 byte is one token. Guarantees that any
// Portuguese text (with accents) survives encode/decode untouched.
class ByteTokenizer {
 public:
  static constexpr int kVocab = 256;

  int vocab_size() const { return kVocab; }
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string decode(const int* ids, int n) const;
};

}  // namespace slmpt
