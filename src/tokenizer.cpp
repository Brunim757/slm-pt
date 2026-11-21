#include "slmpt/tokenizer.hpp"

namespace slmpt {

std::vector<int> ByteTokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  ids.reserve(text.size());
  for (unsigned char c : text) ids.push_back(static_cast<int>(c));
  return ids;
}

std::string ByteTokenizer::decode(const std::vector<int>& ids) const {
  return decode(ids.data(), static_cast<int>(ids.size()));
}

std::string ByteTokenizer::decode(const int* ids, int n) const {
  std::string s;
  s.reserve(n);
  for (int i = 0; i < n; ++i)
    s.push_back(static_cast<char>(static_cast<unsigned char>(ids[i] & 0xFF)));
  return s;
}

}  // namespace slmpt
