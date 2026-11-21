#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "slmpt/rng.hpp"

namespace slmpt {

// Holds the whole corpus as raw bytes and samples fixed-length windows.
class Dataset {
 public:
  bool load(const std::string& path);

  size_t size() const { return data_.size(); }

  // Fills toks/tgts with B*T token ids: toks = window, tgts = window shifted by 1.
  void random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                    std::vector<int>& tgts) const;

 private:
  std::vector<uint8_t> data_;
};

}  // namespace slmpt
