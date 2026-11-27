#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "slmpt/rng.hpp"
#include "slmpt/tokenizer.hpp"

namespace slmpt {

// Holds the whole corpus as token ids and samples fixed-length windows.
//
// The corpus is stored either memory-mapped (the fast path produced by
// `slmpt_bpe`: a 16-byte header + uint16 ids, halving both RAM and disk versus
// int32) or, for a legacy int32 file / the raw-text fallback, as an owned
// vector.  The mapped form is zero-copy: the OS pages the file in on demand, so
// corpus size is no longer bounded by RAM and the 2x int32 load peak is gone.
class Dataset {
 public:
  Dataset() = default;
  ~Dataset();
  Dataset(const Dataset&) = delete;
  Dataset& operator=(const Dataset&) = delete;

  // Loads ids from a file, auto-detecting the SLMPTIDS v1 (uint16) format and
  // falling back to a legacy headerless int32 file.
  bool load_ids(const std::string& path);
  bool load_text(const std::string& path, const BPETokenizer& tok);

  size_t size() const { return n_; }
  int at(size_t i) const {
    return data16_ ? static_cast<int>(data16_[i]) : owned32_[i];
  }
  // uint16 view into the mapping (null when the owned/legacy path was used).
  const uint16_t* data16() const { return data16_; }
  // Materializes the ids as int (for consumers that still want a vector).
  std::vector<int> to_int_vector() const;

  int max_id() const;

  // Fills toks/tgts with B*T ids: toks = window, tgts = window shifted by 1.
  void random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                    std::vector<int>& tgts) const;

 private:
  const uint16_t* data16_ = nullptr;  // points into the mapping (not owned)
  size_t n_ = 0;
  std::vector<int> owned32_;  // legacy int32 / text-fallback storage
  void* map_ = nullptr;
  size_t map_bytes_ = 0;
};

}  // namespace slmpt
