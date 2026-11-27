#include "slmpt/dataset.hpp"

#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "slmpt/ids_format.hpp"

namespace slmpt {

Dataset::~Dataset() {
  if (map_ && map_ != MAP_FAILED) munmap(map_, map_bytes_);
}

bool Dataset::load_ids(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st;
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return false;
  }
  const size_t bytes = static_cast<size_t>(st.st_size);

  unsigned char head[idsfmt::kHeaderBytes];
  const bool have_head =
      bytes >= sizeof(head) &&
      ::pread(fd, head, sizeof(head), 0) == static_cast<ssize_t>(sizeof(head));

  bool is16 = false;
  size_t elems = 0;
  if (have_head && std::memcmp(head, idsfmt::kMagic, 8) == 0) {
    uint32_t dtype = 0;
    std::memcpy(&dtype, head + 8, 4);
    if (dtype == idsfmt::kDtypeU16) {
      is16 = true;
      elems = (bytes - idsfmt::kHeaderBytes) / sizeof(uint16_t);
    }
  }

  if (is16) {
    const size_t map_len = idsfmt::kHeaderBytes + elems * sizeof(uint16_t);
    void* m = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return false;
    map_ = m;
    map_bytes_ = map_len;
    data16_ = reinterpret_cast<const uint16_t*>(
        static_cast<const unsigned char*>(m) + idsfmt::kHeaderBytes);
    n_ = elems;
    return true;
  }

  // Legacy headerless int32 (little-endian) — read into owned storage.
  if (bytes % sizeof(int32_t) != 0) {
    ::close(fd);
    return false;
  }
  const size_t n = bytes / sizeof(int32_t);
  owned32_.resize(n);
  size_t off = 0;
  while (off < bytes) {
    const ssize_t r =
        ::pread(fd, reinterpret_cast<char*>(owned32_.data()) + off, bytes - off, off);
    if (r <= 0) {
      ::close(fd);
      owned32_.clear();
      return false;
    }
    off += static_cast<size_t>(r);
  }
  ::close(fd);
  n_ = n;
  return true;
}

bool Dataset::load_text(const std::string& path, const BPETokenizer& tok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::string text;
  char b[1 << 16];
  size_t n;
  while ((n = std::fread(b, 1, sizeof(b), f)) > 0) text.append(b, n);
  std::fclose(f);
  if (text.empty()) return false;
  owned32_ = tok.encode(text);
  n_ = owned32_.size();
  return !owned32_.empty();
}

std::vector<int> Dataset::to_int_vector() const {
  if (data16_) return std::vector<int>(data16_, data16_ + n_);
  return owned32_;
}

int Dataset::max_id() const {
  int m = 0;
  for (size_t i = 0; i < n_; ++i) {
    const int t = at(i);
    if (t > m) m = t;
  }
  return m;
}

void Dataset::random_batch(Rng& rng, int B, int T, std::vector<int>& toks,
                           std::vector<int>& tgts) const {
  toks.resize(static_cast<size_t>(B) * T);
  tgts.resize(static_cast<size_t>(B) * T);
  const int N = static_cast<int>(n_);
  const int hi = N - T - 1;
  const int span = hi > 0 ? hi : 1;
  for (int b = 0; b < B; ++b) {
    const int off = rng.below(span);
    for (int i = 0; i < T; ++i) {
      toks[static_cast<size_t>(b) * T + i] = at(static_cast<size_t>(off + i));
      tgts[static_cast<size_t>(b) * T + i] = at(static_cast<size_t>(off + i + 1));
    }
  }
}

}  // namespace slmpt
