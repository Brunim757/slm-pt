#include "slmpt/quant.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace slmpt {

namespace {
const char kMagic[8] = {'S', 'L', 'M', 'Q', '0', '0', '0', '1'};

inline int qmax_for(int bits) { return (1 << (bits - 1)) - 1; }

// Packs signed quantized values into the byte stream.
void pack_write(FILE* f, const std::vector<int>& q, int bits) {
  if (bits == 8) {
    std::vector<int8_t> b(q.size());
    for (size_t i = 0; i < q.size(); ++i) b[i] = static_cast<int8_t>(q[i]);
    std::fwrite(b.data(), 1, b.size(), f);
  } else if (bits == 4) {
    const size_t n = q.size();
    std::vector<uint8_t> b((n + 1) / 2, 0);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t v = static_cast<uint8_t>(q[i]) & 0xF;
      if (i % 2 == 0) b[i / 2] |= v;
      else b[i / 2] |= static_cast<uint8_t>(v << 4);
    }
    std::fwrite(b.data(), 1, b.size(), f);
  } else {  // 2 bits
    const size_t n = q.size();
    std::vector<uint8_t> b((n + 3) / 4, 0);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t v = static_cast<uint8_t>(q[i]) & 0x3;
      b[i / 4] |= static_cast<uint8_t>(v << (2 * (i % 4)));
    }
    std::fwrite(b.data(), 1, b.size(), f);
  }
}

void pack_read(FILE* f, std::vector<int>& q, size_t n, int bits) {
  q.resize(n);
  if (bits == 8) {
    std::vector<int8_t> b(n);
    (void)std::fread(b.data(), 1, n, f);
    for (size_t i = 0; i < n; ++i) q[i] = b[i];
  } else if (bits == 4) {
    std::vector<uint8_t> b((n + 1) / 2);
    (void)std::fread(b.data(), 1, b.size(), f);
    for (size_t i = 0; i < n; ++i) {
      uint8_t v = (i % 2 == 0) ? (b[i / 2] & 0xF) : (b[i / 2] >> 4);
      q[i] = (v >= 8) ? v - 16 : v;
    }
  } else {
    std::vector<uint8_t> b((n + 3) / 4);
    (void)std::fread(b.data(), 1, b.size(), f);
    for (size_t i = 0; i < n; ++i) {
      uint8_t v = (b[i / 4] >> (2 * (i % 4))) & 0x3;
      q[i] = (v >= 2) ? static_cast<int>(v) - 4 : v;
    }
  }
}
}  // namespace

bool is_quantized(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  const bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMagic, 8) == 0;
  std::fclose(f);
  return ok;
}

bool save_quantized(const Model& model, const std::string& path, int bits, int group,
                    const std::string& tokenizer_blob, QuantStats* stats) {
  if (bits != 8 && bits != 4 && bits != 2) return false;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const Config& c = model.config();
  const int32_t hdr[6] = {c.vocab, c.n_embd, c.n_layer, c.n_head, c.block, c.batch};
  const int32_t meta[2] = {bits, group};
  std::fwrite(kMagic, 1, 8, f);
  std::fwrite(hdr, sizeof(int32_t), 6, f);
  std::fwrite(meta, sizeof(int32_t), 2, f);
  const int64_t np = static_cast<int64_t>(model.params().size());
  std::fwrite(&np, sizeof(int64_t), 1, f);

  size_t orig = 8 + 24 + 8 + 8;  // magic + hdr + meta + np
  QuantStats st;
  st.bits = bits;
  st.group = group;
  st.total_tensors = static_cast<int>(model.params().size());
  const int qmax = qmax_for(bits);

  for (Param* p : model.params()) {
    const int64_t sz = static_cast<int64_t>(p->data.size());
    std::fwrite(&sz, sizeof(int64_t), 1, f);
    orig += 8 + static_cast<size_t>(sz) * 4;
    const bool doq = (bits < 8 || true) && static_cast<size_t>(sz) >=
                                             static_cast<size_t>(group) * 4;
    const int32_t flags = doq ? 1 : 0;
    std::fwrite(&flags, sizeof(int32_t), 1, f);
    orig += 4;
    if (!doq) {
      std::fwrite(p->data.data(), sizeof(float), p->data.size(), f);
      continue;
    }
    st.quantized_tensors++;
    const size_t n = p->data.size();
    const size_t ng = (n + group - 1) / group;
    std::vector<float> scales(ng);
    std::vector<int> q(n);
    for (size_t g = 0; g < ng; ++g) {
      const size_t lo = g * group;
      const size_t hi = std::min(n, lo + group);
      float mx = 0.f;
      for (size_t i = lo; i < hi; ++i) mx = std::max(mx, std::fabs(p->data[i]));
      const float scale = mx > 0.f ? mx / qmax : 1.f;
      scales[g] = scale;
      for (size_t i = lo; i < hi; ++i) {
        int v = static_cast<int>(std::lround(p->data[i] / scale));
        if (v > qmax) v = qmax;
        if (v < -qmax) v = -qmax;
        q[i] = v;
      }
    }
    const int32_t ng32 = static_cast<int32_t>(ng);
    std::fwrite(&ng32, sizeof(int32_t), 1, f);
    std::fwrite(scales.data(), sizeof(float), scales.size(), f);
    pack_write(f, q, bits);
  }

  const int64_t tl = static_cast<int64_t>(tokenizer_blob.size());
  std::fwrite(&tl, sizeof(int64_t), 1, f);
  if (tl > 0) std::fwrite(tokenizer_blob.data(), 1, tokenizer_blob.size(), f);
  const long end = std::ftell(f);
  std::fclose(f);

  if (stats) {
    st.orig_bytes = orig;
    st.quant_bytes = static_cast<size_t>(end);
    *stats = st;
  }
  return true;
}

bool load_quantized(const std::string& path, Model& model, std::string* tokenizer_blob) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kMagic, 8) != 0) {
    std::fclose(f);
    return false;
  }
  int32_t hdr[6];
  (void)std::fread(hdr, sizeof(int32_t), 6, f);
  int32_t meta[2];
  (void)std::fread(meta, sizeof(int32_t), 2, f);
  const int bits = meta[0], group = meta[1];
  const int qmax = qmax_for(bits);
  int64_t np = 0;
  (void)std::fread(&np, sizeof(int64_t), 1, f);
  if (np != static_cast<int64_t>(model.params().size())) {
    std::fclose(f);
    return false;
  }
  std::vector<int> q;
  for (Param* p : model.params()) {
    int64_t sz = 0;
    (void)std::fread(&sz, sizeof(int64_t), 1, f);
    int32_t flags = 0;
    (void)std::fread(&flags, sizeof(int32_t), 1, f);
    if (sz != static_cast<int64_t>(p->data.size())) {
      std::fclose(f);
      return false;
    }
    if (!flags) {
      (void)std::fread(p->data.data(), sizeof(float), p->data.size(), f);
      continue;
    }
    int32_t ng = 0;
    (void)std::fread(&ng, sizeof(int32_t), 1, f);
    std::vector<float> scales(ng);
    (void)std::fread(scales.data(), sizeof(float), scales.size(), f);
    pack_read(f, q, p->data.size(), bits);
    for (size_t i = 0; i < p->data.size(); ++i) {
      const size_t g = i / group;
      p->data[i] = static_cast<float>(q[i]) * scales[g < scales.size() ? g : scales.size() - 1];
    }
  }
  (void)qmax;
  if (tokenizer_blob) {
    int64_t tl = 0;
    if (std::fread(&tl, sizeof(int64_t), 1, f) == 1 && tl > 0) {
      tokenizer_blob->resize(static_cast<size_t>(tl));
      if (std::fread(&(*tokenizer_blob)[0], 1, static_cast<size_t>(tl), f) !=
          static_cast<size_t>(tl))
        tokenizer_blob->clear();
    }
  }
  std::fclose(f);
  return true;
}

}  // namespace slmpt
