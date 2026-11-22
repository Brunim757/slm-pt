#include "slmpt/serialization.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace slmpt {

namespace {
// v2 magic: RMSNorm + RoPE + SwiGLU architecture, embedded tokenizer.
const char kMagic[8] = {'S', 'L', 'M', 'P', 'T', '0', '2', '\0'};
}  // namespace

bool save_checkpoint(const Model& model, const std::string& path,
                     const std::string& tokenizer_blob) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const Config& c = model.config();
  const int32_t hdr[6] = {c.vocab, c.n_embd, c.n_layer, c.n_head, c.block, c.batch};
  std::fwrite(kMagic, 1, 8, f);
  std::fwrite(hdr, sizeof(int32_t), 6, f);
  const int64_t np = static_cast<int64_t>(model.params().size());
  std::fwrite(&np, sizeof(int64_t), 1, f);
  for (Param* p : model.params()) {
    const int64_t sz = static_cast<int64_t>(p->data.size());
    std::fwrite(&sz, sizeof(int64_t), 1, f);
    std::fwrite(p->data.data(), sizeof(float), p->data.size(), f);
  }
  const int64_t tl = static_cast<int64_t>(tokenizer_blob.size());
  std::fwrite(&tl, sizeof(int64_t), 1, f);
  if (tl > 0) std::fwrite(tokenizer_blob.data(), 1, tokenizer_blob.size(), f);
  std::fclose(f);
  return true;
}

bool peek_config(const std::string& path, Config& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 ||
      (std::memcmp(magic, kMagic, 8) != 0 && std::memcmp(magic, "SLMQ0001", 8) != 0)) {
    std::fclose(f);
    return false;
  }
  int32_t hdr[6];
  if (std::fread(hdr, sizeof(int32_t), 6, f) != 6) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  out.vocab = hdr[0];
  out.n_embd = hdr[1];
  out.n_layer = hdr[2];
  out.n_head = hdr[3];
  out.block = hdr[4];
  out.batch = hdr[5];
  return true;
}

bool load_checkpoint(Model& model, const std::string& path, std::string* tokenizer_blob) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kMagic, 8) != 0) {
    std::fclose(f);
    return false;
  }
  int32_t hdr[6];
  if (std::fread(hdr, sizeof(int32_t), 6, f) != 6) {
    std::fclose(f);
    return false;
  }
  int64_t np = 0;
  if (std::fread(&np, sizeof(int64_t), 1, f) != 1 ||
      np != static_cast<int64_t>(model.params().size())) {
    std::fclose(f);
    return false;
  }
  for (Param* p : model.params()) {
    int64_t sz = 0;
    if (std::fread(&sz, sizeof(int64_t), 1, f) != 1 ||
        sz != static_cast<int64_t>(p->data.size())) {
      std::fclose(f);
      return false;
    }
    if (std::fread(p->data.data(), sizeof(float), p->data.size(), f) != p->data.size()) {
      std::fclose(f);
      return false;
    }
  }
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
