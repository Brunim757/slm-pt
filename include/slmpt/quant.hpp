#pragma once
#include <cstddef>
#include <string>

#include "slmpt/model.hpp"

namespace slmpt {

struct QuantStats {
  size_t orig_bytes = 0;
  size_t quant_bytes = 0;
  int bits = 0;
  int group = 0;
  int quantized_tensors = 0;
  int total_tensors = 0;
  double ratio() const {
    return quant_bytes ? static_cast<double>(orig_bytes) / quant_bytes : 0.0;
  }
};

// Group-wise symmetric quantization (2/4/8 bits) of the model's weight tensors;
// small 1D tensors (RMSNorm gains) stay fp32. The tokenizer blob is embedded so
// the file is self-contained.
bool save_quantized(const Model& model, const std::string& path, int bits, int group,
                    const std::string& tokenizer_blob, QuantStats* stats);

// Load a quantized checkpoint by dequantizing back into float params.
bool load_quantized(const std::string& path, Model& model, std::string* tokenizer_blob);

bool is_quantized(const std::string& path);

}  // namespace slmpt
