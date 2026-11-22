#pragma once
#include <string>

#include "slmpt/model.hpp"

namespace slmpt {

// Checkpoint format: header (magic + architecture) then every param in order,
// then an optional embedded tokenizer blob (so the model is self-contained).
bool save_checkpoint(const Model& model, const std::string& path,
                     const std::string& tokenizer_blob = "");
bool load_checkpoint(Model& model, const std::string& path,
                     std::string* tokenizer_blob = nullptr);

// Reads just the architecture header (so callers can build a matching Model).
bool peek_config(const std::string& path, Config& out);

}  // namespace slmpt
