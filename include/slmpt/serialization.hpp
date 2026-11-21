#pragma once
#include <string>

#include "slmpt/model.hpp"

namespace slmpt {

// Checkpoint format: header (magic + architecture) then every param in order.
bool save_checkpoint(const Model& model, const std::string& path);
bool load_checkpoint(Model& model, const std::string& path);

// Reads just the architecture header (so callers can build a matching Model).
bool peek_config(const std::string& path, Config& out);

}  // namespace slmpt
