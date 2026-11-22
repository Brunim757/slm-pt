#pragma once
#include <string>
#include <vector>

#include "slmpt/model.hpp"
#include "slmpt/tokenizer.hpp"

namespace slmpt {

// A tiny local retrieval index: text chunks + their mean hidden-state embedding
// from the model. Lets the chat ground answers in your own files.
struct RagIndex {
  int dim = 0;
  std::vector<std::string> texts;
  std::vector<std::vector<float>> emb;  // one L2-normalized vector per chunk
};

// Mean-pooled final hidden state of `text` (capped to the model context).
void embed_text(Model& model, const BPETokenizer& tok, const std::string& text,
                std::vector<float>& out);

// Split a text file into chunks (paragraphs) and embed each one.
bool build_index(Model& model, const BPETokenizer& tok, const std::string& text_file,
                 RagIndex& index);

bool save_index(const RagIndex& index, const std::string& path);
bool load_index(const std::string& path, RagIndex& index);

// Top-k cosine similarity search (index embeddings are normalized).
void retrieve(const RagIndex& index, const std::vector<float>& query, int topk,
              std::vector<int>& out_idx, std::vector<float>& out_score);

}  // namespace slmpt
