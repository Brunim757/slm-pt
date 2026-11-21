#pragma once
// Deterministic xorshift64* RNG.

#include <cmath>
#include <cstdint>

namespace slmpt {

class Rng {
 public:
  explicit Rng(uint64_t seed = 12345) : s_(seed ? seed : 0x9E3779B97F4A7C15ull) {}

  uint64_t next() {
    uint64_t x = s_;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s_ = x;
    return x;
  }
  uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
  float uniform() { return (next() >> 40) * (1.0f / 16777216.0f); }  // [0,1)
  int below(int n) { return static_cast<int>(u32() % static_cast<uint32_t>(n)); }
  float normal() {
    float u1 = uniform() + 1e-9f, u2 = uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
  }

 private:
  uint64_t s_;
};

}  // namespace slmpt
