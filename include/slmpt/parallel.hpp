#pragma once
// Tiny parallel_for abstraction (std::thread based, no OpenMP dependency).

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace slmpt {

inline int& thread_count() {
  static int n = 1;
  return n;
}

inline void set_threads(int n) {
  if (n > 0) thread_count() = n;
}

inline int get_threads() { return thread_count(); }

// Runs fn(lo, hi) over [0, n) split into thread_count() chunks.
inline void parallel_for(int n, const std::function<void(int, int)>& fn) {
  if (n <= 0) return;
  int nt = thread_count();
  if (nt <= 1 || n < 128) {
    fn(0, n);
    return;
  }
  if (nt > n) nt = n;
  std::vector<std::thread> pool;
  pool.reserve(nt);
  const int chunk = (n + nt - 1) / nt;
  for (int t = 0; t < nt; ++t) {
    const int lo = t * chunk;
    const int hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    pool.emplace_back([&fn, lo, hi]() { fn(lo, hi); });
  }
  for (auto& th : pool) th.join();
}

}  // namespace slmpt
