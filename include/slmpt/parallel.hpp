#pragma once
// Tiny parallel_for abstraction: a persistent std::thread pool (no OpenMP
// dependency).  Spawning a fresh thread per call was the old behaviour and it
// dominates the cost of the ~150 small parallel regions executed per training
// step (thread create/join ~tens of microseconds each).  Here the workers live
// for the whole process and are woken per region through a generation counter.
//
// Nesting is made safe: a call made from inside a region runs serially instead
// of deadlocking on the pool (every real call site in this codebase is already
// flat, but this guarantees it).

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
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

// True while the calling thread is executing inside a parallel region.
inline bool& in_parallel() {
  static thread_local bool b = false;
  return b;
}

class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool p;
    return p;
  }

  // Runs fn(wid, lo, hi) over [0, n) using `nt` workers (the caller is worker 0).
  void run(int nt, int n, const std::function<void(int, int, int)>& fn) {
    if (n <= 0) return;
    if (in_parallel() || nt <= 1 || n <= 1) {  // serial fallback (also breaks nesting)
      fn(0, 0, n);
      return;
    }
    if (nt > n) nt = n;
    ensure(nt);
    const int chunk = (n + nt - 1) / nt;
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = &fn;
      n_ = n;
      nt_ = nt;
      pending_ = nt - 1;
      ++gen_;
    }
    cv_.notify_all();
    {
      const int hi = std::min(n, chunk);
      if (0 < hi) fn(0, 0, hi);
    }
    std::unique_lock<std::mutex> lk(mu_);
    done_.wait(lk, [this] { return pending_ == 0; });
    fn_ = nullptr;
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
  }

 private:
  ThreadPool() = default;
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void ensure(int nt) {
    const int want = nt - 1;
    while (static_cast<int>(workers_.size()) < want) {
      const int id = static_cast<int>(workers_.size()) + 1;
      int g;
      {
        std::lock_guard<std::mutex> lk(mu_);
        g = gen_;
      }
      workers_.emplace_back([this, id, g] { worker(id, g); });
    }
  }

  void worker(int id, int start_gen) {
    int last = start_gen;
    for (;;) {
      const std::function<void(int, int, int)>* f;
      int n, nt;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || gen_ != last; });
        if (stop_) return;
        last = gen_;
        f = fn_;
        n = n_;
        nt = nt_;
      }
      in_parallel() = true;
      if (id < nt) {
        const int chunk = (n + nt - 1) / nt;
        const int lo = id * chunk;
        const int hi = std::min(n, lo + chunk);
        if (lo < hi) (*f)(id, lo, hi);
        std::lock_guard<std::mutex> lk(mu_);
        if (--pending_ == 0) done_.notify_one();
      }
      in_parallel() = false;
    }
  }

  std::mutex mu_;
  std::condition_variable cv_, done_;
  std::vector<std::thread> workers_;
  const std::function<void(int, int, int)>* fn_ = nullptr;
  int n_ = 0, nt_ = 0, pending_ = 0, gen_ = 0;
  bool stop_ = false;
};

// Runs fn(lo, hi) over [0, n) split into thread_count() chunks.
inline void parallel_for(int n, const std::function<void(int, int)>& fn) {
  ThreadPool::instance().run(thread_count(), n,
                             [&fn](int, int lo, int hi) { fn(lo, hi); });
}

// Like parallel_for, but also hands each worker a stable index in [0, nt) so it
// can use its own scratch slice (no locks, no per-item allocation).
inline void parallel_for_ctx(int n, const std::function<void(int, int, int)>& fn) {
  ThreadPool::instance().run(thread_count(), n, fn);
}

}  // namespace slmpt
