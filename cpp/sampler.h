#ifndef __SAMPLER_INCLUDED__
#define __SAMPLER_INCLUDED__

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include <Eigen/Core>
#include <random>
#include <cstdint>
#include <vector>

// trace_logger.hpp (single-header)
#pragma once
#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <iomanip>
#include <thread>
#include <unordered_map>

#ifndef TRACE_LIMIT_PER_FUNC
#define TRACE_LIMIT_PER_FUNC 20 // stop logging after 20 ENTERs per function
#endif

// ---------------- Logger (thread-safe, basic) ----------------
class Logger
{
public:
  static Logger &instance()
  {
    static Logger inst("calls.txt"); // change path if needed
    return inst;
  }
  void log(const std::string &msg)
  {
    std::lock_guard<std::mutex> lk(m_);
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    ofs_ << std::put_time(&tm, "%F %T")
         << " [tid=" << std::this_thread::get_id() << "] "
         << msg << '\n';
    ofs_.flush();
  }

private:
  explicit Logger(const std::string &file) : ofs_(file, std::ios::app) {}
  std::mutex m_;
  std::ofstream ofs_;
};

// ---------------- Per-function counters ----------------
class TraceCounters
{
public:
  static TraceCounters &instance()
  {
    static TraceCounters inst;
    return inst;
  }

  // Increase and return new count for this function key
  int inc(const std::string &key)
  {
    std::lock_guard<std::mutex> lk(m_);
    int &c = counters_[key];
    ++c;
    return c;
  }

  // Read current count (optional helper)
  int get(const std::string &key)
  {
    std::lock_guard<std::mutex> lk(m_);
    auto it = counters_.find(key);
    return (it == counters_.end()) ? 0 : it->second;
  }

private:
  std::mutex m_;
  std::unordered_map<std::string, int> counters_;
};

// Pick a stable, fully-qualified function identity for C++
#if defined(_MSC_VER)
#define FUNC_ID __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
#define FUNC_ID __PRETTY_FUNCTION__
#else
#define FUNC_ID __func__ // fallback (may collide for overloads)
#endif

// ---------------- RAII tracer with cap & counter ----------------
struct FuncTrace
{
  std::string key;     // function identity
  int enter_count = 0; // this call's count
  bool logged_enter = false;

  explicit FuncTrace(const char *funcsig, const char *file, int line)
      : key(funcsig)
  {
    // increase per-function ENTER count
    enter_count = TraceCounters::instance().inc(key);

    // only log while count <= LIMIT
    if (enter_count <= TRACE_LIMIT_PER_FUNC)
    {
      Logger::instance().log(
          std::string("ENTER #") + std::to_string(enter_count) +
          " " + key + " @ " + file + ":" + std::to_string(line));
      logged_enter = true;
    }
    // else: silently suppress to avoid log spam
    // (If you want a one-time "suppressed" notice, add: if (enter_count == TRACE_LIMIT_PER_FUNC + 1) ...)
  }

  ~FuncTrace()
  {
    // Only print LEAVE if we printed ENTER for this invocation
    if (logged_enter)
    {
      Logger::instance().log(
          std::string("LEAVE #") + std::to_string(enter_count) +
          " " + key);
    }
  }
};

// Put this at the top of any function you want to trace
#ifdef KEYATM_ENABLE_TRACE
#define TRACE_FUNC() FuncTrace __trace(FUNC_ID, __FILE__, __LINE__)
#else
#define TRACE_FUNC() ((void)0)
#endif

namespace sampler {

  void seed_rng(std::uint32_t seed);
  std::mt19937 &rng();

  double slice_uniform(double lower, double upper);

  std::vector<int> shuffled_indexes(int m);

  int rcat(Eigen::VectorXd& prob, int size);
  int rcat_without_normalize(Eigen::VectorXd& prob, double total, int size);

  int rcat_eqsize(int size);
  int rcat_eqprob(double prob, int size);

}

#endif
