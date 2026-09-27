#include "sampler.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
namespace
{

  std::mt19937 &global_rng()
  {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
  }

  inline double rand_open01()
  {
    double u;
    do
    {
      u = std::generate_canonical<double, 53>(global_rng());
    } while (u <= 0.0 || u >= 1.0);
    return u;
  }

  inline int rand_int(int n)
  { // [0, n-1]
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(global_rng());
  }

} // anon

namespace sampler
{

  std::mt19937 &rng()
  {
    return global_rng();
  }

  void seed_rng(std::uint32_t seed)
  {
    global_rng().seed(seed);
  }

  double slice_uniform(double lower, double upper)
  {
    constexpr double eps = 1e-12;
    if (upper <= lower)
      std::swap(lower, upper);
    lower = std::clamp(lower, eps, 1.0 - eps);
    upper = std::clamp(upper, eps, 1.0 - eps);
    const double u = rand_open01();
    const double a = std::nextafter(lower, upper);
    const double b = std::nextafter(upper, lower);
    return a + (b - a) * u;
  }

  std::vector<int> shuffled_indexes(int m)
  {
    std::vector<int> v(m);
    std::iota(v.begin(), v.end(), 0);
    for (int i = 0; i < m - 1; ++i)
    {
      int j = i + rand_int(m - i);
      std::swap(v[i], v[j]);
    }
    return v;
  }

  int rcat(Eigen::VectorXd &prob, int size)
  {
    double u = rand_open01();
    double acc = 0.0;
    for (int i = 0; i < size; ++i)
    {
      acc += prob(i);
      if (u < acc)
        return i;
    }
    return 0;
  }

  int rcat_without_normalize(Eigen::VectorXd &prob, double total, int size)
  {
    double u = rand_open01() * total;
    double acc = 0.0;
    for (int i = 0; i < size; ++i)
    {
      acc += prob(i);
      if (u < acc)
        return i;
    }
    return 0;
  }

  int rcat_eqsize(int size)
  {
    double u = rand_open01();
    double acc = 0.0, p = 1.0 / size;
    for (int i = 0; i < size; ++i)
    {
      acc += p;
      if (u < acc)
        return i;
    }
    return 0;
  }

  int rcat_eqprob(double p, int size)
  {
    double u = rand_open01();
    double acc = 0.0;
    for (int i = 0; i < size; ++i)
    {
      acc += p;
      if (u < acc)
        return i;
    }
    return 0;
  }

} // namespace sampler
