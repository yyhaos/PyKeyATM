#ifndef __utils__INCLUDED__
#define __utils__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <sstream>

using namespace Eigen;
using namespace std;

namespace utils
{
  void calc_PGtheta(const Eigen::MatrixXd &theta_tilda, Eigen::MatrixXd &theta, int num_doc, int num_topics);

  template <typename T>
  std::string to_string_prec(const T val, const int n = 2)
  {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << val;
    return out.str();
  }
}

#endif

