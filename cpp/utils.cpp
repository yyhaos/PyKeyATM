#include "utils.h"

using namespace Eigen;
using namespace std;
namespace utils
{

  void calc_PGtheta(const Eigen::Ref<const Eigen::MatrixXd> &theta_tilde_in,
                    Eigen::Ref<Eigen::MatrixXd> theta,
                    int num_doc, int num_topics)
  {
    assert(theta.rows() == num_doc && theta.cols() == num_topics);
    assert(theta_tilde_in.rows() == num_doc);
    assert(theta_tilde_in.cols() == num_topics || theta_tilde_in.cols() == num_topics - 1);

    const int K = num_topics;
    const int Ktilde = static_cast<int>(theta_tilde_in.cols());

    constexpr double eps = 1e-12;
    Eigen::MatrixXd theta_tilde = theta_tilde_in.array()
                                      .min(1.0 - eps)
                                      .max(eps)
                                      .matrix();

    theta.setZero();

    for (int d = 0; d < num_doc; ++d)
    {
      double remaining = 1.0;

      {
        double p0 = (Ktilde >= 1 ? theta_tilde(d, 0) : 0.0);
        theta(d, 0) = p0;
        remaining *= (1.0 - p0);
      }

      for (int k = 1; k < K - 1; ++k)
      {
        double pk = (k < Ktilde ? theta_tilde(d, k) : 0.0);
        double val = remaining * pk;
        theta(d, k) = val;
        remaining *= (1.0 - pk);
      }

      double last = std::max(0.0, std::min(1.0, remaining));
      theta(d, K - 1) = last;

      double rowsum = theta.row(d).sum();
      if (!(rowsum > 0.0) || !std::isfinite(rowsum))
      {
        theta.row(d).setConstant(1.0 / K);
      }
      else if (std::abs(rowsum - 1.0) > 1e-12)
      {
        theta.row(d) /= rowsum;
      }
    }
  }

} // namespace utils
