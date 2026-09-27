#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "keyATM_meta.h"

namespace keyatm {

class keyATMbase : virtual public keyATMmeta {
 public:
  // Parameters
  int estimate_alpha = 0;
  int store_alpha = 0;

  std::vector<int> topic_ids;
  Eigen::VectorXd keep_current_param;
  double store_loglik = 0.0;
  double newalphallk = 0.0;

  // For alpha_loglik
  Eigen::MatrixXd ndk_a;

  // Constructor
  explicit keyATMbase(const pybind11::dict& model_)
      : keyATMmeta(model_) {}

  // Initialization
  void initialize_specific() override final;

  // Resume
  void resume_initialize_specific() override final;

  // Iteration
  void iteration_single(int it) override;
  void sample_parameters(int it) override final;

  void sample_alpha();
  double alpha_loglik(int k);
  double loglik_total() override;
};

}  // namespace keyatm
