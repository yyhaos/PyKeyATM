#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "keyATM_base.h"
#include "LDA_base.h"
#include "keyATM_meta.h"
#include <pybind11/pybind11.h>

namespace keyatm
{

  class LDAweight : public LDAbase, public keyATMbase
  {
  public:
    //
    // Parameters
    //
    int estimate_alpha;
    int store_alpha;

    // Slice Sampling
    double start, end, previous_p, new_p, newlikelihood, slice_;
    std::vector<int> topic_ids;
    Eigen::VectorXd keep_current_param;
    double store_loglik;
    double newalphallk;

    // in alpha_loglik
    Eigen::MatrixXd ndk_a;

    //
    // Functions
    //
    explicit LDAweight(const pybind11::dict &model_)
        : keyATMmeta(model_), LDAbase(model_), keyATMbase(model_) {}

    // Iteration
    void iteration_single(int it) override final;
    double loglik_total() override final;
  };

} // namespace keyatm
