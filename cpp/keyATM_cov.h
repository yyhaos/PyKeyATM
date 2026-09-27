#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include "sampler.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "keyATM_meta.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <random>
#include <stdexcept>
#include <cmath>

namespace keyatm
{

  class keyATMcov : virtual public keyATMmeta
  {
  public:
    // -----------------
    // Parameters
    // -----------------
    Eigen::MatrixXd Alpha;
    int num_cov = 0;
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd C;

    int mh_use = 0;
    double mu = 0.0;
    double sigma = 0.0;

    std::vector<int> topic_ids;
    std::vector<int> cov_ids;

    double val_min = 0.0;
    double val_max = 0.0;

    // Constructor
    explicit keyATMcov(const pybind11::dict &model_)
        : keyATMmeta(model_) {}

    // -----------------
    // Lifecycle hooks
    // -----------------
    void read_data_specific() override final;
    void initialize_specific() override final;
    void resume_initialize_specific() override final;

    // -----------------
    // Iteration
    // -----------------
    void iteration_single(int it) override;
    void sample_parameters(int it) override final;

    // -----------------
    // Parameter sampling
    // -----------------
    void sample_lambda();
    void sample_lambda_mh();
    void sample_lambda_slice();
    double alpha_loglik();
    double loglik_total() override;

    double likelihood_lambda(int k, int t);
    void proposal_lambda(int k);
  };

} // namespace keyatm
