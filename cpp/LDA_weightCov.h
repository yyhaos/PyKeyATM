#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "LDA_base.h"
#include "keyATM_cov.h"
#include "keyATM_meta.h"
#include <pybind11/pybind11.h>

namespace keyatm
{

  class LDAcov : public LDAbase, public keyATMcov
  {
  public:
    explicit LDAcov(const pybind11::dict &model_)
        : keyATMmeta(model_), LDAbase(model_), keyATMcov(model_) {}

    void iteration_single(int it) override final;
    double loglik_total() override final;
  };

} // namespace keyatm
