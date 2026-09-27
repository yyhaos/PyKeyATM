#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "keyATM_meta.h"

namespace keyatm {

class LDAbase : virtual public keyATMmeta {
 public:
  // Variables
  Eigen::MatrixXd n_kv;
  Eigen::VectorXd n_k;
  Eigen::VectorXd n_k_noWeight;

  // Constructor
  explicit LDAbase(const pybind11::dict& model_)
      : keyATMmeta(model_) {}

  // Functions
  // In LDA, we do not need to read and initialize X
  void read_data_common() override final;
  void initialize_common() override final;
  void parameters_store(int r_index) override final;
  int sample_z(Eigen::VectorXd& alpha, int z, int s,
               int w, int doc_id) override final;
};

}  // namespace keyatm
