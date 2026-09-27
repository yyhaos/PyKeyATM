#pragma once
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "keyATM_meta.h"

namespace keyatm
{

class keyATMhmm : virtual public keyATMmeta
{
public:
  // -----------------
  // Data
  // -----------------
  Eigen::VectorXi time_index;
  int num_time = 0; // number of time segments
  Eigen::VectorXi time_doc_start;
  Eigen::VectorXi time_doc_end;

  // -----------------
  // Parameters
  // -----------------
  // Time-blocked change point HMM (following Chib, 1998)
  int num_states = 0;
  int index_states = 0; // typically num_states - 1
  int store_transition_matrix = 0;

  // Prk: p(state=k | y_{1:t}) forward probs per time block
  Eigen::MatrixXd Prk;     // (num_time, num_states)
  Eigen::VectorXi R_est;   // sampled state index per time block (num_time)
  Eigen::VectorXi R_count; // counts per state (for sampling P)

  // Transition and state/topic params
  Eigen::MatrixXd P_est;  // (num_states, num_states)
  Eigen::MatrixXd alphas; // (num_states, num_topics)

  // Constructor
  explicit keyATMhmm(const pybind11::dict &model_)
      : keyATMmeta(model_) {}

  Eigen::VectorXd logfy; // (num_states)
  Eigen::VectorXd rt_1l;
  Eigen::VectorXd rt_k;
  Eigen::VectorXd logrt_k;

  Eigen::VectorXd state_prob_vec;

  // Sample alpha over states
  Eigen::VectorXi states_start;
  Eigen::VectorXi states_end;

  // Slice sampling helpers
  std::vector<int> topic_ids;
  Eigen::VectorXd keep_current_param;
  Eigen::MatrixXd ndk_a;

  // -----------------
  // Utilities
  // -----------------
  int get_state_index(int doc_id);

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
  void sample_alpha();
  void sample_alpha_state(int state, int state_start, int state_end);
  double alpha_loglik(int k, int state_start, int state_end);

  // HMM steps
  void sample_forward();  // compute Prk
  void sample_backward(); // sample R_est
  void sample_P();        // sample P_est

  // Storage
  void store_R_est();
  void store_P_est();
  void keep_P_est();

  // Likelihoods & logging
  double polyapdfln(int t, Eigen::VectorXd &alpha);
  double loglik_total() override;
  void verbose_special(int r_index) override;
};

} // namespace keyatm
