#include "keyATM_HMM.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <random>
#include <stdexcept>
#include <cmath>

namespace keyatm
{

  void keyATMhmm::read_data_specific()
  {
    TRACE_FUNC();

    num_states = py::cast<int>(model_settings["num_states"]);
    index_states = num_states - 1;

    time_index = py::cast<Eigen::VectorXi>(model_settings["time_index"]);
    num_time = time_index.maxCoeff();
    time_index.array() -= 1;

    time_doc_start = Eigen::VectorXi::Zero(num_time);
    time_doc_end = Eigen::VectorXi::Zero(num_time);

    int index_prev = -1;
    int store_index = 0;
    for (int d = 0; d < num_doc; ++d)
    {
      int index = time_index[d];
      if (index != index_prev)
      {
        time_doc_start[store_index] = d;
        index_prev = index;
        store_index += 1;
        if (store_index > num_time)
        {
          break;
        }
      }
    }

    for (int s = 0; s < num_time - 1; ++s)
    {
      time_doc_end(s) = time_doc_start(s + 1) - 1;
    }
    time_doc_end(num_time - 1) = num_doc - 1;

    store_transition_matrix = py::cast<int>(options_list["store_transition_matrix"]);
  }

  void keyATMhmm::initialize_specific()
  {
    // Initialize R_est
    // Use multinomial distribution (with flat probability)
    // to decide the number of each state
    // and push it into R_est.
    Eigen::VectorXi R_est_num = Eigen::VectorXi::Constant(num_states, 1);
    Eigen::VectorXd R_est_temp = Eigen::VectorXd::Zero(num_states);
    double cumulative = 1.0 / num_states;
    double u;
    int index;
    for (int i = 0; i < num_states; ++i)
    {
      R_est_temp(i) = cumulative * (i + 1);
    }

    auto &rng = sampler::rng();
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    for (int j = 0; j < num_time - num_states; ++j)
    {
      // `num_time - num_states` because all states have at least one time
      // `R_est_num` is initialized to 1
      u = unif01(rng);
      for (int i = 0; i < num_states; ++i)
      {
        if (u < R_est_temp(i))
        {
          index = i;
          break;
        }
      }
      R_est_num(index) += 1;
    }

    R_est = Eigen::VectorXi::Zero(num_time);
    R_count = R_est_num;
    int count;
    index = 0;
    for (int i = 0; i < num_states; ++i)
    {
      count = R_est_num(i);
      for (int j = 0; j < count; ++j)
      {
        R_est(index) = i;
        index += 1;
      }
    }

    // Initializae P_est
    P_est = Eigen::MatrixXd::Zero(num_states, num_states);
    double prob;
    for (int i = 0; i <= (index_states - 1); ++i)
    {
      prob = unif01(rng);
      P_est(i, i) = prob;
      P_est(i, i + 1) = 1 - prob;
    }
    P_est(index_states, index_states) = 1;

    // Initialize alphas
    alphas = Eigen::MatrixXd::Constant(num_states, num_topics, 50.0 / num_topics);

    // Initialize variables we use in the sampling
    Prk = Eigen::MatrixXd::Zero(num_time, num_states);
    logfy = Eigen::VectorXd::Zero(num_states);
    rt_k = Eigen::VectorXd::Zero(num_states);
    logrt_k = Eigen::VectorXd::Zero(num_states);
    state_prob_vec = Eigen::VectorXd::Zero(num_states);

    states_start = Eigen::VectorXi::Zero(num_states);
    states_end = Eigen::VectorXi::Zero(num_states);

    rt_1l = Eigen::VectorXd::Zero(num_states);
  }
  void keyATMhmm::resume_initialize_specific()
  {
    // Resume R_est
    py::list R_iter = stored_values["R_iter"].cast<py::list>();
    auto state_R = R_iter[py::len(R_iter) - 1];
    R_est = py::cast<Eigen::VectorXi>(state_R);

    // Create R_count (the number of each state)
    R_count = Eigen::VectorXi::Zero(num_states);
    for (int t = 0; t < num_time; ++t)
    {
      R_count(R_est(t)) += 1;
    }

    // Resume P_est
    py::list P_iter;
    if (store_transition_matrix)
    {
      P_iter = stored_values["P_iter"].cast<py::list>();
    }
    else
    {
      P_iter = stored_values["P_last"].cast<py::list>();
    }
    auto mat_R = P_iter[py::len(P_iter) - 1];
    P_est = py::cast<Eigen::MatrixXd>(mat_R);

    // Resume alphas
    py::list alpha_iter = stored_values["alpha_iter"].cast<py::list>();
    auto alphas_R = alpha_iter[py::len(alpha_iter) - 1];
    alphas = py::cast<Eigen::MatrixXd>(alphas_R);

    // Initialize variables we use in the sampling
    Prk = Eigen::MatrixXd::Zero(num_time, num_states);
    logfy = Eigen::VectorXd::Zero(num_states);
    rt_k = Eigen::VectorXd::Zero(num_states);
    logrt_k = Eigen::VectorXd::Zero(num_states);
    state_prob_vec = Eigen::VectorXd::Zero(num_states);

    states_start = Eigen::VectorXi::Zero(num_states);
    states_end = Eigen::VectorXi::Zero(num_states);
  }
  int keyATMhmm::get_state_index(const int doc_id)
  {
    // Which time segment the document belongs to
    int t;
    for (t = 0; t < num_time; ++t)
    {
      if (time_doc_start(t) <= doc_id && doc_id <= time_doc_end(t))
      {
        break;
      }
    }
    return R_est(t);
  }
  void keyATMhmm::iteration_single(int it)
  { // Single iteration
    int doc_id_;
    int doc_length;
    int w_, z_, s_;
    int new_z, new_s;
    int w_position;

    doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

    for (int ii = 0; ii < num_doc; ++ii)
    {
      doc_id_ = doc_indexes[ii];

      auto &doc_s = S[doc_id_];
      auto &doc_z = Z[doc_id_];
      const auto &doc_w = W[doc_id_];
      doc_length = doc_each_len[doc_id_];

      alpha = alphas.row(get_state_index(doc_id_)).transpose(); // select alpha for this document

      token_indexes = sampler::shuffled_indexes(doc_length); // shuffle

      // Iterate each word in the document
      for (int jj = 0; jj < doc_length; ++jj)
      {
        w_position = token_indexes[jj];
        s_ = doc_s[w_position];
        z_ = doc_z[w_position];
        w_ = doc_w[w_position];

        new_z = sample_z(alpha, z_, s_, w_, doc_id_);
        doc_z[w_position] = new_z;

        if (keywords[new_z].find(w_) == keywords[new_z].end())
          continue;

        z_ = doc_z[w_position]; // use updated z
        new_s = sample_s(z_, s_, w_, doc_id_);
        doc_s[w_position] = new_s;
      }

    }

    sample_parameters(it);
  }

  void keyATMhmm::verbose_special(int r_index)
  {
    // If there is anything special to show, write here.
  }

  void keyATMhmm::sample_parameters(int it)
  {
    // alpha
    sample_alpha();

    // HMM
    sample_forward();  // calculate Prk
    sample_backward(); // sample R_est
    sample_P();        // sample P_est

    // Store alpha, state, and the transition matrix
    int r_index = it + 1;
    if (r_index % thinning == 0 || r_index == 1 || r_index == iter)
    {
      auto alpha_iter = stored_values["alpha_iter"].cast<py::list>();
      alpha_iter.append(py::cast(alphas));
      stored_values["alpha_iter"] = alpha_iter;

      // Store state
      store_R_est();

      // Store transition matrix
      if (store_transition_matrix)
      {
        store_P_est();
      }
      else
      {
        keep_P_est();
      }
    }
  }
  void keyATMhmm::sample_alpha()
  {

    // Retrieve start and end indexes of states in documents
    int index_start, index_end;
    for (int r = 0; r < num_states; ++r)
    {
      if (r == 0)
      {
        // First state
        // Which time segment correspond to s = 0
        index_start = 0;
        index_end = R_count(r) - 1;

        // Index of documents that belong to s = 0
        states_start(r) = time_doc_start(index_start);
        states_end(r) = time_doc_end(index_end);
        continue;
      }

      index_start = index_end + 1;
      index_end = index_start + R_count(r) - 1;
      states_start(r) = time_doc_start(index_start);
      states_end(r) = time_doc_end(index_end);
    }

    for (int r = 0; r < num_states; ++r)
    {
      sample_alpha_state(r, states_start(r),
                         states_end(r));
    }
  }
  void keyATMhmm::sample_alpha_state(int state, int state_start, int state_end)
  {

    double start, end, previous_p, new_p, newlikelihood, slice_;
    double store_loglik;
    double newalphallk;

    keep_current_param = alpha;
    topic_ids = sampler::shuffled_indexes(num_topics);
    newalphallk = 0.0;
    int k;

    alpha = alphas.row(state).transpose(); // select alpha to update

    auto &rng = sampler::rng();
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    for (int i = 0; i < num_topics; ++i)
    {
      k = topic_ids[i];
      store_loglik = alpha_loglik(k, state_start, state_end);
      start = min_v; // shrinked with shrinkp()
      end = max_v;   // shrinked with shrinkp()

      previous_p = alpha(k) / (1.0 + alpha(k));                                         // shrinkp
      slice_ = store_loglik - 2.0 * std::log(1.0 - previous_p) + std::log(unif01(rng)); // <-- using R random uniform

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time)
      {
        new_p = sampler::slice_uniform(start, end); // <-- using R function above
        alpha(k) = new_p / (1.0 - new_p);           // expandp

        newalphallk = alpha_loglik(k, state_start, state_end);
        newlikelihood = newalphallk - 2.0 * std::log(1.0 - new_p);

        if (slice_ < newlikelihood)
        {
          break;
        }
        else if (previous_p < new_p)
        {
          end = new_p;
        }
        else if (new_p < previous_p)
        {
          start = new_p;
        }
        else
        {
          throw std::runtime_error("Something goes wrong in sample_lambda_slice(). Adjust `A_slice`.");
          alpha(k) = keep_current_param(k);
          break;
        }
      }
    }

    // Set new alpha
    alphas.row(state) = alpha.transpose();
  }
  double keyATMhmm::alpha_loglik(int k, int state_start, int state_end)
  {
    double loglik = 0.0;
    double fixed_part = 0.0;

    ndk_a = n_dk.rowwise() + alpha.transpose(); // Use Eigen Broadcasting
    double alpha_sum_val = alpha.sum();

    fixed_part += mylgamma(alpha_sum_val); // first term numerator
    fixed_part -= mylgamma(alpha(k));      // first term denominator
    // Add prior
    if (k < keyword_k)
    {
      loglik += gammapdfln(alpha(k), eta_1, eta_2);
    }
    else
    {
      loglik += gammapdfln(alpha(k), eta_1_regular, eta_2_regular);
    }

    for (int d = state_start; d <= state_end; ++d)
    {
      loglik += fixed_part;

      // second term numerator
      loglik += mylgamma(ndk_a(d, k));

      // second term denominator
      loglik -= mylgamma(doc_each_len_weighted[d] + alpha_sum_val);
    }
    return loglik;
  }
  void keyATMhmm::sample_forward()
  { // Calculate Prk (num_doc, num_states)
    double logsum;
    int added;
    double loglik;

    for (int t = 0; t < num_time; ++t)
    {
      if (t == 0)
      {
        // First time segment should be the first state
        Prk(0, 0) = 1.0;
        continue;
      }

      // Prepare f in Eq.(6) of Chib (1998)
      for (int r = 0; r < num_states; ++r)
      {
        // f(y_t | ...) in the numerator
        alpha = alphas.row(r).transpose();
        logfy(r) = polyapdfln(t, alpha);
      }

      // Prepare Pst
      rt_1l = Prk.row(t - 1).transpose(); // previous time block
      rt_k = (rt_1l.transpose() * P_est);
      // p(s_{t} = k), summation is done as matrix calculation
      // Note that P has a lot of 0 elements
      // This is a first term of the numerator in Eq.(6)

      // Format numerator and calculate denominator at the same time
      logsum = 0.0;
      added = 0;
      for (int r = 0; r < num_states; ++r)
      {
        if (rt_k(r) != 0.0)
        {
          loglik = std::log(rt_k(r)) + logfy(r);
          logrt_k(r) = loglik;
          logsum = logsumexp(logsum, loglik, (added == 0));
          added += 1;
        }
        else
        {
          logrt_k(r) = 0.0; // place holder
        }
      }

      for (int r = 0; r < num_states; ++r)
      {
        if (rt_k(r) != 0.0)
        {
          Prk(t, r) = std::exp(logrt_k(r) - logsum);
        }
        else
        {
          Prk(t, r) = 0.0;
        }
      }
    }
  }
  double keyATMhmm::polyapdfln(int t, Eigen::VectorXd &alpha)
  { // Polya distribution: log-likelihood
    double loglik = 0.0;

    int doc_start, doc_end;
    doc_start = time_doc_start(t); // starting doc index of time segment t
    doc_end = time_doc_end(t);

    for (int d = doc_start; d <= doc_end; ++d)
    {
      loglik += mylgamma(alpha.sum()) - mylgamma(doc_each_len_weighted[d] + alpha.sum());
      for (int k = 0; k < num_topics; ++k)
      {
        loglik += mylgamma(n_dk(d, k) + alpha(k)) - mylgamma(alpha(k));
      }
    }

    return loglik;
  }
  void keyATMhmm::sample_backward()
  {
    int state_id;

    // sample R_est
    // num_time - 2, because time segment index is (num_time - 1)
    // and we want to start from (time_index - 1)

    R_count = Eigen::VectorXi::Zero(num_states); // reset counter

    // Last document
    R_est(num_time - 1) = index_states;
    R_count(index_states) += 1; // last document

    for (int t = (num_time - 2); 0 <= t; --t)
    {
      state_id = R_est(t + 1);

      state_prob_vec.array() = Prk.row(t).transpose().array() * P_est.col(state_id).array();
      state_prob_vec.array() = state_prob_vec.array() / state_prob_vec.sum();

      state_id = sampler::rcat(state_prob_vec, num_states); // new state id
      R_est(t) = state_id;
      R_count(state_id) += 1;
    }
  }
  void keyATMhmm::sample_P()
  {
    double pii;

    // sample P_est
    // iterate until index_state - 2
    for (int r = 0; r <= (num_states - 2); ++r)
    {
      auto &rng = sampler::rng();
      std::gamma_distribution<double> ga(static_cast<double>(R_count(r)), 1.0);
      std::gamma_distribution<double> gb(2.0, 1.0);
      const double x = ga(rng);
      const double y = gb(rng);
      pii = x / (x + y);
      // First value is 1 + R_count(s) - 1.
      // R_count(s) - 1: the number of transitions from state
      // s to state s in the sequence of state
      // ----------------------------------------------------
      // "-1" because the first count in R_count is
      // the transition from s-1 to s
      // prior is Beta(1,1)

      P_est(r, r) = pii;
      P_est(r, r + 1) = 1.0 - pii;
    }
  }
  void keyATMhmm::store_R_est()
  {
    // Store state
    auto R_iter = stored_values["R_iter"].cast<py::list>();
    R_iter.append(py::cast(R_est));
    stored_values["R_iter"] = R_iter;
  }

  void keyATMhmm::store_P_est()
  {
    // Store transition matrix
    auto P_iter = stored_values["P_iter"].cast<py::list>();
    P_iter.append(py::cast(P_est));
    stored_values["P_iter"] = P_iter;
  }

  void keyATMhmm::keep_P_est()
  {
    // Keep the latest transition matrix
    auto P_last = stored_values["P_last"].cast<py::list>();
    if (py::len(P_last) == 0)
    {
      P_last.append(py::cast(P_est));
    }
    else
    {
      py::list new_P_last;
      new_P_last.append(py::cast(P_est));
      P_last = std::move(new_P_last);
    }
    stored_values["P_last"] = P_last;
  }

  double keyATMhmm::loglik_total()
  {
    double loglik = 0.0;
    int state_id;

    for (int k = 0; k < num_topics; ++k)
    {
      for (int v = 0; v < num_vocab; ++v)
      { // word
        loglik += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
      }

      // word normalization
      loglik += mylgamma(beta * (double)num_vocab) - mylgamma(beta * (double)num_vocab + n_s0_k(k));

      if (k < keyword_k)
      {
        // For keyword topics

        // n_s1_kv
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it)
        {
          loglik += mylgamma(beta_s + it.value() / vocab_weights(it.index())) - mylgamma(beta_s);
        }
        loglik += mylgamma(beta_s * (double)keywords_num[k]) - mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k));

        // Normalization
        loglik += mylgamma(prior_gamma(k, 0) + prior_gamma(k, 1)) - mylgamma(prior_gamma(k, 0)) - mylgamma(prior_gamma(k, 1));

        // s
        loglik += mylgamma(n_s0_k(k) + prior_gamma(k, 1)) - mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1)) + mylgamma(n_s1_k(k) + prior_gamma(k, 0));
      }
    }

    for (int d = 0; d < num_doc; ++d)
    {
      // z
      alpha = alphas.row(get_state_index(d)).transpose(); // Doc alpha, column vector

      loglik += mylgamma(alpha.sum()) - mylgamma(doc_each_len_weighted[d] + alpha.sum());
      for (int k = 0; k < num_topics; ++k)
      {
        loglik += mylgamma(n_dk(d, k) + alpha(k)) - mylgamma(alpha(k));
      }
    }

    // HMM part
    for (int t = 0; t < num_time; ++t)
    {
      state_id = R_est(t);
      loglik += std::log(P_est(state_id, state_id));
    }

    return loglik;
  }

}
