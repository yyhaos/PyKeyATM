#include "keyATM_base.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <random>
#include <stdexcept>
#include <cmath>

namespace keyatm
{

  void keyATMbase::initialize_specific()
  {
    TRACE_FUNC();
    // Load alpha prior from Python list
    nv_alpha = priors_list["alpha"].cast<std::vector<double>>();
    alpha = Eigen::Map<Eigen::VectorXd>(nv_alpha.data(), nv_alpha.size());

    estimate_alpha = options_list["estimate_alpha"].cast<int>();
    store_alpha = estimate_alpha ? 1 : 0;
  }

  void keyATMbase::resume_initialize_specific()
  {
    estimate_alpha = options_list["estimate_alpha"].cast<int>();
    if (estimate_alpha == 0)
    {
      nv_alpha = priors_list["alpha"].cast<std::vector<double>>();
      alpha = Eigen::Map<Eigen::VectorXd>(nv_alpha.data(), nv_alpha.size());
      store_alpha = 0;
    }
    else
    {
      py::gil_scoped_acquire gil;
      py::list alpha_iter = stored_values["alpha_iter"].cast<py::list>();
      std::vector<double> last = alpha_iter[alpha_iter.size() - 1].cast<std::vector<double>>();
      nv_alpha = std::move(last);
      alpha = Eigen::Map<Eigen::VectorXd>(nv_alpha.data(), nv_alpha.size());
      store_alpha = 1;
    }
  }

  void keyATMbase::iteration_single(int it)
  {
    TRACE_FUNC();
    int doc_id_, doc_length;
    int w_, z_, s_, new_z, new_s, w_position;

    // Shuffle document order
    doc_indexes = sampler::shuffled_indexes(num_doc);

    for (int ii = 0; ii < num_doc; ++ii)
    {
      doc_id_ = ii;
      auto &doc_s = S[doc_id_];
      auto &doc_z = Z[doc_id_];
      const auto &doc_w = W[doc_id_];
      doc_length = doc_each_len[doc_id_];
      // Shuffle token order
      token_indexes = sampler::shuffled_indexes(doc_length);

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

        z_ = doc_z[w_position]; // updated z
        new_s = sample_s(z_, s_, w_, doc_id_);
        doc_s[w_position] = new_s;
      }

    }
    sample_parameters(it);
  }

  void keyATMbase::sample_parameters(int it)
  {
    TRACE_FUNC();

    if (estimate_alpha)
      sample_alpha();

    if (!store_alpha)
      return;

    const int r_index = it + 1;
    if (!(r_index % thinning == 0 || r_index == 1 || r_index == iter))
      return;

    std::vector<double> alpha_rvec = alpha_reformat(alpha, num_topics);

    py::gil_scoped_acquire gil;

    py::list alpha_iter;
    if (stored_values.contains("alpha_iter"))
    {
      py::object obj = stored_values["alpha_iter"];
      alpha_iter = py::isinstance<py::list>(obj) ? obj.cast<py::list>() : py::list(obj);
    }
    else
    {
      alpha_iter = py::list();
    }

    alpha_iter.append(py::cast(alpha_rvec));

    stored_values["alpha_iter"] = alpha_iter;
  }

  void keyATMbase::sample_alpha()
  {
    TRACE_FUNC();
    keep_current_param = alpha;
    topic_ids = sampler::shuffled_indexes(num_topics /*, rng*/);
    newalphallk = 0.0;

    auto &rng = sampler::rng();

    auto open01 = [&](std::mt19937 &r)
    {
      double u;
      do
      {
        u = std::generate_canonical<double, 53>(r);
      } while (u <= 0.0 || u >= 1.0);
      return u;
    };
    auto safe01 = [](double p)
    {
      constexpr double eps = 1e-12;
      return std::clamp(p, eps, 1.0 - eps);
    };

    for (int i = 0; i < num_topics; ++i)
    {
      int k = topic_ids[i];
      store_loglik = alpha_loglik(k);
      double start = min_v;
      double end = max_v;

      double previous_p = alpha(k) / (1.0 + alpha(k));

      double u = std::generate_canonical<double, 53>(rng);
      double slice_ = store_loglik - 2.0 * std::log(1.0 - previous_p) + std::log(u);

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time)
      {
        double new_p = sampler::slice_uniform(start, end /*, rng*/);
        alpha(k) = new_p / (1.0 - new_p);

        newalphallk = alpha_loglik(k);
        double newlikelihood = newalphallk - 2.0 * std::log(1.0 - new_p);
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
          throw std::runtime_error("Something goes wrong in sample_lambda_slice().");
          alpha(k) = keep_current_param(k);
          break;
        }
      }
    }
  }

  double keyATMbase::alpha_loglik(int k)
  {
    double loglik = 0.0;
    double fixed_part = 0.0;

    ndk_a = n_dk.rowwise() + alpha.transpose(); // broadcasting
    double alpha_sum_val = alpha.sum();

    fixed_part += mylgamma(alpha_sum_val);
    fixed_part -= mylgamma(alpha(k));

    // Prior
    if (k < keyword_k)
      loglik += gammapdfln(alpha(k), eta_1, eta_2);
    else
      loglik += gammapdfln(alpha(k), eta_1_regular, eta_2_regular);

    for (int d = 0; d < num_doc; ++d)
    {
      loglik += fixed_part;
      loglik += mylgamma(ndk_a(d, k));
      loglik -= mylgamma(doc_each_len_weighted[d] + alpha_sum_val);
    }

    return loglik;
  }

  double keyATMbase::loglik_total()
  {
    double loglik = 0.0;

    // Topic-word part
    for (int k = 0; k < num_topics; ++k)
    {
      for (int v = 0; v < num_vocab; ++v)
      {
        loglik += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
      }

      loglik += mylgamma(beta * num_vocab) - mylgamma(beta * num_vocab + n_s0_k(k));

      if (k < keyword_k)
      {
        // For keyword topics — sparse
        // for (SparseMatrix<double, RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it)
        {
          loglik += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
        }

        loglik += mylgamma(beta_s * keywords_num[k]) -
                  mylgamma(beta_s * keywords_num[k] + n_s1_k(k));

        // Gamma normalization terms
        loglik += mylgamma(prior_gamma(k, 0) + prior_gamma(k, 1)) -
                  mylgamma(prior_gamma(k, 0)) - mylgamma(prior_gamma(k, 1));

        // s (binary variable)
        loglik += mylgamma(n_s0_k(k) + prior_gamma(k, 1)) +
                  mylgamma(n_s1_k(k) + prior_gamma(k, 0)) -
                  mylgamma(n_s0_k(k) + n_s1_k(k) + prior_gamma(k, 0) + prior_gamma(k, 1));
      }
    }

    // Document-topic part
    double alpha_sum = alpha.sum();
    for (int d = 0; d < num_doc; ++d)
    {
      loglik += mylgamma(alpha_sum) - mylgamma(doc_each_len_weighted[d] + alpha_sum);

      for (int k = 0; k < num_topics; ++k)
      {
        loglik += mylgamma(n_dk(d, k) + alpha(k)) - mylgamma(alpha(k));
      }
    }

    return loglik;
  }

} // namespace keyatm
