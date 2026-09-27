#include "LDA_weight.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace keyatm
{

  namespace py = pybind11;

  void LDAweight::iteration_single(int it)
  {
    int s_ = -1;                                      // x_/s not used in weighted LDA
    doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle documents

    for (int ii = 0; ii < num_doc; ++ii)
    {
      const int doc_id_ = doc_indexes[ii];
      auto &doc_z = Z[doc_id_];
      const auto &doc_w = W[doc_id_];
      const int doc_length = doc_each_len[doc_id_];

      token_indexes = sampler::shuffled_indexes(doc_length); // shuffle tokens

      for (int jj = 0; jj < doc_length; ++jj)
      {
        const int w_position = token_indexes[jj];
        const int z_ = doc_z[w_position];
        const int w_ = doc_w[w_position];

        const int new_z = sample_z(alpha, z_, s_, w_, doc_id_);
        doc_z[w_position] = new_z;
      }

    }

    sample_parameters(it);
  }

  double LDAweight::loglik_total()
  {
    double loglik = 0.0;

    // Topic-word part
    for (int k = 0; k < num_topics; ++k)
    {
      for (int v = 0; v < num_vocab; ++v)
      {
        loglik += mylgamma(beta + n_kv(k, v)) - mylgamma(beta);
      }
      loglik += mylgamma(beta * static_cast<double>(num_vocab)) - mylgamma(beta * static_cast<double>(num_vocab) + n_k(k));
    }

    // Document-topic part
    const double alpha_sum = alpha.sum();
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
