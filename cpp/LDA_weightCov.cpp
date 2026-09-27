#include "LDA_weightCov.h"

namespace py = pybind11;

namespace keyatm
{

# define PI_V   3.14159265358979323846  /* pi */

void LDAcov::iteration_single(int it)
{
  const int s_ = -1;
  doc_indexes = sampler::shuffled_indexes(num_doc);

  Alpha = (C * Lambda.transpose()).array().exp();

  for (int ii = 0; ii < num_doc; ++ii)
  {
    const int doc_id_ = doc_indexes[ii];
    auto &doc_z = Z[doc_id_];
    const auto &doc_w = W[doc_id_];
    const int doc_length = doc_each_len[doc_id_];

    token_indexes = sampler::shuffled_indexes(doc_length);

    alpha = Alpha.row(doc_id_).transpose();

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
double LDAcov::loglik_total()
{
  double loglik = 0.0;

  // topic–word part
  for (int k = 0; k < num_topics; ++k)
  {
    for (int v = 0; v < num_vocab; ++v)
    {
      loglik += mylgamma(beta + n_kv(k, v)) - mylgamma(beta);
    }
    loglik += mylgamma(beta * static_cast<double>(num_vocab)) - mylgamma(beta * static_cast<double>(num_vocab) + n_k(k));
  }

  // document–topic part with covariates (alpha depends on C and Lambda)
  Alpha = (C * Lambda.transpose()).array().exp();
  alpha = Eigen::VectorXd::Zero(num_topics);

  for (int d = 0; d < num_doc; ++d)
  {
    alpha = Alpha.row(d).transpose();
    const double alpha_sum = alpha.sum();

    loglik += mylgamma(alpha_sum) - mylgamma(doc_each_len_weighted[d] + alpha_sum);

    for (int k = 0; k < num_topics; ++k)
    {
      loglik += mylgamma(n_dk(d, k) + alpha(k)) - mylgamma(alpha(k));
    }
  }

  const double prior_fixedterm = -0.5 * std::log(2.0 * PI_V * sigma * sigma);

  for (int k = 0; k < num_topics; ++k)
  {
    for (int t = 0; t < num_cov; ++t)
    {
      loglik += prior_fixedterm;
      const double diff = Lambda(k, t) - mu;
      loglik -= (diff * diff) / (2.0 * sigma * sigma);
    }
  }

  return loglik;
}

}
