#include "LDA_base.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <stdexcept>
#include <cmath>

namespace keyatm {

namespace py = pybind11;

void LDAbase::read_data_common()
{
  TRACE_FUNC();

  W = model["W"].cast<std::vector<std::vector<int>>>();
  Z = model["Z"].cast<std::vector<std::vector<int>>>();
  S.clear();
  model["W"] = py::none();
  model["Z"] = py::none();
  vocab = py::cast<std::vector<std::string>>(model["vocab"]);
  regular_k = py::cast<int>(model["no_keyword_topics"]);
  model_fit = model["model_fit"].cast<py::dict>();

  keyword_k = 0;
  num_topics = regular_k;

  num_vocab = static_cast<int>(vocab.size());
  num_doc = static_cast<int>(W.size());

  options_list = model["options"].cast<py::dict>();
  use_weights = py::cast<int>(options_list["use_weights"]);
  slice_A = py::cast<double>(options_list["slice_shape"]);
  store_theta = py::cast<int>(options_list["store_theta"]);
  thinning = py::cast<int>(options_list["thinning"]);
  llk_per = py::cast<int>(options_list["llk_per"]);
  verbose = py::cast<int>(options_list["verbose"]);
  weights_type = py::cast<std::string>(options_list["weights_type"]);

  priors_list = model["priors"].cast<py::dict>();
  beta = py::cast<double>(priors_list["beta"]);
  eta_1 = py::cast<double>(priors_list["eta_1"]);
  eta_2 = py::cast<double>(priors_list["eta_2"]);
  eta_1_regular = py::cast<double>(priors_list["eta_1_regular"]);
  eta_2_regular = py::cast<double>(priors_list["eta_2_regular"]);

  stored_values = model["stored_values"].cast<py::dict>();
  model_settings = model["model_settings"].cast<py::dict>();

  min_v = shrinkp(py::cast<double>(model_settings["slice_min"]));
  max_v = shrinkp(py::cast<double>(model_settings["slice_max"]));
}

void LDAbase::initialize_common()
{
  TRACE_FUNC();
  // Prior values are set in `LDAbase::read_data_common()`

  // Slice sampling initialization
  max_shrink_time = 200;

  //
  // Vocabulary weights
  //
  vocab_weights = Eigen::VectorXd::Constant(num_vocab, 1.0);

  // Construct vocab weights
  for (int doc_id = 0; doc_id < num_doc; ++doc_id)
  {
    const auto &doc_w = W[doc_id];
    const int doc_len = static_cast<int>(doc_w.size());
    doc_each_len.push_back(doc_len);

    for (int w : doc_w)
    {
      vocab_weights(w) += 1.0;
    }
  }
  total_words = static_cast<int>(vocab_weights.sum());

  if (weights_type == "inv-freq" || weights_type == "inv-freq-normalized")
  {
    // Inverse frequency
    weights_invfreq();
  }
  else if (weights_type == "information-theory" || weights_type == "information-theory-normalized")
  {
    // Information theory
    weights_inftheory();
  }

  // Normalize weights
  if (weights_type == "inv-freq-normalized" ||
      weights_type == "information-theory-normalized")
  {
    weights_normalize_total();
  }

  // Do you want to use weights?
  if (use_weights == 0)
  {
    vocab_weights = Eigen::VectorXd::Constant(num_vocab, 1.0);
  }

  //
  // Construct data matrices
  //
  n_kv = Eigen::MatrixXd::Zero(num_topics, num_vocab);
  n_dk = Eigen::MatrixXd::Zero(num_doc, num_topics);
  n_dk_noWeight = Eigen::MatrixXd::Zero(num_doc, num_topics);
  n_k = Eigen::VectorXd::Zero(num_topics);

  total_words_weighted = 0.0;

  for (int doc_id = 0; doc_id < num_doc; ++doc_id)
  {
    const auto &doc_z = Z[doc_id];
    const auto &doc_w = W[doc_id];
    const int doc_len = doc_each_len[doc_id];

    for (int w_position = 0; w_position < doc_len; ++w_position)
    {
      const int z = doc_z[w_position];
      const int w = doc_w[w_position];

      const double vw = vocab_weights(w);
      n_kv(z, w) += vw;
      n_k(z) += vw;
      n_dk(doc_id, z) += vw;
      n_dk_noWeight(doc_id, z) += 1.0;
    }

    const double temp = n_dk.row(doc_id).sum();
    doc_each_len_weighted.push_back(temp);
    total_words_weighted += temp;
  }

  // Use during the iteration
  z_prob_vec = Eigen::VectorXd::Zero(num_topics);
}

void LDAbase::parameters_store(int r_index)
{
  if (store_theta)
    store_theta_iter(r_index);
}

int LDAbase::sample_z(Eigen::VectorXd &alpha, int z, int /*s*/, int w, int doc_id)
{
  int new_z = -1;
  double numerator, denominator;

  // remove current token counts
  n_kv(z, w)              -= vocab_weights(w);
  n_k(z)                  -= vocab_weights(w);
  n_dk(doc_id, z)         -= vocab_weights(w);
  n_dk_noWeight(doc_id, z) -= 1.0;

  // compute full conditional for each topic
  for (int k = 0; k < num_topics; ++k) {
    numerator   = (beta + n_kv(k, w)) * (n_dk(doc_id, k) + alpha(k));
    denominator = (static_cast<double>(num_vocab) * beta + n_k(k));
    z_prob_vec(k) = numerator / denominator;
  }

  // sample new topic assignment
  const double sum = z_prob_vec.sum();
  new_z = sampler::rcat_without_normalize(z_prob_vec, sum, num_topics);

  // add back counts with new topic
  n_kv(new_z, w)              += vocab_weights(w);
  n_k(new_z)                  += vocab_weights(w);
  n_dk(doc_id, new_z)         += vocab_weights(w);
  n_dk_noWeight(doc_id, new_z) += 1.0;

  return new_z;
}

} // namespace keyatm
