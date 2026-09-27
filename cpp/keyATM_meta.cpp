#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "keyATM_meta.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <random> 
#include "sampler.h"

namespace py = pybind11;
using namespace keyatm;

#define PI_V 3.14159265358979323846

// Constructor: Accepts a Python dict and stores it as py::dict internally
keyATMmeta::keyATMmeta(const py::dict &model_){
  model = model_;
}

// Destructor
keyATMmeta::~keyATMmeta() = default;

void keyATMmeta::configure_checkpoint(int every, const py::object& callback)
{
  checkpoint_every = every;
  checkpoint_callback = callback;
}

// Fit: Full run
void keyATMmeta::fit()
{
  TRACE_FUNC();
  read_data();
  initialize();
  iteration();

}

// Resume fit
void keyATMmeta::resume_fit() {
    read_data();
    resume_initialize();
    iteration();
}

// Data reading
void keyATMmeta::read_data()
{
  TRACE_FUNC();
  read_data_common();
  read_data_specific(); // to be implemented in subclass
}

void keyATMmeta::read_data_common()
{
  TRACE_FUNC();

  W = model["W"].cast<std::vector<std::vector<int>>>();
  Z = model["Z"].cast<std::vector<std::vector<int>>>();
  S = model["S"].cast<std::vector<std::vector<int>>>();
  model["W"] = py::none();
  model["Z"] = py::none();
  model["S"] = py::none();
  vocab = py::cast<std::vector<std::string>>(model["vocab"]);
  regular_k = py::cast<int>(model["no_keyword_topics"]);

  py::dict kw_dict = model["keywords"].cast<py::dict>();
  keywords_list.clear();
  for (auto item : kw_dict)
  {
    auto kw_vec = item.second.cast<std::vector<int>>();
    keywords_list.push_back(std::move(kw_vec));
  }

  keyword_k = py::cast<int>(model["keyword_k"]);
  model_fit = model["model_fit"].cast<py::dict>();

  num_topics = keyword_k + regular_k;

  num_vocab = static_cast<int>(vocab.size());
  num_doc = static_cast<int>(W.size());


  options_list = model["options"].cast<py::dict>();
  use_weights = py::cast<int>(options_list["use_weights"]);
  slice_A = py::cast<double>(options_list["slice_shape"]);
  store_theta = py::cast<int>(options_list["store_theta"]);
  store_pi = py::cast<int>(options_list["store_pi"]);
  thinning = py::cast<int>(options_list["thinning"]);
  llk_per = py::cast<int>(options_list["llk_per"]);
  verbose = py::cast<int>(options_list["verbose"]);
  weights_type = py::cast<std::string>(options_list["weights_type"]);

  priors_list = model["priors"].cast<py::dict>();
  beta = py::cast<double>(priors_list["beta"]);

  prior_gamma = py::cast<Eigen::MatrixXd>(priors_list["gamma"]);
  beta_s = py::cast<double>(priors_list["beta_s"]);
  eta_1 = py::cast<double>(priors_list["eta_1"]);
  eta_2 = py::cast<double>(priors_list["eta_2"]);
  eta_1_regular = py::cast<double>(priors_list["eta_1_regular"]);
  eta_2_regular = py::cast<double>(priors_list["eta_2_regular"]);

  model_settings = model["model_settings"].cast<py::dict>();
  stored_values = model["stored_values"].cast<py::dict>();

  min_v = shrinkp(py::cast<double>(model_settings["slice_min"]));
  max_v = shrinkp(py::cast<double>(model_settings["slice_max"]));
}

void keyATMmeta::initialize()
{
  TRACE_FUNC();
  // `common`: common initialization
  // `specific`: model specific initialization
  initialize_common();
  initialize_specific();
}

void keyATMmeta::initialize_common()
{
  TRACE_FUNC();
  max_shrink_time = 200;

  // Step 1: build keyword sets
  for (int ii = 0; ii < keyword_k; ++ii) {
    auto wd_ids = keywords_list[ii];  // std::vector<int>
    keywords_num.push_back(static_cast<int>(wd_ids.size()));

    std::unordered_set<int> keywords_set;
    for (int wid : wd_ids) {
      keywords_set.insert(wid);
    }
    keywords.push_back(keywords_set);
  }

  for (int i = keyword_k; i < num_topics; ++i) {
    keywords_num.push_back(0);
    keywords.emplace_back(std::unordered_set<int>{-1});
  }

  // Step 2: initialize sufficient statistics
  n_s0_kv = MatrixXd::Zero(num_topics, num_vocab);
  n_s1_kv.resize(num_topics, num_vocab);
  n_dk = MatrixXd::Zero(num_doc, num_topics);
  n_dk_noWeight = MatrixXd::Zero(num_doc, num_topics);
  n_s0_k = VectorXd::Zero(num_topics);
  n_s1_k = VectorXd::Zero(num_topics);
  vocab_weights = VectorXd::Constant(num_vocab, 1.0);

  // Step 3: compute vocab weights
  for (int doc_id = 0; doc_id < num_doc; ++doc_id) {
    const auto& doc_w = W[doc_id];
    int doc_len = static_cast<int>(doc_w.size());
    doc_each_len.push_back(doc_len);

    for (int w : doc_w) {
      vocab_weights(w) += 1.0;
    }
  }

  total_words = static_cast<int>(vocab_weights.sum());

  // Step 4: weighting strategy
  if (weights_type == "inv-freq" || weights_type == "inv-freq-normalized") {
    weights_invfreq();
  } else if (weights_type == "information-theory" || weights_type == "information-theory-normalized") {
    weights_inftheory();
  }

  if (weights_type == "inv-freq-normalized" || weights_type == "information-theory-normalized") {
    weights_normalize_total();
  }

  if (use_weights == 0) {
    vocab_weights = VectorXd::Constant(num_vocab, 1.0);
  }

  // Step 5: store vocab_weights back to stored_values
  py::list vocab_weights_py;
  for (int v = 0; v < num_vocab; ++v) {
    vocab_weights_py.append(vocab_weights(v));
  }

  stored_values["vocab_weights"] = vocab_weights_py;
  py::dict stored_dict;
  for (const auto& pair : stored_values) {
      stored_dict[py::str(pair.first)] = pair.second;
  }
  model["stored_values"] = stored_dict;

  // Step 6: build sparse matrix and count statistics
  std::vector<Eigen::Triplet<double>> trip_s1;
  total_words_weighted = 0.0;

  for (int doc_id = 0; doc_id < num_doc; ++doc_id) {
    const auto &doc_s = S[doc_id];
    const auto &doc_z = Z[doc_id];
    const auto &doc_w = W[doc_id];
    int doc_len = doc_each_len[doc_id];

    for (int i = 0; i < doc_len; ++i) {
      int s = doc_s[i];
      int z = doc_z[i];
      int w = doc_w[i];

      double vw = vocab_weights(w);

      if (s == 0) {
        n_s0_kv(z, w) += vw;
        n_s0_k(z) += vw;
      } else {
        trip_s1.emplace_back(z, w, vw);
        n_s1_k(z) += vw;
      }

      n_dk(doc_id, z) += vw;
      n_dk_noWeight(doc_id, z) += 1.0;
    }

    double temp = n_dk.row(doc_id).sum();
    doc_each_len_weighted.push_back(temp);
    total_words_weighted += temp;
  }

  n_s1_kv.setFromTriplets(trip_s1.begin(), trip_s1.end());

  z_prob_vec = VectorXd::Zero(num_topics);
  Vbeta = static_cast<double>(num_vocab) * beta;
  Lbeta_sk = VectorXd::Zero(num_topics);

  for (int k = 0; k < num_topics; ++k) {
    Lbeta_sk(k) = static_cast<double>(keywords_num[k]) * beta_s;
  }
}
void keyATMmeta::weights_invfreq()
{
  TRACE_FUNC();
  // Inverse frequency: vocab_weights = total_words / count
  vocab_weights = (double)total_words / vocab_weights.array();
}

void keyATMmeta::weights_inftheory()
{
  TRACE_FUNC();
  // vocab_weights = -[p(w) - log2(p(w))]
  vocab_weights = vocab_weights.array() / (double)total_words;         // p(w)
  vocab_weights = vocab_weights.array().log();                         // log(p(w))
  vocab_weights = -vocab_weights.array() / std::log(2.0);              // -log2(p(w))
}

void keyATMmeta::weights_normalize_total()
{
  TRACE_FUNC();
  // Normalize weights so that total weighted count = total_words

  double total_weights = 0.0;

  for (int doc_id = 0; doc_id < num_doc; ++doc_id) {
    const auto &doc_w = W[doc_id];
    int doc_len = doc_each_len[doc_id];

    for (int w_pos = 0; w_pos < doc_len; ++w_pos) {
      int w = doc_w[w_pos];
      total_weights += vocab_weights(w);
    }
  }

  double scale = static_cast<double>(total_words) / total_weights;
  vocab_weights *= scale;
}

//
// Initializing using the resume data
//
void keyATMmeta::resume_initialize()
{
  initialize_common();
  resume_initialize_specific();
}

void keyATMmeta::iteration()
{
  TRACE_FUNC();
  iter = options_list["iterations"].cast<int>();
  int iter_new = options_list["iter_new"].cast<int>();
  int iter_start = iter - iter_new;
  constexpr int rate_window_size = 5;
  int rate_window_iterations = 0;
  auto rate_window_start = std::chrono::steady_clock::now();

  if (verbose) std::cout << "[keyATM] Starting at " << iter_start << "/" << iter << std::endl;

  for (int it = iter_start; it < iter; ++it) {
    // dd();
    // if (it==200)
    //   exit(11);
    iteration_single(it);

    int r_index = it + 1;

    if (r_index % llk_per == 0 || r_index == 1 || r_index == iter) {
      sampling_store(r_index);
      verbose_special(r_index);
    }

    if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
      parameters_store(r_index);
    }

    if (checkpoint_every > 0 && r_index % checkpoint_every == 0 &&
        !checkpoint_callback.is_none()) {
      sync_model_state();
      checkpoint_callback(r_index, model);
      model["W"] = py::none();
      model["Z"] = py::none();
      model["S"] = py::none();
    }

    if (verbose) {
    ++rate_window_iterations;
    std::cout << "\r[keyATM] " << r_index << "/" << iter;
    const bool report_rate =
        rate_window_iterations == rate_window_size || r_index == iter;
    if (report_rate) {
      const auto rate_window_end = std::chrono::steady_clock::now();
      const double elapsed_seconds =
          std::chrono::duration<double>(rate_window_end - rate_window_start).count();
      const double seconds_per_iteration =
          elapsed_seconds / static_cast<double>(rate_window_iterations);
      std::ostringstream rate;
      rate << std::fixed << std::setprecision(2) << seconds_per_iteration;
      std::cout << " | avg " << rate.str() << " s/iter (last "
                << rate_window_iterations << ")";
      rate_window_start = rate_window_end;
      rate_window_iterations = 0;
    }
    if (report_rate) {
      std::cout << std::endl;
    } else {
      std::cout << std::flush;
    }

    }

    // Optional: sleep or check interrupt
    // (No need to handle Ctrl+C in C++; Python handles it at top level)
  }

  sync_model_state();
}

void keyATMmeta::sync_model_state()
{
  model["W"] = py::cast(W);
  model["Z"] = py::cast(Z);
  model["S"] = py::cast(S);
  model["model_fit"] = model_fit;
  model["stored_values"] = stored_values;
}

void keyATMmeta::sampling_store(int r_index)
{
  TRACE_FUNC();

  const double loglik = loglik_total();
  const double denom = (total_words_weighted > 0) ? total_words_weighted : 1.0;
  const double perplexity = std::exp(-loglik / denom);

  const std::vector<double> rec{
      static_cast<double>(r_index), loglik, perplexity};

  py::gil_scoped_acquire gil;

  model_fit[py::str(std::to_string(r_index))] = py::cast(rec);
}

void keyATMmeta::parameters_store(int r_index)
{
  TRACE_FUNC();
  if (store_theta)
    store_theta_iter(r_index);

  if (store_pi)
    store_pi_iter(r_index);
}

void keyATMmeta::store_theta_iter(int r_index)
{
  TRACE_FUNC();
  py::list Z_tables_py = stored_values["Z_tables"].cast<py::list>();
  Z_tables_py.append(py::cast(n_dk_noWeight));
  stored_values["Z_tables"] = Z_tables_py;
}

void keyATMmeta::store_pi_iter(int r_index)
{
  TRACE_FUNC();
  py::list pi_vectors_py = stored_values["pi_vectors"].cast<py::list>();

  VectorXd numer = n_s1_k.array() + prior_gamma.col(0).array();
  VectorXd denom = n_s0_k.array() + prior_gamma.col(1).array() + numer.array();
  VectorXd pi = numer.array() / denom.array();

  pi_vectors_py.append(py::cast(pi));
  stored_values["pi_vectors"] = pi_vectors_py;
}

void keyATMmeta::verbose_special(int r_index)
{
  // If there is anything special to show, write here.
}
int keyATMmeta::sample_z(VectorXd &alpha, int z, int s, int w, int doc_id)
{
  int new_z = -1;
  double numerator, denominator, sum;

  assert(0 <= w && w < num_vocab);
  assert(0 <= z && z < num_topics);

  // remove
  if (s == 0)
  {
    n_s0_kv(z, w) -= vocab_weights(w);
    n_s0_k(z) -= vocab_weights(w);
  }
  else if (s == 1)
  {
    auto &ref = n_s1_kv.coeffRef(z, w);
    ref -= vocab_weights(w);
#ifdef DEBUG
    if (ref < -1e-12)
      std::cerr << "n_s1_kv<0 at (" << z << "," << w << "): " << ref << std::endl;
#endif
    n_s1_k(z) -= vocab_weights(w);
  }
  else
  {
    std::cerr << "Error at sample_z: invalid s (remove)" << std::endl;
  }
  n_dk(doc_id, z) -= vocab_weights(w);
  n_dk_noWeight(doc_id, z) -= 1.0;

  z_prob_vec.setZero();

  if (s == 0)
  {
    for (int k = 0; k < num_topics; ++k)
    {
      if (k < keyword_k)
      {
        numerator = (beta + n_s0_kv(k, w)) *
                    (n_s0_k(k) + prior_gamma(k, 1)) *
                    (n_dk(doc_id, k) + alpha(k));
        denominator = (Vbeta + n_s0_k(k)) *
                      (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));
      }
      else
      {
        // Regular topics have no keyword-route mass. Applying the route
        // factor would yield 0/0 when the regular topic is empty.
        numerator = (beta + n_s0_kv(k, w)) *
                    (n_dk(doc_id, k) + alpha(k));
        denominator = Vbeta + n_s0_k(k);
      }
      z_prob_vec(k) = numerator / denominator;
    }
    sum = z_prob_vec.sum();
    // if (sum <= 0.0)
    // {
    // }
    // else
    // {
      if (diagnostic_uniform >= 0.0) {
      double acc = 0.0;
      double draw = diagnostic_uniform * sum;
      new_z = 0;
      for (int k = 0; k < num_topics; ++k) { acc += z_prob_vec(k); if (draw < acc) { new_z = k; break; } }
    } else {
      new_z = sampler::rcat_without_normalize(z_prob_vec, sum, num_topics);
    }
    // }
  }
  else
  { // s == 1
    for (int k = 0; k < num_topics; ++k)
    {
      if (keywords[k].find(w) == keywords[k].end())
      {
        z_prob_vec(k) = 0.0;
        continue;
      }
      double n_s1_kw = n_s1_kv.coeff(k, w);
      numerator = (beta_s + n_s1_kw) *
                  (n_s1_k(k) + prior_gamma(k, 0)) *
                  (n_dk(doc_id, k) + alpha(k));
      denominator = (Lbeta_sk(k) + n_s1_k(k)) *
                    (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));
      z_prob_vec(k) = numerator / denominator;
    }
    sum = z_prob_vec.sum();
    if (sum <= 0.0)
    {
      if (keywords[z].find(w) != keywords[z].end())
      {
        new_z = z;
      }
      else
      {
        std::vector<int> cand;
        cand.reserve(num_topics);
        for (int k = 0; k < num_topics; ++k)
          if (keywords[k].find(w) != keywords[k].end())
            cand.push_back(k);
        if (cand.empty())
          for (int k = 0; k < num_topics; ++k)
            cand.push_back(k);
        new_z = cand[std::rand() % cand.size()];
      }
    }
    else
    {
      if (diagnostic_uniform >= 0.0) {
      double acc = 0.0;
      double draw = diagnostic_uniform * sum;
      new_z = 0;
      for (int k = 0; k < num_topics; ++k) { acc += z_prob_vec(k); if (draw < acc) { new_z = k; break; } }
    } else {
      new_z = sampler::rcat_without_normalize(z_prob_vec, sum, num_topics);
    }
    }
  }

  // add back
  if (s == 0)
  {
    n_s0_kv(new_z, w) += vocab_weights(w);
    n_s0_k(new_z) += vocab_weights(w);
  }
  else if (s == 1)
  {
    n_s1_kv.coeffRef(new_z, w) += vocab_weights(w);
    n_s1_k(new_z) += vocab_weights(w);
  }
  else
  {
    std::cerr << "Error at sample_z: invalid s (add)" << std::endl;
  }
  n_dk(doc_id, new_z) += vocab_weights(w);
  n_dk_noWeight(doc_id, new_z) += 1.0;

  return new_z;
}

int keyATMmeta::sample_s(int z, int s, int w, int doc_id)
{
  int new_s;
  double numerator, denominator;
  double s0_prob;
  double s1_prob;
  double sum;

  // remove data
  if (s == 0)
  {
    n_s0_kv(z, w) -= vocab_weights(w);
    n_s0_k(z) -= vocab_weights(w);
  }
  else
  {
    n_s1_kv.coeffRef(z, w) -= vocab_weights(w);
    n_s1_k(z) -= vocab_weights(w);
  }

  // newprob_s1()
  double n_s1_kw = n_s1_kv.coeff(z, w);
  numerator = (beta_s + n_s1_kw) * (n_s1_k(z) + prior_gamma(z, 0));

  denominator = (Lbeta_sk(z) + n_s1_k(z));
  s1_prob = numerator / denominator;

  // newprob_s0()
  numerator = (beta + n_s0_kv(z, w)) *
              (n_s0_k(z) + prior_gamma(z, 1));

  denominator = (Vbeta + n_s0_k(z));
  s0_prob = numerator / denominator;

  // Normalize
  sum = s0_prob + s1_prob;
  if (sum <= 0.0)
  {
    new_s = s;
  }
  else
  {
    double p1 = s1_prob / sum;
    last_s_prob = Eigen::Vector2d(1.0 - p1, p1);
    auto &rng = sampler::rng();
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    new_s = diagnostic_uniform >= 0.0 ? (diagnostic_uniform <= p1 ? 1 : 0) : ((unif(rng) <= p1) ? 1 : 0);
  }

  // add back data counts
  if (new_s == 0)
  {
    n_s0_kv(z, w) += vocab_weights(w);
    n_s0_k(z) += vocab_weights(w);
  }
  else
  {
    n_s1_kv.coeffRef(z, w) += vocab_weights(w);
    n_s1_k(z) += vocab_weights(w);
  }

  return new_s;
}

// Gamma PDF in log scale: log(p(x; a, b)) where a=shape, b=scale
double keyATMmeta::gammapdfln(const double x, const double a, const double b) {
  return -a * std::log(b) - mylgamma(a) + (a - 1.0) * std::log(x) - x / b;
}

// Beta PDF in normal scale
double keyATMmeta::betapdf(const double x, const double a, const double b) {
  return std::tgamma(a + b) / (std::tgamma(a) * std::tgamma(b)) *
         std::pow(x, a - 1) * std::pow(1.0 - x, b - 1);
}

// Beta PDF in log scale
double keyATMmeta::betapdfln(const double x, const double a, const double b) {
  return (a - 1) * std::log(x) +
         (b - 1) * std::log(1.0 - x) +
         mylgamma(a + b) - mylgamma(a) - mylgamma(b);
}

std::vector<double> keyATMmeta::alpha_reformat(const Eigen::VectorXd& alpha, int num_topics)
{
    std::vector<double> alpha_rvec(num_topics);
    for (int i = 0; i < num_topics; ++i) {
        alpha_rvec[i] = alpha(i);
    }
    return alpha_rvec;
}

double keyATMmeta::gammaln_frac(const double value, const int count) {
  if (count > 19) {
    return mylgamma(value + count) - mylgamma(value);
  } else {
    double gammaln_val = 0.0;
    for (int i = 0; i < count; ++i) {
      gammaln_val += std::log(value + i);
    }
    return gammaln_val;
  }
}

py::dict keyATMmeta::return_model()
{
  TRACE_FUNC();
  return model;
}
