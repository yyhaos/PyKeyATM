#pragma once

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning(disable : 4819)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
namespace py = pybind11;


namespace keyatm {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;

class keyATMmeta {
  public:
    // Parameters
    int iter = 0, llk_per = 0, verbose = 0;
    std::string weights_type;
    double eta_1 = 0.0, eta_2 = 0.0, eta_1_regular = 0.0, eta_2_regular = 0.0;
    int use_weights = 0, store_theta = 0, store_pi = 0, thinning = 1;
    double slice_A = 1.0;
    // Data (Python dicts will be passed in for these)
    py::dict model;

    // Native token state used throughout sampling.
    std::vector<std::vector<int>> W, Z, S;
    std::vector<std::vector<int>> keywords_list;

    std::vector<std::string> vocab; // from Python list/tuple
    std::vector<double> nv_alpha;   // from Python list/ndarray

    // matrices / vectors (Eigen)
    Eigen::MatrixXd prior_gamma;
    double beta = 0.01, beta_s = 0.01, Vbeta = 1.0;
    int regular_k = 0, keyword_k = 0;

    py::dict model_fit;
    std::vector<int> doc_each_len;
    std::vector<double> doc_each_len_weighted;

    int num_vocab = 0, num_doc = 0, total_words = 0;
    double total_words_weighted = 0.0;

    Eigen::MatrixXd beta_s0kv;
    Eigen::SparseMatrix<double, Eigen::RowMajor> beta_s1kv;
    Eigen::VectorXd Vbeta_k, Lbeta_sk;

    py::dict options_list;
    py::dict Z_tables;
    py::dict priors_list;
    py::dict model_settings;
    py::dict stored_values;
    int checkpoint_every = 0;
    py::object checkpoint_callback = py::none();

    Eigen::MatrixXd Z_table;

    // alpha
    int num_topics = 0;
    Eigen::VectorXd alpha;

    std::vector<std::unordered_set<int>> keywords;
    std::vector<int> keywords_num;

    // Latent Variables
    Eigen::MatrixXd n_s0_kv;
    Eigen::SparseMatrix<double, Eigen::RowMajor> n_s1_kv;
    Eigen::MatrixXd n_dk, n_dk_noWeight;
    Eigen::VectorXd n_s0_k, n_s1_k;
    Eigen::VectorXd vocab_weights;

    // Iteration use
    std::vector<int> doc_indexes;
    std::vector<int> token_indexes;

    Eigen::VectorXd z_prob_vec;
    double diagnostic_uniform = -1.0;
    Eigen::VectorXd last_s_prob;
    double min_v = 0.0, max_v = 0.0;
    int max_shrink_time = 0;

    void dd()
    {
      auto print_vec = [](const auto &v, const char *name)
      {
        printf("%s: [", name);
        for (size_t i = 0; i < v.size() && i < 5; i++)
        {
          printf("%g", (double)v[i]);
          if (i + 1 < v.size() && i < 4)
            printf(", ");
        }
        if (v.size() > 5)
          printf(", ...");
        printf("] (size=%zu)\n", v.size());
      };

      auto print_vecvec = [&](const std::vector<std::vector<int>> &vv, const char *name)
      {
        printf("%s: [", name);
        for (size_t i = 0; i < vv.size() && i < 5; i++)
        {
          const auto &vvi = vv[i];
          printf("[");
          for (size_t j = 0; j < vvi.size() && j < 5; j++)
          {
            printf("%d", vvi[j]);
            if (j + 1 < vvi.size() && j < 4)
              printf(", ");
          }
          if (vvi.size() > 5)
            printf(", ...");
          printf("]");
          if (i + 1 < vv.size() && i < 4)
            printf(", ");
        }
        if (vv.size() > 5)
          printf(", ...");
        printf("] (outer size=%zu)\n", vv.size());
      };


      // vocab
      printf("vocab: [");
      for (size_t i = 0; i < vocab.size() && i < 5; i++)
      {
        printf("%s", vocab[i].c_str());
        if (i + 1 < vocab.size() && i < 4)
          printf(", ");
      }
      if (vocab.size() > 5)
        printf(", ...");
      printf("] (size=%zu)\n", vocab.size());

      // prior_gamma
      printf("prior_gamma shape=(%d,%d)\n", (int)prior_gamma.rows(), (int)prior_gamma.cols());
      for (int i = 0; i < std::min(5, (int)prior_gamma.rows()); i++)
      {
        for (int j = 0; j < std::min(5, (int)prior_gamma.cols()); j++)
        {
          printf("%g ", prior_gamma(i, j));
        }
        if (prior_gamma.cols() > 5)
          printf("...");
        printf("\n");
      }
      if (prior_gamma.rows() > 5)
        printf("...\n");

      printf("beta=%g, beta_s=%g, Vbeta=%g, regular_k=%d, keyword_k=%d\n",
             beta, beta_s, Vbeta, regular_k, keyword_k);

      printf("model_fit keys: ");
      for (auto item : model_fit)
      {
        std::string key = py::cast<std::string>(item.first);
        printf("%s ", key.c_str());
      }
      printf("\n");

      print_vec(doc_each_len, "doc_each_len");

      print_vec(doc_each_len_weighted, "doc_each_len_weighted");

      printf("===== END DEBUG PRINT =====\n");
    }
  // Constructor/Destructor
  keyATMmeta() = default;
  explicit keyATMmeta(const py::dict& model_);
  virtual ~keyATMmeta();

  // Main fit routines
  void fit();
  void resume_fit();
  void configure_checkpoint(int every, const py::object& callback);

  // Initialization
  void read_data();
  virtual void read_data_common();
  virtual void read_data_specific() {}

  void initialize();
  virtual void initialize_common();
  virtual void initialize_specific() = 0;

  void weights_invfreq();
  void weights_inftheory();
  void weights_normalize_total();

  void resume_initialize();
  virtual void resume_initialize_specific() {}


  // Sampling
  void iteration();
  void sync_model_state();
  virtual void iteration_single(int it) = 0;
  virtual void sample_parameters(int it) = 0;

  virtual int sample_z(VectorXd& alpha, int z, int s, int w, int doc_id);
  int sample_s(int z, int s, int w, int doc_id);


  void sampling_store(int r_index);
  virtual void parameters_store(int r_index);
  void store_theta_iter(int r_index);
  void store_pi_iter(int r_index);
  virtual void verbose_special(int r_index);
  virtual double loglik_total() = 0;

  // Math Utilities
  double gammapdfln(double x, double a, double b);
  double betapdf(double x, double a, double b);
  double betapdfln(double x, double a, double b);
  std::vector<double> alpha_reformat(const Eigen::VectorXd &alpha, int num_topics);
  double gammaln_frac(double value, int count);

    //
    // Inline functions
    //

    // Slice sampling
    double expand(const double p, const double A)
    {
      return (-(1.0/A) * log((1.0/p) - 1.0));
    };

    double shrink(const double x, const double A)
    {
      return (1.0 / (1.0 + exp(-A*x)));
    };

    double shrinkp(const double x)
    {
      return (x / (1.0 + x));
    };


    // Log-sum-exp
    double logsumexp(double x, double y, bool flg)
    {
      if (flg) return y; // init mode
      if (x == y) return x + 0.69314718055; // log(2)
      double vmin = std::min (x, y);
      double vmax = std::max (x, y);
      if (vmax > vmin + 50) {
        return vmax;
      } else {
        return vmax + std::log (std::exp (vmin - vmax) + 1.0);
      }
    };

    double logsumexp_Eigen(VectorXd &vec, const int size){
      double vmax = vec.maxCoeff();
      double sum = 0.0;

      for(int i = 0; i < size; ++i){
        sum += exp(vec(i) - vmax);
      }

      return vmax + log(sum);
    }

    // Approximations
    double mylgamma(const double x){
      // gammaln_val = 0.0;
      // gammaln_val = lgamma(x);

      // Good approximation when x > 1
      //    x > 1: max abs err: 2.272e-03
      //    x > 0.5: 0.012
      //    x > 0.6: 0.008
      // Abramowitz and Stegun p.257

      if(x < 0.6)
        return (lgamma(x));
      else
        return ((x-0.5)*log(x) - x + 0.91893853320467 + 1/(12*x));
    };

    // Approximations below are not used
    double mypow(const double a, const double b){
      // Reference: https://github.com/ekmett/approximate/blob/master/cbits/fast.c
      // Probably not good to use if b>1.0

      if(b > 1.0)
        return(pow(a,b));

      union { double d; long long x; } u = { a };
      u.x = (long long)(b * (u.x - 4606921278410026770LL) + 4606921278410026770LL);
      return u.d;
    };

    double myexp(const double a){
      // Seems to be not very good
      union { double d; long long x; } u, v;
      u.x = (long long)(3248660424278399LL * a + 0x3fdf127e83d16f12LL);
      v.x = (long long)(0x3fdf127e83d16f12LL - 3248660424278399LL * a);
      return u.d / v.d;
    };

    double mylog(const double a){
      // Looks fine even with large a
      union { double d; long long x; } u = { a };
      return (u.x - 4606921278410026770) * 1.539095918623324e-16;
    };

    // Export
    py::dict return_model();
};

}
