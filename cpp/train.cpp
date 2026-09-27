#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include "train.h"

namespace py = pybind11;
using namespace keyatm;

static void seed_model_rng(const py::dict &model)
{
    py::dict options = py::cast<py::dict>(model["options"]);
    sampler::seed_rng(py::cast<std::uint32_t>(options["seed"]));
}

// -----------------------------
// Python-wrapped Sampler APIs
// -----------------------------

py::dict keyATM_fit_base(py::dict model, bool resume, int checkpoint_every,
                         py::object checkpoint_callback)
{
    TRACE_FUNC();
    seed_model_rng(model);
    keyATMbase m(model);
    m.configure_checkpoint(checkpoint_every, checkpoint_callback);
    if (resume)
        m.resume_fit();
    else
        m.fit();
    model = m.return_model();
    return model;
}

py::dict keyATM_fit_HMM(py::dict model, bool resume, int checkpoint_every,
                        py::object checkpoint_callback) {

    TRACE_FUNC();
    seed_model_rng(model);
    keyATMhmm m(model);
    m.configure_checkpoint(checkpoint_every, checkpoint_callback);
    if (resume)
        m.resume_fit();
    else
        m.fit();
    model = m.return_model();
    return model;
}

py::dict keyATM_fit_cov(py::dict model, bool resume = false)
{
    TRACE_FUNC();
    seed_model_rng(model);
    keyATMcov m(model);
    if (resume) m.resume_fit(); else
        m.fit();
    return m.return_model();
}


template <typename Model>
static py::dict run_diagnostic_sweep(Model& m, Eigen::MatrixXd alpha_by_doc,
                                     const std::vector<double>& uniforms_z,
                                     const std::vector<double>& uniforms_s,
                                     const py::dict& parameters)
{
    m.read_data();
    m.initialize();
    const int D = m.num_doc, K = m.num_topics;
    std::size_t N = 0;
    for (const auto& doc : m.W) N += doc.size();
    if (alpha_by_doc.rows() != D || alpha_by_doc.cols() != K || uniforms_z.size() != N || uniforms_s.size() != N)
        throw std::runtime_error("diagnostic dimensions do not match model");
    if constexpr (std::is_same_v<Model, keyATMbase>) {
        m.alpha = alpha_by_doc.row(0).transpose();
    } else if constexpr (std::is_same_v<Model, keyATMcov>) {
        m.Lambda = parameters["Lambda"].cast<Eigen::MatrixXd>();
        m.Alpha = (m.C * m.Lambda.transpose()).array().exp();
        alpha_by_doc = m.Alpha;
    } else if constexpr (std::is_same_v<Model, keyATMhmm>) {
        m.R_est = parameters["states"].cast<Eigen::VectorXi>();
        m.P_est = parameters["transition"].cast<Eigen::MatrixXd>();
        m.alphas = parameters["alphas"].cast<Eigen::MatrixXd>();
        for (int d = 0; d < D; ++d) alpha_by_doc.row(d) = m.alphas.row(m.get_state_index(d));
    }
    Eigen::MatrixXd z_probs = Eigen::MatrixXd::Zero(N, K);
    Eigen::MatrixXd s_probs = Eigen::MatrixXd::Constant(N, 2, std::numeric_limits<double>::quiet_NaN());
    std::vector<int> z_after, s_after;
    z_after.reserve(N); s_after.reserve(N);
    std::size_t token = 0;
    for (int d = 0; d < D; ++d) for (std::size_t j = 0; j < m.W[d].size(); ++j, ++token) {
        const int w = m.W[d][j], old_z = m.Z[d][j], old_s = m.S[d][j];
        Eigen::VectorXd alpha = alpha_by_doc.row(d).transpose();
        m.diagnostic_uniform = uniforms_z[token];
        const int new_z = m.sample_z(alpha, old_z, old_s, w, d);
        z_probs.row(token) = m.z_prob_vec.transpose();
        int new_s = old_s;
        if (m.keywords[new_z].find(w) != m.keywords[new_z].end()) {
            m.diagnostic_uniform = uniforms_s[token];
            new_s = m.sample_s(new_z, old_s, w, d);
            s_probs.row(token) = m.last_s_prob.transpose();
        }
        m.Z[d][j] = new_z; m.S[d][j] = new_s;
        z_after.push_back(new_z); s_after.push_back(new_s);
    }
    m.diagnostic_uniform = -1.0;
    py::dict out;
    out["z_probs"] = z_probs; out["s_probs"] = s_probs;
    out["z_after"] = z_after; out["s_after"] = s_after;
    out["n_dk"] = m.n_dk; out["n_s0_kv"] = m.n_s0_kv;
    out["n_s1_kv"] = Eigen::MatrixXd(m.n_s1_kv);
    out["n_s0_k"] = m.n_s0_k; out["n_s1_k"] = m.n_s1_k;
    out["log_likelihood"] = m.loglik_total();
    py::dict fixed_parameters;
    if constexpr (std::is_same_v<Model, keyATMbase>) {
        fixed_parameters["alpha_by_doc"] = alpha_by_doc;
    } else if constexpr (std::is_same_v<Model, keyATMcov>) {
        fixed_parameters["Lambda"] = m.Lambda;
        fixed_parameters["alpha_by_doc"] = m.Alpha;
    } else if constexpr (std::is_same_v<Model, keyATMhmm>) {
        fixed_parameters["states"] = m.R_est;
        fixed_parameters["alphas"] = m.alphas;
        fixed_parameters["transition"] = m.P_est;
    }
    out["fixed_parameters"] = fixed_parameters;
    m.model["W"] = m.W; m.model["Z"] = m.Z; m.model["S"] = m.S;
    out["model"] = m.return_model();
    return out;
}

py::dict keyATM_diagnostic_sweep(py::dict model, std::string specification,
                                 Eigen::MatrixXd alpha_by_doc,
                                 std::vector<double> uniforms_z,
                                 std::vector<double> uniforms_s, py::dict parameters)
{
    seed_model_rng(model);
    if (specification == "base") { keyATMbase m(model); return run_diagnostic_sweep(m, alpha_by_doc, uniforms_z, uniforms_s, parameters); }
    if (specification == "cov") { keyATMcov m(model); return run_diagnostic_sweep(m, alpha_by_doc, uniforms_z, uniforms_s, parameters); }
    if (specification == "hmm") { keyATMhmm m(model); return run_diagnostic_sweep(m, alpha_by_doc, uniforms_z, uniforms_s, parameters); }
    throw std::invalid_argument("specification must be base, cov, or hmm");
}

py::dict keyATM_fit_LDA(py::dict model, bool resume = false)
{
    TRACE_FUNC();
    seed_model_rng(model);
    LDAweight m(model);
    if (resume)
        m.resume_fit();
    else
        m.fit();
    return m.return_model();
}

py::dict keyATM_fit_LDAcov(py::dict model, bool resume = false)
{
    TRACE_FUNC();
    seed_model_rng(model);
    LDAcov m(model);
    if (resume) m.resume_fit(); else
        m.fit();
    return m.return_model();
}

py::dict keyATM_fit_LDAHMM(py::dict model, bool resume = false)
{
    TRACE_FUNC();
    seed_model_rng(model);
    LDAhmm m(model);
    if (resume) m.resume_fit(); else
        m.fit();
    return m.return_model();
}
