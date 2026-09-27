#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "sampler.h"
#include "keyATM_base.h"
#include "keyATM_HMM.h"
#include "keyATM_cov.h"
#include "LDA_base.h"
#include "LDA_weight.h"
#include "LDA_weightCov.h"
#include "LDA_weightHMM.h"

namespace py = pybind11;

// Declare keyATM fitting interface
py::dict keyATM_fit_base(py::dict model, bool resume, int checkpoint_every = 0,
                         py::object checkpoint_callback = py::none());

// Uncomment the following when you implement the others:

py::dict keyATM_fit_cov(py::dict model, bool resume);
// py::dict keyATM_fit_covPG(py::dict model, bool resume = false);
py::dict keyATM_fit_HMM(py::dict model, bool resume, int checkpoint_every = 0,
                        py::object checkpoint_callback = py::none());
py::dict keyATM_diagnostic_sweep(py::dict model, std::string specification, Eigen::MatrixXd alpha_by_doc, std::vector<double> uniforms_z, std::vector<double> uniforms_s, py::dict parameters);
py::dict keyATM_fit_LDA(py::dict model, bool resume);
py::dict keyATM_fit_LDAcov(py::dict model, bool resume);
py::dict keyATM_fit_LDAHMM(py::dict model, bool resume);
