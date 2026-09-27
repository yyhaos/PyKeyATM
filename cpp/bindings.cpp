#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "train.h"
#include "read.h"
#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;


PYBIND11_MODULE(keyATM_scr, m) {
    m.doc() = "keyATM training module (compiled from C++)";

    m.def("read_dfm_cpp", &read_dfm_cpp, "Read DFM-like matrix");
    m.def("keyATM_fit_base", &keyATM_fit_base, "Run keyATM base model",
          py::arg("model"), py::arg("resume"),
          py::arg("checkpoint_every") = 0,
          py::arg("checkpoint_callback") = py::none());
    m.def("keyATM_fit_HMM", &keyATM_fit_HMM, "Run keyATM HMM model",
          py::arg("model"), py::arg("resume"),
          py::arg("checkpoint_every") = 0,
          py::arg("checkpoint_callback") = py::none());
    m.def("keyATM_fit_LDA", &keyATM_fit_LDA, "Run keyATM LDA model");
    m.def("keyATM_fit_cov", &keyATM_fit_cov, "Run keyATM cov model");
    m.def("keyATM_diagnostic_sweep", &keyATM_diagnostic_sweep, "Run deterministic keyATM diagnostic sweep");
    m.def("keyATM_fit_LDAcov", &keyATM_fit_LDAcov, "Run keyATM cov model");
    m.def("keyATM_fit_LDAHMM", &keyATM_fit_LDAHMM, "Run keyATM cov model");
}
