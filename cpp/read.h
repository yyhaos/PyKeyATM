#pragma once

#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <random>

// Function declaration
std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>>
read_dfm_cpp(const Eigen::SparseMatrix<int, Eigen::ColMajor>& dfm_ori,
             const std::vector<std::string>& vocab,
             double split);
