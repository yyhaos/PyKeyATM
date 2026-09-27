
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include "read.h"

namespace py = pybind11;
using namespace Eigen;
using std::string;
using std::vector;

std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>>
read_dfm_cpp(const Eigen::SparseMatrix<int, Eigen::ColMajor>& dfm_ori,
             const std::vector<std::string>& vocab,
             double split)
{
    std::vector<std::vector<std::string>> W_raw;
    std::vector<std::vector<std::string>> W_split;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    const Eigen::SparseMatrix<int, Eigen::ColMajor> dfm = dfm_ori.transpose();

    int doc_num = dfm.cols();

    for (int doc_id = 0; doc_id < doc_num; ++doc_id) {
        std::vector<std::string> doc_words;
        std::vector<std::string> doc_words_split;

        for (Eigen::SparseMatrix<int>::InnerIterator it(dfm, doc_id); it; ++it) {
            std::string word_id = vocab[it.row()];
            int count = it.value();

            for (int i = 0; i < count; ++i) {
                double u = dis(gen);
                if (split != 0.0 && u < split) {
                    doc_words_split.push_back(word_id);
                } else {
                    doc_words.push_back(word_id);
                }
            }
        }

        W_raw.push_back(doc_words);
        if (split != 0.0) {
            W_split.push_back(doc_words_split);
        }
    }

    return std::make_pair(W_raw, W_split);
}
