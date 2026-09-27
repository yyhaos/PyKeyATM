import unittest
import numpy as np
from pykeyatm.python.posterior import (keyATM_output_phi_calc_key,
    keyATM_output_phi_calc_lda, keyATM_output_pi)


class PosteriorProbabilityTest(unittest.TestCase):
    def test_empty_lda_topic_retains_prior_and_row(self):
        result = keyATM_output_phi_calc_lda(["a", "a"], [0, 0], ["a", "b"], .1, ["first", "empty"])
        np.testing.assert_allclose(result["phi"].iloc[1], [.5, .5])
        np.testing.assert_array_equal(result["topic_counts"], [2, 0])

    def test_keyword_support_and_empty_regular_topic(self):
        vocab = ["a", "b", "c"]
        gamma = [[1, 1], [1, 1], [0, 0]]
        pi = keyATM_output_pi([[0, 1]], [[1, 1]], gamma)
        np.testing.assert_allclose(pi["Proportion"], [200/3, 200/3, 0])
        model = {"keywords": {"first": [0], "second": [1]},
                 "keywords_raw": {"first": ["a"], "second": ["b"]}}
        result = keyATM_output_phi_calc_key(["a", "b"], [0, 1], [1, 1], pi,
            model["keywords_raw"], vocab, {"beta": .1, "beta_s": .2},
            ["first", "second", "regular"], model)
        expected = np.array([[7/9, 1/9, 1/9], [1/9, 7/9, 1/9], [1/3, 1/3, 1/3]])
        np.testing.assert_allclose(result["phi"], expected)
        np.testing.assert_array_equal(result["topic_counts"], [1, 1, 0])
