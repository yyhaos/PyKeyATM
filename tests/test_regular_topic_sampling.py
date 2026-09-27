import unittest

import numpy as np

from pykeyatm import keyATM


class RegularTopicSamplingTest(unittest.TestCase):
    def test_empty_regular_topic_uses_finite_base_probability(self):
        docs = {
            "W_raw": [["apple", "apple", "pear", "x"], ["pear", "x", "x"]],
            "wd_names": ["apple", "pear", "x"],
            "doc_index": [1, 2],
        }
        fitted = keyATM(
            docs=docs,
            model="base",
            no_keyword_topics=1,
            keywords={"fruit": ["apple", "pear"]},
            options={
                "use_cache": False,
                "seed": 19,
                "iterations": 20,
                "thinning": 2,
                "verbose": False,
                "use_weights": False,
            },
        )

        self.assertTrue(np.isfinite(fitted["model_fit"]["Log Likelihood"]).all())
        self.assertGreater(fitted["topic_counts"][1], 0)


if __name__ == "__main__":
    unittest.main()
