import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pykeyatm import keyATM, keyATM_read, weightedLDA


class PackageSmokeTest(unittest.TestCase):
    def make_docs(self):
        dtm = pd.DataFrame(
            [[3, 1, 0, 2], [0, 2, 3, 1], [2, 0, 1, 3], [1, 2, 0, 2]],
            columns=["policy", "health", "market", "public"],
        )
        return keyATM_read(dtm)

    def fit_keyatm(self, model, model_settings=None):
        return keyATM(
            docs=self.make_docs(),
            model=model,
            no_keyword_topics=1,
            keywords={"policy": ["policy"], "health": ["health"]},
            model_settings=model_settings or {},
            options={
                "seed": 42,
                "iterations": 12,
                "thinning": 2,
                "verbose": False,
                "use_weights": False,
            },
        )

    def run_without_artifacts(self, fit_function):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            try:
                os.chdir(workdir)
                fit = fit_function()
                self.assertEqual(os.listdir(workdir), [])
            finally:
                os.chdir(original_cwd)
        log_likelihood = np.asarray(fit["model_fit"]["Log Likelihood"], dtype=float)
        self.assertTrue(log_likelihood.size > 0)
        self.assertTrue(np.isfinite(log_likelihood).all())

    def test_weighted_models_return_normalized_probabilities(self):
        for model in ("base", "covariates", "dynamic"):
            settings = {}
            if model == "covariates":
                settings = {"covariates_data": pd.DataFrame({"group": [0, 1, 0, 1]})}
            if model == "dynamic":
                settings = {"num_states": 2, "time_index": [1, 1, 2, 2]}
            for weighting in ("information-theory", "information-theory-normalized", "inv-freq", "inv-freq-normalized"):
                for family in ("keyATM", "LDA"):
                    with self.subTest(model=model, weighting=weighting, family=family):
                        options = {"seed": 42, "iterations": 8, "verbose": False, "weights_type": weighting}
                        if family == "keyATM":
                            fit = keyATM(self.make_docs(), model=model, no_keyword_topics=1,
                                         keywords={"policy": ["policy"], "health": ["health"]},
                                         model_settings=settings, options=options)
                        else:
                            fit = weightedLDA(self.make_docs(), model=model, number_of_topics=3,
                                              model_settings=settings, options=options)
                        for field in ("theta", "phi"):
                            values = np.asarray(fit[field], dtype=float)
                            self.assertTrue(np.isfinite(values).all())
                            self.assertTrue((values >= 0).all())
                            np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-10)
                        if model == "dynamic":
                            states = fit["values_iter"]["R_iter_last"]
                            self.assertEqual(states[0], 1)
                            self.assertEqual(states[-1], 2)
                            self.assertTrue(np.isin(np.diff(states), [0, 1]).all())

    def test_public_reader_creates_documents(self):
        docs = self.make_docs()
        self.assertEqual(len(docs["W_raw"]), 4)
        self.assertEqual(set(docs["wd_names"]), {"policy", "health", "market", "public"})

    def test_keyatm_base(self):
        self.run_without_artifacts(lambda: self.fit_keyatm("base"))

    def test_same_seed_reproduces_model_output(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            try:
                os.chdir(workdir)
                first = self.fit_keyatm("base")
                second = self.fit_keyatm("base")
                np.testing.assert_array_equal(
                    first["theta"].to_numpy(), second["theta"].to_numpy()
                )
                self.assertEqual(os.listdir(workdir), [])
            finally:
                os.chdir(original_cwd)

    def test_keyatm_covariates(self):
        settings = {"covariates_data": pd.DataFrame({"group": [0, 1, 0, 1]})}
        self.run_without_artifacts(lambda: self.fit_keyatm("covariates", settings))

    def test_keyatm_dynamic(self):
        settings = {"num_states": 2, "time_index": [1, 1, 2, 2]}
        self.run_without_artifacts(lambda: self.fit_keyatm("dynamic", settings))

    def test_lda_base(self):
        self.run_without_artifacts(lambda: self.fit_lda("base"))

    def fit_lda(self, model, model_settings=None):
        return weightedLDA(
            docs=self.make_docs(),
            model=model,
            number_of_topics=2,
            model_settings=model_settings or {},
            options={
                "seed": 42,
                "iterations": 12,
                "thinning": 2,
                "verbose": False,
                "use_weights": False,
            },
        )

    def test_lda_covariates(self):
        settings = {"covariates_data": pd.DataFrame({"group": [0, 1, 0, 1]})}
        self.run_without_artifacts(lambda: self.fit_lda("covariates", settings))

    def test_lda_dynamic(self):
        settings = {"num_states": 2, "time_index": [1, 1, 2, 2]}
        self.run_without_artifacts(lambda: self.fit_lda("dynamic", settings))


if __name__ == "__main__":
    unittest.main()
