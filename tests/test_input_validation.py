import copy
import subprocess
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

from pykeyatm import keyATM, keyATM_read, weightedLDA


class InputValidationTest(unittest.TestCase):
    def docs(self):
        return keyATM_read(pd.DataFrame([[2, 1], [1, 2]], columns=["a", "b"]))

    def fit(self, **kwargs):
        return weightedLDA(self.docs(), model="base", number_of_topics=2, **kwargs)

    def test_reject_invalid_counts(self):
        for value in (-1, 0.5, np.nan, np.inf, 2**31, 1 + 2j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                keyATM_read(pd.DataFrame([[value, 1]], columns=["a", "b"]))

    def test_reject_invalid_vocabulary(self):
        for columns in (["a", "a"], ["", "b"], [0, 1]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                keyATM_read(pd.DataFrame([[1, 2]], columns=columns))

    def test_reject_invalid_split(self):
        for split in (-0.1, 1, np.inf, np.nan):
            with self.subTest(split=split), self.assertRaises(ValueError):
                keyATM_read(pd.DataFrame([[1]], columns=["a"]), split=split)

    def test_integer_valued_float_counts_and_doc_ids(self):
        dtm = pd.DataFrame({"doc_id": ["d1", "d2"], "a": [2., 1.]})
        docs = keyATM_read(dtm, keep_docnames=True)
        self.assertEqual(docs["docnames"], ["d1", "d2"])
        self.assertEqual(docs["W_raw"], [["a", "a"], ["a"]])
        self.assertIn("doc_id", dtm)

    def test_reject_zero_sampling_intervals(self):
        for option in ("thinning", "llk_per"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.fit(options={option: 0, "iterations": 2})

    def test_reject_invalid_topics(self):
        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                weightedLDA(self.docs(), model="base", number_of_topics=count, options={"iterations": 2})

    def test_reject_invalid_priors_and_slice_bounds(self):
        for priors in ({"beta": -1}, {"alpha": [1, 0]}, {"eta_1_regular": np.nan}):
            with self.subTest(priors=priors), self.assertRaises(ValueError):
                self.fit(priors=priors, options={"iterations": 2})
        with self.assertRaises(ValueError):
            self.fit(model_settings={"slice_min": 2, "slice_max": 1}, options={"iterations": 2})

    def test_fits_do_not_mutate_documents_or_options(self):
        dtm = pd.DataFrame({"doc_id": ["one", "empty", "three"], "a": [2, 0, 1], "b": [1, 0, 2]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            docs = keyATM_read(dtm, keep_docnames=True)
            original = copy.deepcopy(docs)
            options = {"seed": 42, "iterations": 5, "verbose": False}
            for _ in range(2):
                fit = weightedLDA(docs, model="base", number_of_topics=2, options=options)
                self.assertEqual(fit["N"], 2)
                self.assertEqual(list(fit["theta"].index), ["one", "three"])
                self.assertEqual(docs, original)
            self.assertEqual(options, {"seed": 42, "iterations": 5, "verbose": False})

    def test_reader_check_false_can_be_fitted(self):
        docs = keyATM_read(pd.DataFrame([[2, 1], [1, 2]], columns=["a", "b"]), check=False)
        self.assertEqual(weightedLDA(docs, model="base", number_of_topics=2, options={"iterations": 3})["N"], 2)

    def test_empty_corpus_is_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            docs = keyATM_read(pd.DataFrame([[0, 0]], columns=["a", "b"]))
            with self.assertRaisesRegex(ValueError, "nonempty document"):
                weightedLDA(docs, model="base", number_of_topics=2)

    def test_pg_reports_unsupported_sampler(self):
        with self.assertRaisesRegex(NotImplementedError, "DirMulti"):
            keyATM(self.docs(), model="covariates", no_keyword_topics=1,
                   keywords={"a": ["a"]},
                   model_settings={"covariates_data": pd.DataFrame({"x": [0, 1]}), "covariates_model": "PG"})

    def test_verbose_false_is_quiet_including_native_code(self):
        result = subprocess.run([sys.executable, "-c", "from pykeyatm import keyATM, keyATM_read; import pandas as pd; "
            "docs=keyATM_read(pd.DataFrame([[2,1],[1,2]],columns=['a','b'])); "
            "keyATM(docs,model='base',keywords={'a':['a']},no_keyword_topics=1,options={'iterations':3,'verbose':False})"],
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
