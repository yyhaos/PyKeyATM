"""Fit the base keyATM model to a tiny synthetic document-term matrix."""
import pandas as pd

from pykeyatm import keyATM, keyATM_read
from _summary import print_fit_summary


dtm = pd.DataFrame(
    [[3, 1, 0, 2], [0, 2, 3, 1], [2, 0, 1, 3], [1, 2, 0, 2]],
    columns=["policy", "health", "market", "public"],
)
docs = keyATM_read(dtm)
fit = keyATM(
    docs, model="base", no_keyword_topics=1,
    keywords={"policy": ["policy"], "health": ["health"]},
    options={"seed": 42, "iterations": 20, "verbose": False, "use_cache": False},
)
print_fit_summary(fit)


