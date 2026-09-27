"""Fit the LDA baseline to a tiny synthetic document-term matrix."""
import pandas as pd

from pykeyatm import keyATM_read, weightedLDA


dtm = pd.DataFrame(
    [[3, 1, 0, 2], [0, 2, 3, 1], [2, 0, 1, 3], [1, 2, 0, 2]],
    columns=["policy", "health", "market", "public"],
)
docs = keyATM_read(dtm)
fit = weightedLDA(
    docs=docs,
    model="base",
    number_of_topics=3,
    options={"seed": 42, "iterations": 20, "verbose": False, "use_cache": False},
)
print(fit["theta"])
