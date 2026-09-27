"""Fit the dynamic keyATM model to synthetic documents ordered in two states."""
import pandas as pd

from pykeyatm import keyATM, keyATM_read


dtm = pd.DataFrame(
    [[3, 1, 0, 2], [0, 2, 3, 1], [2, 0, 1, 3], [1, 2, 0, 2]],
    columns=["policy", "health", "market", "public"],
)
docs = keyATM_read(dtm)
fit = keyATM(
    docs, model="dynamic", no_keyword_topics=1,
    keywords={"policy": ["policy"], "health": ["health"]},
    model_settings={"num_states": 2, "time_index": [1, 1, 2, 2]},
    options={"seed": 42, "iterations": 20, "verbose": False, "use_cache": False},
)
print(fit["theta"])
