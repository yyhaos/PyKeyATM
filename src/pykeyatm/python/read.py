import pandas as pd
import numpy as np
import os
from scipy.sparse import issparse
from typing import Union, List
import re
import warnings
from scipy.sparse import csc_matrix
from pykeyatm._core import require_core

def _core():
    return require_core()


def keyATM_read(texts: pd.DataFrame,
                encoding: str = "utf-8",
                check: bool = True,
                keep_docnames: bool = False,
                split: float = 0.0):
    """
    Read texts and create a keyATM_docs-like object for Python.
    Expects a DataFrame of nonnegative integer term frequencies.
    An optional doc_id column supplies document names when keep_docnames=True.

    Returns:
        dict with W_raw, W_split, doc_index, wd_names, docnames
    """
    if not isinstance(texts, pd.DataFrame):
        raise TypeError("texts must be a pandas DataFrame of word counts.")
    if not texts.columns.is_unique:
        raise ValueError("DTM columns must have unique vocabulary names.")
    if not isinstance(split, (int, float)) or not np.isfinite(split) or not 0 <= split < 1:
        raise ValueError("split must be finite and in [0, 1).")
    docnames = None
    W_read = {"W_raw": [], "W_split": []}

    # Use first column as docnames if needed
    if 'doc_id' in texts.columns:
        if keep_docnames:
            docnames = texts['doc_id'].astype(str).tolist()
        texts = texts.drop(columns='doc_id')  # drop before converting

    if not np.issubdtype(texts.values.dtype, np.number):
        raise ValueError("DTM must only contain numeric values (word counts).")

    values = texts.to_numpy()
    if not texts.shape[0] or not texts.shape[1]:
        raise ValueError("DTM must contain documents and vocabulary columns.")
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("Word counts must be finite real numbers.")
    if (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Word counts must be nonnegative integers.")
    if (values > np.iinfo(np.int32).max).any():
        raise ValueError("Word counts exceed the supported 32-bit integer range.")
    if not all(isinstance(word, str) and word for word in texts.columns):
        raise ValueError("Vocabulary names must be nonempty strings.")
    dfm = csc_matrix(values.astype(np.int32))
    vocabulary = list(texts.columns)
    W_raw, W_split = _core().read_dfm_cpp(dfm, vocabulary, split)

    if check:
        seen = set()
        wd_names = []
        for doc in W_raw:
            for word in doc:
                if word not in seen:
                    seen.add(word)
                    wd_names.append(word)
        doc_index = get_doc_index(W_raw)
    else:
        wd_names = None
        doc_index = None

    return {
        "W_raw": W_raw,
        "W_split": {"W_raw": W_split if split > 0 else None},
        "doc_index": doc_index,
        "wd_names": wd_names,
        "docnames": docnames,
        "class": ["keyATM_docs"]
    }

def get_doc_index(W_raw, check=False):
    """
    Return indices of non-empty documents in W_raw.
    If check=True, warn if any zero-length docs are found.
    """
    len_list = [len(doc) for doc in W_raw]
    index = list(range(1, len(W_raw) + 1))
    nonzero_index = [i for i, l in zip(index, len_list) if l != 0]
    zero_index = [str(i) for i, l in zip(index, len_list) if l == 0]

    if zero_index:
        warning_msg = f"Document(s) with length 0: {', '.join(zero_index)}"
        if check:
            warning_msg += "\nThis may cause invalid covariates or time indexes. Please review preprocessing steps."
        warnings.warn(warning_msg)

    return nonzero_index
