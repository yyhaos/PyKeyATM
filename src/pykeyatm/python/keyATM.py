import random
import os
from typing import List, Dict, Any, Union
from .model import *
from .posterior import *
import copy
import pickle
def keyATM(
    docs,
    model: str,
    no_keyword_topics: int,
    keywords: List[List[str]] = None,
    model_settings: Dict[str, Any] = None,
    priors: Dict[str, Any] = None,
    options: Dict[str, Any] = None,
    keep: List[str] = None,
):
    """Fit a keyATM model.

    This is the main entry point for fitting keyATM models (e.g., base, covariates,
    dynamic). It mirrors the behavior of the R `keyATM()` function: it constructs a
    model configuration from `docs` and inputs, optionally resumes from a saved
    intermediate state, runs Gibbs sampling via `keyATM_fit()`, and returns a
    `keyATM_output`-like object produced by `keyATM_output()`.

    Notes
    -----
    - Random seed:
      If ``options["seed"]`` is not provided, a random seed is generated and then
      used to seed Python's RNG.
    - Resume:
      If ``options["resume"]`` points to an existing path, the function loads the
      intermediate fitted object and continues fitting.
    - Caching is disabled by default. Set ``options["use_cache"]`` to true to
      read and write a pickle cache (``keyATM_fit.pkl`` by default).

    Parameters
    ----------
    docs
        Documents returned by :func:`keyATM_read` (e.g., converted from a quanteda
        dfm object). This should contain the corpus representation needed by the
        backend fitting routine.
    model
        The keyATM model type. Common values are ``"base"``, ``"covariates"``,
        and ``"dynamic"`` (depending on what your backend supports).
    no_keyword_topics
        Number of regular (non-keyword) topics.
    keywords
        Keywords specification for keyword topics. Typically a mapping or nested
        list structure where each keyword topic is associated with a list of words.
        If ``None``, it defaults to an empty list.
    model_settings
        Model-specific settings. If ``None``, it defaults to an empty dict.
    priors
        Priors for model parameters. If ``None``, it defaults to an empty dict.
    options
        Runtime options controlling fitting and bookkeeping. If ``None``, it
        defaults to an empty dict.

        Common keys (aligned with the R documentation):

        - ``seed``: int
          Random seed. If absent, a seed is sampled.
        - ``iterations``: int
          Number of MCMC iterations. If set to 0, this function returns the
          initialized/fitted object early (initialization-only behavior).
        - ``verbose``: bool
          Whether to print fitting diagnostics (backend-dependent).
        - ``llk_per``: int
          Store log-likelihood/perplexity every ``llk_per`` iterations.
        - ``use_weights``: bool
          Whether to use weights (backend-dependent).
        - ``weights_type``: str
          Weighting scheme (backend-dependent).
        - ``prune``: bool
          Whether to remove keywords that do not appear in the corpus.
        - ``store_theta``: bool or int
          Whether to store document-topic distribution-related sufficient stats
          at the specified thinning interval.
        - ``store_pi``: bool or int
          Whether to store pi-related quantities at the specified thinning interval.
        - ``thinning``: int
          Interval at which selected parameters are stored.
        - ``parallel_init``: bool
          Whether to parallelize initialization (backend-dependent).
        - ``resume``: str or None
          Path used to save/load intermediate results to resume fitting.
        - ``use_cache``: bool
          If True, read and write a pickle cache. Defaults to False.
    keep
        Names of elements to keep in the final output (passed to
        :func:`keyATM_output`). If provided, must be a list of strings.

    Returns
    -------
    dict
        A keyATM output object (Python-side equivalent of the R `keyATM_output`),
        typically containing entries such as:

        - ``keyword_k``: number of keyword topics
        - ``no_keyword_topics``: number of non-keyword topics
        - ``V``: vocabulary size
        - ``N``: number of documents
        - ``model``: model name
        - ``theta``: document-topic proportions (if stored/available)
        - ``phi``: topic-word distributions
        - ``topic_counts``: token counts per topic
        - ``word_counts``: token counts per vocabulary term
        - ``doc_lens``: document lengths
        - ``vocab``: vocabulary
        - ``priors``: priors used
        - ``options``: options used
        - ``keywords_raw``: raw keywords provided
        - ``model_fit``: perplexity/log-likelihood history (if stored/available)
        - ``pi``: last-iteration pi (if stored/available)
        - ``values_iter``: stored values during iterations (if any)
        - ``kept_values``: extra outputs specified by `keep`
        - ``information``: fitting metadata

        If ``options["iterations"] == 0`` (or the backend indicates 0 iterations),
        this function returns the initialized/fitted object early rather than a
        finalized output object.

    See Also
    --------
    keyATM_read
        Read/convert a document-feature matrix into the internal `docs` format.
    keyATM_initialize
        Initialize the model state prior to fitting.
    keyATM_fit
        Run posterior sampling / fitting iterations.
    keyATM_output
        Create a user-facing output object from a fitted state.

    Examples
    --------
    Basic usage (similar to the R examples):

    >>> import pyreadr
    >>> from collections import defaultdict
    >>> import pandas as pd
    >>> from pykeyatm import keyATM, keyATM_read
    >>>
    >>> # Load an R quanteda dfm (.rda) and convert to docs
    >>> result = pyreadr.read_r(r"..\\keyATM\\dfm_mat.rda")
    >>> dfm = list(result.values())[0]
    >>> keyATM_docs = keyATM_read(dfm)
    >>>
    >>> # Build keywords dict from a CSV with columns: topic, word
    >>> df = pd.read_csv("data/keywords.csv")
    >>> kw = defaultdict(list)
    >>> for _, row in df.iterrows():
    ...     kw[row["topic"]].append(row["word"])
    >>> keywords = dict(kw)
    >>>
    >>> options = {"use_cache": False, "seed": 471}
    >>> out = keyATM(
    ...     docs=keyATM_docs,
    ...     model="base",
    ...     no_keyword_topics=5,
    ...     keywords=keywords,
    ...     options=options,
    ... )
    """
    if keep is not None and not isinstance(keep, list):
        raise TypeError("`keep` must be a list of strings")

    if keywords is None:
        keywords = []
    if model_settings is None:
        model_settings = {}
    if priors is None:
        priors = {}
    if options is None:
        options = {}

    options = copy.deepcopy(options)
    keep = None if keep is None else list(keep)
    model_name = full_model_name(model, type="keyATM")

    # Set random seed if not provided
    if "seed" not in options or options["seed"] is None:
        options["seed"] = int(random.random() * 1e5)
    random.seed(options["seed"])

    # Check if resume path exists
    options_copy = copy.deepcopy(options)
    resume_path = options.get("resume", None)
    if resume_path and os.path.exists(resume_path):
        resume = True
        fitted = fitted_load(resume_path)
        exists_iter = fitted.get("used_iter", None)
        fitted = fitted_update_iterations(fitted, options_copy)
        fitted = keyATM_fit(fitted, resume=True)
        used_iter = get_used_iter(fitted, resume, exists=exists_iter)
    else:
        resume = False
        initialized = keyATM_initialize(
            docs, model_name, no_keyword_topics,
            keywords, model_settings, priors, options_copy
        )
        initialized["seed"] = int(options["seed"])
        resume = False
        cache_file = options_copy.get("cache_file", "keyATM_fit.pkl")
        use_cache = bool(options_copy.get("use_cache", False))
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                fitted = pickle.load(f)
        else:
            fitted = keyATM_fit(initialized)
            if use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(fitted, f)

        used_iter = get_used_iter(fitted, resume)
    # Save after training (or initialize-only)
    if resume_path:
        fitted_save(resume_path, fitted, model_name, used_iter)

    # If iterations is 0, return early
    if fitted.get("options", {}).get("iterations", 0) == 0:
        return fitted
    out = keyATM_output(fitted, keep, used_iter)
    return out

def get_used_iter(fitted, resume, exists=None):
    thinning = fitted["options"]["thinning"]
    if resume:
        exists_max = max(exists)
        total_iter = list(range(exists_max + 1, fitted["options"]["iterations"] + 1))
        total_iter = [i for i in total_iter if (i % thinning == 0) or (i == 1) or (i == total_iter[-1])]
        used_iter = exists + total_iter
    else:
        total_iter = list(range(1, fitted["options"]["iterations"] + 1))
        used_iter = [i for i in total_iter if (i % thinning == 0) or (i == 1) or (i == total_iter[-1])]
    return used_iter

def weightedLDA(
    docs,
    model: str,
    number_of_topics: int,
    model_settings: Dict[str, Any] = None,
    priors: Dict[str, Any] = None,
    options: Dict[str, Any] = None,
    keep: List[str] = None,
):
    """Fit a weighted LDA model.

    This function is a Python translation of the R `weightedLDA()` wrapper in the
    keyATM ecosystem. It fits an LDA-style model (i.e., without keyword topics),
    using the same backend fitting pipeline as `keyATM()`:

    1) build a model name via `full_model_name(model, type="lda")`
    2) initialize the model state via `keyATM_initialize(...)` with empty keywords
    3) run posterior sampling via `keyATM_fit(...)`
    4) optionally resume from an intermediate state via `fitted_load(...)`
    5) construct a user-facing result via `keyATM_output(...)`

    Notes
    -----
    - Random seed:
      If ``options["seed"]`` is not provided (or is None), a random seed is
      generated and used to seed Python's RNG.
    - Resume:
      If ``options["resume"]`` points to an existing file path, the function loads
      the intermediate state and continues fitting.
    - Cache (local development helper):
      If ``options["use_cache"]`` is truthy and a pickle cache exists (default:
      ``weightedLDA_fit.pkl`` or ``options["cache_file"]``), the cached fitted object
      is loaded instead of running a new fit.

    Parameters
    ----------
    docs
        Documents returned by :func:`keyATM_read` (e.g., created from a DFM/DTM).
        This object should contain the corpus representation expected by the backend.
    model
        The weighted LDA model type. Common values are ``"base"``, ``"covariates"``,
        and ``"dynamic"`` (subject to backend support).
    number_of_topics
        Number of topics to fit (all are regular topics; there are no keyword topics).
    model_settings
        Model-specific settings (backend-dependent). If ``None``, defaults to ``{}``.
    priors
        Priors for model parameters (backend-dependent). If ``None``, defaults to ``{}``.
    options
        Runtime options controlling fitting and bookkeeping. If ``None``, defaults to ``{}``.

        Common keys (aligned with the R `weightedLDA()` docs and `keyATM()` options):

        - ``seed``: int
          Random seed. If absent, a seed is sampled.
        - ``iterations``: int
          Number of MCMC iterations. If 0, this function returns the initialized/fitted
          object early (initialization-only behavior).
        - ``thinning``: int
          Interval at which selected parameters are stored (used by `get_used_iter`).
        - ``resume``: str or None
          Path used to save/load intermediate results to resume fitting.
        - ``use_cache``: bool
          If True and the cache file exists, load cached fitted object.
        - ``cache_file``: str
          Pickle cache path (default: ``"weightedLDA_fit.pkl"``).
        - Other keys supported by your backend (e.g., verbose, llk_per, store_theta, ...).
    keep
        Names of elements to keep in the final output (passed to :func:`keyATM_output`).
        If provided, must be a list of strings.

    Returns
    -------
    dict
        A keyATM-style output object (Python-side equivalent of the R `keyATM_output`)
        containing LDA-relevant entries such as:

        - ``V``: vocabulary size
        - ``N``: number of documents
        - ``model``: model name
        - ``theta``: document-topic proportions (if stored/available)
        - ``phi``: topic-word distributions
        - ``topic_counts``: token counts per topic
        - ``word_counts``: token counts per vocabulary term
        - ``doc_lens``: document lengths
        - ``vocab``: vocabulary
        - ``priors``: priors used
        - ``options``: options used
        - ``keywords_raw``: typically empty/None for LDA models
        - ``model_fit``: perplexity/log-likelihood history (if stored/available)
        - ``values_iter``: stored values during iterations (if any)
        - ``number_of_topics``: number of topics (added here to mirror the R wrapper)
        - ``kept_values``: extra outputs specified by `keep`
        - ``information``: fitting metadata

        This wrapper also removes keyword-topic specific fields (``keyword_k`` and
        ``no_keyword_topics``) from the returned dict, mirroring the R behavior.

        If ``options["iterations"] == 0`` (or the backend indicates 0 iterations),
        this function returns the initialized/fitted object early rather than a
        finalized output object.

    See Also
    --------
    keyATM
        Main keyATM wrapper (keyword-assisted topic modeling).
    keyATM_read
        Read/convert a DFM/DTM into the internal `docs` format.
    keyATM_initialize
        Initialize the model state prior to fitting.
    keyATM_fit
        Run posterior sampling / fitting iterations.
    keyATM_output
        Create a user-facing output object from a fitted state.

    Examples
    --------
    Fit a weighted LDA model from a scikit-learn DFM:

    >>> import pandas as pd
    >>> import nltk
    >>> from nltk.corpus import stopwords
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from pykeyatm.read import keyATM_read
    >>> from pykeyatm.keyATM import weightedLDA
    >>>
    >>> nltk.download("stopwords")
    >>> stop_words = stopwords.words("english") + ["may", "shall", "can", "must", "upon", "with", "without"]
    >>> vectorizer = CountVectorizer(
    ...     lowercase=True,
    ...     stop_words=stop_words,
    ...     token_pattern=r"(?u)\\b[a-zA-Z]{3,}\\b",
    ...     min_df=2,
    ... )
    >>> df = pd.read_csv("../keyATM/inaugural_58.csv")
    >>> X = vectorizer.fit_transform(df["text"])
    >>> dfm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df["doc_id"])
    >>> docs = keyATM_read(dfm)
    >>>
    >>> options = {"use_cache": True, "seed": 472}
    >>> out = weightedLDA(
    ...     docs=docs,
    ...     model="base",
    ...     number_of_topics=5,
    ...     options=options,
    ... )
    """

    # ---- type checks & defaults ----
    if keep is not None and not isinstance(keep, list):
      raise TypeError("`keep` must be a list of strings")

    if model_settings is None:
        model_settings = {}
    if priors is None:
        priors = {}
    if options is None:
        options = {}

    options = copy.deepcopy(options)
    keep = None if keep is None else list(keep)
    model_name = full_model_name(model, type="lda")

    # seed
    if "seed" not in options or options.get("seed") is None:
        options["seed"] = int(random.random() * 1e5)
    random.seed(options["seed"])

    # helper for JSON dumping complex numpy types (align with your keyATM)
    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        else:
            # fallback: let json raise
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # make a copy to pass into inner functions (like your keyATM)
    options_copy = copy.deepcopy(options)

    # ---- resume branch ----
    resume_path = options.get("resume", None)
    if resume_path and os.path.exists(resume_path):
        resume = True
        fitted = fitted_load(resume_path)
        exists_iter = fitted.get("used_iter", None)
        fitted = fitted_update_iterations(fitted, options_copy)
        fitted = keyATM_fit(fitted, resume=True)
        used_iter = get_used_iter(fitted, resume, exists=exists_iter)
    else:
        # ---- fresh run branch ----
        resume = False
        initialized = keyATM_initialize(
            docs, model_name, number_of_topics,
            keywords={},                      # weightedLDA: no keywords
            model_settings=model_settings,
            priors=priors,
            options=options_copy
        )
        # keep seed in initialized for reproducibility if downstream expects it
        try:
            initialized["seed"] = int(options["seed"])
        except Exception:
            pass

        # cache: use a distinct file for LDA
        cache_file = options_copy.get("cache_file", "weightedLDA_fit.pkl")
        use_cache = bool(options_copy.get("use_cache", False))

        if use_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                fitted = pickle.load(f)
        else:
            fitted = keyATM_fit(initialized)
            if use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(fitted, f)

        used_iter = get_used_iter(fitted, resume)

    # save resume snapshot if requested
    if "resume" in options and resume_path:
        fitted_save(resume_path, fitted, model_name, used_iter)

    # iterations == 0: return raw fitted (align with R early-return)
    iters = 0
    try:
        iters = int(fitted.get("options", {}).get("iterations", 0))
    except Exception:
        pass
    if iters == 0:
        return fitted

    # produce output
    out = keyATM_output(fitted, keep, used_iter)
    # align R behavior
    if isinstance(out, dict):
        out["number_of_topics"] = number_of_topics
        # R sets these to NULL; in Python we can just drop them if present
        out.pop("no_keyword_topics", None)
        out.pop("keyword_k", None)

    return out
