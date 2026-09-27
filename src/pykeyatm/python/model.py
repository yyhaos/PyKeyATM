import random
import copy
import warnings
from functools import partial
from .read import get_doc_index
import os
from collections import defaultdict
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import json
from .utils import *
import weakref
from typing import Union
from pykeyatm._core import require_core

def _core():
    return require_core()



def keyATM_initialize(docs,
                      model: str,
                      no_keyword_topics: int,
                      keywords=None,
                      model_settings=None,
                      priors=None,
                      options=None):
    """
    Python equivalent of keyATM_initialize (internal use).
    """
    if keywords is None:
        keywords = []
    if model_settings is None:
        model_settings = {}
    if priors is None:
        priors = {}
    if options is None:
        options = {}

    # Initialization must not change caller-owned documents or settings.
    docs = dict(docs)
    model_settings = copy.deepcopy(model_settings)
    priors = copy.deepcopy(priors)
    options = copy.deepcopy(options)

    # Check
    if isinstance(no_keyword_topics, bool) or not isinstance(no_keyword_topics, int) or no_keyword_topics < 0:
        raise ValueError("`no_keyword_topics` must be a nonnegative integer.")

    info = {
        "models_keyATM": ["base", "cov", "hmm"],
        "models_lda": ["lda", "ldacov", "ldahmm"]
    }

    if model not in info["models_keyATM"] + info["models_lda"]:
        raise ValueError("Please select a correct model.")

    keywords = check_arg(keywords, "keywords", model, info)

    # doc_index
    if docs.get("doc_index") is None:
        info["use_doc_index"] = get_doc_index(docs["W_raw"])
    else:
        info["use_doc_index"] = docs["doc_index"]
        if len(info["use_doc_index"]) != len(docs["W_raw"]):
            warnings.warn("Some documents have 0 length. Please review the preprocessing steps.")

    docs["W_raw"] = [docs["W_raw"][i - 1] for i in info["use_doc_index"]]  # 1-based index in R
    if docs.get("docnames") is not None:
        docs["docnames"] = [docs["docnames"][i - 1] for i in info["use_doc_index"]]
    info["num_doc"] = len(docs["W_raw"])
    info["keyword_k"] = len(keywords)
    info["total_k"] = info["keyword_k"] + no_keyword_topics
    if info["total_k"] < 1:
        raise ValueError("At least one topic is required.")
    if info["num_doc"] < 1:
        raise ValueError("At least one nonempty document is required.")

    model_settings = check_arg(model_settings, "model_settings", model, info)
    priors = check_arg(priors, "priors", model, info)
    options = check_arg(options, "options", model, info)

    info["parallel_init"] = options.get("parallel_init", False)
    if info["parallel_init"]:
        print("ℹ️ Parallel initialization is enabled. Be sure to set up `multiprocessing` or similar plan.")

    # Initialization
    random.seed(options["seed"])
    np.random.seed(options["seed"])

    if docs.get("wd_names") is None:
        seen = set()
        wd_names = []
        for doc in docs["W_raw"]:
            for w in doc:
                if w not in seen:
                    seen.add(w)
                    wd_names.append(w)
        info["wd_names"] = wd_names
    else:
        info["wd_names"] = docs["wd_names"]

    if not info["wd_names"] or not all(isinstance(w, str) and w for w in info["wd_names"]):
        raise ValueError("Vocabulary must contain nonempty strings.")
    if len(set(info["wd_names"])) != len(info["wd_names"]):
        raise ValueError("Vocabulary names must be unique.")
    info["wd_map"] = myhashmap(info["wd_names"], list(range(len(info["wd_names"]))))

    # W: word index sequences
    if info["parallel_init"]:
        from multiprocessing import Pool
        with Pool() as pool:
            W = pool.map(partial(myhashmap_getvec, info["wd_map"]), docs["W_raw"])
    else:
        W = [myhashmap_getvec(info["wd_map"], x) for x in docs["W_raw"]]


    # check keywords
    keywords = check_keywords(info["wd_names"], keywords, options.get("prune", None))
    keywords_raw = keywords
    keywords_id = {
        topic: myhashmap_getvec(info["wd_map"], ws)
        for topic, ws in keywords.items()
    }
    info["keywords_id"] = [i for ids in keywords_id.values() for i in ids]


    # Assign S and Z
    if model in info["models_keyATM"]:
        res = make_sz_key(W, keywords, info)
    else:
        res = make_sz_lda(W, info)
    

    S, Z = res["S"], res["Z"]

    
    del res

    # Stored values
    stored_values = {
        "vocab_weights": [-1] * len(info["wd_names"]),
        "doc_index": info["use_doc_index"],
        "keyATMdoc_meta": {k: v for k, v in docs.items() if k not in ["W_raw", "doc_index"]}
    }

    if model in ["base", "lda"] and options.get("estimate_alpha", False):
        stored_values["alpha_iter"] = []

    if model in ["hmm", "ldahmm"]:
        options["estimate_alpha"] = True
        stored_values["alpha_iter"] = []
        stored_values["R_iter"] = []
        if options.get("store_transition_matrix", False):
            stored_values["P_iter"] = []
        else:
            stored_values["P_last"] = []

    if model in ["cov", "ldacov"]:
        stored_values["Lambda_iter"] = []

    if model in info["models_keyATM"] and options.get("store_pi", False):
        stored_values["pi_vectors"] = []

    if options.get("store_theta", False):
        stored_values["Z_tables"] = []
        stored_values["theta_PG"] = []

    # Build keyATM model
    key_model = {
        "W": W,
        "Z": Z,
        "S": S,
        "model": abb_model_name(model),
        "keywords": keywords_id,
        "keywords_raw": keywords_raw,
        "no_keyword_topics": no_keyword_topics,
        "keyword_k": len(keywords_raw),
        "vocab": info["wd_names"],
        "model_settings": model_settings,
        "priors": priors,
        "options": options,
        "stored_values": stored_values,
        "model_fit": {},
        "call": f"keyATM_initialize({model})",
        "seed": options["seed"]
    }
    
    key_model["class"] = ["keyATM_model"] + [model] + ["list"]

    keyATM_initialized = {
        "model": key_model,
        "model_name": model,
        "class": ["keyATM_initialized"]
    }

    return keyATM_initialized


def myhashmap(keys, values):
    """
    Construct a mapping from keys to values.
    Equivalent to R's hashmap from words to indices.
    """
    return dict(zip(keys, values))

def myhashmap_getvec(mapping, sequence):
    """
    Given a mapping (word -> id), convert a list of words into a list of ids.
    """
    return [mapping[w] for w in sequence if w in mapping]


def make_sz_key(W: List[List[int]], keywords: List[List[int]], info: Dict[str, Any]):
    key_wdids = info["keywords_id"]
    total_k = int(info["total_k"])
    cat_ids = [
        topic_id
        for topic_id, topic in enumerate(keywords)
        for _ in range(len(keywords[topic]))
    ]
    keyword_map = defaultdict(list)
    for wid, topic_id in zip(key_wdids, cat_ids):
        keyword_map[wid].append(topic_id)

    def make_sz(doc):
        zz = np.random.randint(0, total_k, size=len(doc)).tolist()
        ss = [0] * len(doc)
        for position, token in enumerate(doc):
            topic_choices = keyword_map.get(token)
            if topic_choices is not None:
                zz[position] = random.choice(topic_choices)
                ss[position] = int(np.random.random() < 0.7)
        return ss, zz

    parallel_init = info['parallel_init']
    if parallel_init:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            initialized = list(executor.map(make_sz, W))
    else:
        initialized = [make_sz(doc) for doc in W]

    S = [state[0] for state in initialized]
    Z = [state[1] for state in initialized]

    return {"S": S, "Z": Z}

def make_sz_lda(W: List[List[int]], info: Dict[str, Any]):
    total_k = int(info["total_k"])
    parallel_init = bool(info.get("parallel_init", False))

    seed = info.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    topicvec = list(range(total_k))

    def make_z(doc: List[int]) -> List[int]:
        if not doc:
            return []
        return np.random.choice(topicvec, size=len(doc), replace=True).astype(int).tolist()

    if parallel_init:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            Z = list(executor.map(make_z, W))
    else:
        Z = [make_z(doc) for doc in W]

    return {"S": [], "Z": Z}


def keyATM_fit(
    keyATM_initialized: dict,
    resume: bool = False,
    checkpoint_every: int = 0,
    checkpoint_callback=None,
) -> dict:
    """
    Fit a keyATM model, handling resume if needed.
    """
    model_classes = keyATM_initialized.get("class", [])

    if "keyATM_initialized" not in model_classes and "keyATM_resume" not in model_classes:
        raise ValueError("The input is not an initialized object.")

    key_model = keyATM_initialized["model"]
    model_name = keyATM_initialized["model_name"]
    # iterations = key_model["options"]["iter_new"]

    if "keyATM_resume" in model_classes:
        iterations = int(key_model["options"]["iter_new"])
        if key_model["options"].get("verbose", False):
            print(
                f"Fitting the model: adding {iterations} iteration(s) "
                f"to the existing {key_model['options']['iterations'] - iterations} iteration(s) "
                f"for a total of {key_model['options']['iterations']} iteration(s)"
            )
    else:
        seed = key_model["options"].get("seed", 1234)
        random.seed(seed)
        np.random.seed(seed)

    # Model-specific dispatch
    

    def clean_for_pybind(obj):
        """
        Convert nested dict/list containing numpy types into pure Python objects.
        """
        def convert(o):
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif hasattr(o, '__dict__'):
                return vars(o)
            if isinstance(o, weakref.ReferenceType):
                return None
            raise TypeError(f"Type {type(o)} not serializable")

        return json.loads(json.dumps(obj, default=convert))

    token_state = {name: key_model.pop(name) for name in ("W", "Z", "S")}
    key_model = clean_for_pybind(key_model)
    key_model.update(token_state)
    del token_state
    # for k, v in key_model.items():
    
    if model_name == "base":
        key_model = _core().keyATM_fit_base(
            key_model, resume, checkpoint_every, checkpoint_callback
        )
    elif model_name == "hmm":
        key_model = _core().keyATM_fit_HMM(
            key_model, resume, checkpoint_every, checkpoint_callback
        )
    elif model_name == "lda":
        key_model = _core().keyATM_fit_LDA(key_model, resume)
    elif model_name == "ldacov":
        key_model = _core().keyATM_fit_LDAcov(key_model, resume)
    elif model_name == "ldahmm":
        key_model = _core().keyATM_fit_LDAHMM(key_model, resume)
    elif model_name == "cov":
        cov_type = key_model["model_settings"].get("covariates_model", "PG")
        if cov_type == "PG":
            raise NotImplementedError("PyKeyATM supports the DirMulti covariate sampler; PG is not implemented.")
        elif cov_type == "DirMulti":
            key_model = _core().keyATM_fit_cov(key_model, resume)
        else:
            raise ValueError(f"Unknown covariates model: {cov_type}")
    else:
        raise ValueError("Please check `model`.")
    # Set class if not already set
    if "keyATM_fitted" not in key_model.get("class", []):
        key_model.setdefault("class", [])
        key_model["class"] = ["keyATM_fitted"] + key_model["class"]

    return key_model


def check_pybind_compatibility(obj, path="root", show = True):
    primitive_types = (int, float, bool, str, type(None))
    
    if isinstance(obj, primitive_types):
        if(show):
            print(f"=== {path}: is (type: {type(obj)})")   
        return True
    elif isinstance(obj, np.ndarray):
        if(show):
            print(f"=== {path}: is (type: {type(obj)})")   
        return True
    elif isinstance(obj, dict):
        all_ok = True
        for k, v in obj.items():
            if not isinstance(k, str):
                print(f"[ERROR] {path}: Key '{k}' is not a string (type: {type(k)})")
                all_ok = False
            if not check_pybind_compatibility(v, f"{path}.{k}"):
                all_ok = False
        return all_ok
    elif isinstance(obj, (list, tuple)):
        all_ok = True
        for i, v in enumerate(obj):
            if not check_pybind_compatibility(v, f"{path}[{i}]", i == 0):
                all_ok = False
        
        if(show):
            print(f"=== {path}: is (type: {type(obj)})")   
        return all_ok
    elif isinstance(obj, np.generic):  # numpy scalar
        if(show):
            print(f"=== {path}: is (type: {type(obj)})")   
        return True
    else:
        print(f"[ERROR] {path}: Unsupported type {type(obj)}")
        return False



def check_arg(obj, name, model, info=None):
    if info is None:
        info = {}

    if name == "keywords":
        return check_arg_keywords(obj, model, info)
    elif name == "model_settings":
        return check_arg_model_settings(obj, model, info)
    elif name == "priors":
        return check_arg_priors(obj, model, info)
    elif name == "options":
        return check_arg_options(obj, model, info)
    elif name == "vb_options":
        return check_arg_vboptions(obj, model, info)


def check_arg_keywords(keywords, model, info):
    if not isinstance(keywords, dict):
        raise TypeError("Expected keywords to be a list.")

    if len(keywords) == 0 and model in info.get("models_keyATM", []):
        raise ValueError("Please provide keywords.")

    if len(keywords) != 0 and model in info.get("models_lda", []):
        raise ValueError("This model does not take keywords.")

    # Name of keywords topic
    if model in info.get("models_keyATM", []):
        for x in keywords:
            if not isinstance(x, str) and not all(isinstance(i, str) for i in x):
                raise TypeError("Each keyword entry must be a string or list of strings.")

        # Emulate R's names using a dict with generated keys
        if isinstance(keywords, dict):
            # Keep original names if present
            new_keywords = {
                f"{i+1}_{k}" if k else f"{i+1}": v
                for i, (k, v) in enumerate(keywords.items())
            }
        else:
            # Convert list to dict with default names
            new_keywords = {
                str(i + 1): kw for i, kw in enumerate(keywords)
            }
        keywords = new_keywords

    return keywords

def show_unused_arguments(obj, name, allowed_arguments):
    unused_input = [k for k in obj.keys() if k not in allowed_arguments]
    if unused_input:
        raise ValueError(
            f"keyATM doesn't recognize some of the arguments in {name}: " +
            ", ".join(unused_input)
        )

def check_arg_model_settings(obj: dict, model: str, info: dict) -> dict:
    if not isinstance(obj, dict):
        raise TypeError("model_settings should be a dict.")

    allowed_arguments = []

    # ---------- Slice Sampling for base/lda/hmm/ldahmm ----------
    if model in ["base", "lda", "hmm", "ldahmm"]:
        if "slice_min" not in obj or obj["slice_min"] is None:
            obj["slice_min"] = 1e-9
        else:
            if not isinstance(obj["slice_min"], (int, float)) or obj["slice_min"] <= 0:
                raise ValueError("`model_settings$slice_min` should be a positive numeric value.")

        if "slice_max" not in obj or obj["slice_max"] is None:
            obj["slice_max"] = 100
        else:
            if not isinstance(obj["slice_max"], (int, float)) or obj["slice_max"] <= 0:
                raise ValueError("`model_settings$slice_max` should be a positive numeric value.")

        allowed_arguments += ["slice_min", "slice_max"]

    # ---------- Covariate models ----------
    if model in ["cov", "ldacov"]:
        if "covariates_data" not in obj or obj["covariates_data"] is None:
            raise ValueError("Please provide `obj['covariates_data']`.")

        if obj.get("covariates_model", "DirMulti") == "PG":
            raise NotImplementedError("PyKeyATM supports the DirMulti covariate sampler; PG is not implemented.")

        # subset rows by use_doc_index (1-based in R; convert to 0-based)
        use_doc_index = np.array(info["use_doc_index"]) - 1
        cov_df = pd.DataFrame(obj["covariates_data"])
        cov_df = cov_df.iloc[use_doc_index, :].copy()

        # check dimensions
        if cov_df.shape[0] != info["num_doc"]:
            raise ValueError("The row of `model_settings$covariates_data` should be the same as the number of documents.")

        # check missing
        if cov_df.isna().to_numpy().sum() != 0:
            raise ValueError("Covariate data should not contain missing values.")

        # covariates_formula may be None → do not change the matrix
        covariates_formula = obj.get("covariates_formula", None)

        # standardize option
        if "standardize" not in obj or obj["standardize"] is None:
            obj["standardize"] = "non-factor"
        if obj["standardize"] not in ["all", "none", "non-factor"]:
            raise ValueError('Unknown option in `standardize`. It should be one of "all", "none", or "non-factor".')

        # standardize / model.matrix equivalent
        covariates_data_use = covariates_standardize(cov_df, obj["standardize"], covariates_formula)
        obj["covariates_data_use"] = covariates_data_use

        # validate regressability (no singular / NA coefficients)
        temp = pd.DataFrame(covariates_data_use).copy()
        temp["y"] = np.random.normal(size=temp.shape[0])

        # try least squares; if it fails or yields NaNs, raise
        X_mat = temp.drop(columns=["y"]).to_numpy()
        y_vec = temp["y"].to_numpy()
        try:
            coef, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
            if np.isnan(coef).any():
                raise ValueError("Covariates are invalid.")
        except Exception:
            raise ValueError("Covariates are invalid.")

        # Slice Sampling Settings for covariate models
        if "slice_min" not in obj or obj["slice_min"] is None:
            obj["slice_min"] = -5.0
        else:
            if not isinstance(obj["slice_min"], (int, float)):
                raise ValueError("`model_settings$slice_min` should be a numeric value.")

        if "slice_max" not in obj or obj["slice_max"] is None:
            obj["slice_max"] = 5.0
        else:
            if not isinstance(obj["slice_max"], (int, float)):
                raise ValueError("`model_settings$slice_max` should be a numeric value.")

        # MH option
        if "mh_use" not in obj or obj["mh_use"] is None:
            obj["mh_use"] = 0
        else:
            obj["mh_use"] = int(obj["mh_use"])
            if obj["mh_use"] not in (0, 1):
                raise ValueError("`model_settings$mh_use` should be TRUE/FALSE (0/1)")

        # Model
        if "covariates_model" not in obj or obj["covariates_model"] is None:
            if model == "cov":
                obj["covariates_model"] = "DirMulti"
            if model == "ldacov":
                obj["covariates_model"] = "DirMulti"

        if obj["covariates_model"] != "DirMulti" and model == "ldacov":
            raise ValueError("Use Diricule-Multinomial model for LDA covariates.")

        if obj["covariates_model"] not in ["PG", "DirMulti"]:
            raise ValueError("Undefined model. `covariates_model` option in `model_settings` take `PG` or `DirMulti`.")

        if obj["covariates_model"] == "PG":
            K = info["total_k"]
            X = np.asarray(obj["covariates_data_use"])
            M = X.shape[1]  # Number of covariates
            D = X.shape[0]  # Number of documents

            Sigma_Lambda = np.eye(K - 1)
            # PG_Lambda: M x (K-1)
            if M == 1:
                obj.setdefault("PG_params", {})
                obj["PG_params"]["PG_Lambda"] = np.random.multivariate_normal(
                    mean=np.zeros(K - 1), cov=Sigma_Lambda
                ).reshape(1, K - 1)
            else:
                obj.setdefault("PG_params", {})
                obj["PG_params"]["PG_Lambda"] = np.random.multivariate_normal(
                    mean=np.zeros(K - 1), cov=Sigma_Lambda, size=M
                )

            obj["PG_params"]["PG_SigmaPhi"] = np.eye(K - 1)

            Mu = X @ obj["PG_params"]["PG_Lambda"]  # D x (K-1)
            Sigma = np.eye(K - 1)
            # Phi: D x (K-1)
            Phi = np.vstack([
                np.random.multivariate_normal(mean=Mu[d, :], cov=Sigma) for d in range(D)
            ])
            obj["PG_params"]["PG_Phi"] = Phi
            obj["PG_params"]["theta_tilda"] = np.exp(Phi) / (1.0 + np.exp(Phi))
            obj["PG_params"]["theta_last"] = np.zeros((D, K))
            obj["PG_params"]["Lambda_list"] = []
            obj["PG_params"]["Sigma_list"] = []

        allowed_arguments += [
            "covariates_data", "covariates_data_use", "slice_min", "slice_max", "mh_use",
            "covariates_model", "PG_params", "covariates_formula", "standardize", "info"
        ]


    # ---------- HMM and LDA-HMM ----------
    if model in ["hmm", "ldahmm"]:
        if "num_states" not in obj:
            raise ValueError("`model_settings$num_states` is not provided.")
        if isinstance(obj["num_states"], bool) or not isinstance(obj["num_states"], int) or obj["num_states"] < 1:
            raise ValueError("num_states must be a positive integer.")
        if "time_index" not in obj:
            raise ValueError("`model_settings$time_index` is not provided.")
            
        use_doc_index = np.array(info["use_doc_index"]) - 1
        time_index = np.array(obj["time_index"])[use_doc_index]

        if len(time_index) != info["num_doc"]:
            raise ValueError("Length of `time_index` must match number of documents.")
        if time_index.min() != 1 or time_index.max() > info["num_doc"]:
            raise ValueError("`time_index` should start from 1 and not exceed num_doc.")
        if time_index.max() < obj["num_states"]:
            raise ValueError("`num_states` must not exceed the max of `time_index`.")

        diff = np.diff(time_index)
        if not np.all(np.isin(diff, [0, 1])):
            raise ValueError("`time_index` does not increment by 1.")

        if not np.isfinite(time_index).all() or not np.equal(time_index, np.floor(time_index)).all():
            raise ValueError("time_index must contain finite integer period labels.")
        obj["time_index"] = time_index.astype(int)

        allowed_arguments += ["num_states", "time_index"]

    if not np.isfinite(obj["slice_min"]) or not np.isfinite(obj["slice_max"]) or obj["slice_min"] >= obj["slice_max"]:
        raise ValueError("slice_min must be finite and smaller than slice_max.")
    show_unused_arguments(obj, "`model_settings`", allowed_arguments)
    return obj

def check_arg_priors(obj: dict, model: str, info: dict) -> dict:
    if not isinstance(obj, dict):
        raise TypeError("priors should be a dict.")

    allowed_arguments = ["beta", "eta_1", "eta_2", "eta_1_regular", "eta_2_regular"]

    # gamma (prior of pi)
    if model in info.get("models_keyATM", []):
        if "gamma" not in obj or obj["gamma"] is None:
            obj["gamma"] = np.ones((info["total_k"], 2))

        gamma = obj.get("gamma")
        if gamma is not None:
            gamma = np.array(gamma)
            if gamma.shape != (info["total_k"], 2):
                raise ValueError("Check the dimension of `priors$gamma` (should be total_k x 2)")

            if not np.isfinite(gamma).all() or (gamma[:info["keyword_k"]] <= 0).any():
                raise ValueError("Keyword-topic gamma priors must be finite and positive.")
            if info["keyword_k"] < info["total_k"]:
                # Regular topics should have 0 prior
                gamma[info["keyword_k"]:info["total_k"], :] = 0
                obj["gamma"] = gamma

        allowed_arguments.append("gamma")

    # beta
    if "beta" not in obj:
        obj["beta"] = 0.01

    # beta_s (for keyATM)
    if model in info.get("models_keyATM", []):
        if "beta_s" not in obj:
            obj["beta_s"] = 0.1
        allowed_arguments.append("beta_s")

    # alpha (for base/lda)
    if model in ["base", "lda"]:
        if "alpha" not in obj or obj["alpha"] is None:
            obj["alpha"] = [1.0 / info["total_k"]] * info["total_k"]
        if len(obj["alpha"]) != info["total_k"]:
            raise ValueError(f"Starting alpha must be a vector of length {info['total_k']}")
        allowed_arguments.append("alpha")

    # eta family
    obj.setdefault("eta_1", 1.0)
    obj.setdefault("eta_2", 1.0)
    obj.setdefault("eta_1_regular", 2.0)
    obj.setdefault("eta_2_regular", 1.0)

    for name in ("beta", "beta_s", "eta_1", "eta_2", "eta_1_regular", "eta_2_regular"):
        if name in obj and (not np.isscalar(obj[name]) or not np.isfinite(obj[name]) or obj[name] <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    if "alpha" in obj and (not np.isfinite(obj["alpha"]).all() or (np.asarray(obj["alpha"]) <= 0).any()):
        raise ValueError("alpha must contain finite positive values.")
    show_unused_arguments(obj, "`priors`", allowed_arguments)
    return obj

def check_arg_options(obj: dict, model: str, info: dict) -> dict:
    if not isinstance(obj, dict):
        raise TypeError("options should be a dict.")

    allowed_arguments = [
        "seed", "llk_per", "thinning", "iterations", "iter_new", "verbose",
        "use_weights", "weights_type", "prune", "store_theta", "slice_shape",
        "parallel_init", "resume", "use_cache", "cache_file"
    ]

    seed = obj.get("seed", 1234)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    obj["seed"] = seed

    # llk_per
    obj.setdefault("llk_per", 10)
    if not isinstance(obj["llk_per"], int) or obj["llk_per"] <= 0:
        raise ValueError("Invalid value in `options$llk_per`")

    # verbose
    obj["verbose"] = int(obj.get("verbose", 0))
    if obj["verbose"] not in [0, 1]:
        raise ValueError("Invalid value in `options$verbose`")

    # thinning
    obj.setdefault("thinning", 5)
    if not isinstance(obj["thinning"], int) or obj["thinning"] <= 0:
        raise ValueError("Invalid value in `options$thinning`")

    # iterations
    obj.setdefault("iterations", 1500)
    obj["iter_new"] = obj["iterations"]
    if not isinstance(obj["iterations"], int) or obj["iterations"] < 0:
        raise ValueError("Invalid value in `options$iterations`")

    # store_theta
    obj["store_theta"] = int(obj.get("store_theta", 0))
    if obj["store_theta"] not in [0, 1]:
        raise ValueError("Invalid value in `options$store_theta`")

    # store_pi (for keyATM)
    if model in info.get("models_keyATM", []):
        obj["store_pi"] = int(obj.get("store_pi", 0))
        if obj["store_pi"] not in [0, 1]:
            raise ValueError("Invalid value in `options$store_pi`")
        allowed_arguments.append("store_pi")

    # estimate_alpha (for base/lda)
    if model in ["base", "lda"]:
        obj["estimate_alpha"] = int(obj.get("estimate_alpha", 1))
        if obj["estimate_alpha"] not in [0, 1]:
            raise ValueError("Invalid value in `options$estimate_alpha`")
        allowed_arguments.append("estimate_alpha")

    # slice_shape
    obj.setdefault("slice_shape", 1.2)
    if not isinstance(obj["slice_shape"], (float, int)) or obj["slice_shape"] < 0:
        raise ValueError("Invalid value in `options$slice_shape`")

    # use_weights
    obj["use_weights"] = int(obj.get("use_weights", 1))
    if obj["use_weights"] not in [0, 1]:
        raise ValueError("Invalid value in `options$use_weights`")

    # weights_type
    obj.setdefault("weights_type", "information-theory")
    if obj["weights_type"] not in [
        "information-theory", "information-theory-normalized",
        "inv-freq", "inv-freq-normalized"
    ]:
        raise ValueError("Invalid value in `options$weights_type`")

    # prune
    obj["prune"] = int(obj.get("prune", 1))
    if obj["prune"] not in [0, 1]:
        raise ValueError("Invalid value in `options$prune`")

    # store_transition_matrix (for hmm/ldahmm)
    if model in ["hmm", "ldahmm"]:
        obj.setdefault("store_transition_matrix", 0)
        if obj["store_transition_matrix"] not in [0, 1]:
            raise ValueError("Invalid value in `options$store_transition_matrix`")
        allowed_arguments.append("store_transition_matrix")

    # parallel_init
    if "parallel_init" not in obj:
        obj["parallel_init"] = False
    elif obj["parallel_init"] not in [0, 1, False, True]:
        raise ValueError("`options$parallel_init` should be TRUE/FALSE (0/1)")

    # resume
    obj.setdefault("resume", "")

    show_unused_arguments(obj, "`options`", allowed_arguments)
    return obj

def check_keywords(unique_words, keywords, prune):
    import warnings

    if isinstance(keywords, dict):
        keywords_flat = [kw for topic in keywords.values() for kw in topic]
    elif isinstance(keywords, list):
        keywords_flat = [kw for topic in keywords for kw in topic]
    else:
        raise TypeError("keywords must be a list or dict")

    non_existent = [kw for kw in keywords_flat if kw not in unique_words]

    if prune:
        if non_existent:
            warnings.warn(
                f"Some keywords are pruned because they do not appear in the documents: {non_existent}"
            )
        if isinstance(keywords, dict):
            keywords = {
                k: [w for w in v if w not in non_existent]
                for k, v in keywords.items()
            }
        else:
            keywords = [
                [w for w in topic if w not in non_existent]
                for topic in keywords
            ]

    else:
        if non_existent:
            raise ValueError(
                f"Some keywords are not found in the documents: {non_existent}"
            )

    if isinstance(keywords, dict):
        empty_topics = [k for k, v in keywords.items() if len(v) == 0]
    else:
        empty_topics = [str(i) for i, v in enumerate(keywords) if len(v) == 0]

    if empty_topics:
        raise ValueError(f"All keywords are pruned. Please check: {', '.join(empty_topics)}")

    return keywords
