import numpy as np
import pandas as pd
import re
import patsy

def abb_model_name(fullname):
    if fullname in ["base", "lda"]:
        return "base"
    elif fullname in ["cov", "ldacov"]:
        return "covariates"
    elif fullname in ["hmm", "ldahmm"]:
        return "dynamic"
    else:
        raise ValueError("Invalid full model name.")

def full_model_name(model: str, type: str = "keyATM") -> str:
    valid_models = ["base", "covariates", "dynamic"]
    valid_types = ["keyATM", "lda"]

    if model not in valid_models:
        raise ValueError(f"Please select a correct model. Got '{model}'")

    if type not in valid_types:
        raise ValueError(f"Please select a correct type. Got '{type}'")

    if type == "keyATM":
        if model == "base":
            return "base"
        elif model == "covariates":
            return "cov"
        elif model == "dynamic":
            return "hmm"
        else:
            raise ValueError("Please select a correct model.")
    elif type == "lda":
        if model == "base":
            return "lda"
        elif model == "covariates":
            return "ldacov"
        elif model == "dynamic":
            return "ldahmm"
        else:
            raise ValueError("Please select a correct model.")
    else:
        raise ValueError("Please select a correct type.")

def extract_full_model_name(obj):
    cls = obj.get("class", [])
    if "base" in cls:
        return "base"
    elif "cov" in cls:
        return "cov"
    elif "hmm" in cls:
        return "hmm"
    elif "lda" in cls:
        return "lda"
    elif "ldacov" in cls:
        return "ldacov"
    elif "ldahmm" in cls:
        return "ldahmm"
    else:
        raise Exception("Unknown model class")

def check_arg_type(arg, expected_type: str, message: str = None):
    """
    Checks if the argument has the expected class name in arg["class"].

    Parameters:
    - arg: dictionary-like object that should contain a "class" key
    - expected_type: the expected class name as a string
    - message: optional custom error message

    Raises:
    - TypeError: if the class does not match
    - KeyError: if "class" key is missing from arg
    """
    try:
        arg_classes = arg["class"]
    except KeyError:
        raise TypeError("Input argument does not contain a 'class' field.")

    if isinstance(arg_classes, str):
        arg_classes = [arg_classes]

    if expected_type not in arg_classes:
        if message is None:
            raise TypeError(f"`arg` is not a {expected_type}")
        else:
            raise TypeError(message)
    
def standardize(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    return (x - m) / s if s > 0 else x * 0.0

def covariates_standardize(data, type_, cov_formula=None):
    if cov_formula is None:
        return np.asarray(data)
    else:
        covariates_data_use = patsy.dmatrix(cov_formula, data=data, return_type="dataframe")

    if type_ == "none":
        return covariates_data_use.to_numpy()

    colnames_keep = covariates_data_use.columns.tolist()
    intercept_cols = {"(Intercept)", "Intercept"}

    if type_ == "all":
        standardize_cols = [c for c in colnames_keep if c not in intercept_cols]

    elif type_ == "non-factor":
        if isinstance(data, pd.DataFrame):
            factor_cols = [c for c in data.columns if pd.api.types.is_categorical_dtype(data[c])]
        else:
            factor_cols = []
        pattern = re.compile(
            "|".join(
                [r"^\(Intercept\)$", r"^Intercept$"]
                + [fr"^{re.escape(fc)}" for fc in factor_cols]
            )
        )
        standardize_cols = [c for c in colnames_keep if not pattern.search(c)]

    else:
        raise ValueError('Unknown option in `standardize`. It should be one of "all", "none", or "non-factor".')

    if len(standardize_cols) == 0:
        return covariates_data_use.to_numpy()
    else:
        standardized = []
        for col in colnames_keep:
            col_data = covariates_data_use[col].to_numpy()
            if col in standardize_cols:
                standardized.append(standardize(col_data))
            else:
                standardized.append(col_data)
        covariates_data_use = np.column_stack(standardized)
        return covariates_data_use
