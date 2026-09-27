import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import *

def plot_modelfit(x, start=1):
    """
    Show a diagnosis plot of log-likelihood and perplexity.

    Parameters
    ----------
    x : dict-like
        keyATM_output object, must contain "model_fit" (DataFrame or convertible).
    start : int
        Starting iteration (default=1).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    modelfit_long : pd.DataFrame
        The reshaped dataframe used for plotting.
    """
    check_arg_type(x, expected_type="keyATM_output")
    modelfit = x["model_fit"]

    if not isinstance(start, int):
        raise ValueError("`start` must be an integer.")

    if not isinstance(modelfit, pd.DataFrame):
        modelfit = pd.DataFrame(modelfit)

    if start is not None:
        modelfit = modelfit.loc[modelfit["Iteration"] >= start, :]

    modelfit_long = modelfit.melt(
        id_vars="Iteration",
        var_name="Measures",
        value_name="Value"
    )

    sns.set(style="whitegrid")
    g = sns.FacetGrid(modelfit_long, col="Measures", col_wrap=2, sharey=False, height=3)
    g.map_dataframe(sns.lineplot, x="Iteration", y="Value", color="C0")
    g.map_dataframe(sns.scatterplot, x="Iteration", y="Value", s=10, color="C0")
    g.set_axis_labels("Iteration", "Value")
    g.set_titles("{col_name}")
    g.fig.suptitle("Model Fit", y=1.02)

    return g.fig, modelfit_long
