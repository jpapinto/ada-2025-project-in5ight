import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_log_log_model(
    df,
    x_col,
    y_col,
    weight_col=None,
    log_base=10,
    verbose=True,
):
    """
    Fit a general weighted log–log linear regression model:
        log(y) = a + b * log(x) + ε

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column name for the independent variable (e.g. subscribers).
    y_col : str
        Column name for the dependent variable (e.g. efficiency).
    weight_col : str or None
        Optional column with sample weights.
    log_base : int
        Logarithm base to use (default 10).
    verbose : bool
        Whether to print model summary.

    Returns
    -------
    data : pd.DataFrame
        Copy of df with added columns:
        - log_x
        - log_y
        - pred_log_y
        - residual
        - zscore

        Model parameters stored in:
        data.attrs["model"] = {"a": a, "b": b}
    """
    mask = (df[x_col] > 0) & (df[y_col] > 0)
    data = df.loc[mask].copy()

    data["log_x"] = np.log(data[x_col]) / np.log(log_base)
    data["log_y"] = np.log(data[y_col]) / np.log(log_base)

    X = data[["log_x"]].values
    y = data["log_y"].values
    weights = data[weight_col].values if weight_col and weight_col in data.columns else None

    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    a, b = model.intercept_, model.coef_[0]

    data["pred_log_y"] = model.predict(X)
    data["residual"] = data["log_y"] - data["pred_log_y"]
    data["zscore"] = (data["residual"] - data["residual"].mean()) / data["residual"].std()

    if verbose:
        print(f"log–log model fitted: log(y) = {a:.4f} + {b:.4f} * log(x)")

    data.attrs["model"] = {"a": a, "b": b}
    return data