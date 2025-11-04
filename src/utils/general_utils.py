import numpy as np


def weighted_percentiles(values, weights, percentiles=(50,90)):
    """
    Compute weighted percentiles of a distribution.
    """
    values, weights = np.array(values), np.array(weights)
    sorted_asc = np.argsort(values)

    values, weights = values[sorted_asc], weights[sorted_asc]

    cumulative_sum_weights = np.cumsum(weights)

    cut_off = cumulative_sum_weights[-1] * np.array(percentiles) / 100.0

    return {p: np.interp(c, cumulative_sum_weights, values) for p, c in zip(percentiles, cut_off)}


def weighted_percentile_of_value(values, weights, value):
    """
    Compute the weighted percentile rank of a given value.

    Example: for value=1_000_000 subscribers,
    returns what percentile (0–100) that count corresponds to.
    """
    values = np.array(values)
    weights = np.array(weights)

    sorted_asc = np.argsort(values)

    values, weights = values[sorted_asc], weights[sorted_asc]

    cumulative_sum_weights = np.cumsum(weights)
    cumulative_frac = cumulative_sum_weights / cumulative_sum_weights[-1]

    percentile = np.interp(value, values, cumulative_frac) * 100
    return percentile


def select_gamma_threshold(residuals, method="zscore", top_frac=0.05, verbose=True):
    """
    Automatically select a threshold (gamma) for identifying outperformers.

    Parameters
    ----------
    residuals : pd.Series or np.ndarray
        Residuals from the log–log regression.
    method : {"zscore", "percentile", "IQR"}
        Strategy to determine the threshold:
        - "zscore": mean + k * std (empirical rule)
        - "percentile": top fraction (e.g., 5%)
        - "IQR": upper whisker (Q3 + 1.5*IQR)
    top_frac : float
        For percentile mode, defines the top fraction to label as outperformers.
    verbose : bool
        Whether to print the chosen threshold.

    Returns
    -------
    gamma : float
        Residual threshold above which channels are labeled outperformers.
    """
    residuals = np.asarray(residuals.dropna() if hasattr(residuals, "dropna") else residuals)

    if method == "zscore":
        gamma = residuals.mean() + 2 * residuals.std()

    elif method == "percentile":
        gamma = np.percentile(residuals, 100 * (1 - top_frac))

    elif method == "IQR":
        q1, q3 = np.percentile(residuals, [25, 75])
        gamma = q3 + 1.5 * (q3 - q1)

    else:
        raise ValueError("method must be one of {'zscore', 'percentile', 'IQR'}")

    if verbose:
        print(f" Selected gamma={gamma:.4f} using method='{method}'")

    return gamma
