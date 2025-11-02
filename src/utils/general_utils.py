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