import numpy as np
from scipy.stats import mannwhitneyu

def weighted_percentiles(values, weights, percentiles=(50, 90)):
    """
    Compute weighted percentiles of a distribution.

    Args:
        values (array-like): The data values.
        weights (array-like): The weights associated with each value.
        percentiles (tuple, optional): Percentiles to compute (0-100). Defaults to (50, 90).

    Returns:
        dict: A dictionary where keys are the requested percentiles and values are the computed weighted percentiles.
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

    Args:
        values (array-like): The data values.
        weights (array-like): The weights associated with each value.
        value (float): The value to compute the percentile rank for.

    Returns:
        float: The weighted percentile rank of the given value (0-100).
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

    Args:
        residuals (pd.Series or np.ndarray): Residuals from the log-log regression.
        method (str, optional): Strategy to determine the threshold. One of {'zscore', 'percentile', 'IQR'}. Defaults to 'zscore'.
            - 'zscore': mean + k * std (empirical rule).
            - 'percentile': top fraction (e.g., 5%).
            - 'IQR': upper whisker (Q3 + 1.5*IQR).
        top_frac (float, optional): For 'percentile' mode, defines the top fraction to label as outperformers. Defaults to 0.05.
        verbose (bool, optional): Whether to print the chosen threshold. Defaults to True.

    Returns:
        float: Residual threshold above which channels are labeled outperformers.

    Raises:
        ValueError: If the method is not one of {'zscore', 'percentile', 'IQR'}.
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

def calculate_smd(df, treat_col, covariate):
    """
    Calculates Standardized Mean Difference (SMD) to check balance between groups.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        treat_col (str): Column name for the treatment indicator (0/1 or False/True).
        covariate (str): Column name for the variable to compare.
        
    Returns:
        float: The SMD value.
    """
    treated = df[df[treat_col] == 1][covariate]
    control = df[df[treat_col] == 0][covariate]
    
    if len(treated) == 0 or len(control) == 0:
        return 0.0
        
    mean_diff = treated.mean() - control.mean()
    std_pool = np.sqrt((treated.var() + control.var()) / 2)
    
    return mean_diff / std_pool if std_pool > 0 else 0.0

def perform_mannwhitney_test(df, group_col, metric_col, alternative='greater', group1_label="Group 1", group2_label="Group 2"):
    """
    Performs a Mann-Whitney U test and prints a formatted summary.
    
    Args:
        df (pd.DataFrame): The dataframe.
        group_col (str): The binary column defining the groups (1 vs 0).
        metric_col (str): The metric to test.
        alternative (str): 'greater', 'less', or 'two-sided'.
        group1_label (str): Label for group=1 (e.g., 'Outperformers').
        group2_label (str): Label for group=0 (e.g., 'Control').
    
    Returns:
        tuple: (statistic, p_value)
    """
    group1 = df[df[group_col] == 1][metric_col].dropna()
    group2 = df[df[group_col] == 0][metric_col].dropna()
    
    stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    
    print(f"\n--- Mann-Whitney U Test: {metric_col} ---")
    print(f"Hypothesis: {group1_label} is {alternative} than {group2_label}")
    print(f"{group1_label} (Mean): {group1.mean():.2f}")
    print(f"{group2_label} (Mean): {group2.mean():.2f}")
    print(f"Statistic: {stat:.1f}, P-Value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant.")
    else:
        print("Result: Not Statistically Significant.")
        
    return stat, p_value