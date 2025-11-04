import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_matrices(df_list, names, figsize=(16, 16), cmap="coolwarm"):
    """
    Plot correlation matrices for multiple DataFrames.

    Args:
        df_list (list[pd.DataFrame]): List of DataFrames to plot.
        names (list[str]): List of names corresponding to each DataFrame.
        figsize (tuple, optional): Figure size for the grid.
        cmap (str, optional): Color map for heatmaps.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    def _plot_correlation_matrix(ax, df, title):
        corr = df.corr(numeric_only=True)
        if corr.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
            return
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=ax)
        ax.set_title(title)

    n = len(df_list)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, df, name in zip(axes, df_list, names):
        numeric_cols = df.select_dtypes(include=["float64", "int64"])
        _plot_correlation_matrix(ax, numeric_cols, f"Correlation Matrix: {name}")

    for ax in axes[len(df_list):]:
        ax.axis("off")

    plt.tight_layout()
    return fig





def plot_log_histogram(
    df, column, percentiles=None, figsize=(8, 5),
    bins=np.logspace(3, 8, 100), color="gray", alpha=0.6,
    title=None, xlabel=None, ylabel="Count"
):
    """
    Plot a histogram with a log-scaled x-axis and optional percentile markers.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the numeric column to plot.
        percentiles (dict[str, float], optional): Dict of {label: value} for percentile markers.
        figsize (tuple, optional): Size of the figure.
        bins (array-like, optional): Bin edges for histogram.
        color (str, optional): Color of histogram bars.
        alpha (float, optional): Transparency of histogram bars.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
    """
    data = df[column].dropna()

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor="white")

    # Add percentile lines
    if percentiles:
        for label, value in percentiles.items():
            plt.axvline(value, linestyle="--", linewidth=1.5, label=f"{label} = {value:,.0f}")

    plt.xscale("log")
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.title(title or f"Distribution of {column}")
    plt.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_residual_distribution(
    residuals,
    gamma_percentile=97,
    bins=60,
    color="gray",
    title="Distribution of Efficiency Residuals",
    figsize=(8, 5),
):
    """
    Plot the residual distribution with key thresholds.

    Parameters
    ----------
    residuals : pd.Series or np.ndarray
        Residual values from the log–log efficiency model.
    gamma_percentile : float
        Percentile threshold (e.g. 97 means top 3% outperformers).
    bins : int
        Number of histogram bins.
    color : str
        Base color for histogram.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    residuals = np.asarray(residuals.dropna() if hasattr(residuals, "dropna") else residuals)
    mean, std = residuals.mean(), residuals.std()
    gamma = np.percentile(residuals, gamma_percentile)

    plt.figure(figsize=figsize)
    sns.histplot(residuals, bins=bins, kde=True, color=color, alpha=0.6)
    plt.axvline(mean, color="blue", linestyle="--", label="Mean")
    plt.axvline(mean + 2 * std, color="red", linestyle="--", label="+2σ (z≈2)")
    plt.axvline(gamma, color="lime", linestyle="--", label=f"{gamma_percentile}th percentile")
    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Residual (log-scale deviation)")
    plt.ylabel("Channel count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"📊 gamma ({gamma_percentile}th percentile) = {gamma:.4f}")
    return gamma