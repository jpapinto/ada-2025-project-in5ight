import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patheffects as patheffects

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
    gamma=None,
    bins=60,
    color="gray",
    title="Distribution of Efficiency Residuals",
    figsize=(8, 5),
):
    """
    Plot the residual distribution with visual thresholds.
    Optionally include a custom gamma line (e.g., 97th percentile).

    Parameters
    ----------
    residuals : pd.Series or np.ndarray
        Residual values from the log–log efficiency model.
    gamma : float or None
        Optional threshold value (e.g., precomputed percentile).
    bins : int
        Number of histogram bins.
    color : str
        Histogram color.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    residuals = np.asarray(residuals.dropna() if hasattr(residuals, "dropna") else residuals)
    mean, std = residuals.mean(), residuals.std()

    plt.figure(figsize=figsize)
    sns.histplot(residuals, bins=bins, kde=True, color=color, alpha=0.6)

    # Reference lines
    plt.axvline(mean, color="dodgerblue", linestyle="--", label="Mean")
    plt.axvline(mean + 2 * std, color="salmon", linestyle="--", label="+2σ (z=2)")
    if gamma is not None:
        plt.axvline(gamma, color="limegreen", linestyle="--", label="Selected γ threshold")

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Residual (log-scale deviation)")
    plt.ylabel("Channel count")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()





def plot_category_bubble(
    df,
    size_col,
    value_col,
    category_col,
    color_col=None,
    size_scale=25,
    log_x=True,
    title="Category-Level Bubble Chart",
    xlabel="Average Size (log scale)",
    ylabel="Average Value",
    cmap="coolwarm"
):
    import matplotlib.patheffects as patheffects

    grouped = (
        df.groupby(category_col)
        .agg(
            avg_value=(value_col, "mean"),
            avg_size=(size_col, "mean"),
            color_metric=(color_col if color_col else size_col, "mean"),
            count=(category_col, "count"),
        )
        .reset_index()
    )

    sizes = np.sqrt(grouped["count"]) * size_scale

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        grouped["avg_size"],
        grouped["avg_value"],
        s=sizes,
        c=grouped["color_metric"],
        cmap=cmap,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    if log_x:
        plt.xscale("log")

    for _, row in grouped.iterrows():
        plt.text(
            row["avg_size"], row["avg_value"], row[category_col],
            fontsize=8.5, ha="center", va="center", weight="bold",
            color="white",  # uniform text color
            path_effects=[patheffects.withStroke(linewidth=0.8, foreground="black", alpha=0.8)]
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label(color_col if color_col else size_col, rotation=270, labelpad=15)

    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()





def plot_timeseries_trend(
    df,
    date_col,
    value_col,
    title="Time Series Trend",
    xlabel=None,
    ylabel=None,
    figsize=(12, 6),
    smooth=True,
    window=6,
    color="seagreen",
    alpha=0.25,        # transparency of the shaded band
    band_width=0.5,    # fraction of SD to display (e.g., 0.5 = ±0.5 SD)
):
    """
    Plot a time series trend with mean and a narrower standard deviation band.
    """

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    # Aggregate mean and std per time point
    trend = (
        data.groupby(date_col)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg", "std": "std"})
    )

    # Optional smoothing
    if smooth:
        trend["avg"] = trend["avg"].rolling(window, min_periods=1).mean()
        trend["std"] = trend["std"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=figsize)

    # Plot the main mean line
    plt.plot(trend[date_col], trend["avg"], color=color, linewidth=2.5, label="Mean")

    # Plot a smaller SD band
    plt.fill_between(
        trend[date_col],
        trend["avg"] - band_width * trend["std"],
        trend["avg"] + band_width * trend["std"],
        color=color,
        alpha=alpha,
        label=f"±{band_width:.1f} SD"
    )

    plt.title(title, weight="bold")
    plt.xlabel(xlabel if xlabel else date_col)
    plt.ylabel(ylabel if ylabel else value_col)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()







def plot_channel_comparison(
    df,
    target_channel,
    avg_channel_id,
    efficiency_col="efficiency_4week_filled",
    week_col="week_start",
    window=6,
    figsize=(12, 6),
):
    """
    Plot engagement efficiency of an outperformer channel versus an average baseline channel over time.

    Args:
        df: DataFrame containing timeseries data with columns [channel, efficiency_col, week_col].
        target_channel: Channel ID of the outperformer.
        avg_channel_id: Channel ID of the average baseline channel.
        efficiency_col: Column name for efficiency metric.
        week_col: Column name for week start date.
        window: Rolling window size for smoothing.
        figsize: Figure size for the plot.
    """

    target_ts = df.loc[df["channel"] == target_channel].copy()
    avg_ts = df.loc[df["channel"] == avg_channel_id].copy()

    target_ts.loc[:, "eff_smooth"] = (
        target_ts[efficiency_col].rolling(window, min_periods=1).mean()
    )
    avg_ts.loc[:, "eff_smooth"] = (
        avg_ts[efficiency_col].rolling(window, min_periods=1).mean()
    )

    plt.figure(figsize=figsize)

    plt.plot(
        target_ts[week_col],
        target_ts["eff_smooth"],
        color="seagreen",
        linewidth=2.5,
        label="Outperformer Channel",
    )

    plt.plot(
        avg_ts[week_col],
        avg_ts["eff_smooth"],
        color="red",
        linewidth=2,
        label="Average Gaming Channel (gaps = no uploads)",
    )

    plt.yscale("log")
    plt.title("Outperformer vs. Average Gaming Channel Over Time", weight="bold")
    plt.xlabel("Week")
    plt.ylabel("4-Week Engagement Efficiency (log scale)")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()