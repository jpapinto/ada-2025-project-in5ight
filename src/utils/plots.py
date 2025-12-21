import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.colors import TwoSlopeNorm




def plot_correlation_matrix(axis, dataframe, title, colormap="coolwarm"):
        correlation_matrix = dataframe.corr(numeric_only=True)
        if correlation_matrix.empty:
            axis.axis("off")
            axis.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
            return
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=colormap, cbar=True, ax=axis)
        axis.set_title(title)



def plot_correlation_matrices(dataframes, dataframe_names, figsize=(16, 16), colormap="coolwarm"):
    """
    Plot correlation matrices for multiple DataFrames.

    Args:
        dataframes (list[pd.DataFrame]): List of DataFrames to plot.
        dataframe_names (list[str]): List of names corresponding to each DataFrame.
        figsize (tuple, optional): Figure size for the grid. Defaults to (16, 16).
        colormap (str, optional): Color map for heatmaps. Defaults to "coolwarm".

    Returns:
        matplotlib.figure.Figure: The created figure containing the correlation matrices.
    """


    num_dataframes = len(dataframes)
    num_columns = 2
    num_rows = (num_dataframes + num_columns - 1) // num_columns

    figure, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=figsize)
    axes = axes.flatten() if num_dataframes > 1 else [axes]

    for axis, dataframe, name in zip(axes, dataframes, dataframe_names):
        numeric_columns = dataframe.select_dtypes(include=["float64", "int64"])
        plot_correlation_matrix(axis, numeric_columns, f"Correlation Matrix: {name}", colormap=colormap)

    for axis in axes[len(dataframes):]:
        axis.axis("off")

    plt.tight_layout()
    return figure



def plot_streamed_correlation(
    stream_fn,
    path,
    nrows=None,
    stream_kwargs=None,
    title="Correlation Matrix (streamed)",
    figsize=(8, 6),
    colormap="coolwarm",
):
    """
    Compute and plot a correlation matrix from a streamed DataFrame source.
    Numeric columns are inferred from the first non-empty chunk.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    stream_kwargs = stream_kwargs or {}

    n = 0
    mean = None
    M2 = None
    numeric_columns = None

    for chunk in stream_fn(path=path, **stream_kwargs):
        # Infer numeric columns once
        if numeric_columns is None:
            numeric_columns = (
                chunk
                .select_dtypes(include=["int64", "float64"])
                .columns
                .tolist()
            )

            if not numeric_columns:
                raise ValueError("No numeric columns found in streamed data")

        chunk = chunk[numeric_columns].dropna()
        if chunk.empty:
            continue

        # Respect nrows cap
        if nrows is not None and n + len(chunk) > nrows:
            chunk = chunk.iloc[: nrows - n]

        x = chunk.to_numpy(dtype=np.float64)
        k = x.shape[0]

        if k == 0:
            break

        if mean is None:
            mean = x.mean(axis=0)
            M2 = np.zeros((x.shape[1], x.shape[1]))

        delta = x.mean(axis=0) - mean
        mean += delta * k / (n + k)
        M2 += (x - mean).T @ (x - mean)
        n += k

        if nrows is not None and n >= nrows:
            break

    if n < 2:
        raise ValueError("Not enough data to compute correlation")

    cov = M2 / (n - 1)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    corr_df = pd.DataFrame(
        corr,
        index=numeric_columns,
        columns=numeric_columns,
    )

    fig, ax = plt.subplots(figsize=figsize)
    plot_correlation_matrix(ax, corr_df, title, colormap=colormap)
    plt.tight_layout()

    return fig





def plot_log_histogram(
    dataframe, column_name, percentile_markers=None, figsize=(8, 5),
    bins=np.logspace(3, 8, 100), bar_color="gray", transparency=0.6,
    plot_title=None, x_axis_label=None, y_axis_label="Count"
):
    """
    Plot a histogram with a log-scaled x-axis and optional percentile markers.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the numeric column to plot.
        percentile_markers (dict[str, float], optional): Dict of {label: value} for percentile markers. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 5).
        bins (array-like, optional): Bin edges for histogram. Defaults to np.logspace(3, 8, 100).
        bar_color (str, optional): Color of histogram bars. Defaults to "gray".
        transparency (float, optional): Transparency of histogram bars. Defaults to 0.6.
        plot_title (str, optional): Plot title. Defaults to None.
        x_axis_label (str, optional): X-axis label. Defaults to None.
        y_axis_label (str, optional): Y-axis label. Defaults to "Count".

    Returns:
        None
    """
    data = dataframe[column_name].dropna()

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=bar_color, alpha=transparency, edgecolor="white")

    # Add percentile lines if provided
    if percentile_markers:
        for label, value in percentile_markers.items():
            plt.axvline(value, linestyle="--", linewidth=1.5, label=f"{label} = {value:,.0f}")

    plt.xscale("log")
    plt.xlabel(x_axis_label or column_name)
    plt.ylabel(y_axis_label)
    plt.title(plot_title or f"Distribution of {column_name}")
    plt.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    plt.legend()
    plt.tight_layout()

def plot_residual_distribution(
    residual_values,
    threshold_gamma=None,
    num_bins=60,
    bar_color="gray",
    plot_title="Distribution of Efficiency Residuals",
    figsize=(8, 5),
):
    """
    Plot the residual distribution with visual thresholds.

    Args:
        residual_values (pd.Series or np.ndarray): Residual values from the log-log efficiency model.
        threshold_gamma (float, optional): Optional threshold value (e.g., precomputed percentile). Defaults to None.
        num_bins (int, optional): Number of histogram bins. Defaults to 60.
        bar_color (str, optional): Histogram color. Defaults to "gray".
        plot_title (str, optional): Plot title. Defaults to "Distribution of Efficiency Residuals".
        figsize (tuple, optional): Figure size. Defaults to (8, 5).

    Returns:
        None
    """
    residual_values = np.asarray(residual_values.dropna() if hasattr(residual_values, "dropna") else residual_values)
    mean_value, std_dev = residual_values.mean(), residual_values.std()

    plt.figure(figsize=figsize)
    sns.histplot(residual_values, bins=num_bins, kde=True, color=bar_color, alpha=0.6)

    # Add reference lines for mean and standard deviation
    plt.axvline(mean_value, color="dodgerblue", linestyle="--", label="Mean")
    plt.axvline(mean_value + 2 * std_dev, color="salmon", linestyle="--", label="+2σ (z=2)")
    if threshold_gamma is not None:
        plt.axvline(threshold_gamma, color="limegreen", linestyle="--", label="Selected γ threshold")

    plt.title(plot_title, fontsize=13, weight="bold")
    plt.xlabel("Residual (log-scale deviation)")
    plt.ylabel("Channel count")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_category_bubble(
    dataframe,
    size_column,
    value_column,
    category_column,
    color_column=None,
    size_scaling_factor=25,
    use_log_scale_x=True,
    plot_title="Category-Level Bubble Chart",
    x_axis_label="Average Size (log scale)",
    y_axis_label="Average Value",
    color_map="coolwarm"
):
    """
    Plot a bubble chart for category-level data.
    """
    # Compute group averages
    grouped_data = (
        dataframe.groupby(category_column)
        .agg(
            avg_value=(value_column, "mean"),
            avg_size=(size_column, "mean"),
            count=(category_column, "count"),
        )
        .reset_index()
    )

    # Handle color metric
    if color_column:
        # If numeric or boolean → take mean; if categorical → take mode
        if np.issubdtype(dataframe[color_column].dtype, np.number) or dataframe[color_column].dtype == bool:
            grouped_data["color_metric"] = dataframe.groupby(category_column)[color_column].mean().values
        else:
            grouped_data["color_metric"] = (
                dataframe.groupby(category_column)[color_column]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                .values
            )
    else:
        grouped_data["color_metric"] = grouped_data["avg_size"]

    # Bubble sizes scaled by sqrt(count)
    bubble_sizes = np.sqrt(grouped_data["count"]) * size_scaling_factor

    plt.figure(figsize=(10, 6))
    scatter_plot = plt.scatter(
        grouped_data["avg_size"],
        grouped_data["avg_value"],
        s=bubble_sizes,
        c=grouped_data["color_metric"],
        cmap=color_map if np.issubdtype(grouped_data["color_metric"].dtype, np.number) else None,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    if use_log_scale_x:
        plt.xscale("log")

    # Annotate category names
    for _, row in grouped_data.iterrows():
        plt.text(
            row["avg_size"], row["avg_value"], row[category_column],
            fontsize=8.5, ha="center", va="center", weight="bold",
            color="white",
            path_effects=[patheffects.withStroke(linewidth=0.8, foreground="black", alpha=0.8)]
        )

    # Add colorbar only if numeric
    if np.issubdtype(grouped_data["color_metric"].dtype, np.number):
        colorbar = plt.colorbar(scatter_plot)
        colorbar.set_label(color_column or size_column, rotation=270, labelpad=15)

    plt.title(plot_title, fontsize=14, weight="bold")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
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
    alpha=0.25,
    band_width=0.5,
):
    """
    Plot a time series trend with mean and a narrower standard deviation band.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        date_col (str): Column name for dates.
        value_col (str): Column name for values.
        title (str, optional): Plot title. Defaults to "Time Series Trend".
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        smooth (bool, optional): Whether to smooth the trend. Defaults to True.
        window (int, optional): Rolling window size for smoothing. Defaults to 6.
        color (str, optional): Line color. Defaults to "seagreen".
        alpha (float, optional): Transparency of the shaded band. Defaults to 0.25.
        band_width (float, optional): Fraction of standard deviation for the band. Defaults to 0.5.

    Returns:
        None
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    trend = (
        data.groupby(date_col)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg", "std": "std"})
    )

    if smooth:
        trend["avg"] = trend["avg"].rolling(window, min_periods=1).mean()
        trend["std"] = trend["std"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=figsize)

    plt.plot(trend[date_col], trend["avg"], color=color, linewidth=2.5, label="Mean")

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
        df (pd.DataFrame): DataFrame containing timeseries data with columns [channel, efficiency_col, week_col].
        target_channel (str): Channel ID of the outperformer.
        avg_channel_id (str): Channel ID of the average baseline channel.
        efficiency_col (str, optional): Column name for efficiency metric. Defaults to "efficiency_4week_filled".
        week_col (str, optional): Column name for week start date. Defaults to "week_start".
        window (int, optional): Rolling window size for smoothing. Defaults to 6.
        figsize (tuple, optional): Figure size for the plot. Defaults to (12, 6).

    Returns:
        None
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

def plot_comparison_boxplot(
    data, 
    x_col, 
    y_col, 
    hue_col=None, 
    title="Group Comparison", 
    xlabel="Group", 
    ylabel="Value", 
    palette=['#332288', '#CC6677'],
    showfliers=False,
    figsize=(10, 6),
    xticklabels=None
):
    """
    Plots a boxplot to compare distributions between groups.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        showfliers=showfliers,
        legend=False
    )
    
    plt.title(title, weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', alpha=0.3)
    
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        
    plt.show()

def plot_dual_metric_boxplot(
    data, 
    x_col, 
    metric1, 
    metric2, 
    titles, 
    ylabels, 
    xticklabels=None, 
    palette=['#332288', '#CC6677'],
    figsize=(16, 6)
):
    """
    Plots two boxplots side-by-side for dual-metric analysis.
    
    Args:
        data (pd.DataFrame): The dataset.
        x_col (str): Grouping column.
        metric1 (str): Column name for left plot.
        metric2 (str): Column name for right plot.
        titles (list): [Title 1, Title 2].
        ylabels (list): [Y-label 1, Y-label 2].
        xticklabels (list, optional): Custom labels for x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1
    sns.boxplot(
        data=data, x=x_col, y=metric1, hue=x_col, 
        palette=palette, showfliers=False, ax=axes[0], legend=False
    )
    axes[0].set_title(titles[0], weight='bold')
    axes[0].set_ylabel(ylabels[0])
    axes[0].grid(True, axis='y', alpha=0.3)
    if xticklabels:
        axes[0].set_xticks(range(len(xticklabels)))
        axes[0].set_xticklabels(xticklabels)

    # Plot 2
    sns.boxplot(
        data=data, x=x_col, y=metric2, hue=x_col, 
        palette=palette, showfliers=False, ax=axes[1], legend=False
    )
    axes[1].set_title(titles[1], weight='bold')
    axes[1].set_ylabel(ylabels[1])
    axes[1].grid(True, axis='y', alpha=0.3)
    if xticklabels:
        axes[1].set_xticks(range(len(xticklabels)))
        axes[1].set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.show()


def plot_growth_trajectory(
    trajectory_df,
    x_col='week_num',
    y_col='cum_subs',
    hue_col='is_outperformer',
    title="Cumulative Growth Trajectory",
    xlabel="Weeks Relative to Start",
    ylabel="Cumulative Subscriber Gain",
    palette=['#332288', '#CC6677'], # Default: [Control, Outperformer]
    figsize=(12, 6)
):
    """
    Plots the median cumulative growth trajectory for two groups using Seaborn.
    
    Args:
        trajectory_df (pd.DataFrame): Dataframe containing the aggregated trajectory data.
        x_col (str): Column for relative time (x-axis).
        y_col (str): Column for the metric (y-axis).
        hue_col (str): Column defining the groups (Control vs Outperformer).
        palette (list): Colors for [Group 0, Group 1].
    """
    plt.figure(figsize=figsize)
    
    # Plot Lines
    sns.lineplot(
        data=trajectory_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        linewidth=3,
        legend=False
    )
    
    # Add custom legend manually to ensure correct labels
    custom_lines = [
        Line2D([0], [0], color=palette[0], lw=3),
        Line2D([0], [0], color=palette[1], lw=3)
    ]
    plt.legend(custom_lines, ['Control (Median)', 'Outperformer (Median)'], loc='upper left', frameon=False)

    plt.title(title, weight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    # Small annotation to identify the break-even point
    plt.axvline(x=160, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def pretty_name(v: str) -> str:
    """
    Converts internal variable names into nicer, human-readable labels for plots and tables.

    Notes:
        - 'log_duration' is displayed as 'log(duration)'.
        - Variables like 'pace8_median' become 'pace (8w)'.
        - Variables like 'cons8_median' become 'consistency (8w)'.
        - Unrecognized names are returned unchanged.

    Args:
        v (str): Variable name.

    Returns:
        str: A prettified label suitable for plot axes and table headers.
    """
    # nicer labels in plots
    if v == "log_duration":
        return "log(duration)"
    if v.startswith("pace") and v.endswith("_median"):
        w = v.replace("pace", "").replace("_median", "")
        return f"pace ({w}w)"
    if v.startswith("cons") and v.endswith("_median"):
        w = v.replace("cons", "").replace("_median", "")
        return f"consistency ({w}w)"
    return v


def _prep_struct_cols(
    d: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
):
    d = d.copy().replace([np.inf, -np.inf], np.nan)
    d["log_duration"] = np.log(pd.to_numeric(d[duration_col], errors="coerce").where(d[duration_col] > 0))

    cols = [pace_col, cons_col, "log_duration"]
    X = (
        d[cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return X, cols


def _compute_vif_series(X: pd.DataFrame) -> pd.Series:
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty or len(X) < 5:
        return pd.Series({c: np.nan for c in X.columns})

    Xv = X.assign(const=1.0)
    vifs = {}
    for i, c in enumerate(Xv.columns):
        if c == "const":
            continue
        vifs[c] = float(variance_inflation_factor(Xv.values, i))
    return pd.Series(vifs)


def vif_table_small_mid(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
) -> pd.DataFrame:
    """
    Computes a Small vs Mid VIF table for structural predictors (pace, consistency, log_duration).

    Notes:
        - Builds log_duration as log(duration_col) for duration_col > 0, else NaN.
        - Drops rows with NaNs/infs in any of the structural predictors before VIF computation.
        - If there are fewer than 5 usable rows for a tier, VIFs are returned as NaN for that tier.
        - Output columns are renamed using pretty_name for readability.

        Internal helper definitions used here:
            _prep_struct_cols(d, pace_col, cons_col, duration_col):
                Returns (X, cols) where X is the cleaned numeric design matrix containing
                [pace_col, cons_col, log_duration], and cols is the list of column names.
            _compute_vif_series(X):
                Returns a Series mapping column name -> VIF, computed by adding a constant and
                using variance_inflation_factor. Returns NaNs if X is too small.

    Args:
        small_df (pd.DataFrame): Data for the Small tier.
        mid_df (pd.DataFrame): Data for the Mid tier.
        pace_col (str): Column name for posting pace predictor.
        cons_col (str): Column name for posting consistency predictor.
        duration_col (str): Column name for duration used to compute log_duration.
            Defaults to 'duration_median'.

    Returns:
        pd.DataFrame: A 2-row table indexed by tier with VIF values for each structural predictor.
    """
    rows = []
    for tier, d in [("small", small_df), ("mid", mid_df)]:
        X, cols = _prep_struct_cols(d, pace_col, cons_col, duration_col)
        vif = _compute_vif_series(X)
        rows.append(pd.Series(vif, name=tier))

    out = pd.DataFrame(rows)
    out = out.rename(columns={c: pretty_name(c) for c in out.columns})
    return out


def _pval_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _bin_edges_global(s: pd.Series, bins: int = 8, q_low: float = 0.02, q_high: float = 0.98) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.array([0.0, 1.0])

    lo = float(s.quantile(q_low))
    hi = float(s.quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(s.min())
        hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.array([0.0, 1.0])

    edges = np.unique(np.linspace(lo, hi, bins + 1))
    if len(edges) < 3:
        edges = np.unique(np.linspace(float(s.min()), float(s.max()), bins + 1))
    return edges


def _centers_and_ranges(edges: np.ndarray):
    centers = (edges[:-1] + edges[1:]) / 2.0
    ranges = list(zip(edges[:-1], edges[1:]))
    return centers, ranges


def _full_categories_from_edges(edges: np.ndarray):
    centers, _ = _centers_and_ranges(edges)
    return pd.cut(pd.Series(centers), bins=edges, include_lowest=True).astype("category").cat.categories


def _grid_fixed_edges(df: pd.DataFrame, x: str, y: str, z: str, x_edges: np.ndarray, y_edges: np.ndarray):
    d = df[[x, y, z]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d[z] = pd.to_numeric(d[z], errors="coerce")
    d = d.dropna()

    if d.empty:
        return None

    x_centers, x_ranges = _centers_and_ranges(x_edges)
    y_centers, y_ranges = _centers_and_ranges(y_edges)

    d["xbin"] = pd.cut(d[x], bins=x_edges, include_lowest=True)
    d["ybin"] = pd.cut(d[y], bins=y_edges, include_lowest=True)

    full_x = _full_categories_from_edges(x_edges)
    full_y = _full_categories_from_edges(y_edges)

    # >>> observed=True to silence the warning and match future default
    grid = d.pivot_table(index="ybin", columns="xbin", values=z, aggfunc="median", observed=True)
    cnt  = d.pivot_table(index="ybin", columns="xbin", values=z, aggfunc="size",   observed=True)

    grid = grid.reindex(index=full_y, columns=full_x)
    cnt  = cnt.reindex(index=full_y, columns=full_x)

    Z = grid.values.astype(float)
    N = cnt.values.astype(float)

    # flip so y increases bottom -> top visually
    Z = Z[::-1, :]
    N = N[::-1, :]
    y_centers = y_centers[::-1]
    y_ranges = y_ranges[::-1]

    return Z, N, x_centers, y_centers, x_ranges, y_ranges


def plot_corr_small_mid(
    small_df,
    mid_df,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
    cmap="coolwarm",
):
    """
    Plots absolute correlation heatmaps of structural predictors for Small and Mid tiers.

    Notes:
        - Computes log_duration as log(duration_col) for duration_col > 0, else NaN.
        - Drops rows with NaNs/infs in [pace_col, cons_col, log_duration] before computing correlations.
        - Displays absolute correlations in [0, 1].
        - Produces two side-by-side heatmaps with a shared colorbar.

    Args:
        small_df (pd.DataFrame): Data for the Small tier.
        mid_df (pd.DataFrame): Data for the Mid tier.
        pace_col (str): Column name for posting pace predictor.
        cons_col (str): Column name for posting consistency predictor.
        duration_col (str): Column name for duration used to compute log_duration.
            Defaults to 'duration_median'.
        cmap: Matplotlib colormap passed to seaborn. Defaults to 'coolwarm'.

    Returns:
        None: Displays the plot via matplotlib.
    """
    Xs, cols = _prep_struct_cols(small_df, pace_col, cons_col, duration_col)
    Xm, _ = _prep_struct_cols(mid_df, pace_col, cons_col, duration_col)

    corr_s = Xs.corr().abs()
    corr_m = Xm.corr().abs()

    labels = [pretty_name(c) for c in cols]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    # shared colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])

    for ax, (title, corr, show_cbar) in zip(
        axes,
        [("Small Tier", corr_s, True), ("Mid Tier", corr_m, False)],
    ):
        sns.heatmap(
            corr.values,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Correlations (absolute) by tier", y=1.02)
    plt.show()


def plot_effects_bars(
    effects_df: pd.DataFrame,
    tier_order=("small", "mid"),
    var_order=("log_duration", "cons8_median", "pace8_median"),
    eff_col="%ΔE_p25->p75",
    lo_col="%ΔE_CI_low",
    hi_col="%ΔE_CI_high",
    whisker_color="0.45",  # gray
    palette=['#332288', '#CC6677'],
):
    """
    Plots grouped horizontal bars of p25->p75 effects with asymmetric confidence intervals.

    Notes:
        - Expects effects_df to contain one row per (tier, var) with effect and CI columns.
        - Filters to tiers in tier_order (if present) and vars in var_order (if present).
          If none match, falls back to sorting unique tiers/vars found in the data.
        - Builds a wide table by pivoting on ['var'] x ['tier'] using the first value per cell.
        - Uses a deterministic mapping tier -> color from the provided palette.

    Args:
        effects_df (pd.DataFrame): Effect table with columns including:
            'tier', 'var', eff_col, lo_col, hi_col.
        tier_order (tuple): Preferred ordering of tiers in the plot. Defaults to ('small', 'mid').
        var_order (tuple): Preferred ordering of variables in the plot.
            Defaults to ('log_duration', 'cons8_median', 'pace8_median').
        eff_col (str): Column name for the point estimate. Defaults to '%ΔE_p25->p75'.
        lo_col (str): Column name for the lower CI bound. Defaults to '%ΔE_CI_low'.
        hi_col (str): Column name for the upper CI bound. Defaults to '%ΔE_CI_high'.
        whisker_color (str): Matplotlib color for CI whiskers. Defaults to '0.45'.
        palette (list): List of colors used for tiers in order. Defaults to two colors.

    Returns:
        None: Displays the plot via matplotlib.
    """
    df = effects_df.copy()
    if df.empty:
        print("[warn] effects_df is empty")
        return

    df["tier"] = df["tier"].astype(str)
    tiers = [t for t in tier_order if t in set(df["tier"])]
    if not tiers:
        tiers = sorted(df["tier"].unique())

    df["var"] = df["var"].astype(str)
    vars_keep = [v for v in var_order if v in set(df["var"])]
    if not vars_keep:
        vars_keep = sorted(df["var"].unique())

    df = df[df["tier"].isin(tiers) & df["var"].isin(vars_keep)].copy()
    if df.empty:
        print("[warn] no rows after filtering")
        return

    wide = df.pivot_table(index="var", columns="tier", values=eff_col, aggfunc="first").reindex(vars_keep)
    lo = df.pivot_table(index="var", columns="tier", values=lo_col, aggfunc="first").reindex(vars_keep)
    hi = df.pivot_table(index="var", columns="tier", values=hi_col, aggfunc="first").reindex(vars_keep)

    y = np.arange(len(vars_keep))
    bar_h = 0.35
    offsets = np.linspace(-bar_h / 2, bar_h / 2, num=len(tiers))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axvline(0, linewidth=1, color="0.6")

    # map tier -> color deterministically
    tier_colors = {t: palette[i % len(palette)] for i, t in enumerate(tiers)}

    for j, tier in enumerate(tiers):
        vals = wide[tier].to_numpy(dtype=float)
        lo_vals = lo[tier].to_numpy(dtype=float)
        hi_vals = hi[tier].to_numpy(dtype=float)

        yy = y + offsets[j]
        ax.barh(
            yy, vals,
            height=bar_h / max(len(tiers), 1) * 1.8,
            label=tier,
            alpha=0.90,
            color=tier_colors[tier],
        )

        xerr_low = vals - lo_vals
        xerr_hi = hi_vals - vals
        ax.errorbar(
            vals, yy,
            xerr=[xerr_low, xerr_hi],
            fmt="none",
            ecolor=whisker_color,
            linewidth=2,
            capsize=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_name(v) for v in vars_keep])
    ax.set_xlabel("% change in efficiency")
    ax.set_title("Structural factor effects by tier (p25 → p75, back-translated)")
    ax.legend(title="tier", frameon=False)
    ax.grid(True, axis="x", linewidth=0.8, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_lo = df[lo_col].to_numpy(dtype=float)
    all_hi = df[hi_col].to_numpy(dtype=float)
    xmin = float(np.nanmin(all_lo))
    xmax = float(np.nanmax(all_hi))
    pad = 0.10 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    plt.tight_layout()
    plt.show()


def plot_coef_bars(
    coefs_df: pd.DataFrame,
    tier_order=("small", "mid"),
    var_order=("log_duration", "cons8_median", "pace8_median"),
    whisker_color="0.45",  # gray
    palette=['#332288', '#CC6677'],
):
    """
    Plots grouped horizontal bars of coefficients with 95% CI whiskers and p-value stars.

    Notes:
        - Computes CI as beta ± 1.96 * se.
        - P-value stars are assigned using:
            p < 0.001: '***', p < 0.01: '**', p < 0.05: '*', else ''.
        - Filters to tiers in tier_order (if present) and vars in var_order (if present).
          If none match, falls back to sorting unique tiers/vars found in the data.
        - Uses a deterministic mapping tier -> color from the provided palette.

        Internal helper definition used here:
            _pval_stars(p):
                Returns '', '*', '**', or '***' based on p-value thresholds above.

    Args:
        coefs_df (pd.DataFrame): Coefficient table with columns including:
            'tier', 'var', 'beta', 'se', 'pval'.
        tier_order (tuple): Preferred ordering of tiers in the plot. Defaults to ('small', 'mid').
        var_order (tuple): Preferred ordering of variables in the plot.
            Defaults to ('log_duration', 'cons8_median', 'pace8_median').
        whisker_color (str): Matplotlib color for CI whiskers. Defaults to '0.45'.
        palette (list): List of colors used for tiers in order. Defaults to two colors.

    Returns:
        None: Displays the plot via matplotlib.
    """
    df = coefs_df.copy()
    if df.empty:
        print("[warn] coefs_df is empty")
        return

    df["tier"] = df["tier"].astype(str)
    tiers = [t for t in tier_order if t in set(df["tier"])]
    if not tiers:
        tiers = sorted(df["tier"].unique())

    df["var"] = df["var"].astype(str)
    vars_keep = [v for v in var_order if v in set(df["var"])]
    if not vars_keep:
        vars_keep = sorted(df["var"].unique())

    df = df[df["tier"].isin(tiers) & df["var"].isin(vars_keep)].copy()
    if df.empty:
        print("[warn] no rows after filtering")
        return

    df["ci_low"] = df["beta"] - 1.96 * df["se"]
    df["ci_high"] = df["beta"] + 1.96 * df["se"]
    df["stars"] = df["pval"].apply(_pval_stars)

    wide = df.pivot_table(index="var", columns="tier", values="beta", aggfunc="first").reindex(vars_keep)
    lo = df.pivot_table(index="var", columns="tier", values="ci_low", aggfunc="first").reindex(vars_keep)
    hi = df.pivot_table(index="var", columns="tier", values="ci_high", aggfunc="first").reindex(vars_keep)
    stars = df.pivot_table(index="var", columns="tier", values="stars", aggfunc="first").reindex(vars_keep)

    y = np.arange(len(vars_keep))
    bar_h = 0.35
    offsets = np.linspace(-bar_h / 2, bar_h / 2, num=len(tiers))

    xmin = float(np.nanmin(df["ci_low"].to_numpy(dtype=float)))
    xmax = float(np.nanmax(df["ci_high"].to_numpy(dtype=float)))
    xrng = (xmax - xmin) if xmax > xmin else 1.0
    xpad = 0.30 * xrng
    dx = 0.06 * xrng

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axvline(0, linewidth=1, color="0.6")

    tier_colors = {t: palette[i % len(palette)] for i, t in enumerate(tiers)}

    for j, tier in enumerate(tiers):
        vals = wide[tier].to_numpy(dtype=float)
        lo_vals = lo[tier].to_numpy(dtype=float)
        hi_vals = hi[tier].to_numpy(dtype=float)
        st = stars[tier].to_numpy()

        yy = y + offsets[j]
        ax.barh(
            yy, vals,
            height=bar_h / max(len(tiers), 1) * 1.8,
            label=tier,
            alpha=0.90,
            color=tier_colors[tier],
        )

        xerr_low = vals - lo_vals
        xerr_hi = hi_vals - vals
        ax.errorbar(
            vals, yy,
            xerr=[xerr_low, xerr_hi],
            fmt="none",
            ecolor=whisker_color,
            linewidth=2,
            capsize=4,
        )

        yshift = 0.06 * (bar_h / max(len(tiers), 1) * 1.8)
        for yi, hi_i, star in zip(yy, hi_vals, st):
            if isinstance(star, str) and star:
                ax.text(hi_i + dx, yi + yshift, star, va="center", ha="left", fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_name(v) for v in vars_keep])
    ax.set_xlabel("coefficient (HC3 95% CI)")
    ax.set_title("WLS coefficients by tier (HC3)  (* p<0.05, ** p<0.01, *** p<0.001)")
    ax.legend(title="tier", frameon=False)
    ax.grid(True, axis="x", linewidth=0.8, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    plt.tight_layout()
    plt.show()


def plot_recipe_maps_3x2(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pairs=(
        ("cons8_median", "log_duration", "Consistency × Duration"),
        ("pace8_median", "log_duration", "Pace × Duration"),
        ("pace8_median", "cons8_median", "Pace × Consistency"),
    ),
    z_col: str = "residual",
    bins: int = 8,
    min_cell_n: int = 8,
    low_opacity: float = 0.30,
    q_low: float = 0.02,
    q_high: float = 0.98,
    scale_q: float | None = 0.98,
    title: str = "Heatmaps of efficiency residuals by structural factor pairs",
    cmap="coolwarm",
):
    """
    Plots 3x2 heatmaps of median z_col values over binned structural-factor pairs for Small and Mid tiers.

    Notes:
        - Uses global bin edges across both tiers so that grids align between Small and Mid.
        - For each (x, y) pair and tier, computes the median z_col per (xbin, ybin) cell.
        - Cells with 0 < N < min_cell_n are drawn with reduced opacity, while cells with N >= min_cell_n
          are drawn at full opacity.
        - Color scaling is symmetric around 0 (TwoSlopeNorm), using either:
            - max absolute value over reliable cells (scale_q is None), or
            - the scale_q quantile of absolute values over reliable cells (default 0.98).
        - Tick labels show bin centers (every second tick) for readability.

        Internal helper definitions used here:
            _bin_edges_global(s, bins, q_low, q_high):
                Computes bin edges using [q_low, q_high] quantiles with fallbacks to min/max.
            _centers_and_ranges(edges):
                Returns (centers, ranges) for successive edge pairs.
            _full_categories_from_edges(edges):
                Returns the full pd.CategoricalIndex produced by cutting bin centers with edges.
            _grid_fixed_edges(df, x, y, z, x_edges, y_edges):
                Returns (Z, N, x_centers, y_centers, x_ranges, y_ranges) where:
                    Z is the median grid, N is the count grid, and centers/ranges describe bins.
                Returns None if the tier data has no usable rows after numeric coercion and dropna.

    Args:
        small_df (pd.DataFrame): Data for the Small tier.
        mid_df (pd.DataFrame): Data for the Mid tier.
        pairs (tuple): Sequence of (x_col, y_col, label) specifying feature pairs and display label.
        z_col (str): Column name for the value aggregated into heatmap cells. Defaults to 'residual'.
        bins (int): Number of bins per axis. Defaults to 8.
        min_cell_n (int): Minimum per-cell count to treat a cell as reliable (full opacity).
            Defaults to 8.
        low_opacity (float): Alpha used for low-count cells. Defaults to 0.30.
        q_low (float): Lower quantile used for global binning. Defaults to 0.02.
        q_high (float): Upper quantile used for global binning. Defaults to 0.98.
        scale_q (float | None): Quantile of |z| over reliable cells used to set symmetric color limits.
            If None, uses the max |z| over reliable cells. Defaults to 0.98.
        title (str): Figure title.
        cmap: Matplotlib colormap passed to seaborn. Defaults to 'coolwarm'.

    Returns:
        None: Displays the plot via matplotlib.
    """
    # validate
    for dname, d in [("small_df", small_df), ("mid_df", mid_df)]:
        if z_col not in d.columns:
            raise KeyError(f"{dname} missing required column: {z_col}")
    for x, y, _ in pairs:
        for dname, d in [("small_df", small_df), ("mid_df", mid_df)]:
            if x not in d.columns or y not in d.columns:
                raise KeyError(f"{dname} missing feature columns: {x}, {y}")

    sns.set_theme(style="white")

    tiers = [("small", small_df), ("mid", mid_df)]

    # global bin edges across tiers so grids align
    all_df = pd.concat([small_df, mid_df], ignore_index=True)
    feats = sorted(set([x for x, _, _ in pairs] + [y for _, y, _ in pairs]))
    edges = {f: _bin_edges_global(all_df[f], bins=bins, q_low=q_low, q_high=q_high) for f in feats}

    # precompute grids + global scale from reliable cells
    grids = {}
    reliable_vals = []
    for tier_name, d in tiers:
        for x, y, pair_title in pairs:
            out = _grid_fixed_edges(d, x, y, z_col, edges[x], edges[y])
            grids[(tier_name, pair_title)] = out
            if out is None:
                continue
            Z, N, *_ = out
            Zr = np.where(N >= min_cell_n, Z, np.nan) if (min_cell_n and min_cell_n > 1) else Z
            v = Zr[np.isfinite(Zr)]
            if v.size:
                reliable_vals.append(v)

    if not reliable_vals:
        zmin, zmax = -0.1, 0.1
    else:
        vv = np.concatenate(reliable_vals)
        a = np.abs(vv)
        m = float(np.nanmax(a)) if scale_q is None else float(np.nanquantile(a, scale_q))
        if not np.isfinite(m) or m <= 0:
            m = 0.1
        zmin, zmax = -m, m

    norm = TwoSlopeNorm(vmin=zmin, vcenter=0.0, vmax=zmax)

    fig, axes = plt.subplots(len(pairs), 2, figsize=(11.2, 9.6))

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        top=0.92,
        bottom=0.12,
        wspace=0.08,
        hspace=0.42,
    )

    for r, (x, y, pair_title) in enumerate(pairs):
        for c, (tier_name, _) in enumerate(tiers):
            ax = axes[r, c]
            out = grids.get((tier_name, pair_title))

            x_label = pretty_name(x)
            y_label = pretty_name(y)

            if out is None:
                ax.set_title(f"{tier_name.title()} tier" if r == 0 else "")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            Z, N, x_centers, y_centers, *_ = out

            if min_cell_n and min_cell_n > 1:
                Z_hi = np.where((N >= min_cell_n) & np.isfinite(Z), Z, np.nan)
                Z_lo = np.where((N > 0) & (N < min_cell_n) & np.isfinite(Z), Z, np.nan)
            else:
                Z_hi = Z.copy()
                Z_lo = np.full_like(Z, np.nan)

            x_pos = np.arange(len(x_centers))
            y_pos = np.arange(len(y_centers))

            x_keep = np.arange(0, len(x_centers), 2, dtype=int)
            y_keep = np.arange(0, len(y_centers), 2, dtype=int)

            xticks = x_pos[x_keep]
            yticks = y_pos[y_keep]

            xticklabels = [f"{x_centers[i]:.2f}" for i in x_keep]
            yticklabels = [f"{y_centers[i]:.2f}" for i in y_keep]

            sns.heatmap(
                Z_hi,
                ax=ax,
                cmap=cmap,
                norm=norm,
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
                linewidths=0.0,
            )

            if np.isfinite(Z_lo).any():
                sns.heatmap(
                    Z_lo,
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    cbar=False,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    linewidths=0.0,
                    alpha=low_opacity,
                )

            if r == 0:
                ax.set_title(f"{tier_name.title()} tier")

            ax.set_xlabel(x_label, labelpad=6)
            ax.set_ylabel(y_label, labelpad=6)

            ax.set_xticks(xticks + 0.5)
            ax.set_yticks(yticks + 0.5)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
            ax.set_yticklabels(yticklabels, rotation=45, ha="right")

            if c == 1:
                ax.annotate(
                    pair_title,
                    xy=(1.06, 0.5),
                    xycoords="axes fraction",
                    rotation=-90,
                    va="center",
                    ha="left",
                    fontsize=13,
                    fontweight="bold",
                )

    # shared colorbar
    cax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"{z_col}\n0 = as expected")

    fig.suptitle(title, fontsize=16)
    plt.show()
